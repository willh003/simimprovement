import numpy as np
import pytest
import torch

from polaris.latent_rl.algos import DSRLSAC
from polaris.latent_rl.checkpoint import SteeringConfig
from polaris.latent_rl.encoders import make_encoder
from polaris.latent_rl.steered_policy import SteeredPolicy
from polaris.rl.chunk_env import ChunkEnv
from polaris.rl.evaluation import evaluate
from polaris.rl.stats import EpisodeStats, MeanAccumulator, RollingMean

H = 4  # chunk length


class NoiseBase:
    noise_shape = (H, 2)

    def query(self, obs, noise=None):
        return noise.copy()


class FakeEnv:
    """Progress rises 0.25 per step; success after `success_at` steps, else time-out at `max_steps`."""

    def __init__(self, success_at=6, max_steps=10):
        self.success_at, self.max_steps = success_at, max_steps

    def _info(self):
        # 4 criteria reached at t=1..4 (if the episode is a "good" one), success once the last is reached at success_at
        n = min(4, self.t) if self.success_at else 0
        metrics = {f"reached/c{i}": float(i < n) for i in range(4)}
        return {"rubric": {"progress": n / 4, "success": bool(self.success_at) and self.t >= self.success_at, "metrics": metrics}}

    def reset(self, **kwargs):
        self.t = 0
        return {}, self._info()

    def step(self, action, expensive=True):
        self.t += 1
        return {}, None, [False], [self.t >= self.max_steps], self._info()


class FakeClient:
    open_loop_horizon = H

    def reset(self):
        pass

    def build_request(self, obs, instruction):
        request = {"observation/joint_position": np.zeros(7), "observation/gripper_position": np.zeros(1)}
        return request, np.zeros((8, 16, 3), np.uint8)

    @staticmethod
    def postprocess_action(action):
        return action


def make_chunk_env(env, deterministic=False):
    cfg = SteeringConfig(noise_shape=(H, 2), steer_horizon=H, feat_dim=8, encoder="droid_proprio")
    steered = SteeredPolicy(DSRLSAC(8, cfg.steer_dim).actor, cfg, make_encoder("droid_proprio"), NoiseBase(), deterministic)
    return ChunkEnv(env, FakeClient(), steered, "task", gamma=0.9, success_bonus=1.0)


def run_episode(chunk_env, record=False):
    chunk_env.reset()
    stats, frames = EpisodeStats(), []
    while True:
        t = chunk_env.step(record=record)
        stats.add(t)
        frames += t.frames
        assert t.reward == pytest.approx(sum(t.reward_terms.values()))
        if t.episode_over:
            return stats.summary(), frames


def test_reward_terms_sum_to_total_and_summarize():
    s, frames = run_episode(make_chunk_env(FakeEnv(success_at=6)))
    assert all(s[f"return/c{i}"] == pytest.approx(0.25) for i in range(4))
    assert s["return/success_bonus"] == pytest.approx(1.0) and s["return"] == pytest.approx(2.0)
    assert s["success/all"] == 1.0 and all(s[f"success/c{i}"] == 1.0 for i in range(4))
    assert s["length_chunks"] == 2 and s["length_steps"] == 6
    assert frames == []  # nothing recorded unless asked


def test_partial_progress_gives_partial_success():
    s, _ = run_episode(make_chunk_env(FakeEnv(success_at=0, max_steps=8)))
    assert s["success/all"] == 0.0 and s["return"] == 0.0

    env = FakeEnv(success_at=9, max_steps=8)  # reaches all 4 criteria by t=4 but never "succeeds"
    s, _ = run_episode(make_chunk_env(env))
    assert s["success/all"] == 0.0 and s["success/c3"] == 1.0 and s["return"] == pytest.approx(1.0)


def test_recording_yields_one_frame_per_state():
    s, frames = run_episode(make_chunk_env(FakeEnv(success_at=0, max_steps=10)), record=True)
    assert s["success/all"] == 0.0 and s["success/c0"] == 0.0 and s["length_steps"] == 10
    assert len(frames) == 11  # initial state + one per env step


def test_evaluate_aggregates_and_records():
    metrics, videos = evaluate(make_chunk_env(FakeEnv(success_at=6), deterministic=True), [{}, {}, {}], n_videos=2)
    assert metrics["success/all"] == 1.0 and metrics["success/c3"] == 1.0 and metrics["return"] == pytest.approx(2.0)
    assert len(videos) == 2 and len(videos[0]) == 7


def test_mean_accumulator():
    acc = MeanAccumulator()
    acc.add({"a": 1.0, "b": torch.tensor(2.0)})
    acc.add({"a": 3.0})
    assert acc.pop_means() == {"a": 2.0, "b": 2.0}
    assert acc.pop_means() == {}


def test_wandb_logger_offline(tmp_path):
    from polaris.rl.wandb_logger import WandbLogger

    logger = WandbLogger("test", tmp_path, {"x": 1}, mode="offline")
    logger.log({"loss": 1.0}, step=0, prefix="train/")
    logger.log_video("eval/heldout/video_0", [np.zeros((16, 16, 3), np.uint8)] * 5, step=0)
    logger.finish()
    assert (tmp_path / "videos" / "eval_heldout_video_0_0.mp4").exists()


def test_rolling_mean():
    r = RollingMean(window=2)
    r.add({"a": 1.0})
    r.add({"a": 0.0})
    assert r.add({"a": 0.0}) == {"a": 0.0}


class FakeRewardManager:
    def __init__(self, env):
        self.env = env

    def get_active_iterable_terms(self, idx):
        # sim-evals scales these differently (x15) from the env reward
        return [("can_in_mug", [15.0 * self.env.in_mug]), ("can_near_mug", [15.0 * 0.25 * self.env.near])]


class FakeSimEvalsEnv:
    """Hover (near) for steps 2-3, success (in mug) from step 4, time-out at 10."""

    def __init__(self):
        self.unwrapped = self
        self.reward_manager = FakeRewardManager(self)

    def reset(self, **kwargs):
        self.t = 0
        return {}, {}

    def step(self, action):
        self.t += 1
        self.in_mug, self.near = float(self.t >= 4), float(2 <= self.t < 4)
        return {}, torch.tensor([self.in_mug + 0.25 * self.near]), [False], [self.t >= 10], {}


def test_sim_evals_task_terms_success_and_stats():
    from polaris.rl.tasks import SimEvalsTask

    chunk_env = make_chunk_env(FakeEnv())
    chunk_env.env, chunk_env.task, chunk_env.success_bonus = FakeSimEvalsEnv(), SimEvalsTask(), 0.0
    s, _ = run_episode(chunk_env)
    assert s["return/can_in_mug"] == pytest.approx(1.0) and s["return/can_near_mug"] == pytest.approx(0.5)
    assert s["success/all"] == 1.0 and s["success/can_near_mug"] == 1.0 and s["length_steps"] == 4
    assert "final_progress" not in s
