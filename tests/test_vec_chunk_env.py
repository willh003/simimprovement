import numpy as np
import pytest
import torch

from polaris.latent_rl.checkpoint import SteeringConfig
from polaris.latent_rl.encoders import make_encoder
from polaris.policy.droid_jointpos_client import SimEvalsJointPosClient
from polaris.rl.evaluation import evaluate_vec
from polaris.rl.stats import EpisodeStats
from polaris.rl.tasks import VecSimEvalsTask
from polaris.rl.vec_chunk_env import VecChunkEnv

H = 4


class FakeRewardManager:
    active_terms = ["in_bin", "near_bin"]

    def __init__(self, n):
        self._step_reward = torch.zeros(n, 2)


class FakeVecEnv:
    """N auto-resetting envs. Env i: 'near' one step before success at step success_at[i] (None: never), time-out at max_steps.

    Like Isaac: the reward manager buffer holds term / step_dt, and returned obs are post-reset for envs that ended.
    """

    step_dt = 0.5
    device = "cpu"

    def __init__(self, success_at=(2, None, 4), max_steps=6):
        self.unwrapped = self
        self.success_at, self.max_steps, self.n = success_at, max_steps, len(success_at)
        self.reward_manager = FakeRewardManager(self.n)
        self.t = np.zeros(self.n, int)

    def _obs(self):
        joint = torch.tensor(100.0 + self.t, dtype=torch.float32)[:, None].repeat(1, 7)
        img = torch.zeros(self.n, 8, 8, 3, dtype=torch.uint8)
        return {"policy": {"arm_joint_pos": joint, "gripper_pos": torch.zeros(self.n, 1), "external_cam": img, "wrist_cam": img}}

    def reset(self):
        self.t[:] = 0
        return self._obs(), {}

    def step(self, action):
        self.last_action = action.numpy().copy()
        self.t += 1
        s = np.array([a if a is not None else -10 for a in self.success_at])
        in_bin, near = (self.t == s).astype(float), (self.t == s - 1).astype(float)
        self.reward_manager._step_reward = torch.tensor(np.stack([in_bin, 0.25 * near], 1) / self.step_dt, dtype=torch.float32)
        term, trunc = in_bin > 0, (self.t >= self.max_steps) & ~(in_bin > 0)
        self.t[term | trunc] = 0  # auto-reset
        rew = torch.tensor((in_bin + 0.25 * near), dtype=torch.float32)
        return self._obs(), rew, torch.tensor(term), torch.tensor(trunc), {}


class CountingTask(VecSimEvalsTask):
    def __init__(self):
        super().__init__("in_bin")
        self.refreshes = 0

    def refresh_obs(self, env):
        self.refreshes += 1
        return env._obs()


class GripperClosedBase:
    """Chunk = noise's first 8 dims with gripper 0.9 (-> closed after postprocessing)."""

    noise_shape = (H, 8)

    def query_batch(self, obs_list, noise=None):
        chunk = noise.copy()
        chunk[..., -1] = 0.9
        return chunk


def make_venv(**kw):
    client = SimEvalsJointPosClient.__new__(SimEvalsJointPosClient)  # request building only, no server
    client.open_loop_horizon = H
    cfg = SteeringConfig(noise_shape=(H, 8), steer_horizon=2, feat_dim=8, encoder="droid_proprio")
    env, task = FakeVecEnv(**kw), CountingTask()
    return VecChunkEnv(env, client, GripperClosedBase(), make_encoder("droid_proprio"), cfg, "task", 0.9, 0.0, task), env, task


def test_chunk_masks_ended_envs_and_holds():
    venv, env, task = make_venv()
    feat = venv.reset()
    assert feat.shape == (3, 8) and venv.num_envs == 3
    t = venv.step(np.zeros((3, 16), np.float32), record=2)
    assert t.steps.tolist() == [2, 4, 4]
    assert t.discount == pytest.approx(0.9 ** np.array([2, 4, 4]))
    assert t.reward == pytest.approx([1.25, 0.0, 1.25])  # terms rescaled by step_dt to the env reward
    assert t.reward_terms["in_bin"] == pytest.approx([1.0, 0.0, 1.0])
    assert t.terminated.tolist() == [True, False, True] and t.success.tolist() == [True, False, True]
    assert not t.truncated.any()
    assert env.last_action[0, -1] == 0.0 and env.last_action[1, -1] == 1.0  # env 0 holds (gripper open) after success
    assert np.allclose(env.last_action[0, :7], 100.0 + 1)  # hold = the new episode's current joint pose
    assert task.refreshes == 1  # env 2 reset on the last step of the chunk
    assert len(t.frames) == 2 and len(t.frames[0]) == 2 and len(t.frames[1]) == 4
    assert t.next_feat[2, 0] == 100.0  # next feat of env 2 is the fresh episode

    t2 = venv.step(np.zeros((3, 16), np.float32))
    assert t2.truncated.tolist() == [False, True, False] and t2.steps[1] == 2 and not t2.terminated[1]

    s = EpisodeStats()
    s.add(t.unbatch(0))
    summary = s.summary()
    assert summary["success/all"] == 1.0 and summary["return"] == pytest.approx(1.25)
    assert summary["success/near_bin"] == 1.0 and summary["length_steps"] == 2


def test_evaluate_vec():
    venv, _, _ = make_venv()
    metrics, videos = evaluate_vec(venv, lambda f: np.zeros((len(f), 16), np.float32), n_videos=2)
    assert metrics["episodes"] == 3
    assert metrics["success/all"] == pytest.approx(2 / 3)
    assert metrics["length_steps"] == pytest.approx((2 + 6 + 4) / 3)
    assert len(videos) == 2 and len(videos[0]) == 2 and len(videos[1]) == 6
