"""Train a DSRL-style noise-steering policy for the frozen pi0.5 DROID policy (served by openpi) with PPO on many
parallel sim-evals envs (CubeBowl / CanMug / BananaBin).

All envs query the openpi server in one batched `infer_batch` call per chunk (docs/claude/openpi_batching.md) and run
the chunk in lockstep; cameras are tiled (docs/claude/parallel_envs_bench.md). See docs/claude/steering_training.md.

  isaacpy scripts/train_steering.py --environment CubeBowl --num-envs 128 --policy.port 8000
  isaacpy scripts/train_steering.py --environment CubeBowl --num-envs 8 --dummy-policy --wandb-mode disabled  # no server
"""

import argparse
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import tyro

from polaris.config import PolicyArgs
from polaris.latent_rl.algos import PPOConfig


@dataclass
class TrainArgs:
    environment: str  # sim-evals task: CubeBowl | CanMug | BananaBin
    policy: PolicyArgs
    num_envs: int = 128
    cam_scale: float = 0.5  # camera resolution multiplier (0.5 = 640x360; 0.25 hurts the base policy, see server_and_resolution_bench.md)
    server_batch_size: int | None = None  # requests per infer_batch call (None: all envs at once); keep fixed, JAX recompiles per size
    headless: bool = True
    instruction: str | None = None
    noise_shape: tuple[int, int] = (15, 32)  # pi05_droid_jointpos_polaris (action_horizon=15, action_dim=32)
    encoder: str = "droid_proprio"  # droid_proprio | droid_vision
    vision_backbone: str = "facebook/dinov2-small"  # droid_vision only
    steer_horizon: int | None = None  # default: open_loop_horizon
    z_bound: float = 3.0
    gamma: float = 0.99  # per env step; a chunk of k steps is discounted by gamma^k
    success_bonus: float = 0.0  # the success reward term already pays 1
    ppo: PPOConfig = field(default_factory=PPOConfig)
    rl_device: str = "cuda"  # PPO nets + rollout buffer (CPU updates are ~100x slower next to Isaac)
    rollout_chunks: int = 8  # chunks per env per PPO iteration (batch = num_envs * rollout_chunks transitions)
    iterations: int = 300
    save_every: int = 25  # iterations
    eval_every_min: float = 20.0  # wall-clock minutes of training between evals (checked after each PPO iteration, so it
    # overshoots by up to one iteration); eval = one deterministic episode in every env. Independent of num_envs/rollout_chunks
    eval_at_start: bool = True
    eval_videos: int = 2
    dummy_policy: bool = False  # no server: base policy holds the arm pose (smoke test of env + RL plumbing)
    seed: int = 0
    wandb_project: str = "polaris-steering"
    wandb_entity: str | None = None  # default: your wandb default entity
    runs_root: str = "runs/steering"  # run folder: <runs_root>/<environment>-DDMM-HHMM-<uuid6>; same string is the wandb run name
    wandb_mode: str = "online"  # online | offline | disabled


class HoldPoseBase:
    """BatchChunkPolicy stand-in for --dummy-policy: every chunk holds the request's joint pose, gripper open."""

    def __init__(self, noise_shape: tuple[int, int]):
        self.noise_shape = noise_shape

    def query_batch(self, obs_list, noise=None):
        joint = np.stack([np.concatenate([r["observation/joint_position"], np.zeros(1)]) for r in obs_list])
        return np.repeat(joint[:, None], self.noise_shape[0], axis=1)


def main(args: TrainArgs):
    parser = argparse.ArgumentParser()
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = args.headless
    from isaaclab.app import AppLauncher

    AppLauncher(args_cli)

    from polaris.environments.simeval_parallel import make_parallel_env
    from polaris.latent_rl.algos import DSRLPPO
    from polaris.latent_rl.checkpoint import SteeringConfig, save_checkpoint
    from polaris.latent_rl.encoders import make_encoder
    from polaris.latent_rl.rollout import RolloutBuffer
    from polaris.policy.droid_jointpos_client import SimEvalsJointPosClient
    from polaris.policy.steered_client import ServerChunkPolicy
    from polaris.rl.evaluation import evaluate_vec
    from polaris.rl.stats import EpisodeStats, MeanAccumulator, RollingMean
    from polaris.rl.tasks import SIM_EVALS_INSTRUCTIONS, SIM_EVALS_SUCCESS_TERMS, VecSimEvalsTask, is_sim_evals
    from polaris.rl.vec_chunk_env import VecChunkEnv
    from polaris.rl.wandb_logger import WandbLogger

    if not is_sim_evals(args.environment):
        raise ValueError(f"parallel training supports the sim-evals tasks {list(SIM_EVALS_INSTRUCTIONS)}, got {args.environment}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    success_term = SIM_EVALS_SUCCESS_TERMS[args.environment]
    env, _ = make_parallel_env(args.environment, args.num_envs, args.cam_scale, success_term=success_term)
    instruction = args.instruction or SIM_EVALS_INSTRUCTIONS[args.environment]

    if args.dummy_policy:
        client = SimEvalsJointPosClient.__new__(SimEvalsJointPosClient)  # request building only, no server connection
        client.args, client.open_loop_horizon = args.policy, args.policy.open_loop_horizon
        base = HoldPoseBase(args.noise_shape)
    else:
        client = SimEvalsJointPosClient(args.policy)
        base = ServerChunkPolicy(client, args.noise_shape, args.server_batch_size)
    encoder_kwargs = {"backbone": args.vision_backbone} if args.encoder == "droid_vision" else {}
    encoder = make_encoder(args.encoder, **encoder_kwargs)
    cfg = SteeringConfig(
        noise_shape=args.noise_shape, steer_horizon=args.steer_horizon or client.open_loop_horizon, feat_dim=encoder.feat_dim,
        encoder=args.encoder, encoder_kwargs=encoder_kwargs, z_bound=args.z_bound, hidden=args.ppo.hidden,
        base_policy="dummy" if args.dummy_policy else f"{args.policy.host}:{args.policy.port}",
    )
    ppo = DSRLPPO(cfg.feat_dim, cfg.steer_dim, cfg.z_bound, args.ppo, args.rl_device)
    venv = VecChunkEnv(env, client, base, encoder, cfg, instruction, args.gamma, args.success_bonus, VecSimEvalsTask(success_term))
    buf = RolloutBuffer(args.rollout_chunks, args.num_envs, cfg.feat_dim, cfg.steer_dim, args.rl_device)

    run_name = f"{args.environment}-{datetime.now():%d%m-%H%M}-{uuid.uuid4().hex[:6]}"
    out_dir = Path(args.runs_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"run folder / wandb run name: {out_dir}")
    logger = WandbLogger(args.wandb_project, out_dir, asdict(args), args.wandb_entity, run_name, args.wandb_mode)

    def save(name: str):
        save_checkpoint(out_dir / name, ppo.actor, cfg)  # loadable by SteeredPolicy / the Steered client
        torch.save(ppo.state_dict(), out_dir / name / "ppo.pt")

    def run_eval(step: int):
        t0 = time.perf_counter()
        metrics, videos = evaluate_vec(venv, lambda f: ppo.act(f, deterministic=True)[0], args.eval_videos)
        metrics["time_s"] = time.perf_counter() - t0
        logger.log(metrics, step, prefix="eval/fixed/")
        for i, frames in enumerate(videos):
            logger.log_video(f"eval/fixed/video_{i}", frames, step)
        print(f"eval @ {step}: success {metrics['success/all']:.3f}, return {metrics['return']:.3f} ({metrics['time_s']:.0f}s)")

    transitions = env_steps = episodes = 0
    evaled_at_it = -1
    recent = RollingMean(window=args.num_envs)
    if args.eval_at_start:
        run_eval(0)
    last_eval_end = time.perf_counter()
    feat = venv.reset()
    ep_stats = [EpisodeStats() for _ in range(args.num_envs)]
    for it in range(1, args.iterations + 1):
        t0 = time.perf_counter()
        buf.reset()
        finished, n_finished, success_ma = MeanAccumulator(), 0, {}
        for _ in range(args.rollout_chunks):
            z, logp, value = ppo.act(feat)
            t = venv.step(z)
            buf.add(feat, z, logp, value, t.reward, t.discount, t.terminated, t.episode_over, t.truncated)
            for i in range(args.num_envs):
                ep_stats[i].add(t.unbatch(i))
                if t.episode_over[i]:
                    summary = ep_stats[i].summary()
                    finished.add(summary)
                    n_finished += 1
                    success_ma = recent.add({k: v for k, v in summary.items() if k.startswith("success/")})
                    ep_stats[i].reset()
                    episodes += 1
            feat = t.next_feat
            env_steps += int(t.steps.sum())
        transitions += len(buf)
        t_rollout = time.perf_counter() - t0
        buf.compute_returns(ppo.value(feat), args.ppo.gae_lambda)
        train_metrics = ppo.update(buf)
        t_update = time.perf_counter() - t0 - t_rollout
        perf = dict(
            iteration=it, env_steps=env_steps, episodes=episodes, transitions_per_sec=len(buf) / (t_rollout + t_update),
            rollout_s=t_rollout, update_s=t_update,
        )
        logger.log({**train_metrics, **perf}, transitions, prefix="train/")
        if n_finished:  # means over the episodes that ended during this iteration
            ep_metrics = {**finished.pop_means(), "episodes_finished": n_finished}
            logger.log({**ep_metrics, **{f"{k}_ma": v for k, v in success_ma.items()}}, transitions, prefix="rollout/")
        print(
            f"it {it} transitions {transitions} ({perf['transitions_per_sec']:.1f}/s) episodes {episodes} "
            f"success_ma {success_ma.get('success/all', float('nan')):.3f} kl {train_metrics.get('approx_kl', 0):.4f}"
        )
        if time.perf_counter() - last_eval_end >= args.eval_every_min * 60:
            run_eval(transitions)
            last_eval_end = time.perf_counter()
            evaled_at_it = it
            feat = venv.reset()  # eval used the same envs; on-policy, so only the partial episodes are dropped
            for s in ep_stats:
                s.reset()
        if it % args.save_every == 0:
            save(f"ckpt_{it}")
    if evaled_at_it != args.iterations:
        run_eval(transitions)
    save("ckpt_final")
    logger.finish()
    sys.stdout.flush()
    os._exit(0)  # Isaac hangs on normal shutdown


if __name__ == "__main__":
    main(tyro.cli(TrainArgs))
