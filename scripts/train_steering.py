"""Train a DSRL-style noise-steering policy for the frozen pi0.5 DROID policy (served by openpi)."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import tyro
from isaaclab.app import AppLauncher

from polaris.config import PolicyArgs


@dataclass
class TrainArgs:
    environment: str
    run_folder: str
    policy: PolicyArgs
    headless: bool = True
    initial_conditions_file: str | None = None
    instruction: str | None = None
    noise_shape: tuple[int, int] = (16, 32)  # pi05 DROID (action_horizon, action_dim); verify vs. server
    encoder: str = "droid_proprio"  # droid_proprio | droid_vision
    vision_backbone: str = "facebook/dinov2-small"  # droid_vision only
    steer_horizon: int | None = None  # default: open_loop_horizon
    z_bound: float = 3.0
    gamma: float = 0.99
    success_bonus: float = 1.0
    total_chunks: int = 5000
    warmup_chunks: int = 200  # sample from the initial actor (~N(0,I)) before any update
    batch_size: int = 256
    updates_per_chunk: int = 1
    heldout: int = 5  # last N initial conditions are never used for training
    save_every: int = 500
    seed: int = 0


def main(args: TrainArgs):
    parser = argparse.ArgumentParser()
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = args.headless
    simulation_app = AppLauncher(args_cli).app

    from isaaclab_tasks.utils import parse_env_cfg
    from polaris.environments.manager_based_rl_splat_environment import ManagerBasedRLSplatEnv  # noqa: F401
    from polaris.latent_rl.algos import DSRLSAC
    from polaris.latent_rl.buffer import ReplayBuffer
    from polaris.latent_rl.checkpoint import SteeringConfig, save_checkpoint
    from polaris.latent_rl.encoders import make_encoder
    from polaris.latent_rl.steered_policy import SteeredPolicy
    from polaris.policy.droid_jointpos_client import DroidJointPosClient
    from polaris.policy.steered_client import ServerChunkPolicy
    from polaris.rl.chunk_env import ChunkEnv
    from polaris.utils import load_eval_initial_conditions

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    env_cfg = parse_env_cfg(args.environment, device="cuda", num_envs=1, use_fabric=True)
    env = gym.make(args.environment, cfg=env_cfg)
    instruction, conditions = load_eval_initial_conditions(env.usd_file, args.initial_conditions_file)
    instruction = args.instruction or instruction
    train_conditions = conditions[: max(1, len(conditions) - args.heldout)]
    print(f"{len(train_conditions)} train / {len(conditions) - len(train_conditions)} held-out initial conditions")

    client = DroidJointPosClient(args.policy)
    encoder_kwargs = {"backbone": args.vision_backbone} if args.encoder == "droid_vision" else {}
    encoder = make_encoder(args.encoder, **encoder_kwargs)
    steer_h = args.steer_horizon or client.open_loop_horizon
    cfg = SteeringConfig(
        noise_shape=args.noise_shape, steer_horizon=steer_h, feat_dim=encoder.feat_dim,
        encoder=args.encoder, encoder_kwargs=encoder_kwargs, z_bound=args.z_bound, base_policy=f"{args.policy.host}:{args.policy.port}",
    )
    algo = DSRLSAC(cfg.feat_dim, cfg.steer_dim, cfg.z_bound)
    buf = ReplayBuffer(cfg.feat_dim, cfg.steer_dim)
    # Same SteeredPolicy class as eval; training samples (deterministic=False).
    steered = SteeredPolicy(algo.actor, cfg, encoder, ServerChunkPolicy(client, args.noise_shape), deterministic=False)
    chunk_env = ChunkEnv(env, client, steered, instruction, args.gamma, args.success_bonus)

    out_dir = Path(args.run_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks, episode, ep_return, successes = 0, 0, 0.0, []
    chunk_env.reset(object_positions=train_conditions[rng.integers(len(train_conditions))])
    while chunks < args.total_chunks:
        t = chunk_env.step()
        buf.add(t.feat, t.z, t.reward, t.next_feat, t.done, t.discount)
        ep_return += t.reward
        chunks += 1
        metrics = {}
        if chunks >= args.warmup_chunks and len(buf) >= args.batch_size:
            for _ in range(args.updates_per_chunk):
                metrics = algo.update(buf.sample(args.batch_size))
        if t.episode_over:
            successes.append(float(t.info["rubric"]["success"]))
            episode += 1
            print(f"ep {episode} chunks {chunks} return {ep_return:.3f} success_ma10 {np.mean(successes[-10:]):.2f} {metrics}")
            ep_return = 0.0
            chunk_env.reset(object_positions=train_conditions[rng.integers(len(train_conditions))])
        if chunks % args.save_every == 0:
            save_checkpoint(out_dir / f"ckpt_{chunks}", algo.actor, cfg)
    save_checkpoint(out_dir / "ckpt_final", algo.actor, cfg)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main(tyro.cli(TrainArgs))
