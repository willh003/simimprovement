"""Train a DSRL-style noise-steering policy for the frozen pi0.5 DROID policy (served by openpi)."""

import argparse
import time
import uuid
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import tyro
from isaaclab.app import AppLauncher

from polaris.config import PolicyArgs


@dataclass
class TrainArgs:
    environment: str  # e.g. DROID-FoodBussing (splat + rubric) or {CubeBowl,CanMug,BananaBin} (plain render, no splat)
    policy: PolicyArgs
    headless: bool = True
    initial_conditions_file: str | None = None
    instruction: str | None = None
    noise_shape: tuple[int, int] = (15, 32)  # pi05_droid_jointpos_polaris (action_horizon=15, action_dim=32)
    encoder: str = "droid_proprio"  # droid_proprio | droid_vision
    vision_backbone: str = "facebook/dinov2-small"  # droid_vision only
    steer_horizon: int | None = None  # default: open_loop_horizon
    z_bound: float = 3.0
    gamma: float = 0.99
    success_bonus: float | None = None  # default: 1.0 for rubric envs, 0.0 for sim-evals (its success term already pays 1)
    total_chunks: int = 5000
    warmup_chunks: int = 200  # sample from the initial actor (~N(0,I)) before any update
    batch_size: int = 256
    updates_per_chunk: int = 1
    heldout: int = 5  # last N initial conditions are never used for training
    save_every: int = 500
    seed: int = 0
    wandb_project: str = "polaris-steering"
    wandb_entity: str | None = None  # default: your wandb default entity
    runs_root: str = "runs/steering"  # run folder: <runs_root>/<environment>-DDMM-HHMM-<uuid6>; same string is the wandb run name
    wandb_mode: str = "online"  # online | offline | disabled
    log_every: int = 10  # chunks between train/* logs (update metrics are averaged over the interval)
    eval_every: int = 500  # chunks; eval runs at the next episode boundary
    eval_at_start: bool = True
    eval_train_conditions: int = 5  # first N train ICs also evaluated (eval/train/*) next to eval/heldout/*; sim-evals: episodes in eval/fixed/*
    eval_videos: int = 2  # recorded episodes per eval split (splat rendered every step)


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
    from polaris.policy.droid_jointpos_client import DroidJointPosClient, SimEvalsJointPosClient
    from polaris.policy.steered_client import ServerChunkPolicy
    from polaris.rl.chunk_env import ChunkEnv
    from polaris.rl.evaluation import evaluate
    from polaris.rl.tasks import RubricSplatTask, SimEvalsTask, SIM_EVALS_INSTRUCTIONS, SIM_EVALS_SUCCESS_TERMS, is_sim_evals
    from polaris.rl.stats import EpisodeStats, MeanAccumulator, RollingMean
    from polaris.rl.wandb_logger import WandbLogger
    from polaris.utils import load_eval_initial_conditions

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    env_cfg = parse_env_cfg(args.environment, device="cuda", num_envs=1, use_fabric=True)
    env = gym.make(args.environment, cfg=env_cfg)
    if is_sim_evals(args.environment):
        # fixed layout (no initial-condition randomization): every episode resets to the default scene
        task, client_cls = SimEvalsTask(SIM_EVALS_SUCCESS_TERMS[args.environment]), SimEvalsJointPosClient
        instruction = args.instruction or SIM_EVALS_INSTRUCTIONS[args.environment]
        train_conditions = [{}]
        eval_splits = {"fixed": [{}] * args.eval_train_conditions}
        print("sim-evals env: single fixed layout")
    else:
        task, client_cls = RubricSplatTask(), DroidJointPosClient
        instruction, conditions = load_eval_initial_conditions(env.usd_file, args.initial_conditions_file)
        instruction = args.instruction or instruction
        train_conditions = conditions[: max(1, len(conditions) - args.heldout)]
        eval_splits = {"heldout": conditions[len(train_conditions) :], "train": train_conditions[: args.eval_train_conditions]}
        eval_splits = {name: conds for name, conds in eval_splits.items() if conds}
        print(f"{len(train_conditions)} train / {len(conditions) - len(train_conditions)} held-out initial conditions")
    success_bonus = task.default_success_bonus if args.success_bonus is None else args.success_bonus

    client = client_cls(args.policy)
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
    chunk_env = ChunkEnv(env, client, steered, instruction, args.gamma, success_bonus, task)
    # Eval: same dynamics, actor mean (matches the Steered client with deterministic=True).
    eval_steered = SteeredPolicy(algo.actor, cfg, encoder, steered.base, deterministic=True)
    eval_env = ChunkEnv(env, client, eval_steered, instruction, args.gamma, success_bonus, task)

    run_name = f"{args.environment}-{datetime.now():%d%m-%H%M}-{uuid.uuid4().hex[:6]}"
    out_dir = Path(args.runs_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"run folder / wandb run name: {out_dir}")
    logger = WandbLogger(args.wandb_project, out_dir, asdict(args), args.wandb_entity, run_name, args.wandb_mode)

    def run_eval(step: int):
        for split, conds in eval_splits.items():
            metrics, videos = evaluate(eval_env, conds, args.eval_videos)
            logger.log(metrics, step, prefix=f"eval/{split}/")
            for i, frames in enumerate(videos):
                logger.log_video(f"eval/{split}/video_{i}", frames, step)
            print(f"eval/{split} @ {step}: {metrics}")

    def reset_train():
        chunk_env.reset(object_positions=train_conditions[rng.integers(len(train_conditions))])
        ep_stats.reset()

    chunks, env_steps, episode = 0, 0, 0
    ep_stats, update_acc, recent = EpisodeStats(), MeanAccumulator(), RollingMean(window=10)
    next_eval = args.eval_every
    if args.eval_at_start:
        run_eval(0)
    reset_train()
    t_log, chunks_log = time.perf_counter(), 0
    while chunks < args.total_chunks:
        t = chunk_env.step()
        buf.add(t.feat, t.z, t.reward, t.next_feat, t.done, t.discount)
        ep_stats.add(t)
        chunks += 1
        env_steps += t.steps
        if chunks >= args.warmup_chunks and len(buf) >= args.batch_size:
            for _ in range(args.updates_per_chunk):
                update_acc.add(algo.update(buf.sample(args.batch_size)))
        if chunks % args.log_every == 0:
            now = time.perf_counter()
            perf = dict(buffer_size=len(buf), env_steps=env_steps, chunks_per_sec=(chunks - chunks_log) / (now - t_log))
            logger.log({**update_acc.pop_means(), **perf}, chunks, prefix="train/")
            t_log, chunks_log = now, chunks
        if t.episode_over:
            ep_metrics = ep_stats.summary()
            episode += 1
            success_ma10 = recent.add({k: v for k, v in ep_metrics.items() if k.startswith("success/")})
            logger.log({**ep_metrics, **{f"{k}_ma10": v for k, v in success_ma10.items()}, "episode": episode}, chunks, prefix="rollout/")
            print(f"ep {episode} chunks {chunks} return {ep_metrics['return']:.3f} success_ma10 {success_ma10['success/all']:.2f}")
            if chunks >= next_eval:
                t_eval = time.perf_counter()
                run_eval(chunks)
                t_log += time.perf_counter() - t_eval  # keep eval time out of chunks_per_sec
                next_eval += args.eval_every * ((chunks - next_eval) // args.eval_every + 1)
            reset_train()
        if chunks % args.save_every == 0:
            save_checkpoint(out_dir / f"ckpt_{chunks}", algo.actor, cfg)
    run_eval(chunks)
    save_checkpoint(out_dir / "ckpt_final", algo.actor, cfg)
    logger.finish()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main(tyro.cli(TrainArgs))
