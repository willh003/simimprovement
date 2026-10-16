"""Train a DSRL-style noise-steering policy for the frozen pi0.5 DROID policy (served by openpi) with PPO on many
parallel sim-evals envs (CubeBowl / CanMug / BananaBin).

All envs query the openpi server in one batched `infer_batch` call per chunk (docs/claude/current/openpi_server.md) and run
the chunk in lockstep; cameras are tiled (docs/claude/experiments/env_scaling_bench.md). See docs/claude/current/steering_training.md.

  isaacpy scripts/train_steering.py --environment CubeBowl --num-envs 128 --policy.port 8000
  isaacpy scripts/train_steering.py --environment CubeBowl --num-envs 8 --dummy-policy --wandb-mode disabled  # no server
"""

import copy
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
    environment: str  # sim-evals task: CubeBowl | CanMug | BananaBin | MultiTask (all three at once, envs split evenly)
    policy: PolicyArgs
    num_envs: int = 128
    cam_scale: float = 0.5  # camera resolution multiplier (0.5 = 640x360; 0.25 hurts the base policy, see docs/claude/experiments/openpi_server_and_resolution.md)
    server_batch_size: int | None = None  # requests per infer_batch call (None: all envs at once); keep fixed, JAX recompiles per size
    headless: bool = True
    instruction: str | None = None
    noise_shape: tuple[int, int] = (15, 32)  # pi05_droid_jointpos_polaris (action_horizon=15, action_dim=32)
    encoder: str = "droid_resnet50"  # droid_resnet50 | droid_vision (dinov2) | droid_proprio
    dino_backbone: str = "facebook/dinov2-small"  # HF model id; only used when encoder == droid_vision
    finetune_encoder: bool = True  # droid_resnet50 / droid_vision: train the backbone end-to-end with PPO (--no-finetune-encoder: frozen, features cached)
    residual_scale: float = 0.01  # radians: the actor also outputs r in (-1,1) per arm joint per executed step (tanh of a
    # Gaussian), added as residual_scale * r to the absolute joint targets. 0 disables it (plain DSRL).
    privileged: bool = True  # actor + critic also see ground-truth sim state (object pose/velocity, goal = container pose); the
    # base policy never does. --no-privileged: image/proprio features only. Checkpoints trained with it need the state at deployment
    steer_horizon: int | None = None  # default: open_loop_horizon
    broadcast_noise: bool = True  # the actor predicts one noise vector (action_dim), broadcast over the steered rows of the
    # chunk noise (the first steer_horizon rows; the rest stay N(0, I)). False: a separate noise per steered row
    z_bound: float = 3.0
    gamma: float = 0.99  # per env step; a chunk of k steps is discounted by gamma^k
    success_bonus: float = 0.0  # the success reward term already pays 1
    ppo: PPOConfig = field(default_factory=PPOConfig)
    critic_warmup_iters: int = 0  # leading iterations that only fit the critic (actor frozen; rollouts use the initial ~N(0, I) actor)
    warmup_train_encoder: bool = False  # also train the finetuned encoder with the value loss during warmup (default: frozen)
    warmup_min_explained_var: float | None = None  # end warmup early once the critic's explained_var exceeds this
    rl_device: str = "cuda"  # PPO nets + rollout buffer (CPU updates are ~100x slower next to Isaac)
    rollout_chunks: int = 8  # chunks per env per PPO iteration (batch = num_envs * rollout_chunks transitions)
    iterations: int = 300  # total, so with --resume the run continues from the checkpoint's iteration up to this
    resume: str | None = None  # checkpoint dir of an earlier run (<run>/ckpt_N): restores actor / critic / optimizer (/ encoder) and starts at iteration N
    # (new run folder + wandb run; the rollout buffer, env state and RNG are not restored, the config must match the checkpoint's)
    save_every: int = 25  # iterations
    eval_every_min: float = 30.0  # wall-clock minutes of training between evals (checked after each PPO iteration, so it
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


def find_nonfinite(**arrays) -> dict[str, np.ndarray]:
    """{name: env indices} of the (N, ...) arrays with a NaN/inf in that env's row."""
    out = {}
    for name, a in arrays.items():
        if a is not None:
            bad = np.flatnonzero(~np.isfinite((a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)).astype(np.float64).reshape(len(a), -1)).all(-1))
            if len(bad):
                out[name] = bad
    return out


def main(args: TrainArgs):
    from polaris.launch import launch_isaac

    launch_isaac(args.headless)

    from polaris.environments.simeval_parallel import make_parallel_env
    from polaris.latent_rl.algos import DSRLPPO
    from polaris.latent_rl.checkpoint import SteeringConfig, save_checkpoint
    from polaris.latent_rl.encoders import make_encoder
    from polaris.latent_rl.rollout import RolloutBuffer
    from polaris.policy.droid_jointpos_client import SimEvalsJointPosClient
    from polaris.policy.steered_client import ServerChunkPolicy
    from polaris.rl.evaluation import evaluate_vec
    from polaris.rl.privileged import MultiTaskPrivileged, SimEvalsPrivileged
    from polaris.rl.stats import EpisodeStats, MeanAccumulator, RollingMean
    from polaris.rl.tasks import MULTITASK_ENVS, SIM_EVALS_INSTRUCTIONS, SIM_EVALS_SUCCESS_TERMS, VecSimEvalsTask, is_sim_evals
    from polaris.rl.vec_chunk_env import VecChunkEnv
    from polaris.rl.wandb_logger import WandbLogger

    if not is_sim_evals(args.environment):
        raise ValueError(f"parallel training supports the sim-evals tasks {list(SIM_EVALS_INSTRUCTIONS)}, got {args.environment}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    success_term = SIM_EVALS_SUCCESS_TERMS[args.environment]
    env, _ = make_parallel_env(args.environment, args.num_envs, args.cam_scale, success_term=success_term)
    tasks = MULTITASK_ENVS.get(args.environment)  # task names of a multitask env, else None
    multitask = tasks is not None
    if multitask:
        from polaris.environments.simeval_droid import task_ids

        if args.num_envs % len(tasks):
            print(f"WARNING: --num-envs {args.num_envs} is not divisible by {len(tasks)}: the task split is uneven (off by one env)")
        tid = task_ids(env.unwrapped, len(tasks)).cpu().numpy()
        instruction = [SIM_EVALS_INSTRUCTIONS[tasks[k]] for k in tid]  # one prompt per env
        task_envs = {name: np.flatnonzero(tid == k) for k, name in enumerate(tasks)}
        print(f"MultiTask: envs per task {({n: len(i) for n, i in task_envs.items()})}")
    else:
        instruction = args.instruction or SIM_EVALS_INSTRUCTIONS[args.environment]
        tid, task_envs = None, None

    if args.dummy_policy:
        client = SimEvalsJointPosClient.__new__(SimEvalsJointPosClient)  # request building only, no server connection
        client.args, client.open_loop_horizon = args.policy, args.policy.open_loop_horizon
        base = HoldPoseBase(args.noise_shape)
    else:
        client = SimEvalsJointPosClient(args.policy)
        base = ServerChunkPolicy(client, args.noise_shape, args.server_batch_size)
    priv_fn = (MultiTaskPrivileged if multitask else SimEvalsPrivileged)(env, success_term) if args.privileged else None
    image_encoder = args.encoder in ("droid_resnet50", "droid_vision")
    finetune = args.finetune_encoder and image_encoder
    encoder_kwargs = {"finetune": finetune} if image_encoder else {}
    if args.encoder == "droid_vision":
        encoder_kwargs["backbone"] = args.dino_backbone
    encoder = make_encoder(args.encoder, **encoder_kwargs, **({"device": args.rl_device} if image_encoder else {}))
    cfg = SteeringConfig(
        noise_shape=args.noise_shape, steer_horizon=args.steer_horizon or client.open_loop_horizon, feat_dim=encoder.feat_dim,
        encoder=args.encoder, encoder_kwargs=encoder_kwargs, z_bound=args.z_bound, hidden=args.ppo.hidden, broadcast_noise=args.broadcast_noise,
        base_policy="dummy" if args.dummy_policy else f"{args.policy.host}:{args.policy.port}",
        residual_scale=args.residual_scale, residual_horizon=client.open_loop_horizon,
        priv_dim=priv_fn.dim if priv_fn else 0,
    )
    ppo = DSRLPPO(cfg.feat_dim, cfg.steer_dim, cfg.z_bound, args.ppo, args.rl_device, cfg.res_dim, encoder if finetune else None, cfg.priv_dim)
    venv = VecChunkEnv(env, client, base, encoder, cfg, instruction, args.gamma, args.success_bonus, VecSimEvalsTask(success_term), priv_fn)
    buf = RolloutBuffer(
        args.rollout_chunks, args.num_envs, cfg.feat_dim, cfg.act_dim, args.rl_device, img_shape=(2, 224, 224, 3) if finetune else None,
        priv_dim=cfg.priv_dim,
    )

    run_name = f"{args.environment}-{datetime.now():%d%m-%H%M}-{uuid.uuid4().hex[:6]}"
    out_dir = Path(args.runs_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"run folder / wandb run name: {out_dir}")
    logger = WandbLogger(args.wandb_project, out_dir, asdict(args), args.wandb_entity, run_name, args.wandb_mode)

    def save(name: str):
        save_checkpoint(out_dir / name, ppo.actor, cfg, encoder if finetune else None)  # loadable by SteeredPolicy / the Steered client
        torch.save(ppo.state_dict(), out_dir / name / "ppo.pt")

    def run_eval(step: int):
        t0 = time.perf_counter()
        metrics, videos = evaluate_vec(venv, lambda f: ppo.act(f, deterministic=True, priv=venv.priv)[0], args.eval_videos, task_envs)
        metrics["time_s"] = time.perf_counter() - t0
        logger.log(metrics, step, prefix="eval/fixed/")
        for i, frames in enumerate(videos):
            logger.log_video(f"eval/fixed/video_{i}", frames, step)
        print(f"eval @ {step}: success {metrics['success/all']:.3f}, return {metrics['return']:.3f} ({metrics['time_s']:.0f}s)")

    transitions = env_steps = episodes = 0
    start_it = 0
    if args.resume:
        ppo.load_state_dict(torch.load(Path(args.resume) / "ppo.pt", map_location=args.rl_device))
        start_it = int(Path(args.resume).name.split("_")[-1])
        transitions = start_it * args.num_envs * args.rollout_chunks  # logging step offset, so the curves continue
        print(f"resumed from {args.resume}: starting at iteration {start_it + 1}")
    evaled_at_it = -1
    recent = RollingMean(window=args.num_envs)
    recent_task = {n: RollingMean(window=len(i)) for n, i in task_envs.items()} if multitask else {}
    warmup_iters = args.critic_warmup_iters  # counted by the iteration index (there is no resume path in this script)
    if args.eval_at_start:
        run_eval(transitions)
    last_eval_end = time.perf_counter()
    feat = venv.reset()
    ep_stats = [EpisodeStats() for _ in range(args.num_envs)]
    for it in range(start_it + 1, args.iterations + 1):
        t0 = time.perf_counter()
        buf.reset()
        finished, n_finished, success_ma, task_success_ma = MeanAccumulator(), 0, {}, {}
        bad_log = []  # (chunk, field, env ids): non-finite / out-of-range rollout data, skips this iteration's update
        for _ in range(args.rollout_chunks):
            z, logp, value = ppo.act(feat, priv=venv.priv)
            priv = venv.priv  # privileged state of the obs `feat` was computed from
            packed = venv.packed  # images/proprio of the obs `feat` was computed from (venv.step replaces them)
            t = venv.step(z)
            bad = find_nonfinite(reward=t.reward, feat=t.next_feat, value=value, logp=logp)
            if priv_fn is not None and len(priv_fn.bad_envs):  # raw state non-finite / out of range (priv_fn sanitizes its output)
                if len(priv_fn.nonfinite_envs):
                    bad["priv_nonfinite"] = priv_fn.nonfinite_envs
                else:  # clipped, so the data stays usable: warn only
                    print(f"it {it} chunk {len(buf)}: priv out of range (clipped) in envs {priv_fn.bad_envs.tolist()}: {priv_fn.raw_bad}")
            for name, ids in bad.items():
                bad_log.append((len(buf), name, ids.tolist()))
                print(f"it {it} chunk {len(buf)}: bad {name} in envs {ids.tolist()}"
                      + (f"\n  raw priv rows: {priv_fn.raw_bad}" if name == "priv_nonfinite" else ""))
            buf.add(feat, z, logp, value, t.reward, t.discount, t.terminated, t.episode_over, t.truncated, packed, priv)
            for i in range(args.num_envs):
                ep_stats[i].add(t.unbatch(i))
                if t.episode_over[i]:
                    summary = ep_stats[i].summary()
                    finished.add(summary)
                    if multitask:  # per-task copies of the headline metrics, rollout/<task>/...
                        name = tasks[tid[i]]
                        finished.add({f"{name}/{k}": summary[k] for k in ("success/all", "return", "length_steps")})
                        for k_, v_ in recent_task[name].add({"success/all": summary["success/all"]}).items():
                            task_success_ma[f"{name}/{k_}_ma"] = v_
                    n_finished += 1
                    success_ma = recent.add({k: v for k, v in summary.items() if k.startswith("success/")})
                    ep_stats[i].reset()
                    episodes += 1
            feat = t.next_feat
            env_steps += int(t.steps.sum())
        transitions += len(buf)
        t_rollout = time.perf_counter() - t0
        last_value = ppo.value(feat, venv.priv)
        if priv_fn is not None and len(priv_fn.nonfinite_envs):
            bad_log.append((len(buf), "priv_nonfinite(last)", priv_fn.nonfinite_envs.tolist()))
        for name, ids in find_nonfinite(last_value=last_value, last_feat=feat).items():
            bad_log.append((len(buf), name, ids.tolist()))
        buf.compute_returns(last_value, args.ppo.gae_lambda)
        if not bad_log:  # also catch bad data that only shows up after GAE
            bad_log = [(len(buf), k, v.tolist()) for k, v in find_nonfinite(adv=buf.d["adv"].T, ret=buf.d["ret"].T).items()]
        if bad_log:
            torch.save(dict(it=it, bad=bad_log, buf=dict(buf.d), last_value=last_value), out_dir / f"bad_rollout_{it}.pt")
            print(f"it {it}: bad rollout data {bad_log}; update skipped, dumped bad_rollout_{it}.pt, envs reset")
            logger.log({"bad_rollouts": 1, "iteration": it}, transitions, prefix="train/")
            feat = venv.reset()
            for s_ in ep_stats:
                s_.reset()
            continue
        warming = it <= warmup_iters
        snapshot = copy.deepcopy(ppo.state_dict())  # restored if the update leaves non-finite weights (run MultiTask-0810-0751-0d2b88 went NaN at it 60 and every later update was skipped)
        train_metrics = ppo.update(buf, train_policy=not warming, warmup_train_encoder=args.warmup_train_encoder)
        if not all(torch.isfinite(p_).all() for m_ in (ppo.actor, ppo.critic, *([ppo.encoder] if ppo.encoder is not None else [])) for p_ in m_.parameters()):
            ppo.load_state_dict(snapshot)
            print(f"it {it}: update produced non-finite weights, reverted to the pre-update weights")
            logger.log({"nan_updates": 1, "iteration": it}, transitions, prefix="train/")
            del snapshot
            feat = venv.reset()
            for s_ in ep_stats:
                s_.reset()
            continue
        del snapshot
        if warming and args.warmup_min_explained_var is not None and train_metrics["explained_var"] > args.warmup_min_explained_var:
            warmup_iters = it  # critic is good enough: policy updates start next iteration
        t_update = time.perf_counter() - t0 - t_rollout
        perf = dict(
            iteration=it, env_steps=env_steps, episodes=episodes, transitions_per_sec=len(buf) / (t_rollout + t_update),
            rollout_s=t_rollout, update_s=t_update,
            gpu_peak_gb=torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0,
        )
        logger.log({**train_metrics, **perf}, transitions, prefix="train/")
        if n_finished:  # means over the episodes that ended during this iteration
            ep_metrics = {**finished.pop_means(), "episodes_finished": n_finished}
            logger.log({**ep_metrics, **{f"{k}_ma": v for k, v in success_ma.items()}, **task_success_ma}, transitions, prefix="rollout/")
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
