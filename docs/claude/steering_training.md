# Steering training (DSRL-style noise steering of pi0.5) — `scripts/train_steering.py`

Logging/eval details live in [wandb_logging.md](wandb_logging.md). This doc covers the training algorithm and data flow.

## Idea
The base policy (pi0.5 DROID, served by openpi, **frozen**) generates an action chunk by denoising flow-matching noise `noise ∈ R^(action_horizon × action_dim)` (default `(15, 32)`). Normally the noise is `N(0, I)`. We train a small actor `π(z | feat)` that *chooses* the noise for the first `steer_horizon` timesteps (the "steered slice"; the rest stays `N(0, I)`). The base model is a black box: noise in → chunk out. RL (SAC) is run in this latent-noise action space ("diffusion steering via RL").

- RL **action** = `z`, flat vector of size `steer_dim = steer_horizon * noise_shape[1]` (`steer_horizon` defaults to the client's `open_loop_horizon`).
- RL **observation** = `feat` from an encoder over the openpi request dict.
- One RL step = one action chunk (SMDP): the chunk's `open_loop_horizon` env steps are executed with the chunk, transition discount is `gamma^k` (k = env steps actually run; fewer if the episode ended mid-chunk).

## Components
| File | Role |
|---|---|
| `scripts/train_steering.py` | Entry point (tyro `TrainArgs`), builds everything, runs the loop |
| `latent_rl/networks.py` | `SteeringActor` (tanh-squashed Gaussian), `TwinQ` (two MLPs on `[feat, z]`), 2×256 hidden |
| `latent_rl/algos/dsrl_sac.py` | `DSRLSAC`: SAC updates with `act` / `update(batch)` |
| `latent_rl/buffer.py` | `ReplayBuffer` (cap 100k; `feat, z, reward, next_feat, done, discount`) |
| `latent_rl/encoders.py` | `droid_proprio` (8-d: 7 joints + gripper) or `droid_vision` (frozen DINOv2 CLS of exterior+wrist + proprio) |
| `latent_rl/steered_policy.py` | `SteeredPolicy`: actor + frozen base → chunk. Same class for train, eval, and the `Steered` inference client |
| `policy/steered_client.py` | `ServerChunkPolicy` (adapts `DroidJointPosClient` to `ChunkPolicy`), `SteeredClient` for deployment from a checkpoint |
| `rl/chunk_env.py` | `ChunkEnv`: runs one chunk in the Isaac env, computes reward terms |
| `rl/evaluation.py`, `rl/stats.py`, `rl/wandb_logger.py` | eval + logging |
| `latent_rl/checkpoint.py` | `ckpt_<n>/actor.pt` + `config.json` (`SteeringConfig`); only the actor is saved |

## Actor
`trunk(feat) → (mean, log_std)`; `u = mean + exp(log_std)·ε`; `z = bound·tanh(u/bound)` with `bound = z_bound = 3`. `log_std` clamped to [-5, 2]. `logp` includes the tanh Jacobian. The final layer is **zero-initialised**, so the untrained actor outputs ≈ `N(0, I)` (for |u| ≪ bound, z ≈ u) = the base policy's own noise distribution; training starts from base-policy behavior. Deterministic mode uses `u = mean`.

## Environment & reward (`ChunkEnv`)
1. `reset(object_positions=...)` resets the env, the client, builds the openpi request, encodes `feat`.
2. `step()`: `SteeredPolicy.step` samples `z`, builds noise (`z` in the first `steer_horizon` rows, `randn` elsewhere), queries the server for a chunk; then executes `open_loop_horizon` env steps (stopping early on success/termination/timeout). Only the last step of a chunk does the expensive splat render (needed for the next query), unless recording.
3. Reward = one sparse term per rubric criterion (`1/N` the first time that criterion's `reached/<name>` flag turns on, so terms sum to rubric progress) + `success_bonus` (default 1.0) on overall success. Total reward = sum of terms.
4. `done` = true termination (success or env terminated) — cuts bootstrapping. Time-outs set `episode_over` but **not** `done` (bootstrap through truncation). `discount = gamma^k`.

## SAC update (`DSRLSAC.update`)
Standard twin-Q SAC with SMDP discount:
- target: `y = r + discount·(1-done)·( min(Q1',Q2')(s', z') − α·logp(z'|s') )`, `z' ~ π(·|s')`
- critic loss: MSE(Q1, y) + MSE(Q2, y)
- actor loss: `mean(α·logp − min(Q1,Q2)(s, z~π))`
- target critics: Polyak `tau = 0.005`
- α: fixed at `init_alpha = 0.1` by default (`target_entropy=None`); set `SACConfig.target_entropy` to auto-tune (not exposed in `TrainArgs`). lr 3e-4 for all optimizers.
- Note: `train_steering.py` constructs `DSRLSAC` on CPU with default `SACConfig`; the only SAC knobs exposed via CLI are `z_bound`, `gamma`, `batch_size`, `updates_per_chunk`.

## Training loop
```
env = single-env Isaac task; conditions = initial object positions from the scene
train_conditions = all but last `heldout` (5); eval splits: heldout + first `eval_train_conditions` train ICs
(optional) eval at step 0
reset to random train IC
while chunks < total_chunks:
    t = chunk_env.step()                  # actor samples z (stochastic), executes chunk
    buffer.add(t.feat, t.z, t.reward, t.next_feat, t.done, t.discount)
    if chunks >= warmup_chunks and len(buf) >= batch_size:
        updates_per_chunk × algo.update(buf.sample(batch_size))
    every log_every chunks: log train/* (update metrics averaged over interval)
    if t.episode_over: log rollout/*; maybe eval (if chunks >= next_eval, at the episode boundary); reset to random train IC
    every save_every chunks: save_checkpoint
final eval + ckpt_final
```
Notes:
- **Warmup** is not a separate random policy: for the first `warmup_chunks` (200) chunks no updates happen, so the zero-init actor samples ≈ `N(0,I)` (base behavior) and fills the buffer.
- Updates are synchronous with data collection (UTD = `updates_per_chunk`, default 1), one env, no parallelism.
- Eval uses `SteeredPolicy(..., deterministic=True)` (actor mean) via a second `ChunkEnv` sharing the same env/client; training uses the stochastic actor. Eval time is excluded from `chunks_per_sec`.
- The x-axis for all wandb metrics is chunks.

## Key `TrainArgs`
`environment`, `runs_root`, `policy` (PolicyArgs: host/port of openpi server), `noise_shape`, `encoder` (`droid_proprio|droid_vision`), `steer_horizon`, `z_bound`, `gamma` (0.99), `success_bonus`, `total_chunks` (5000), `warmup_chunks` (200), `batch_size` (256), `updates_per_chunk`, `heldout`, `save_every`, `eval_every`, `eval_videos`, wandb options.

## Using a trained checkpoint
Set the `Steered` client with `policy.steering_ckpt=runs/steering/<run>/ckpt_X` (and `deterministic`) for evaluation; `SteeredPolicy.from_checkpoint` rebuilds the encoder from `config.json`, and checks `noise_shape` matches the base policy. Pi server helper scripts: `scripts/start_pi_server.sh`, `scripts/test_pi_server.sh`.

## Other env backends
Training also supports the sim-evals scene-2 env (no splat, fixed layout): see [simevals_env.md](simevals_env.md).

## Run folder / naming
Each run writes to `<runs_root>/<environment>-DDMM-HHMM-<uuid6>` (default `runs/steering/...`, e.g. `runs/steering/CanMug-0110-1423-a3f9c1`). That exact string is also the wandb run name. Contents: `ckpt_<n>/`, `ckpt_final/`, `videos/`, `wandb/`. The path is printed at start.
