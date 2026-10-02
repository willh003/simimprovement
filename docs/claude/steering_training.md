# Steering training (DSRL-PPO on parallel sim-evals envs) — `scripts/train_steering.py`

Logging details: [wandb_logging.md](wandb_logging.md) (written for the SAC version; the logger classes are unchanged).
Parallel envs / tiled cameras: [parallel_envs_bench.md](parallel_envs_bench.md). Batched server: [openpi_batching.md](openpi_batching.md).

The trainer used to be single-env DSRL-**SAC** (also worked on splat/rubric envs). It is now on-policy **PPO** over N
parallel envs (128-256), **sim-evals tasks only** (CubeBowl / CanMug / BananaBin). The splat renderer only handles env 0,
so splat envs can't run in parallel. The SAC pieces (`DSRLSAC`, `ReplayBuffer`, `ChunkEnv`, `evaluate`) are still in the
library and tested, but no script uses them now (see git history before this change for the SAC trainer).

## Idea (unchanged)
Frozen pi0.5 DROID (openpi server) makes an action chunk by denoising flow-matching noise `(15, 32)`. The actor
`π(z | feat)` picks the noise for the first `steer_horizon` rows (default = `open_loop_horizon` = 8; the rest stay
`N(0, I)`). RL action = `z` (`steer_dim = steer_horizon·32`); one RL step = one chunk (SMDP, discount `gamma^k`, k = env
steps actually run).

## Components
| File | Role |
|---|---|
| `scripts/train_steering.py` | tyro `TrainArgs` (`--ppo.*` nested `PPOConfig`), builds everything, runs the PPO loop |
| `environments/simeval_parallel.py` | `make_parallel_env`: N envs, `TiledCameraCfg` cams (`cam_scale`), cam2 dropped, **success termination** |
| `rl/vec_chunk_env.py` | `VecChunkEnv` (action z (N, steer_dim) → `VecChunkTransition`), `.unbatch(i)` → `ChunkTransition` |
| `rl/tasks.py` | `VecSimEvalsTask`: vectorized reward terms / success / reached, `hold_action`, `refresh_obs` |
| `rl/evaluation.py` | `evaluate_vec`: one deterministic episode in every env |
| `latent_rl/algos/dsrl_ppo.py` | `DSRLPPO` + `PPOConfig` |
| `latent_rl/rollout.py` | `RolloutBuffer` (T × N), SMDP GAE |
| `latent_rl/networks.py` | `SteeringActor` (+ `dist`, `evaluate(feat, z)`), `ValueNet`, `TwinQ` (SAC) |
| `latent_rl/steered_policy.py` | `build_noise` (single or batched), `SteeredPolicy` (deployment, unchanged) |
| `latent_rl/encoders.py` | `encode_batch`; `DroidVisionEncoder.batch` (ViT in mini-batches of 64) |
| `policy/droid_jointpos_client.py` | `SimEvalsJointPosClient.build_requests` (batched), `query_chunks` (`infer_batch`, fixed sub-batches), `postprocess_actions` |
| `policy/steered_client.py` | `ServerChunkPolicy.query_batch` (also `BatchChunkPolicy`) |

Layering: `latent_rl/` = torch/numpy only. `rl/` = env wrappers with no Isaac imports (tested with fakes in
`tests/test_vec_chunk_env.py`). Isaac-specific code is only in `environments/simeval_parallel.py` and the script.

## VecChunkEnv: one chunk in N envs
1. `noise = build_noise(z)` (N, 15, 32) → **one** `base.query_batch(requests, noise)` → chunks (N, 15, 8).
2. Run `open_loop_horizon` env steps in lockstep (gripper binarized). Isaac **auto-resets** an env inside `env.step` when
   it terminates or times out. An env that ends mid-chunk is **masked** for the rest of the chunk: its reward and k stop,
   it gets `hold_action` (the new episode's arm pose, gripper open), and its next chunk starts in the fresh episode.
   So a new episode can begin with ≤7 hold steps (counted against its 450-step limit).
3. **Success must be a termination.** sim-evals only has `time_out`, so `make_parallel_env(success_term=...)` adds
   `terminations.success` = the success reward term's func/params `> 0`. `VecChunkEnv` raises if success fires without
   termination.
4. Stale cameras: Isaac renders *before* resetting. If an env resets on the last executed step of the chunk,
   `task.refresh_obs` re-renders, forces a sensor update, and recomputes obs. Isaac's own `num_rerenders_on_reset` would
   pay this on every reset; here it is only paid when needed. Envs that reset earlier get rendered by the hold steps.
5. Reward terms: `reward_manager._step_reward * step_dt` (N, n_terms). This sums exactly to the env reward and needs
   no per-env `get_active_iterable_terms` loop. `reached` flags are per env, cleared when that env resets.
6. Returns `next_feat` (for ended envs: the new episode's first feat), `reward`, `reward_terms`, `terminated`,
   `truncated` (time-out without termination), `episode_over`, `discount = gamma^k`, `steps`, `success`, `reached`, and
   `frames` (every step's model view for the first `record` envs; the terminal frame is lost to the auto-reset).

## PPO (`DSRLPPO`, `RolloutBuffer`)
- Actor = the same `SteeringActor` as SAC (tanh-squashed Gaussian, state-dependent log_std, zero-init last layer → starts
  at ≈N(0, I) = base behavior). Checkpoints stay compatible with `SteeredPolicy.from_checkpoint` / the `Steered` client.
  `evaluate(feat, z)` inverts the squash (`u = b·atanh(z/b)`, clamped) for the ratio; entropy = pre-tanh Gaussian entropy.
- Critic `ValueNet` (2×256 MLP). One Adam (lr 3e-4, eps 1e-5) over both; loss = clipped PG + `vf_coef`·value MSE
  − `ent_coef`·entropy; grad-norm clip 0.5; `epochs` × `minibatches`, early stop when mean approx-KL > `target_kl`.
  Advantages normalized per minibatch. Value clipping optional (`value_clip`, default off).
- **SMDP GAE:** `δ_t = r_t + disc_t·(1−over_t)·V_{t+1} − V_t`, `A_t = δ_t + disc_t·λ·(1−over_t)·A_{t+1}` (λ per chunk).
- **Time-outs:** the terminal obs is lost to the auto-reset, so the buffer bootstraps rsl_rl-style:
  `r += gamma^k·V(s_t)` when `truncated & ~terminated`. Terminations (success) don't bootstrap.
- Defaults (`PPOConfig`): epochs 5, minibatches 4, clip 0.2, vf_coef 0.5, ent_coef 0, target_kl 0.02, λ 0.95.
  None of these are tuned yet.

## Training loop
```
env = make_parallel_env(task, num_envs, cam_scale, success_term)   # tiled cams, success terminates
(optional) eval at 0;  feat = venv.reset()
for it in 1..iterations:
    rollout_chunks × { z, logp, v = ppo.act(feat); t = venv.step(z); buf.add(...); per-env EpisodeStats }
    buf.compute_returns(ppo.value(feat), λ);  ppo.update(buf)
    log train/* (PPO stats + transitions_per_sec, rollout_s, update_s, env_steps, episodes)
    log rollout/* = means over episodes that ended this iteration (+ success/*_ma over the last num_envs episodes)
    every eval_every_min minutes (checked per iteration): evaluate_vec (deterministic actor mean, 1 episode per env) → eval/fixed/*; then reset all envs
    every save_every: ckpt_<it>
final eval (if not just done) + ckpt_final; os._exit(0) (Isaac hangs on shutdown)
```
- wandb x-axis = **transitions** (chunks summed over envs), not iterations.
- Eval reuses the training envs. Afterwards all envs are reset, which is fine for on-policy PPO (only the partial
  training episodes are dropped). Eval cost ≈ one full episode (~57 chunk rounds).
- Checkpoint dir: `actor.pt` + `config.json` (`SteeringConfig`, same as before) + `ppo.pt` (actor, critic, optimizer).

## Throughput expectations
The single batched openpi server does ~20 obs/s, so each chunk round takes about `num_envs/20` s (128 envs ≈ 6.4 s,
256 ≈ 13 s) plus 8 env steps (~0.3-0.4 s each at 128-256 envs, 640x360). Iteration with `rollout_chunks=8` at 128 envs
should take roughly 1-1.5 min for 1024 transitions. This is an estimate and hasn't been measured with the server.
`--server-batch-size` splits the batch into fixed-size `infer_batch` calls. Keep it constant: JAX recompiles for each
new batch size, and the first call compiles for minutes.

## Key `TrainArgs`
`environment`, `policy` (host/port/open_loop_horizon), `num_envs` (128), `cam_scale` (0.5), `server_batch_size`,
`encoder`, `steer_horizon`, `z_bound`, `gamma` (0.99/step), `success_bonus` (0), `ppo.*`, `rl_device` (cuda; CPU updates took seconds next to Isaac),
`rollout_chunks` (8), `iterations` (300), `eval_every_min` (20 min of wall clock, independent of envs/chunks), `save_every` (25 iterations), `eval_videos`, `dummy_policy`
(base holds the arm pose, no server: smoke test), wandb/run-folder options.

## Usage
```
isaacpy scripts/train_steering.py --environment CubeBowl --num-envs 128 --policy.port 8000 --policy.host "0.0.0.0"
isaacpy scripts/train_steering.py --environment CubeBowl --num-envs 8 --dummy-policy --rollout-chunks 4 \
    --iterations 2 --eval-every-min 0 --no-eval-at-start --wandb-mode disabled     # smoke test without a server
```
Run folder: `<runs_root>/<environment>-DDMM-HHMM-<uuid6>` (also the wandb run name).

## Using a trained checkpoint
Unchanged: `Steered` client with `policy.steering_ckpt=<run>/ckpt_X` (+ `deterministic`).

## Tests
`tests/test_ppo.py` (logp inversion, batched noise, GAE with SMDP discount / truncation / termination, PPO bandit),
`tests/test_vec_chunk_env.py` (fake auto-resetting N-env: masking, hold action, gamma^k, refresh, unbatch, evaluate_vec).
Run: `isaacpy -m pytest tests/`.

## Residual steering + finetuned ResNet50 encoder

**Action space (verified):** the pi0.5 `pi05_droid_jointpos_polaris` chunk is ABSOLUTE joint positions (openpi's `AbsoluteActions`
adds the current state back), and the sim `JointPositionActionCfg(use_default_offset=False, scale=1)` treats it as an absolute
target. Gripper is dim 7 (binarised at 0.5). So a residual is in radians on dims 0-6.

**Residual (`--residual-scale`, default 0.01; 0 = plain DSRL).** The actor's action becomes `[z (steer_dim), r (res_dim)]`,
`res_dim = open_loop_horizon * 7`. `r` is a squashed Gaussian in (-1, 1) (same tanh logp machinery; per-dim `bound` buffer in
`SteeringActor`, 1 for residual dims). The env adds `residual_scale * r` to the arm joints of chunk steps `[:residual_horizon]`
(`apply_residual` in `latent_rl/steered_policy.py`, used by `VecChunkEnv.step`, `SteeredPolicy.step`). Rollout buffer `z` holds the
full action. `SteeringConfig` stores `residual_scale` / `residual_horizon`. Logged: `train/res_abs_mean` (unit scale; x scale = rad).

**Encoder (`--encoder droid_resnet50`, default).** Shared ImageNet ResNet50 over exterior+wrist, pooled 2048-d each, layer-normed, +
proprio -> 4104-d. `--finetune-encoder` (default true; `--no-finetune-encoder` freezes and caches features like dino). Finetune mode:
`RolloutBuffer(img_shape=...)` also stores uint8 images + proprio; `DSRLPPO.update` re-encodes each minibatch (no-grad, microbatched),
runs the PPO loss on the feature leaf, then recomputes each microbatch with grad and backprops `d loss/d feat` (bounded memory).
Separate Adam group `ppo.encoder_lr` (3e-5), separate grad clip, `ppo.encoder_microbatch` (32). BatchNorm is always eval mode. bf16
autocast. Checkpoints add `encoder.pt`; `SteeredPolicy.from_checkpoint` loads it. Pretrained weights come from torchvision
(`IMAGENET1K_V2`, downloaded to the torch hub cache - warm it on a node with internet).
