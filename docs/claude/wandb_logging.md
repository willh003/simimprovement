# wandb logging & in-training eval (`scripts/train_steering.py`)

## Module layout (logging is decoupled from training logic)
| File | Role | wandb? |
|---|---|---|
| `src/polaris/rl/chunk_env.py` | `ChunkTransition.reward_terms` (one term per rubric criterion + `success_bonus`; `reward == sum`), `.steps`, `.frames` (only with `step(record=True)`, which splat-renders every env step) | no |
| `src/polaris/rl/stats.py` | `EpisodeStats` (per-episode sums per term + success flags), `MeanAccumulator` (interval means of update metrics), `RollingMean` (moving success rates) | no |
| `src/polaris/rl/evaluation.py` | `evaluate(chunk_env, conditions, n_videos)` → mean metrics + recorded frame lists | no |
| `src/polaris/rl/wandb_logger.py` | `WandbLogger.log / log_video / finish`; mp4s also saved to `<run_folder>/videos/` | **only file** |
| `src/polaris/latent_rl/algos/dsrl_sac.py` | `update()` returns extra diagnostics | no |

## Reward terms & success metrics
The sparse reward terms are the rubric criteria (`Rubric.criteria`, named `<idx>_<checker name>` e.g. `0_reach_sponge`, `2_sponge_in_pan`; checkers set `__name__`). `Rubric.evaluate` exposes `metrics["reached/<name>"]` (max-ever 0/1 within an episode). `ChunkEnv` gives each criterion reward 1/N the first time it is reached (they sum to the old progress delta, so total reward is unchanged) plus `success_bonus` on overall success.
- Reward curves: `rollout/return/<criterion>` (per-episode return of each term), `rollout/return/success_bonus`, `rollout/return` (total).
- Success curves: `rollout/success/<criterion>` (0/1 per episode: criterion reached) and `rollout/success/all` (task success), plus `rollout/success/*_ma10` moving averages (= success rates). Eval: `eval/<split>/success/*` are rates over the ICs.
To add a reward term: add a key to `terms` in `ChunkEnv.step`; for a new success metric add a `success/<name>` entry in `EpisodeStats.summary`.

## Metric namespaces (x-axis = wandb `step` = chunks)
- `train/*` every `log_every` chunks: SAC update metrics averaged over the interval (`q_loss, actor_loss, alpha, alpha_loss` (only if auto-tuned), `q_mean, q_std, q_target_mean, td_abs, logp, z_abs_mean, reward_mean`) + `buffer_size, env_steps, chunks_per_sec` (eval time excluded).
- `rollout/*` per training episode: `return`, `return/<term>`, `success/<metric>`, `success/<metric>_ma10`, `final_progress`, `length_chunks`, `length_steps`, `episode`.
- `eval/{heldout,train}/*`: same as rollout, but means over ICs (`success/*` are rates) + `video_{i}`.

## Eval behavior
- Uses a second `ChunkEnv` on the same env/client with a `deterministic=True` `SteeredPolicy` sharing the training actor (same semantics as the `Steered` client in `scripts/eval.py`).
- Splits: `heldout` = last `heldout` ICs, `train` = first `eval_train_conditions` train ICs.
- Runs at chunk 0 (`eval_at_start`), then at the first episode boundary after every `eval_every` chunks (never interrupts a training episode), then once at the end.
- The first `eval_videos` episodes per split render every step (smooth video, 15 fps). Video frame = 224x448 model view (exterior | wrist).

## Environment notes
- Run inside the UWLab apptainer container (`/gscratch/weirdlab/will/polaris_overlay.sh`, python = `/isaac-sim/python.sh`). wandb 0.25.0 is already installed there. Auth comes from `~/.netrc`. The repo `.venv` and `uv run` don't work on the login node (glibc too old for isaaclab).
- There is no system ffmpeg in the container. `WandbLogger` points mediapy at imageio-ffmpeg's bundled binary.
- Tests: `tests/test_rl_logging.py` (fake env/client, no Isaac or server needed).
