# Steering training metrics reference

What each metric logged by `scripts/train_steering.py` means and how to read it. Metrics go to wandb and to
`runs/steering/<run>/metrics.csv` (long format: `step, metric, value`). Run-independent; see `steering_training.md`
for the algorithm itself.

**x-axis (`step`)** = cumulative *transitions* (one action chunk in one env). One PPO iteration adds
`num_envs * rollout_chunks` transitions (default 128 * 8 = 1024). Metric names are `<prefix>/<name>`.

Sources: `ppo.update()` in `latent_rl/algos/dsrl_ppo.py` (train/*), `EpisodeStats`/`RollingMean` in `rl/stats.py` (rollout/*),
`evaluate_vec` (eval/*), and the loop in `train_steering.py`.

## `train/*` — one row per PPO iteration

Update metrics are the mean over all epochs x minibatches of that iteration, except the last group (batch statistics),
which are computed once on the whole rollout buffer after the update.

### Optimization health
| Metric | Meaning | How to read it |
|---|---|---|
| `pg_loss` | Clipped PPO surrogate loss `max(-A*r, -A*clip(r))`. | Hovers near 0 and is usually slightly negative. Not a progress signal. |
| `v_loss` | Critic MSE `0.5*(V - return)^2` (or value-clipped variant if `value_clip` set). | Scales with reward/return magnitude. Spikes when a batch contains unusually large returns. |
| `approx_kl` | `mean((r-1) - log r)`, an estimate of KL(old‖new) during the update. | If its epoch mean exceeds `target_kl` the epoch loop stops. Persistently near `target_kl` means updates are being capped. |
| `clipfrac` | Fraction of samples with `\|r-1\| > clip`. | ~0.1–0.2 is normal. Very high = policy moving too fast per update (lower lr / more minibatch noise). |
| `epochs` | PPO epochs actually run (max `cfg.epochs`). | Stuck below max means `target_kl` early stopping is binding every iteration. |
| `grad_norm` | Global grad norm of actor+critic **before** clipping to `max_grad_norm`. | Always > clip means the step size is set by lr and the clip, not gradient magnitude. |
| `enc_grad_norm` | Same for the finetuned encoder. | Only logged when the encoder is trained. |

### Critic quality
| Metric | Meaning | How to read it |
|---|---|---|
| `explained_var` | `1 - Var(ret - V)/Var(ret)` over the batch (computed after the update). | 1 = perfect, 0 = no better than a constant, <0 = worse than constant. NaN if returns are constant. With sparse reward it is noisy and often low. |
| `value_mean` | Mean critic prediction `V(s)`. | Should track `return_mean`. |
| `return_mean` | Mean GAE(lambda) return target. | Rises as the policy collects more reward. |

### Policy / action distribution
| Metric | Meaning | How to read it |
|---|---|---|
| `entropy` | Mean entropy of the pre-tanh Gaussian over the steering latent. | The absolute value scales with latent dimension; only the *trend* matters. Decreases as the policy commits. |
| `std_mean` | Mean actor std (`exp(log_std)`) over the batch. | Starts at ~1 (the diffusion prior). Staying ~1 means the actor has barely sharpened. |
| `z_abs_mean` | Mean `\|z\|` of the sampled steering latent (steer part only). | For N(0,1) this is ~0.8. Drift away from that shows the actor moving off the prior (up = larger-norm noise). |
| `res_abs_mean` | Mean `\|·\|` of the residual-action part of the action. | Only logged when a residual head is enabled. |

### Signal strength
| Metric | Meaning | How to read it |
|---|---|---|
| `reward_mean` | Mean per-chunk reward in the batch. | Directly shows how sparse the reward is. Exactly 0 means the batch has no learning signal. |
| `adv_std` | Std of the *raw* advantages (before per-minibatch normalization). | Near 0 = nothing to learn from; normalization then amplifies noise into full-size gradients. |

### Counters and performance
| Metric | Meaning |
|---|---|
| `iteration` | PPO iteration index (1-based). |
| `env_steps` | Cumulative sim steps across all envs. |
| `episodes` | Cumulative finished episodes. |
| `transitions_per_sec` | `len(buf) / (rollout_s + update_s)`. Excludes eval time. |
| `rollout_s` | Wall time collecting the batch (base policy inference + sim). Usually dominates. |
| `update_s` | Wall time of the PPO update. |

## `rollout/*` — training (stochastic-policy) episodes

Logged only for iterations in which at least one episode finished. Values are **means over episodes that ended during
that iteration**, so with long episodes they can be based on just 1–5 episodes and are very noisy.

| Metric | Meaning |
|---|---|
| `return` | Sum of per-chunk rewards over the episode (all reward terms plus bonus). |
| `return/<term>` | Same, per reward term (e.g. `obj_near_container`, `obj_in_container`, `success_bonus`). Terms are defined by the task; the sparse ones are partial-credit rewards for reaching a criterion. |
| `success/all` | 0/1 overall task success of the episode (the env's success termination). |
| `success/<term>` | 0/1 whether that sparse criterion was reached; the mean is the criterion's success rate. |
| `success/all_ma`, `success/<term>_ma` | **Rolling mean** over the last `num_envs` finished episodes. Use these for plots: they are the smooth success-rate curve. |
| `length_chunks`, `length_steps` | Episode length in action chunks / sim steps. Shorter on success (early termination), so falling length usually accompanies rising success. |
| `episodes_finished` | How many episodes ended this iteration (also the sample size of the means above). |
| `final_progress` | Rubric progress at the end of the episode, if the task provides one. |

Note: `success_bonus` is 0 unless configured, in which case `success/all` and the success-based terms are the only
success signal.

## `eval/fixed/*` — deterministic evaluation

Run every `eval_every_min` minutes of wall-clock time (plus at step 0 if `eval_at_start`). One deterministic-policy
episode in every env (`episodes` = `num_envs`), so each value is a mean over `num_envs` episodes with a fixed set of
initial conditions per run. It uses the same keys as `rollout/*` (`return`, `return/<term>`, `success/*`,
`length_chunks`, `length_steps`) without the `_ma` variants, plus:

| Metric | Meaning |
|---|---|
| `episodes` | Number of eval episodes (= `num_envs`). |
| `time_s` | Wall time of the eval. Can be a large fraction of total runtime. |

Eval is deterministic (mean action), so it can be better or worse than `rollout/*`, which samples from the policy.
The eval step is the transition count at which it ran, so it lines up with the train/rollout x-axis.

## Quick diagnostics

- **Learning?** `eval/fixed/success/all` and `rollout/success/all_ma` rising; `length_steps` falling.
- **Policy barely moving?** `std_mean` ~1, `entropy` flat, `z_abs_mean` ~0.8 with `approx_kl` still near `target_kl` and `epochs` below max: updates are noise-limited, not step-size-limited.
- **No signal in batch?** `reward_mean` = 0 and tiny `adv_std`; consider larger `rollout_chunks`.
- **Critic problems?** Fluctuating `explained_var`, `v_loss` spikes. Try a larger batch or `value_clip`.
- **Regression after a peak?** Compare `eval/fixed/success/all` across checkpoints (`ckpt_*`) and keep the best one.
