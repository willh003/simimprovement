# docs/claude: notes for Claude agents

Three kinds of pages; keep each fact in one place and link instead of repeating.

- **`current/`**: how the repo works now. Update these when behaviour changes.
  - `running.md`: apptainer, GPUs, Isaac hang gotchas, `scripts/cluster/` slurm launcher, standard smoke test.
  - `openpi_server.md`: starting the policy server, cluster pitfalls, the batched `infer_batch` API.
  - `envs_simevals.md`: CubeBowl / CanMug / BananaBin, randomized / walled / multitask variants, `run_pi.py`.
  - `envs_molmo.md`: generated MolmoSpaces envs (asset pipeline, success criterion, findings).
  - `steering_training.md`: DSRL-PPO trainer (`scripts/train_steering.py`): design, flags, residual / encoder / warmup / privileged state.
  - `steering_metrics.md`: what every logged metric means, plus the logging code layout.
- **`experiments/`**: dated measurements and the evidence behind design choices (append new results; don't rewrite history).
  `env_scaling_bench.md` (parallel envs / tiled cameras), `openpi_batching_bench.md`, `openpi_server_and_resolution.md`, `walled_envs.md`. Figures in `images/` (regenerate with `scripts/plot_*.py`).
- **`handoffs/`**: self-contained task descriptions for another agent (login node vs apptainer). Delete or fold into `current/` once done.

Repo-level docs for users (README, `docs/custom_*.md`, `docs/checkpoints_and_envs.md`) are outside this folder.
