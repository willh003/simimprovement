# Launching training runs from a login node (agent workflow)

Login nodes have no GPU or `/isaac-sim`, so runs are slurm jobs. `scripts/cluster/`:

- `run_experiment.py NAME [--time H] [--pi-addr HOST:PORT] [--dry-run] -- <train_steering.py args>`
  Checks free weirdlab GPUs (hyakalloc, via `~/bin/show_weird_compute.py`). If no L40/L40s (trainer) or no a40/l40/l40s
  (pi server) is free it submits nothing and exits 2 ("NOT AVAILABLE"). Otherwise submits two jobs: `pi-NAME`
  (openpi server, random free port, writes `host:port` to `pi_server.addr` when ready) and `train-NAME` (waits for that file,
  runs `train_steering.py` in the apptainer with `--policy.host/--policy.port` appended, cancels the pi job on exit).
  `--dummy-policy` in the train args skips the server; `--pi-addr` reuses a running one.
- `exp_status.sh [NAME]` job states + log tails + latest `it N` line; `exp_stop.sh NAME` cancels both jobs.
- `apptainer_exec.sh "<cmd>"`: non-interactive `polaris_overlay.sh` (ro overlay, runs `~/polaris_setup.sh` first).
- State/logs: `runs/cluster/NAME/{pi.log,train.log,jobs.json,pi_server.addr}`; run output in `runs/steering/...`.

Example: `scripts/cluster/run_experiment.py cb1 --time 12 -- --environment CubeBowl --num-envs 128`

Notes
- Klone ignores comma-separated partition lists (uses the first only), so the launcher picks one partition per job
  (pi: a40 -> l40 -> l40s; trainer: l40 -> l40s). Jobs use account weirdlab, so they are not preempted.
- Server and trainer are on different nodes; this works over the cluster network.
- Startup takes ~10 min (model load + Isaac); initial eval of 8 envs ~2 min. Pass `--no-eval-at-start` for smoke tests.
- Isaac hangs at exit, but `train_steering.py` calls `os._exit(0)`, so jobs end by themselves.
- wandb works online in batch jobs (`~/.netrc`); use `--wandb-mode disabled` for smoke tests.
- Smoke test used: `-- --environment CubeBowl --num-envs 8 --iterations 2 --no-eval-at-start --wandb-mode disabled`.
