# Running things: apptainer, GPUs, slurm launcher

Isaac Sim only runs inside the apptainer, on an L40 (preferred) or L40s GPU. Login nodes have no GPU and no `/isaac-sim`, and the
apptainer has no `sbatch`, so unattended runs are slurm jobs launched from a login node.

## Interactive (inside the apptainer)
- Enter: `/gscratch/weirdlab/will/polaris_overlay.sh`. Check you are inside with `ls /isaac-sim/python.sh`; `isaacpy` is the bashrc alias for it.
- GPUs (aliases in `~/.bashrc`): `show_weird_compute` lists free compute, `compute [flags]` allocates. Use `-w` (weirdlab GPUs); ckpt GPUs
  work but can be preempted. Isaac needs l40/l40s; the openpi server can use a40. Put the pi server on a different GPU than Isaac when possible.
- Without Isaac (quick scripts, tests that don't import Isaac): `/gscratch/weirdlab/will/venvs/ogbench/bin/python`.
- **Isaac hangs at exit and on renderer errors/OOM.** Scripts end with `os._exit(0)`; if output stalls, check for a hang and
  terminate with `ctrl+]`. Sweeps: run detached (`setsid ... < /dev/null`) with `timeout -s KILL`, poll for the output file, kill by PID.
  A crashed process can linger holding GPU memory (`kill -9`). Don't `pkill -f <pattern>` when the pattern is in your own command line
  (it kills your shell); use `pgrep` + `kill`, or `pkill -f "[s]erve_polic[y]"`.
- One Isaac env (`gym.make`) per process: a second one hangs.
- Log noise: `NGX isn't enabled` is harmless; grep for `Traceback` / `out of memory` / `Failed to allocate` rather than `Error`.

## Launching training runs from a login node: `scripts/cluster/`
- `run_experiment.py NAME [--time H] [--pi-addr HOST:PORT] [--dry-run] -- <train_steering.py args>`
  Checks free weirdlab GPUs (hyakalloc via `~/bin/show_weird_compute.py`). If no L40/L40s (trainer) or no a40/l40/l40s (pi server)
  is free it submits nothing and exits 2 ("NOT AVAILABLE"); retry later. Otherwise it submits `pi-NAME` (openpi server on a random
  free port, writes `host:port` to `pi_server.addr` when ready) and `train-NAME` (waits for that file, runs `train_steering.py` in the
  apptainer with `--policy.host/--policy.port` appended, cancels the pi job on exit). `--dummy-policy` in the train args skips the
  server; `--pi-addr` reuses a running one.
- `exp_status.sh [NAME]`: job states, log tails, latest `it N` line. `exp_stop.sh NAME`: cancels both jobs.
- `apptainer_exec.sh "<cmd>"`: non-interactive `polaris_overlay.sh` (ro overlay, runs `~/polaris_setup.sh` first).
- State/logs: `runs/cluster/NAME/{pi.log,train.log,jobs.json,pi_server.addr}`; run output in `runs/steering/<env>-DDMM-HHMM-<id>`.

Standard single-task launch (flags explained in `steering_training.md`):
```
scripts/cluster/run_experiment.py cb1 --time 12 -- --environment CubeBowl --num-envs 128 --cam-scale 0.5 \
    --critic-warmup-iters 5 --rollout-chunks 32 --residual-scale 0.0
```
Smoke test: `-- --environment CubeBowl --num-envs 8 --iterations 2 --no-eval-at-start --wandb-mode disabled`.

Notes
- Klone ignores comma-separated partition lists (uses the first only), so the launcher picks one partition per job
  (pi: a40 -> l40 -> l40s; trainer: l40 -> l40s). Jobs use account weirdlab, so they are not preempted.
- Server and trainer run on different nodes over the cluster network.
- Startup takes ~10 min (model load + Isaac); the initial eval of 8 envs ~2 min. `train_steering.py` calls `os._exit(0)`, so jobs end by themselves.
- wandb works online in batch jobs (`~/.netrc`); use `--wandb-mode disabled` for smoke tests.
- If `train.log` stops changing for a long time while the job is running, check for a hang.
- Server setup (manual start, pitfalls): `openpi_server.md`.
