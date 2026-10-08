# Task: launch a steering training run on `Molmo-cup-bowl`

For an agent on a **login node** (slurm available; the Isaac apptainer has no sbatch). Background: `cluster_runs.md`, `steering_training.md`, `molmo_envs.md`.

## Settings (the standard single-task ones, see `steering_training.md`)
`--num-envs 128 --cam-scale 0.5 --critic-warmup-iters 5 --rollout-chunks 32 --residual-scale 0.0` (plain DSRL, no residual).

## Steps
1. Check the scenes exist (gitignored, generated in the repo): `ls assets/simevals/molmo_cup_bowl.usd assets/simevals/molmo_cup_bowl.json assets/molmo/cache/objects/thor/20260128/Cup_1`.
   If missing, they must be rebuilt inside the apptainer (`docs/claude/molmo_envs.md`: `scripts/download_molmo_assets.py` from a venv with `molmospaces-resources`, then `isaacpy scripts/make_molmo_scenes.py`).
2. Optional smoke test (~15 min of startup), same launcher with a throwaway name:
   `scripts/cluster/run_experiment.py cupbowl-smoke --time 1 -- --environment Molmo-cup-bowl --num-envs 8 --iterations 2 --no-eval-at-start --wandb-mode disabled`
   (the pi server + Isaac startup takes ~10 min; check `scripts/cluster/exp_status.sh cupbowl-smoke`; stop with `exp_stop.sh cupbowl-smoke`). Expect `it 1`, `it 2` lines, no traceback.
3. Launch the real run:
   `scripts/cluster/run_experiment.py cupbowl1 --time 12 -- --environment Molmo-cup-bowl --num-envs 128 --cam-scale 0.5 --critic-warmup-iters 5 --rollout-chunks 32 --residual-scale 0.0`
   If it prints "NOT AVAILABLE" (no free L40 / L40s trainer GPU, or no a40/l40/l40s for the pi server) nothing was submitted: retry later.
4. Monitor: `scripts/cluster/exp_status.sh cupbowl1` (job states, log tails, latest `it N` line). Logs/state: `runs/cluster/cupbowl1/{pi.log,train.log,jobs.json}`; run output in `runs/steering/Molmo-cup-bowl-<DDMM-HHMM>-<id>`. Isaac hangs at exit but `train_steering.py` calls `os._exit(0)`, so jobs end themselves; if `train.log` stops changing for a long time while the job is still running, check it is not hung.
5. Report back: the run folder, wandb run (if any), and early metrics (`rollout/success/all_ma`, `eval/fixed/*`; metric glossary in `steering_metrics.md`).

## Notes
- The env is `Molmo-cup-bowl` (put the cup in the bowl): reward is the sparse `obj_in_container` term (1 per step while the cup rests in the bowl), episodes end on success or after 30 s. Resets randomize cup and bowl xy within the ranges in `assets/simevals/molmo_cup_bowl.json` (`x_range`, `y_range`). The success criterion was checked with `scripts/check_molmo_success.py` (all 5 cases ok).
- Do not run more than one Isaac env per process (a second `gym.make` hangs); use separate runs for other Molmo envs (`Molmo-<obj>-<cont>`, listed in `molmo_envs.md`).
- The cup-bowl env has not been run through the trainer since the object-origin / threshold / bin changes (only an earlier 2-iteration dummy-policy smoke train), hence the optional smoke step.
