# Handoff: launch a steering run on `Molmo-cup-bowl`

For an agent on a **login node** (slurm available; the apptainer has no sbatch). Background: `../current/running.md`, `../current/steering_training.md`, `../current/envs_molmo.md`.
Env: put the cup in the bowl; sparse `obj_in_container` reward, episodes end on success or after 30 s; resets randomize cup and bowl xy within `x_range` / `y_range` of
`assets/simevals/molmo_cup_bowl.json`. Success was checked with `scripts/check_molmo_success.py` (5/5 cases). The env has only had an earlier 2-iteration dummy-policy smoke train
(before the object-origin / threshold / bin changes), hence the optional smoke step.

1. Check the (gitignored) scenes exist: `ls assets/simevals/molmo_cup_bowl.usd assets/simevals/molmo_cup_bowl.json assets/molmo/cache/objects/thor/20260128/Cup_1`. If missing, rebuild inside the
   apptainer (`../current/envs_molmo.md` pipeline steps 2-3).
2. Optional smoke test (~15 min startup): `scripts/cluster/run_experiment.py cupbowl-smoke --time 1 -- --environment Molmo-cup-bowl --num-envs 8 --iterations 2 --no-eval-at-start --wandb-mode disabled`;
   expect `it 1`, `it 2` lines and no traceback (`exp_status.sh cupbowl-smoke`, stop with `exp_stop.sh`).
3. Launch with the standard single-task settings: `scripts/cluster/run_experiment.py cupbowl1 --time 12 -- --environment Molmo-cup-bowl --num-envs 128 --cam-scale 0.5 --critic-warmup-iters 5 --rollout-chunks 32 --residual-scale 0.0`.
   "NOT AVAILABLE" means nothing was submitted; retry later.
4. Monitor with `scripts/cluster/exp_status.sh cupbowl1`; output in `runs/steering/Molmo-cup-bowl-<DDMM-HHMM>-<id>`.
5. Report the run folder, wandb run, and early metrics (`rollout/success/all_ma`, `eval/fixed/*`; glossary in `../current/steering_metrics.md`).
