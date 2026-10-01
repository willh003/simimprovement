# sim-evals envs (scenes 1–3) for steering training (no splat)

`environment={CubeBowl,CanMug,BananaBin}` trains on the plain-Isaac-render sim-evals DROID env (CubeBowl: "put the cube in the bowl", CanMug: "put the can in the mug", BananaBin: "put banana in the bin"), instead of a splat/rubric PolaRiS env.

## What was ported
`third_party/sim-evals/src/sim_evals/environments/{droid_environment,nvidia_droid}.py` → `src/polaris/environments/simeval_droid.py`, `simeval_robot.py` (copies; only imports/asset path changed). Assets (scene/robot USDs, backgrounds; ~89 MB, gitignored) live in `<repo>/assets/simevals` (moved out of `third_party/sim-evals/`, which is no longer needed; optional override `SIM_EVALS_ASSETS`). `CanMugEnvCfg` = `EnvCfg` + `set_scene("2")`; registered in `environments/__init__.py` as `CanMug` with plain `ManagerBasedRLEnv`. If sim-evals' env file changes, re-sync the copy.

## Backends (`src/polaris/rl/tasks.py`)
`ChunkEnv` takes a `task` that does reset/step and extracts reward terms + success:
- `RubricSplatTask` (default, old behaviour): rubric criteria → 1/N-per-criterion sparse terms.
- `SimEvalsTask`: terms = `reward_manager.get_active_iterable_terms(0)` rescaled so they sum to the env reward. Terms: `can_in_mug` (1/step while the can rests in the mug; **success** when > 0) and `can_near_mug` (0.25/step shaping while above the mug, not yet in). `success/<term>` = term ever fired in the episode.

`ChunkTransition` now carries `success` and `reached`; `EpisodeStats` uses those (not `info["rubric"]`), and `final_progress` is only logged when the info has a rubric.

## Client
`SimEvalsJointPosClient` (`policy/droid_jointpos_client.py`, registered `DroidJointPosSimEvals`) reads `obs["policy"]` cameras (`external_cam`, `wrist_cam`, full-res → resized to 224 by `build_request`) instead of `obs["splat"]`. `train_steering.py` picks it automatically for the sim-evals env ids (`is_sim_evals` in `rl/tasks.py`: ids in `SIM_EVALS_INSTRUCTIONS`).

## Behaviour differences from the splat envs
- **Single fixed layout**: env resets to the default scene (`reset_scene_to_default`); no initial conditions, no held-out split. Eval = `eval_train_conditions` episodes of the same layout, logged as `eval/fixed/*`. Variation only from the policy/base-noise stochasticity.
- First reset does the double-reset (material loading) from `run_eval.py`.
- Episode ends on success, timeout (30 s = 450 steps ≈ 56 chunks at horizon 8).
- `success_bonus` defaults to 0 here (the `can_in_mug` term already pays 1); 1.0 for rubric envs.
- Hovering reward (`can_near_mug`) accrues each step while above the mug, so it can be farmed (max ≈0.25/step); watch `return/can_near_mug` vs `success/all`.
- `record=True` is not more expensive here (no splat), so eval videos cost nothing extra.

## Status
All three tasks (CanMug, CubeBowl, BananaBin) were confirmed working in the container by the user (env builds, rollouts run with the openpi server). Unit tests (`tests/test_rl_logging.py::test_sim_evals_task_*`) use fakes. The `get_active_iterable_terms` values are on a different scale from the env reward, so both `SimEvalsTask` and `run_pi.py` rescale terms to sum to the reward.

Train (inside the Isaac container, openpi server on the same node; output goes to `runs/steering/<env>-DDMM-HHMM-<uuid>`, see steering_training.md):
`python scripts/train_steering.py --environment CanMug --policy.open-loop-horizon 8 --policy.port 8000 ...`

## Checking an env/policy: `scripts/run_pi.py`
Port of sim-evals' `run_eval.py` onto the polaris env/client. Saves `all_cams.png`, `episode_<n>_{success,fail}.mp4`, `episode_<n>_rewards.txt` under `runs/simevals_eval/<date>/<time>`. Video side panel: step, SUCCESS/FAIL, `return: current / total / max` (max = peak running return), and each reward term's value on the current step (no cumulative). `--dummy-policy` needs no server (holds pose, gripper open) and only checks env build/reset/step/render/video.
```
cd /gscratch/weirdlab/will/polaris && export PYTHONPATH=$PWD/src:$PYTHONPATH
/isaac-sim/python.sh scripts/run_pi.py --environment CanMug --episodes 1 --dummy-policy      # env only
/isaac-sim/python.sh scripts/run_pi.py --environment CubeBowl --episodes 2 --policy.port 8000  # with openpi server
```

## CubeBowl (scene 1) and BananaBin (scene 3)
sim-evals only defines rewards for scene 2, so for 1 and 3 `set_scene` builds the same two terms (`obj_in_container` = success, `obj_near_container` = 0.25 shaping) from `SCENE_TASKS` in `simeval_droid.py`: name keywords for the object/container rigid bodies and `object_in_container` thresholds (`xy_threshold`, `success_z_max`). Instruction and success-term tables are in `rl/tasks.py`.
Rigid bodies in the scene USDs: scene 1 `rubiks_cube`, `_24_bowl`; scene 2 `_10_potted_meat_can`, `_25_mug`; scene 3 `_11_banana`, `small_KLT_visual_collision` (the bin). Names resolve by keyword (error if 0 or >1 match).
**Thresholds are hand-set** (not tuned) from the initial poses (table top z≈0.08; bowl origin z 0.078, bin origin z 0.125): CubeBowl xy<0.05, z∈[0,0.08]; BananaBin xy<0.08, z∈[-0.08,0.1] (negative z_min because banana rests on the bin floor, below its origin). Also requires the object to be nearly still. Verify with `run_pi.py --environment <Task>` and the reward printouts, or by hand-placing the object, then adjust `SCENE_TASKS`.
