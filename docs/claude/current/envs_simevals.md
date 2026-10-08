# sim-evals envs (plain Isaac render, no splat)

Env ids: `CubeBowl` (scene 1, "put the cube in the bowl"), `CanMug` (scene 2), `BananaBin` (scene 3), plus the variants below and the
generated Molmo envs (`envs_molmo.md`). These are what `train_steering.py` trains on (see `steering_training.md`; launch recipe in `running.md`).

## Code map
- `src/polaris/environments/simeval_droid.py`, `simeval_robot.py`: copies of `third_party/sim-evals/.../{droid_environment,nvidia_droid}.py` (only imports / asset
  path changed; re-sync if sim-evals changes). Assets (scene/robot USDs, backgrounds, ~89 MB, gitignored) live in `<repo>/assets/simevals` (override with `SIM_EVALS_ASSETS`).
- `environments/__init__.py`: gym registration. `environments/simeval_parallel.py::make_parallel_env`: N envs with tiled cameras + success termination.
- `rl/tasks.py`: instruction / success-term tables (`SIM_EVALS_INSTRUCTIONS`, `MULTITASK_ENVS`, ...), `SimEvalsTask` / `VecSimEvalsTask`.
- `rl/privileged.py`: ground-truth state for actor/critic. `policy/droid_jointpos_client.py::SimEvalsJointPosClient` (registered `DroidJointPosSimEvals`):
  reads `obs["policy"]` cameras (`external_cam`, `wrist_cam`, resized to 224) and picks one prompt per env. `train_steering.py` selects it for sim-evals ids (`is_sim_evals`).

## Tasks, reward and success
`SCENE_TASKS` (`simeval_droid.py`) holds per scene the name keywords of the object/container rigid bodies (resolved by keyword; error on 0 or >1 match)
and the `object_in_container` thresholds. sim-evals only defines rewards for scene 2; `set_scene` builds the same terms for scenes 1 and 3.
- Single sparse term `obj_in_container`: 1 per step while the object rests in the container (xy < `xy_threshold`, `z_min` < z < `success_z_max`, speed small). It is the success term.
  The old `obj_near_container` hover shaping term (0.25/step, farmable) was removed from all scenes.
- `SimEvalsTask` rescales `reward_manager.get_active_iterable_terms(0)` so the terms sum to the env reward (as does `run_pi.py`); `success/<term>` = term ever fired.
  The vectorized path reads `reward_manager._step_reward * step_dt` instead (`steering_training.md`).
- Rigid bodies: scene 1 `rubiks_cube`, `_24_bowl`; scene 2 `_10_potted_meat_can`, `_25_mug`; scene 3 `_11_banana`, `small_KLT_visual_collision` (bin).
- **Thresholds are hand-set** from initial poses (table top z≈0.08; bowl origin z 0.078, bin origin z 0.125): CubeBowl xy<0.05, z∈[0,0.08]; BananaBin xy<0.08,
  z∈[-0.08,0.1] (negative `z_min`: the banana rests below the bin origin). Verify with `run_pi.py` or by hand-placing the object, then adjust `SCENE_TASKS`.
- Episode ends on success or timeout (30 s = 450 steps ≈ 56 chunks at horizon 8). `success_bonus` defaults to 0 (the term already pays 1); 1.0 for rubric envs.
- Default layout is fixed (`reset_scene_to_default`; first reset does the double-reset for material loading), so variance is policy noise only, unless a randomized variant is used.

## Variants
- **`CubeBowlRandom`**: CubeBowl + `randomize_pair_xy` (event after the default reset): object and container xy offset uniformly x∈±0.15, y∈±0.2 m,
  random object yaw, resampled while the pair is <0.15 m apart. Ranges are guesses; check reachability with `run_pi.py`.
- **`*Walled`** (`CubeBowlWalled`, `CanMugWalled`, `BananaBinWalled`, `MultiTaskWalled`, `MultiTaskObjaverseWalled`): `SceneCfg.add_walls()` boxes every env in, because with 7 m env
  spacing `external_cam(_2)` otherwise see neighbouring envs' tables and robots. 4 visual-only cuboid walls (env-local x∈[-0.8,0.9], y∈[-0.62,0.62], z∈[-0.5,1.4], open top,
  dark grey 0.12, roughness 0.9; the front wall touches the table front edge, side walls sit just behind the external cams, back wall behind the robot) and a `room_light` SphereLight
  (intensity 60000, r=0.3, at (0.5, 0, 1.2)) because walls block the dome light. `add_walls(left_wall=False)` drops the +y wall. The defaults are "walls v2"; the evidence is in
  `../experiments/walled_envs.md`.
- **`MultiTask`**: CubeBowl + CanMug + BananaBin in one env (`MultiTaskEnvCfg`, `MULTITASK_TASKS`).
  - Scene: `ensure_multitask_scene()` builds `assets/simevals/scene_multi.usd` on first use (gitignored): scene 1 + the mug/can specs of scene 2 + the banana/bin specs of scene 3
    (`Sdf.CopySpec`, keeps collision overrides). All six objects exist in every env.
  - Env `i` runs task `i % n_tasks`, so use `--num-envs` divisible by the task count (else the split is off by one; the train script warns). `task_ids(env)` is cached on the env.
  - Resets (`randomize_multitask`): the env's object + container get uniform xy in x∈[0.30,0.62], y∈[-0.30,0.30] (table is x 0.20-0.90, y -0.48-0.52) and random yaw, resampled until
    centres are >= `min_dist` apart (CubeBowl/CanMug 0.16, BananaBin 0.30). Other tasks' objects are parked at x=-2, z=-5 (out of every camera). Box and `min_dist` are guesses.
  - Reward/success: one `obj_in_container` term (`multitask_in_container`) dispatching to each env's task (`MULTITASK_SPECS`); `make_parallel_env` turns it into the termination.
  - Prompts: one per env. Privileged state: `MultiTaskPrivileged` (the usual 16 floats + task one-hot).
  - Metrics: the usual `rollout/*`, `eval/fixed/*` plus per-task `<Task>/success/all`, `<Task>/return`, `<Task>/length_steps` (rollout: `<Task>/success/all_ma`).
- **`MultiTaskObjaverse`**: 11 tasks = CubeBowl, CanMug, BananaBin + the 8 objaverse Molmo envs (`Molmo-*2-*2`: bottle2-pot2, candle2-pan2, carrot2-pot2, pen2-mug2, pen2-pot2,
  potato2-pot2, spatula2-pan2, tomato2-pan2). Tasks come from `MULTITASK_ENVS[env]` in `rl/tasks.py`; use `--num-envs` divisible by 11 (e.g. 132).
  - Scene: `ensure_multitask_molmo_scene()` builds `assets/simevals/scene_multi_molmo.usd` (gitignored) = `scene_multi.usd` + the `/World/obj_*` / `/World/cont_*` bodies of the Molmo
    scenes (shared containers pot2 / pan2 copied once, pen2 shared by mug2 and pot2): 16 rigid bodies, all parked except each env's pair. `_rigid_name` ignores `obj_*` / `cont_*` bodies for sim-evals keywords.
  - Resets: sim-evals tasks as MultiTask; Molmo tasks as their single-task env (default xy + the json's `x_range` / `y_range`, json `min_dist`, random object yaw, fixed container orientation).
  - **Status: written but NOT run** (no pxr / GPU on the login node): scene build, rigid-body discovery and reset are unverified. Smoke test:
    `scripts/cluster/run_experiment.py moa-smoke --time 1 -- --environment MultiTaskObjaverse --num-envs 22 --iterations 2 --no-eval-at-start --wandb-mode disabled`.
- `MultiTaskWalled` was launched as `mtwall1` (`--num-envs 129`, residual 0.1) without a smoke test.

## Checking an env/policy: `scripts/run_pi.py`
Port of sim-evals' `run_eval.py`. Saves `all_cams.png`, `episode_<n>_{success,fail}.mp4`, `episode_<n>_rewards.txt` under `runs/simevals_eval/<date>/<time>`. The video side panel shows step,
SUCCESS/FAIL, `return: current / total / max`, and each reward term's value on the current step. `--dummy-policy` needs no server (holds pose, gripper open) and only checks build/reset/step/render/video.
```
export PYTHONPATH=$PWD/src:$PYTHONPATH
isaacpy scripts/run_pi.py --environment CanMug --episodes 1 --dummy-policy
isaacpy scripts/run_pi.py --environment CubeBowl --episodes 2 --policy.port 8000
```
`scripts/run_pi_parallel.py` evaluates many envs at once (tiled cameras, one batched server call per chunk); `scripts/render_walls.py`, `render_noise_test.py` and `render_molmo_envs.py` are render checks.

## Status
CubeBowl, CanMug, BananaBin were confirmed working in the container with the openpi server. Unit tests (`tests/test_rl_logging.py::test_sim_evals_task_*`) use fakes.
