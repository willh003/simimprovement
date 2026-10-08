# Plan: Generate many pick-and-place envs from MolmoSpaces assets

## Context
Goal: train RL on many "object -> container" envs like BananaBin/CubeBowl. Findings:
- Polaris envs are not per-env code. Each is a scene USD (`assets/simevals/sceneN.usd`, rigid bodies as direct `/World` children with `xformOp:translate/orient`) plus an entry in `SCENE_TASKS` (`src/polaris/environments/simeval_droid.py:40`: object/container name keywords, xy/z thresholds), an EnvCfg calling `set_scene`, gym registration (`environments/__init__.py:20`), and instruction/success-term entries in `rl/tasks.py:10-24`. Reward, privileged state, and train_steering all work off `object_in_container` automatically. `MultiTask` already combines several tasks in one scene (`ensure_multitask_scene`, `MULTI_EXTRA_PRIMS`).
- `/gscratch/weirdlab/will/molmospaces` is just the allenai/molmospaces repo; **no assets on disk**. It has no Isaac manipulation envs, only asset tooling. USD (.usda) objaverse assets are already converted upstream and downloadable per-UID via `scripts/assets/usda_downloader.py::install_objaverse_uid(uid)` (HF dataset `allenai/molmospaces`, prefix `isaac`, version 20260128). Metadata (category, mass, bbox, license) comes from `molmo_spaces.utils.object_metadata.ObjectMeta` and CLIP search from `object_retriever.ObjectRetriever().query("a bowl")`.
- So the real work: download a few objects, check their USD is usable as a Polaris rigid body, then auto-generate scene USDs + registrations from a small manifest.

## Steps (minimal, incremental)
1. **Download 4-5 assets only** (2-3 pickables e.g. mug/banana-like/toy, 2 containers e.g. bowl/bin/box) into `assets/molmo/` (own dir, outside the repo's git). Install `molmospaces-resources` into the ogbench venv (no Isaac needed) and use `install_objaverse_uid` + `ObjectRetriever`/`ObjectMeta` to pick UIDs; record chosen UIDs, category, bbox, mass in `assets/molmo/manifest.json`. Fall back to the thor set (always-installed, has Bowl/Box/Cup/Mug/Plate) for clean containers if objaverse containers are poor.
2. **Inspect inside apptainer (isaacpy)**: write `scripts/inspect_usd.py` (also fixes the missing `inspect_simevals_scene.py` referenced in errors) to print per-asset: prim hierarchy, RigidBodyAPI/CollisionAPI presence, collision approximation, mass, bbox, up-axis/units. Decide what normalization is needed (scale to realistic size, collision approx = convexDecomposition for containers, convexHull for objects, mass default, rest orientation).
3. **Scene generator** `scripts/make_molmo_scenes.py` (run in apptainer): for each (object, container) pair in the manifest, copy `scene1.usd` as a base (table etc., same approach as `ensure_multitask_scene`), delete `rubiks_cube`/`_24_bowl`, add the molmo object/container as `/World/<obj>` and `/World/<cont>` references with RigidBodyAPI, applied scale, collision approximations, and init xy poses placed like scene1 (object on table, container ~0.2 m apart; container z from bbox). Output `assets/simevals/molmo_<obj>_<cont>.usd`. Keep sidecar json with name keywords, bbox-derived thresholds (xy_threshold ~ container inner radius, z_min/z_max from bboxes), and instruction ("put the {obj category} in the {container category}").
4. **Dynamic registration**: small change in `simeval_droid.py` / `environments/__init__.py` / `rl/tasks.py` so `SCENE_TASKS`, `SIM_EVALS_INSTRUCTIONS`, `SIM_EVALS_SUCCESS_TERMS`, and gym registration are additionally populated from the generated sidecar json files (ids like `Molmo-<obj>-<cont>`), instead of hand-editing 4 files per env. Reuse `set_scene`, `object_in_container`, and existing randomization (`randomize_pair_xy`) so resets vary.
5. **Verify one env**: `isaacpy scripts/run_pi.py --environment Molmo-X-Y --episodes 1 --dummy-policy`, look at renders (object stable on table, container not exploding, success fires when object manually placed). Then smoke train: `scripts/cluster/run_experiment.py ... -- --environment Molmo-X-Y --num-envs 8 --iterations 2 --no-eval-at-start --wandb-mode disabled`.
6. **Defer** (only after 1 env works): batched multi-env training across all generated pairs. Extend the MultiTask mechanism (single combined scene, per-env task id, `MULTITASK_SPECS`) to read from the manifest; note objects need different bodies per env, so this follows the existing park-at-(-2,-1,-5) trick. Also scale to more UIDs.
7. Write `docs/claude/molmo_envs.md` describing the pipeline, and update `docs/claude/simevals_env.md` link.

## Critical files
- Modify: `src/polaris/environments/simeval_droid.py`, `environments/__init__.py`, `src/polaris/rl/tasks.py`
- New: `scripts/inspect_usd.py`, `scripts/make_molmo_scenes.py`, `scripts/download_molmo_assets.py`, `assets/molmo/manifest.json`, `docs/claude/molmo_envs.md`
- Reference only: molmospaces `scripts/assets/usda_downloader.py`, `molmo_spaces/utils/object_retriever.py`, `object_metadata.py`

## Risks
- Objaverse USD may lack collision/rigid-body schema, wrong scale, or non-upright rest pose; step 2 decides normalization. Concave containers need convex decomposition; objaverse bins may be poorly modeled.
- Some objaverse licenses are non-commercial; set `MLSPACES_LICENSE_POLICY` if that matters.
- Downloader needs network (HF) from a node that has it; apptainer likely lacks the pip package, so do downloads from the venv outside it.

## Verification
Step 5 above (dummy-policy run with renders + manual success check, then 2-iteration smoke train).
