# MolmoSpaces pick-and-place envs (`Molmo-<obj>-<cont>`)

Plan: `molmo_envs_plan.md` (steps 1-5 + 7 done; step 6, one batched env over all pairs, deferred).
Envs so far (all thor assets, `Pot_1`/`Pan_1`/`bin_1`/`Plate_1` containers): apple-bowl, mug-bowl, cup-bowl, cellphone-pot, candle-pan,
spatula-pan, cd-bin, tomato-pot, egg-plate, remote-bin, keychain-plate, pen-pot, potato-pan (ids `Molmo-<obj>-<cont>`). Same reward / privileged state / training as the
other sim-evals envs (`simevals_env.md`): `obj_in_container` term, resets via `randomize_pair_xy` (x±0.15, y±0.2, `min_dist` from the spec).

## Pipeline
1. `assets/molmo/manifest.json` (gitignored dir): `objects` / `containers` (source `thor`|`objaverse`, id, category, mass, `rot_x_deg`, optional
   `scale`) and `pairs`.
2. `scripts/download_molmo_assets.py`: installs the assets into `assets/molmo/cache` (+ `isaac-assets/` symlinks). Needs `molmospaces-resources`,
   so run it from a throwaway venv, NOT the Isaac python (`uv venv v && uv pip install --python v/bin/python molmospaces-resources==0.0.3a2`;
   set `UV_CACHE_DIR` if the default is read-only). The resource manager setup always installs the whole thor set (~3.8 GB). Objaverse UIDs are
   fetched per entry (untested: the CLIP search / `ObjectMeta` of the molmospaces repo was not used, pick UIDs by hand).
3. `isaacpy scripts/make_molmo_scenes.py [--pairs apple:bowl]`: writes `assets/simevals/molmo_<obj>_<cont>.usd` + `.json` sidecar. Scene = scene1
   (table, light) without cube/bowl + `/World/obj_<obj>`, `/World/cont_<cont>` (rigid-body Xform, mass from manifest) with the asset
   referenced in a child Xform (rotated by `rot_x_deg`, origin moved to the bbox bottom centre, so body z = height above the table).
   Nested RigidBody/Articulation/PhysicsScene prims of the asset are disabled. Thor assets ship primitive colliders (bowl 20 boxes, mug 10) which
   are kept; assets without colliders get convexHull (object) / convexDecomposition (container). Poses reuse scene 1's cube/bowl xy.
   Thresholds: `xy_threshold = 0.35 * min container footprint`, `z_min=-0.01`, `success_z_max = container height` (object origin = bottom).
4. Registration: `polaris/rl/molmo_specs.py` loads the sidecars (no Isaac imports); `rl/tasks.py` (instructions, success term),
   `simeval_droid.py` (`SCENE_TASKS`, `MolmoEnvCfg`, scene file `<scene>.usd` for names starting `molmo_`) and `environments/__init__.py` (gym ids)
   pick them up automatically. New pair = edit the manifest, rerun step 3, no code changes.
5. `scripts/inspect_usd.py <usd...>`: prim tree, physics schemas, collision approx, mass, bbox (runs under a headless Isaac app).

Objaverse envs (keys with suffix `2` in the manifest, `source: objaverse`, id `obja_<uid>`): bottle2-pot2, candle2-pan2, spatula2-pan2, tomato2-pan2,
potato2-pot2, carrot2-pot2, pen2-mug2, pen2-pot2. UID lists: `assets/molmo/isaac_objaverse_uids.txt` (all 129,709 UIDs of the Isaac USD set),
`assets/molmo/isaac_objaverse_index.csv` (category, size, mass, description, license from the objathor metadata; the metadata itself is in
`assets/molmo/meta`, fetched with `R2RemoteStorage("mujoco-thor-resources")`, source `objathor_metadata` 20260129). Browse candidates with
`scripts/download_molmo_assets.py --uids ...` + `scripts/render_usd_assets.py OUT label=path ...` (one process, 3/4-view renders).
Manifest `lay_flat: true` (extra 90 deg about y) puts a tall thin object on its side, `roll_deg` rolls it about its long axis (used for pen2, carrot2, spatula2; the spatula needs roll 90 to rest on its flat face).
Objaverse USDs are in metres and Y-up (height on raw Y): `rot_x_deg=90`, no scale needed, no colliders (generator adds hull / decomposition).

## Success criterion (all Molmo envs)
`object_in_container` (`simeval_droid.py`), thresholds auto-derived by `make_molmo_scenes.py` into each sidecar json (`thresholds`, `success_z_max`):
object body origin = its bbox **centre** (container origin = bottom centre), so a toppled object is still judged by where its middle is. Success iff
xy distance to the container origin < `0.48 * min(container core size)`, `0 < z < success_z_max` (= container height + 0.6 (shallow containers, <0.12 m) or 0.25 (deep) x the
object's thinnest dimension, so an object resting ON the rim does not count), and speed < 0.1 m/s. `success_z_max` for the bins etc. is in the json.
Bins (`bin_1`) use `"collision": "walls"` (invisible floor + 4 wall boxes from the mesh bbox, wall 12 mm), `scale 0.5`, mass 1 kg, `com_frac` 0.15 (low centre of mass):
mesh colliders (convex decomposition fills the floor, SDF explodes with thin objects) and the asset's primitive colliders (sink 5 cm into the table) were all bad.
`scripts/check_molmo_success.py ENV OUT` (one env per process) drops the object into the container (success_center / success_offset) and places it beside,
hovering above, and far from it (fail_*), evaluates the criterion and saves images to `runs/molmo_images/success_checks/<env>/` (+ `montage.png`, `results.json`).
Results: 20/21 envs match on all 5 cases; `Molmo-tomato2-pan2` success_offset mismatches (the round tomato bounces out of the shallow pan, a test limitation).
Success drops align the object's long axis with the container's (the remote bridges the bin rim otherwise); a spatula (0.3 m) is longer than the pan body (0.27 m).

## Findings
- Thor USDs declare `upAxis=Z` but the geometry is **Y-up** (bowl, mug, cup lie on their side): they need `rot_x_deg=90`. The apple is round and
  needs 0 (check orientation with `run_pi.py --dummy-policy` `all_cams.png` for every new asset).
- Table top in scene 1 is z≈0.045 (the docs' 0.08 was off); the generator computes it from the table bbox.
- Thor Box_* are 0.43x0.32x0.49 m articulated cardboard boxes: too big, not used.

- Thor Pot_1 / Pan_1 are shallow square dishes with a long handle (bbox 0.29x0.5 / 0.28x0.45): the xy origin is the centre of the bottom 25 % slice of
  the mesh vertices (so the handle does not shift it), and `xy_threshold` uses that core size.
- `bin_*` are 0.5x0.34x0.6 m wheelie bins: manifest `scale: 0.5`. Their primitive colliders gave an unstable body, so the manifest uses
  `"collision": "mesh"` (convex decomposition of the meshes, primitives disabled). The tall bin can land under the gripper's home pose (arm nudges it).
- The spec json has `x_range`/`y_range`: reset offsets (default +-0.15 / +-0.2) clamped so object and container footprints stay on the table.
- Multiple envs in one Isaac process hang on the second `gym.make`; render one env per process (`scripts/render_molmo_envs.py OUT env_id`, images in `runs/molmo_images/`).
- Orientation: every thor asset except apple / tomato / potato needed `rot_x_deg=90` (Y-up geometry); tomato / potato are roughly round so 0 looks fine.

Launch instructions for a training run: `launch_molmo_cupbowl_run.md`.

## Verification (done)
`run_pi.py --environment Molmo-X-bowl --dummy-policy`: objects upright and stable. Hand-placing the object 3 cm above the bowl origin makes
`obj_in_container` fire (apple, cup). Smoke train: `scripts/train_steering.py --environment Molmo-cup-bowl --dummy-policy --num-envs 8
--iterations 2 --no-eval-at-start --wandb-mode disabled` runs (2 its + final eval) inside the apptainer. Not yet verified: the mug env's manual
success check, thresholds/mass tuning, and the cluster launch with a real policy.
