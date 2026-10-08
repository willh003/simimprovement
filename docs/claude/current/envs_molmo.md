# MolmoSpaces pick-and-place envs (`Molmo-<obj>-<cont>`)

Generated from MolmoSpaces assets, same reward / privileged state / training as the other sim-evals envs (`envs_simevals.md`): sparse `obj_in_container` term, resets via
`randomize_pair_xy` (default xy ± `x_range`/`y_range` from the spec json, `min_dist` from the spec). Adding a pair = edit the manifest and rerun step 3 below; no code changes.
Not done: one batched env over all pairs for arbitrary pairs (only `MultiTaskObjaverse`, see `envs_simevals.md`).

Thor envs (`Pot_1`/`Pan_1`/`bin_1`/`Plate_1`/bowl containers): apple-bowl, mug-bowl, cup-bowl, cellphone-pot, candle-pan, spatula-pan, cd-bin, tomato-pot, egg-plate, remote-bin,
keychain-plate, pen-pot, potato-pan.
Objaverse envs (manifest keys with suffix `2`, `source: objaverse`, id `obja_<uid>`): bottle2-pot2, candle2-pan2, spatula2-pan2, tomato2-pan2, potato2-pot2, carrot2-pot2, pen2-mug2, pen2-pot2.

## Pipeline
1. `assets/molmo/manifest.json` (gitignored dir): `objects` / `containers` (source `thor`|`objaverse`, id, category, mass, `rot_x_deg`, optional `scale`, `lay_flat`, `roll_deg`,
   `collision`, `com_frac`) and `pairs`.
2. `scripts/download_molmo_assets.py`: installs assets into `assets/molmo/cache` (+ `isaac-assets/` symlinks). Needs `molmospaces-resources`, so run it from a throwaway venv, NOT the
   Isaac python (`uv venv v && uv pip install --python v/bin/python molmospaces-resources==0.0.3a2`; set `UV_CACHE_DIR` if the default is read-only). Setup always installs the whole thor
   set (~3.8 GB). Objaverse UIDs are fetched per entry and picked by hand (the molmospaces CLIP search / `ObjectMeta` was not used).
3. `isaacpy scripts/make_molmo_scenes.py [--pairs apple:bowl]`: writes `assets/simevals/molmo_<obj>_<cont>.usd` + `.json` sidecar. Scene = scene1 (table, light) without cube/bowl +
   `/World/obj_<obj>`, `/World/cont_<cont>` (rigid-body Xform, manifest mass) with the asset referenced in a child Xform (rotated by `rot_x_deg`, origin moved to the bbox bottom
   centre, so body z = height above the table; the table top is computed from the table bbox, ~0.045). Nested RigidBody/Articulation/PhysicsScene prims of the asset are disabled.
   Thor primitive colliders are kept; assets without colliders get convexHull (object) / convexDecomposition (container). Poses reuse scene 1's cube/bowl xy.
4. Registration is automatic: `polaris/rl/molmo_specs.py` loads the sidecars (no Isaac imports); `rl/tasks.py`, `simeval_droid.py` (`SCENE_TASKS`, `MolmoEnvCfg`, scene file
   `<scene>.usd` for names starting `molmo_`) and `environments/__init__.py` pick them up.
5. `scripts/inspect_usd.py <usd...>`: prim tree, physics schemas, collision approximation, mass, bbox (headless Isaac app).

Objaverse browsing: `assets/molmo/isaac_objaverse_uids.txt` (all 129,709 UIDs of the Isaac USD set), `assets/molmo/isaac_objaverse_index.csv` (category, size, mass, description,
license from the objathor metadata in `assets/molmo/meta`, fetched with `R2RemoteStorage("mujoco-thor-resources")`, source `objathor_metadata` 20260129). Candidates:
`scripts/download_molmo_assets.py --uids ...` then `scripts/render_usd_assets.py OUT label=path ...` (one process, 3/4-view renders).

## Asset findings (orientation, colliders)
- Thor USDs declare `upAxis=Z` but the geometry is **Y-up**: every thor asset except apple / tomato / potato (round) needs `rot_x_deg=90`. Objaverse USDs are metres, Y-up: `rot_x_deg=90`,
  no scale, no colliders (the generator adds hull / decomposition). Check the orientation of every new asset with `run_pi.py --dummy-policy` (`all_cams.png`).
- `lay_flat: true` (extra 90 deg about y) puts a tall thin object on its side; `roll_deg` rolls it about its long axis (pen2, carrot2, spatula2; the spatula needs roll 90 to rest on its flat face).
- Thor Box_* (0.43x0.32x0.49 m articulated cardboard) are too big, not used.
- Thor Pot_1 / Pan_1 are shallow square dishes with a long handle (bbox 0.29x0.5 / 0.28x0.45): the xy origin is the centre of the bottom 25 % slice of mesh vertices (the handle does not shift it)
  and the thresholds use that core size.
- `bin_*` are 0.5x0.34x0.6 m wheelie bins: `scale 0.5`, mass 1 kg, `com_frac` 0.15, `"collision": "walls"` (invisible floor + 4 wall boxes from the mesh bbox, wall 12 mm). Mesh colliders
  (convex decomposition fills the floor; SDF explodes with thin objects) and the asset's primitive colliders (sink 5 cm into the table) were all bad. The tall bin can land under the gripper's home pose.
- The spec json's `x_range` / `y_range` are the reset offsets (default ±0.15 / ±0.2) clamped so object and container footprints stay on the table.
- One env per Isaac process: render with `scripts/render_molmo_envs.py OUT env_id` (images in `runs/molmo_images/`).

## Success criterion (all Molmo envs)
`object_in_container` (`simeval_droid.py`); thresholds are auto-derived by `make_molmo_scenes.py` into each sidecar (`thresholds`, `success_z_max`). The object body origin is its bbox **centre**
(container origin = bottom centre), so a toppled object is still judged by where its middle is. Success iff xy distance to the container origin < `0.48 * min(container core size)`,
`0 < z < success_z_max` (= container height + 0.6 x (shallow containers, <0.12 m) or 0.25 x (deep) the object's thinnest dimension, so an object resting ON the rim does not count), and speed < 0.1 m/s.

`scripts/check_molmo_success.py ENV OUT` (one env per process) drops the object into the container (`success_center` / `success_offset`) and places it beside, hovering above, and far from it
(`fail_*`), evaluates the criterion and saves images to `runs/molmo_images/success_checks/<env>/` (+ `montage.png`, `results.json`). Results: 20/21 envs match on all 5 cases;
`Molmo-tomato2-pan2` `success_offset` mismatches (the round tomato bounces out of the shallow pan; a test limitation). Success drops align the object's long axis with the container's
(the remote bridges the bin rim otherwise); a spatula (0.3 m) is longer than the pan body (0.27 m).

## Verification status
Done: `run_pi.py --environment Molmo-X-bowl --dummy-policy` (objects upright and stable); hand-placing the object 3 cm above the bowl origin fires `obj_in_container` (apple, cup);
2-iteration dummy-policy smoke train on cup-bowl (before the object-origin / threshold / bin changes). Not verified: the mug env's manual success check, thresholds/mass tuning, and a cluster
launch with a real policy (`../handoffs/launch_molmo_cupbowl.md`).
