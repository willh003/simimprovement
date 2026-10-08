# Walled sim-evals envs (`CubeBowlWalled`, `CanMugWalled`, `BananaBinWalled`)

Why: with parallel envs (7 m spacing) `external_cam` / `external_cam_2` see other envs' tables and robots on the horizon (verified by rendering; mostly the interior envs of the grid). The walled envs box each env in.

Implementation (`SceneCfg.add_walls` in `simeval_droid.py`, `*WalledEnvCfg`, registered in `environments/__init__.py`, tables in `rl/tasks.py`):
- 4 visual-only cuboid walls (no collision), env-local bounds x∈[-0.8,0.9], y∈[-0.62,0.62], z∈[-0.5,1.4], open top, dark grey 0.12 (was 0.75 white-ish; the Franka was invisible against it), roughness 0.9. Front wall touches the table's front edge; side walls sit just behind the external cams (y=±0.57, so they cannot be flush with the table edges at -0.48/0.52); back wall is behind the robot.
- Walls block the shared dome light (table renders black, wrist cam noisy), so each env also gets a `room_light` SphereLight (intensity 60000, r=0.3) at (0.5, 0, 1.2). Tried: emissive walls (noisy wrist cam, worse), 20000 intensity (table darker). The polished table still looks darker in `external_cam` than the unwalled env (specular reflection of the room), mean image brightness 189/195/160 vs 177/179/176 (ext/ext2/wrist) unwalled.
- Check render: a script like the one in this session (`make_parallel_env(..., keep_cam2=True)`, save `obs["policy"][cam][i]` for several envs).

## Experiment: eval with vs without walls (pi05_droid_jointpos, 128 envs, cam_scale 0.5, 1 round, open-loop horizon 8)
`scripts/run_pi_parallel.py`, outputs in `runs/walls_exp/<Env>/summary.json` (videos for 2 envs each). Fixed single layout, so variance is policy noise only. Server shared the GPU (mem fraction 0.25, `--max-batch-size 32`; 0.4 OOMed at 128 envs).

| task | no walls | walls | Fisher p |
|---|---|---|---|
| CubeBowl | 120/128 (93.8%) | 127/128 (99.2%) | 0.036 |
| BananaBin | 3/128 (2.3%) | 25/128 (19.5%) | <0.001 |
| CanMug | 8/128 (6.2%) | 1/128 (0.8%) | 0.036 |

Single run per condition (no repeats/seeds). Walls change more than "no neighbours" (background, lighting, table reflections), so this is not a clean attribution to the leaked robots.

## How the experiment was run (reproduce from scratch, inside the apptainer on one L40)
No sbatch and no `/gscratch` inside the container, so `scripts/start_pi_server.sh` fails (read-only cache dir). Start the server by hand with a writable cache (network works; first start downloads ~12 GB and takes ~10 min):
```
cd third_party/openpi
TQDM_DISABLE=1 OPENPI_DATA_HOME=<writable dir> XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 nohup .venv/bin/python scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config=pi05_droid_jointpos_polaris --policy.dir=gs://openpi-assets/checkpoints/pi05_droid_jointpos > serve.log 2>&1 &
# ready when `echo > /dev/tcp/localhost/8000` succeeds (the websocket "InvalidMessage" error in serve.log from that probe is harmless)
```
Fraction 0.4 OOMs Isaac at 128 envs; 0.25 gives ~12 GB used by the server, ~40 GB total with Isaac.
Then run each env sequentially (~12-20 min each; Isaac `os._exit`s itself at the end):
```
export PYTHONPATH=$PWD/src:$PYTHONPATH
for t in CubeBowl BananaBin CanMug; do for v in "" Walled; do
  /isaac-sim/python.sh scripts/run_pi_parallel.py --environment ${t}${v} --num-envs 128 --cam-scale 0.5 --episodes 1 \
    --num-videos 2 --policy.port 8000 --max-batch-size 32 --run-folder runs/walls_exp/${t}${v} > runs/walls_exp/${t}${v}.log 2>&1
done; done
```
Read `runs/walls_exp/<Env>/summary.json` (`success_rate`). Gotchas: `pkill -f <pattern>` can kill your own shell if the pattern is in the command line (use `pgrep` + `kill`); a crashed Isaac process can linger holding GPU memory (`kill -9`).
Wall/lighting render check: build with `make_parallel_env(env, 32, 0.5, keep_cam2=True)`, reset twice, step zeros 5x, save `obs["policy"][cam][i]` for envs 0,1,n//2,n-2,n-1 (all three cams, concatenated), and compare mean brightness to the unwalled env.

## Experiment v2: tight dark-grey walls (runs/walls_exp2/, same settings, 128 envs, separate pi-server and sim L40 nodes)
Pi server via `scripts/cluster/pi_server_job.sbatch` (full GPU, `--max-batch-size 128`), sim via `runs/walls_exp2/sim_job.sbatch` (+`in_container.sh`). Render check: `scripts/render_walls.py` (images in `runs/walls_exp2/render/`; mean brightness ext/ext2/wrist ~147/154/127 CubeBowl). No-walls numbers are from the table above (env unchanged).

| task | no walls | walls v1 | walls v2 |
|---|---|---|---|
| CubeBowl | 120/128 | 127/128 | 121/128 (94.5%) |
| BananaBin | 3/128 | 25/128 | 31/128 (24.2%) |
| CanMug | 8/128 | 1/128 | 0/128 |

## Render noise diagnosis (CanMugWalled, `scripts/render_noise_test.py`, output `runs/walls_exp2/noise/`)
Speckle in walled CanMug/BananaBin comes from the single small `room_light` (sphere r=0.3): direct-light sampling noise, worst in `external_cam_2` (noise score = mean |img - 5x5 blur|; ext/ext2/wrist):
| variant | brightness | noise score |
|---|---|---|
| base walled (step 5/30/100) | 129/114/123 | 3.4/**15.3**/6.0 (no change with more warm-up steps) |
| `/rtx/...sampledLighting/samplesPerPixel`=8 | 138/125/155 | 3.7/14.9/2.9 (ext2 not fixed) |
| `--rendering-mode quality` | 133/6/4 | broken (ext2/wrist go black) |
| one big light (r=0.8, 60000) | 200/201/162 | 3.3/3.0/2.6 but overexposed/white |
| 4 lights (r=0.2, 15000 each) | 96/88/122 | 2.8/2.1/2.3 clean, a bit dark |
Unwalled reference scores ~7 (texture detail, not speckle). Fix candidate: several/larger lights with higher intensity (~25-30k each); not applied yet, so the v2 experiment results above used the noisy single light.

## Experiment v3: 4-light rooms (runs/walls_exp3/; `add_walls` now spawns room_light0-3, r=0.2, 28000 each)
Render check (`render_noise_test.py`, 1 warm-up step): brightness ext/ext2/wrist CanMug 119/109/99, BananaBin 135/134/122, CubeBowl 135/139/123; mostly clean (BananaBin ext2 still shows speckle on the far wall in some envs).
| task | no walls | walls v2 (1 light, noisy) | walls v3 (4 lights) |
|---|---|---|---|
| CubeBowl | 120/128 | 121/128 | 98/128 (76.6%) |
| BananaBin | 3/128 | 31/128 | 20/128 (15.6%) |
| CanMug | 8/128 | 0/128 | 0/128 |

## Experiment v4: left (+y) wall removed (runs/walls_exp4/; `add_walls(left_wall=False)` is now the default)
The +y wall sits right behind `external_cam` (y=+0.57) and rendered very fuzzy in it. Without it, `external_cam` sees the dome HDRI background on its left (no neighbour envs seen); the dome light also lights the room again (brightness ext/ext2/wrist: CubeBowl 144/150/131, BananaBin 175/174/160, CanMug 123/113/90). Wrist cam still shows the (fuzzy) +x/-y walls; CanMug ext2 is still noisy (score 10).
| task | no walls | v2 (1 light) | v3 (4 lights) | v4 (4 lights, no left wall) |
|---|---|---|---|---|
| CubeBowl | 120/128 | 121/128 | 98/128 | 124/128 (96.9%) |
| BananaBin | 3/128 | 31/128 | 20/128 | 11/128 (8.6%) |
| CanMug | 8/128 | 0/128 | 0/128 | 2/128 (1.6%) |

## Decision: default setup reverted to "walls v2" (1 light r=0.3 / 60000, all 4 walls, dark grey 0.12)
It performed best in the experiments above (CubeBowl 121, BananaBin 31, CanMug 0). 4-light and `left_wall=False` code paths: the left wall is still switchable (`left_wall=False`); the 4-light variant lives only in `scripts/render_noise_test.py --light multi`.
