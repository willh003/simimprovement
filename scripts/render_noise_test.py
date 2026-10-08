"""Diagnose speckle noise in walled envs. One config per process (Isaac allows one sim per process).
  isaacpy scripts/render_noise_test.py --environment CanMugWalled --tag base --warmups 5 30 100 [--light big|multi] [--spp N] [--rendering-mode quality]
Writes runs/walls_exp2/noise/<tag>.png (rows = warm-up step counts, env 1; cols = ext, ext2, wrist) and prints a noise score
(mean |img - 5x5 box blur| over all envs, per cam; lower = smoother)."""
import argparse, os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--environment", default="CanMugWalled")
p.add_argument("--tag", required=True)
p.add_argument("--num-envs", type=int, default=32)
p.add_argument("--warmups", type=int, nargs="+", default=[5, 30])
p.add_argument("--light", default="default", choices=["default", "big", "multi"])
p.add_argument("--spp", type=int, default=None)
p.add_argument("--rendering-mode", default=None)
a = p.parse_args()

from isaaclab.app import AppLauncher
args_cli, _ = argparse.ArgumentParser().parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True
if a.rendering_mode:
    args_cli.rendering_mode = a.rendering_mode
AppLauncher(args_cli)

import mediapy, numpy as np, torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from polaris.environments import simeval_droid
from polaris.environments.simeval_parallel import make_parallel_env

if a.light != "default":
    orig = simeval_droid.SceneCfg.add_walls

    def patched(self, *args, **kw):
        orig(self, *args, **kw)
        if a.light == "big":  # one large soft light
            self.room_light = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/room_light",
                spawn=sim_utils.SphereLightCfg(intensity=60000.0, radius=0.8),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, 0.0, 1.3)))
        else:  # four lights spread over the room
            for k, (x, y) in enumerate([(0.1, -0.4), (0.1, 0.4), (0.7, -0.4), (0.7, 0.4)]):
                setattr(self, f"room_light{k}", AssetBaseCfg(prim_path=f"{{ENV_REGEX_NS}}/room_light{k}",
                    spawn=sim_utils.SphereLightCfg(intensity=15000.0, radius=0.2),
                    init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, 1.2))))
        if a.light == "multi":
            self.room_light = None
    simeval_droid.SceneCfg.add_walls = patched

import carb.settings
st = carb.settings.get_settings()
if a.spp is not None:
    st.set("/rtx/directLighting/sampledLighting/samplesPerPixel", a.spp)
    st.set("/rtx/reflections/sampledLighting/samplesPerPixel", a.spp)
print("rtx settings:", {k: st.get(k) for k in ["/rtx/directLighting/sampledLighting/samplesPerPixel", "/rtx/post/aa/op", "/rtx/post/dlss/execMode", "/rtx/rendermode"]}, flush=True)

env, cams = make_parallel_env(a.environment, a.num_envs, 0.5, keep_cam2=True)
env.reset(); obs, _ = env.reset()
act = torch.zeros(env.action_space.shape, device="cuda")


def noise(x):  # x: (N,H,W,3) uint8
    x = x.float().permute(0, 3, 1, 2)
    return float((x - torch.nn.functional.avg_pool2d(x, 5, 1, 2)).abs().mean())


out = Path("runs/walls_exp2/noise"); out.mkdir(parents=True, exist_ok=True)
rows, step = [], 0
for w in sorted(a.warmups):
    while step < w:
        obs, *_ = env.step(act); step += 1
    rows.append(np.concatenate([obs["policy"][c][1].cpu().numpy() for c in cams], axis=1))
    print(a.tag, f"step {w}", "brightness", [round(float(obs["policy"][c].float().mean()), 1) for c in cams],
          "noise", [round(noise(obs["policy"][c]), 2) for c in cams], flush=True)
mediapy.write_image(out / f"{a.tag}.png", np.concatenate(rows, axis=0))
os._exit(0)
