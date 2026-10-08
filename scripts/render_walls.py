"""Render check for the walled envs: saves ext/ext2/wrist images for a few envs and prints mean brightness.
  isaacpy scripts/render_walls.py --environment CubeBowlWalled --out runs/walls_exp2/render"""
import argparse, os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--environment", nargs="+", default=["CubeBowlWalled"])
p.add_argument("--num-envs", type=int, default=32)
p.add_argument("--out", default="runs/walls_exp2/render")
a = p.parse_args()

from isaaclab.app import AppLauncher
args_cli, _ = argparse.ArgumentParser().parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True
AppLauncher(args_cli)

import mediapy, numpy as np, torch
from polaris.environments.simeval_parallel import make_parallel_env

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for name in a.environment:
    env, cams = make_parallel_env(name, a.num_envs, 0.5, keep_cam2=True)
    env.reset(); obs, _ = env.reset()
    for _ in range(5):
        obs, *_ = env.step(torch.zeros(env.action_space.shape, device="cuda"))
    n = a.num_envs
    rows = [np.concatenate([obs["policy"][c][i].cpu().numpy() for c in cams], axis=1) for i in (0, 1, n // 2, n - 1)]
    mediapy.write_image(out / f"{name}.png", np.concatenate(rows, axis=0))
    print(name, "brightness", [float(obs["policy"][c].float().mean()) for c in cams], flush=True)
    env.close()
os._exit(0)
