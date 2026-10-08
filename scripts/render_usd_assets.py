"""Render each USD asset from a 3/4 view (as-is, no re-orientation) into OUT_DIR/<label>.png plus info.json (upAxis, bbox size).

`isaacpy scripts/render_usd_assets.py OUT_DIR label=path.usda [label=path ...]`. Used to eyeball candidate MolmoSpaces assets
(textures, orientation, size) before adding them to assets/molmo/manifest.json.
"""
import argparse
import json
import os
import sys

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("assets", nargs="+", help="label=path")
    ap.add_argument("--size", type=int, default=512)
    a = ap.parse_args()
    from isaaclab.app import AppLauncher

    AppLauncher(argparse.Namespace(enable_cameras=True, headless=True))
    import cv2
    import isaaclab.sim as sim_utils
    import numpy as np
    import torch
    from isaaclab.sensors import Camera, CameraCfg
    from pxr import Gf, Usd, UsdGeom

    os.makedirs(a.out, exist_ok=True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device="cuda:0"))
    sim_utils.DomeLightCfg(intensity=350.0).func("/World/Dome", sim_utils.DomeLightCfg(intensity=350.0))
    sim_utils.DistantLightCfg(intensity=3500.0).func("/World/Key", sim_utils.DistantLightCfg(intensity=3500.0), translation=(0, 0, 5))
    cam = Camera(CameraCfg(prim_path="/World/Cam", height=a.size, width=a.size, data_types=["rgb"],
                           spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.001, 100.0))))
    sim.reset()
    stage = sim.stage
    info = {}
    for spec in a.assets:
        label, path = spec.split("=", 1)
        up = UsdGeom.GetStageUpAxis(Usd.Stage.Open(path))
        sim_utils.UsdFileCfg(usd_path=os.path.abspath(path)).func("/World/asset", sim_utils.UsdFileCfg(usd_path=os.path.abspath(path)))
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        r = cache.ComputeWorldBound(stage.GetPrimAtPath("/World/asset")).ComputeAlignedRange()
        lo, hi = np.array(r.GetMin()), np.array(r.GetMax())
        c, size = (lo + hi) / 2, hi - lo
        dist = 1.6 * np.linalg.norm(size) + 0.05
        eye = c + dist * np.array([0.6, -0.7, 0.45])
        cam.set_world_poses_from_view(torch.tensor(eye, dtype=torch.float32, device="cuda:0")[None], torch.tensor(c, dtype=torch.float32, device="cuda:0")[None])
        for _ in range(8):  # textures stream in over a few frames
            sim.step()
            cam.update(1 / 60)
        img = cam.data.output["rgb"][0, ..., :3].cpu().numpy().copy()
        cv2.putText(img, label[-40:], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        cv2.imwrite(os.path.join(a.out, f"{label}.png"), img[..., ::-1])
        info[label] = dict(upAxis=up, bbox_size=[round(float(x), 3) for x in size], path=path)
        print("rendered", label, info[label]["bbox_size"], up, flush=True)
        stage.RemovePrim("/World/asset")
    json.dump(info, open(os.path.join(a.out, "info.json"), "w"), indent=1)
    sys.stdout.flush()
    os._exit(0)  # the Isaac app hangs on a normal exit
