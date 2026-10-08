"""Save the initial camera images (ext cam 1, ext cam 2, wrist) of Molmo envs: `isaacpy scripts/render_molmo_envs.py OUT_DIR [env ids...]` (default all)."""
import argparse
import os
import sys

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("envs", nargs="*")
    a = ap.parse_args()
    from isaaclab.app import AppLauncher

    app = AppLauncher(argparse.Namespace(enable_cameras=True, headless=True)).app
    import gymnasium as gym
    import mediapy
    import numpy as np
    from isaaclab_tasks.utils import parse_env_cfg

    import polaris.environments  # noqa: F401
    from polaris.rl.molmo_specs import MOLMO_SPECS

    os.makedirs(a.out, exist_ok=True)
    for eid in a.envs or MOLMO_SPECS:
        env = gym.make(eid, cfg=parse_env_cfg(eid, device="cuda", num_envs=1, use_fabric=True))
        env.reset()
        obs, _ = env.reset()  # second render cycle loads materials
        cams = [obs["policy"][n][0].cpu().numpy() for n in ("external_cam", "external_cam_2", "wrist_cam")]
        mediapy.write_image(os.path.join(a.out, f"{eid}.png"), np.concatenate(cams, axis=1))
        print("saved", eid, flush=True)
        env.close()
    sys.stdout.flush()
    os._exit(0)  # the Isaac app hangs on a normal exit
