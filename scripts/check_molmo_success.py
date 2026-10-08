"""Check the success criterion (object_in_container) of a Molmo env on hand-made success / failure placements and save images.

`isaacpy scripts/check_molmo_success.py Molmo-apple-bowl OUT_DIR` writes OUT_DIR/<env>/<case>.png (both external cams, label =
expected vs measured) and results.json. One env per process (Isaac hangs on a second gym.make).
Cases: success_center / success_offset (object dropped into the container), fail_beside (on the table next to it),
fail_hover (above the container, still falling), fail_far (far side of the table).
"""
import argparse
import json
import os
import sys

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("out")
    a = ap.parse_args()
    from isaaclab.app import AppLauncher

    AppLauncher(argparse.Namespace(enable_cameras=True, headless=True))
    import cv2
    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    import polaris.environments  # noqa: F401
    from polaris.environments.simeval_droid import object_in_container
    from polaris.rl.molmo_specs import MOLMO_SPECS

    spec = MOLMO_SPECS[a.env]
    env = gym.make(a.env, cfg=parse_env_cfg(a.env, device="cuda", num_envs=1, use_fabric=True))
    obs, _ = env.reset()
    obs, _ = env.reset()
    u, sc = env.unwrapped, env.unwrapped.scene
    params = dict(u.cfg.rewards.obj_in_container.params)
    obj, cont = sc[params["object_cfg"].name], sc[params["container_cfg"].name]
    org = sc.env_origins[0]
    dev = "cuda"
    hold = torch.cat([obs["policy"]["arm_joint_pos"].clone(), torch.zeros(1, 1, device=dev)], -1)
    cpos = cont.data.default_root_state[0, :3].clone()
    robot = sc["robot"]
    ee = (robot.data.body_pos_w[0, robot.body_names.index("base_link")] - org)  # gripper (held pose): keep the container away from it
    cy = float(ee[1]) + (0.3 if float(ee[1]) < 0.05 else -0.3)
    cpos[:2] = torch.tensor([0.55, float(np.clip(cy, -0.3, 0.35))], device=dev)  # fixed container spot
    ch = spec["container_size"][2]
    osz, csz = spec["object_size"], spec["container_size"]
    thr = spec["thresholds"]["xy_threshold"]

    def put(asset, pos, quat):
        pose = torch.cat([(pos + org)[None], quat[None]], -1)
        asset.write_root_pose_to_sim(pose)
        asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))

    oq, cq = obj.data.default_root_state[0, 3:7].clone(), cont.data.default_root_state[0, 3:7].clone()
    oz = obj.data.default_root_state[0, 2]
    side = 0.5 * max(csz[:2]) + 0.5 * max(osz[:2]) + 0.05
    top = ch + 0.5 * max(osz)  # object-centre height at which the object's bottom is level with the container rim (origin = bbox centre)
    cases = {  # name: (expected, object xyz offset from container origin, env steps before checking)
        "success_center": (True, (0.0, 0.0, top + 0.03), 120),
        "success_offset": (True, (0.4 * thr * 0.7, 0.4 * thr * 0.7, top + 0.03), 120),
        "fail_beside": (False, (-side, 0.0, float(oz - cpos[2])), 40),
        "fail_hover": (False, (0.0, 0.0, top + 0.12), 1),
        "fail_far": (False, (-0.25, -0.45, float(oz - cpos[2])), 40),
    }
    odir = os.path.join(a.out, a.env)
    os.makedirs(odir, exist_ok=True)
    results = {}
    # success drops: turn the object 90 deg about z when its long side is across the container's long side
    turn = (osz[0] > osz[1]) != (csz[0] > csz[1])
    qz = torch.tensor([0.7071068, 0.0, 0.0, 0.7071068], device=dev)
    def qmul(a, b):
        w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
        return torch.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2, w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])
    for name, (expect, off, n) in cases.items():
        put(cont, cpos, cq)
        put(obj, cpos + torch.tensor(off, device=dev), qmul(qz, oq) if (turn and name.startswith("success")) else oq)
        for _ in range(n):
            obs, *_ = env.step(hold)
        val = float(object_in_container(u, **params)[0])
        rel = (obj.data.root_pos_w[0] - cont.data.root_pos_w[0]).tolist()
        ok = (val > 0) == expect
        results[name] = dict(expected=expect, measured=bool(val > 0), ok=ok, rel_pos=[round(x, 3) for x in rel],
                             speed=round(float(obj.data.root_lin_vel_w[0].norm()), 3))
        cams = [obs["policy"][c][0].cpu().numpy() for c in ("external_cam", "external_cam_2")]
        img = cv2.resize(np.concatenate(cams, 1), None, fx=0.5, fy=0.5).copy()
        txt = f"{a.env} {name}: expected {'SUCCESS' if expect else 'FAIL'}, criterion {'SUCCESS' if val > 0 else 'FAIL'}"
        bar = np.full((34, img.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, txt + (" [OK]" if ok else " [MISMATCH]"), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 140, 0) if ok else (200, 0, 0), 2)
        cv2.imwrite(os.path.join(odir, f"{name}.png"), np.concatenate([bar, img], 0)[..., ::-1])
        print(a.env, name, results[name], flush=True)
    json.dump(dict(thresholds=spec["thresholds"], success_z_max=spec["success_z_max"], cases=results), open(os.path.join(odir, "results.json"), "w"), indent=1)
    sys.stdout.flush()
    os._exit(0)  # the Isaac app hangs on a normal exit
