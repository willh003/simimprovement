"""Benchmark: GPU memory + env throughput of a sim-evals env vs num_envs (no policy server needed).

One process per N (Isaac does not free memory cleanly), e.g.:
    for n in 1 2 4 8 16; do isaacpy scripts/bench_envs.py --environment BananaBin --num-envs $n; done
Prints one `BENCH {json}` line. Run with the openpi server up to measure under real memory conditions.
Memory = device-wide used delta (driver level, includes Omniverse/PhysX allocations torch cannot see).
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

import gymnasium as gym
import torch
import tyro
from isaaclab.app import AppLauncher


@dataclass
class BenchArgs:
    environment: str = "BananaBin"
    num_envs: int = 1
    chunks: int = 20  # timed chunks (each = `horizon` env steps)
    horizon: int = 8
    cam_scale: float = 1.0  # multiply all camera resolutions (0.5 -> 640x360)
    drop_cam2: bool = False  # remove the unused external_cam_2
    tiled: bool = False  # use TiledCameraCfg (batched rendering)
    headless: bool = True


def used_mib() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2**20


def main(args: BenchArgs):
    args_cli, _ = argparse.ArgumentParser().parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = args.headless
    simulation_app = AppLauncher(args_cli).app

    from isaaclab_tasks.utils import parse_env_cfg
    import polaris.environments  # noqa: F401  (registers envs)

    base_used = used_mib()
    env_cfg = parse_env_cfg(args.environment, device="cuda", num_envs=args.num_envs, use_fabric=True)
    cams = ["external_cam", "external_cam_2", "wrist_cam"]
    if args.drop_cam2:
        env_cfg.scene.external_cam_2 = None
        env_cfg.observations.policy.external_cam_2 = None
        cams.remove("external_cam_2")
    for name in cams:
        cam = getattr(env_cfg.scene, name)
        cam.width, cam.height = int(cam.width * args.cam_scale), int(cam.height * args.cam_scale)
        if args.tiled:  # one batched render product per camera type instead of one per env
            from isaaclab.sensors import TiledCameraCfg

            cam = TiledCameraCfg(
                prim_path=cam.prim_path, height=cam.height, width=cam.width, data_types=cam.data_types,
                spawn=cam.spawn, offset=TiledCameraCfg.OffsetCfg(pos=cam.offset.pos, rot=cam.offset.rot, convention=cam.offset.convention),
            )
            setattr(env_cfg.scene, name, cam)
    env = gym.make(args.environment, cfg=env_cfg)
    env.reset()
    env.reset()  # second reset loads materials (as in SimEvalsTask)
    after_build = used_mib()

    action = torch.zeros(env.action_space.shape, device="cuda")  # (N, 8): joint-pos targets + gripper
    for _ in range(args.horizon):  # warm-up
        env.step(action)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.chunks * args.horizon):
        env.step(action)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    steps = args.chunks * args.horizon
    result = dict(
        **asdict(args),
        build_mib=round(after_build - base_used),
        peak_mib=round(used_mib() - base_used),
        total_used_mib=round(used_mib()),
        step_ms=round(1000 * dt / steps, 1),
        env_steps_per_s=round(steps * args.num_envs / dt, 1),
        chunks_per_s_per_env=round(args.chunks / dt, 2),
    )
    print("BENCH " + json.dumps(result), flush=True)
    os._exit(0)  # Isaac hangs on normal shutdown


if __name__ == "__main__":
    main(tyro.cli(BenchArgs))
