"""
Example script for running 10 rollouts of a DROID policy on the example environment.

Usage:

First, make sure you download the simulation assets and unpack them into the root directory of this package.

Then, in a separate terminal, launch the policy server on localhost:8000 
-- make sure to set XLA_PYTHON_CLIENT_MEM_FRACTION to avoid JAX hogging all the GPU memory.

For example, to launch a pi0-FAST-DROID policy (with joint position control), 
run the command below in a separate terminal from the openpi "karl/droid_policies" branch:

XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid_jointpos --policy.dir=s3://openpi-assets-simeval/pi0_fast_droid_jointpos

Finally, run the evaluation script:

python run_eval.py --episodes 10 --headless
"""

import numpy as np
import tyro
import argparse
import gymnasium as gym
import torch
import cv2
import mediapy
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from sim_evals.inference.droid_jointpos import Client as DroidJointPosClient
from sim_evals.inference.dummy import Client as DummyClient


SUCCESS_TERM = "can_in_mug"  # reward term that defines task success


def overlay_frame(frame, step, term_vals, term_cums, running_return, ep_return, success, success_step):
    """Return `frame` with a text panel appended to its right (the images are not covered or resized).

    Shows step, SUCCESS/FAIL, and each reward term's current value and running total.
    """
    frame = np.ascontiguousarray(frame)
    h, w = frame.shape[:2]
    scale = max(h / 480, 0.4)
    th = 1 if scale < 0.8 else 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    lh = int(38 * scale)
    panel = np.zeros((h, int(560 * scale), 3), dtype=np.uint8)

    reached = success and success_step is not None and step >= success_step
    status, status_color = ("SUCCESS", (0, 220, 0)) if reached else (
        ("FAIL", (255, 60, 60)) if not success else ("...", (200, 200, 200))
    )
    rows = [
        (f"step {step}", (255, 255, 255)),
        (status + ("" if not success or reached else f" (at {success_step})"), status_color),
        (f"return: {running_return:.2f} / {ep_return:.2f}", (255, 255, 255)),
        ("reward terms (now / total)", (160, 160, 160)),
    ]
    for name, v in term_vals.items():
        color = (0, 255, 0) if v > 0 else (255, 120, 120) if v < 0 else (255, 255, 255)
        rows.append((f"{name}: {v:+.2f} / {term_cums[name]:.2f}", color))
    for i, (text, color) in enumerate(rows):
        cv2.putText(panel, text, (8, lh * (i + 1)), font, scale * 0.9, color, th, cv2.LINE_AA)
    if reached:
        cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, h - 1), (0, 200, 0), max(int(4 * scale), 2))
    return np.concatenate([frame, panel], axis=1)


def main(
        episodes:int = 10,
        headless: bool = True,
        scene: int = 2,
        remote_host: str = "localhost",
        remote_port: int = 8000,
        dummy_policy: bool = False,
        ):
    # launch omniverse app with arguments (inside function to prevent overriding tyro)
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # All IsaacLab dependent modules should be imported after the app is launched
    import sim_evals.environments # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg


    # Initialize the env
    env_cfg = parse_env_cfg(
        "DROID",
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )
    instruction = None
    match scene:
        case 1:
            instruction = "put the cube in the bowl"
        case 2:
            instruction = "put the can in the mug"
        case 3:
            instruction = "put banana in the bin"
        case _:
            raise ValueError(f"Scene {scene} not supported")
        
    env_cfg.set_scene(scene)
    env = gym.make("DROID", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset() # need second render cycle to get correctly loaded materials


    video_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    video_dir.mkdir(parents=True, exist_ok=True)

    # Save the initial camera views so the scene assets can be checked visually
    cam_names = ["external_cam", "external_cam_2", "wrist_cam"]
    cam_imgs = [obs["policy"][name][0].detach().cpu().numpy() for name in cam_names]
    for name, img in zip(cam_names, cam_imgs):
        mediapy.write_image(video_dir / f"scene{scene}_{name}.png", img)
    mediapy.write_image(video_dir / f"scene{scene}_all_cams.png", np.concatenate(cam_imgs, axis=1))
    print(f"Saved initial camera images to {video_dir}")
    video = []
    ep = 0
    if dummy_policy:
        client = DummyClient()
    else:
        # Blocks until a policy server is reachable at remote_host:remote_port
        print(f"Connecting to policy server at {remote_host}:{remote_port} ...")
        client = DroidJointPosClient(remote_host=remote_host, remote_port=remote_port)
    max_steps = env.env.max_episode_length
    with torch.no_grad():
        for ep in range(episodes):
            ep_return = 0.0
            rewards = []
            term_hist = []  # per step: {term name: weighted value}
            for _ in tqdm(range(max_steps), desc=f"Episode {ep+1}/{episodes}"):
                ret = client.infer(obs, instruction)
                if not headless:
                    cv2.imshow("Right Camera", cv2.cvtColor(ret["viz"], cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                video.append(ret["viz"])
                action = torch.tensor(ret["action"])[None]
                obs, rew, term, trunc, _ = env.step(action)
                r = float(rew[0])
                rewards.append(r)
                term_hist.append(
                    dict(
                        (name, vals[0])
                        for name, vals in env.unwrapped.reward_manager.get_active_iterable_terms(0)
                    )
                )
                ep_return += r
                if r != 0:
                    tqdm.write(f"  step {len(rewards) - 1}: reward {r:.3f}")
                if term or trunc:
                    break

            # An episode is successful if the success term fired at any step (shaping terms don't count)
            succ = [t.get(SUCCESS_TERM, r) for t, r in zip(term_hist, rewards)]
            success = any(v > 0 for v in succ)
            success_step = next((i for i, v in enumerate(succ) if v > 0), None)
            print(f"Episode {ep+1} return: {ep_return} -> {'SUCCESS' if success else 'FAIL'}")
            np.savetxt(video_dir / f"episode_{ep}_rewards.txt", np.array(rewards), fmt="%.6f")
            client.reset()
            running = np.cumsum(rewards)
            cums = {}
            video_out = []
            for i, f in enumerate(video[: len(rewards)]):
                for k, v in term_hist[i].items():
                    cums[k] = cums.get(k, 0.0) + v
                video_out.append(
                    overlay_frame(f, i, term_hist[i], dict(cums), running[i], ep_return, success, success_step)
                )
            video = video_out
            suffix = "success" if success else "fail"
            mediapy.write_video(
                video_dir / f"episode_{ep}_{suffix}.mp4",
                video,
                fps=15,
            )
            video = []

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    args = tyro.cli(main)
