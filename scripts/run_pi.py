"""Roll out a DROID policy in a sim-evals scene ported into polaris ({CubeBowl,CanMug,BananaBin}).

Port of third_party/sim-evals/run_eval.py, used to check the ported env/client before steering training.
Needs an openpi server on --policy.host/--policy.port unless --dummy-policy is set.

  python scripts/eval_simevals.py --episodes 2 --policy.port 8000 [--environment CubeBowl]
  python scripts/eval_simevals.py --episodes 1 --dummy-policy     # no server: holds pose, gripper open
"""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import gymnasium as gym
import mediapy
import numpy as np
import torch
import tyro
from tqdm import tqdm

from polaris.config import PolicyArgs
from polaris.rl.tasks import SIM_EVALS_INSTRUCTIONS, SIM_EVALS_SUCCESS_TERMS



def overlay_frame(frame, step, term_vals, running_return, ep_return, max_return, success, success_step):
    """Return `frame` with a text panel appended to its right (the images are not covered or resized).

    Shows step, SUCCESS/FAIL, return (current / total / max running return) and each reward term's value this step.
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
        (f"return: {running_return:.2f} / {ep_return:.2f} / {max_return:.2f}", (255, 255, 255)),
        ("reward terms (this step)", (160, 160, 160)),
    ]
    for name, v in term_vals.items():
        color = (0, 255, 0) if v > 0 else (255, 120, 120) if v < 0 else (255, 255, 255)
        rows.append((f"{name}: {v:+.2f}", color))
    for i, (text, color) in enumerate(rows):
        cv2.putText(panel, text, (8, lh * (i + 1)), font, scale * 0.9, color, th, cv2.LINE_AA)
    if reached:
        cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, h - 1), (0, 200, 0), max(int(4 * scale), 2))
    return np.concatenate([frame, panel], axis=1)


class DummyClient:
    """Server-free: hold the current arm pose, gripper open."""

    def reset(self):
        pass

    def infer(self, obs, instruction, return_viz=False):
        policy = obs["policy"]
        action = np.concatenate([policy["arm_joint_pos"].detach().cpu().numpy(), np.zeros(1)])
        viz = np.concatenate(
            [cv2.resize(policy[n][0].detach().cpu().numpy(), (224, 224)) for n in ("external_cam", "wrist_cam")], axis=1
        )
        return action, viz


def main(
    policy: PolicyArgs = PolicyArgs(),
    environment: str = "CanMug",
    episodes: int = 1,
    headless: bool = True,
    dummy_policy: bool = False,
    run_folder: str | None = None,  # default: runs/simevals_eval/<date>/<time>
):
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    simulation_app = AppLauncher(args_cli).app

    import polaris.environments  # noqa: F401  (registers *)
    from isaaclab_tasks.utils import parse_env_cfg
    from polaris.policy.droid_jointpos_client import SimEvalsJointPosClient

    env = gym.make(environment, cfg=parse_env_cfg(environment, device="cuda", num_envs=1, use_fabric=True))
    instruction, success_term = SIM_EVALS_INSTRUCTIONS[environment], SIM_EVALS_SUCCESS_TERMS[environment]
    obs, _ = env.reset()
    obs, _ = env.reset()  # need second render cycle to get correctly loaded materials

    now = datetime.now()
    video_dir = Path(run_folder) if run_folder else Path("runs/simevals_eval") / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
    video_dir.mkdir(parents=True, exist_ok=True)
    cam_names = ["external_cam", "external_cam_2", "wrist_cam"]
    cam_imgs = [obs["policy"][name][0].detach().cpu().numpy() for name in cam_names]
    mediapy.write_image(video_dir / "all_cams.png", np.concatenate(cam_imgs, axis=1))
    print(f"Saved initial camera images to {video_dir}")

    client = DummyClient() if dummy_policy else SimEvalsJointPosClient(policy)
    max_steps = env.unwrapped.max_episode_length
    with torch.no_grad():
        for ep in range(episodes):
            client.reset()
            video, rewards, term_hist, ep_return = [], [], [], 0.0
            for _ in tqdm(range(max_steps), desc=f"Episode {ep + 1}/{episodes}"):
                action, viz = client.infer(obs, instruction, return_viz=True)
                video.append(viz)
                obs, rew, term, trunc, _ = env.step(torch.tensor(action)[None])
                r = float(rew[0])
                rewards.append(r)
                # the reward manager's per-term values are scaled differently from rew; rescale so they sum to rew
                raw = {n: float(v[0]) for n, v in env.unwrapped.reward_manager.get_active_iterable_terms(0)}
                total = sum(raw.values())
                term_hist.append({n: v * (r / total if total else 0.0) for n, v in raw.items()})
                ep_return += r
                if r != 0:
                    tqdm.write(f"  step {len(rewards) - 1}: reward {r:.3f}")
                if term or trunc:
                    break
            succ = [t.get(success_term, r) for t, r in zip(term_hist, rewards)]
            success = any(v > 0 for v in succ)
            success_step = next((i for i, v in enumerate(succ) if v > 0), None)
            print(f"Episode {ep + 1} return: {ep_return} -> {'SUCCESS' if success else 'FAIL'}")
            np.savetxt(video_dir / f"episode_{ep}_rewards.txt", np.array(rewards), fmt="%.6f")
            running = np.cumsum(rewards)
            max_return = float(running.max())
            video_out = [
                overlay_frame(f, i, term_hist[i], running[i], ep_return, max_return, success, success_step)
                for i, f in enumerate(video[: len(rewards)])
            ]
            mediapy.write_video(video_dir / f"episode_{ep}_{'success' if success else 'fail'}.mp4", video_out, fps=15)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    tyro.cli(main)
