"""Roll out a DROID policy on many parallel sim-evals envs at once and report the success rate across all of them.

Parallel version of scripts/run_pi.py. Cameras are converted to `TiledCameraCfg` (one batched render product per camera
instead of one per env), which lifts the ~48-camera cap of `CameraCfg` (see docs/claude/parallel_envs_bench.md).
Needs an openpi server on --policy.host/--policy.port unless --dummy-policy is set. All envs are queried in one batched
`infer_batch` call every `open_loop_horizon` steps (needs the batched openpi server, docs/claude/openpi_batching.md).

  isaacpy scripts/run_pi_parallel.py --num-envs 32 --environment BananaBin --policy.port 8000
  isaacpy scripts/run_pi_parallel.py --num-envs 8 --dummy-policy     # no server: holds pose, gripper open
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import mediapy
import numpy as np
import torch
import tyro
from tqdm import tqdm

from polaris.config import PolicyArgs
from polaris.rl.tasks import SIM_EVALS_INSTRUCTIONS, SIM_EVALS_SUCCESS_TERMS

def main(
    policy: PolicyArgs = PolicyArgs(),
    environment: str = "CanMug",
    num_envs: int = 16,
    episodes: int = 1,  # rounds; each round runs one episode in every env, so num_envs * episodes episodes in total
    cam_scale: float = 0.5,  # multiply camera resolutions (1.0 = 1280x720, 0.5 = 640x360); memory scales with this
    keep_cam2: bool = False,  # external_cam_2 is unused by the policy client, so it is dropped by default
    num_videos: int = 4,  # save a model-view video for the first K envs of each round
    headless: bool = True,
    dummy_policy: bool = False,
    max_batch_size: int = 128,  # obs per server message; larger env counts are split (a huge batch can OOM the server)
    run_folder: str | None = None,  # default: runs/simevals_parallel/<date>/<time>
):
    from isaaclab.app import AppLauncher

    args_cli, _ = argparse.ArgumentParser().parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    AppLauncher(args_cli)

    from polaris.environments.simeval_parallel import make_parallel_env

    env, cams = make_parallel_env(environment, num_envs, cam_scale, keep_cam2)
    instruction, success_term = SIM_EVALS_INSTRUCTIONS[environment], SIM_EVALS_SUCCESS_TERMS[environment]
    obs, _ = env.reset()
    obs, _ = env.reset()  # need second render cycle to get correctly loaded materials

    now = datetime.now()
    out_dir = Path(run_folder) if run_folder else Path("runs/simevals_parallel") / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    mediapy.write_image(
        out_dir / "all_cams.png", np.concatenate([obs["policy"][c][0].cpu().numpy() for c in cams], axis=1)
    )
    print(f"Saved env 0 camera images to {out_dir}")

    horizon = policy.open_loop_horizon or 8
    if dummy_policy:
        client = None
    else:
        from polaris.policy.droid_jointpos_client import DroidJointPosClient

        client = DroidJointPosClient(policy).client  # one connection; all envs go in one infer_batch call
    from openpi_client import image_tools

    def make_request(ext, wrist, joint_pos, gripper):
        ext, wrist = (image_tools.resize_with_pad(x, 224, 224) for x in (ext, wrist))
        req = {
            "observation/exterior_image_1_left": ext,
            "observation/wrist_image_left": wrist,
            "observation/joint_position": joint_pos,
            "observation/gripper_position": gripper,
            "prompt": instruction,
        }
        return req, np.concatenate([ext, wrist], axis=1)

    max_steps = env.unwrapped.max_episode_length
    successes, returns, round_stats = [], [], []
    with torch.no_grad():
        for ep in range(episodes):
            success = np.zeros(num_envs, dtype=bool)
            ep_return = np.zeros(num_envs)
            done = np.zeros(num_envs, dtype=bool)  # env finished its episode (terminated/truncated); later steps ignored
            chunks, videos = None, [[] for _ in range(min(num_videos, num_envs))]
            for t in tqdm(range(max_steps), desc=f"Round {ep + 1}/{episodes} ({num_envs} envs)"):
                if t % horizon == 0:
                    joint_pos = obs["policy"]["arm_joint_pos"].cpu().numpy()
                    gripper = obs["policy"]["gripper_pos"].cpu().numpy()
                    ext = obs["policy"]["external_cam"].cpu().numpy()
                    wrist = obs["policy"]["wrist_cam"].cpu().numpy()
                    reqs, viz = zip(*(make_request(ext[i], wrist[i], joint_pos[i], gripper[i]) for i in range(num_envs)))
                    if client is None:
                        chunks = np.stack([np.concatenate([jp, np.zeros(1)])[None].repeat(horizon, 0) for jp in joint_pos])
                    else:
                        chunks = np.stack([r["actions"] for r in client.infer_batch(list(reqs), max_batch_size)])  # (N, T, 8)
                    for v, frame in zip(videos, viz):
                        v.extend([frame] * horizon)  # model view only refreshes when the policy is queried
                action = chunks[:, t % horizon].copy()
                action[:, -1] = (action[:, -1] > 0.5).astype(np.float32)  # binarize gripper, as the client does
                obs, rew, term, trunc, _ = env.step(torch.as_tensor(action, dtype=torch.float32, device="cuda"))
                rew = rew.cpu().numpy()
                # the reward manager's per-term values are scaled differently from rew, but their sign is preserved
                fired = np.array(
                    [
                        dict(env.unwrapped.reward_manager.get_active_iterable_terms(i)).get(success_term, [0.0])[0] > 0
                        for i in range(num_envs)
                    ]
                )
                success |= fired & ~done
                ep_return += np.where(done, 0.0, rew)
                done |= (term | trunc).cpu().numpy()
                if done.all():
                    break
            successes.append(success)
            returns.append(ep_return)
            round_stats.append(dict(round=ep, success_rate=float(success.mean()), mean_return=float(ep_return.mean())))
            print(f"Round {ep + 1}: success {success.sum()}/{num_envs} = {success.mean():.1%}, mean return {ep_return.mean():.3f}")
            for i, v in enumerate(videos):
                mediapy.write_video(out_dir / f"round{ep}_env{i}_{'success' if success[i] else 'fail'}.mp4", v[: t + 1], fps=15)

    all_success = np.concatenate(successes)
    summary = dict(
        environment=environment,
        num_envs=num_envs,
        episodes=len(all_success),
        success_rate=float(all_success.mean()),
        mean_return=float(np.concatenate(returns).mean()),
        rounds=round_stats,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{environment}: success rate {all_success.sum()}/{len(all_success)} = {all_success.mean():.1%}  (saved to {out_dir})")

    sys.stdout.flush()
    os._exit(0)  # Isaac hangs on normal shutdown


if __name__ == "__main__":
    tyro.cli(main)
