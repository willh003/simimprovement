import numpy as np
import mediapy
import torch
import argparse
import gymnasium as gym
from isaaclab.app import AppLauncher
# This must be done before importing anything with dependency on Isaaclab
# >>>> Isaac Sim App Launcher <<<<
parser = argparse.ArgumentParser()
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
# >>>> Isaac Sim App Launcher <<<<

import polaris.environments
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from polaris.environments.manager_based_rl_splat_environment import ManagerBasedRLSplatEnv
from polaris.utils import load_eval_initial_conditions

env_cfg = parse_env_cfg(
    "DROID-FoodBussing",
    device="cuda",
    num_envs=1,
    use_fabric=True,
)
env: ManagerBasedRLSplatEnv = gym.make("DROID-FoodBussing", cfg=env_cfg)   # type: ignore
language_instruction, initial_conditions = load_eval_initial_conditions(env.usd_file)
obs, info = env.reset(object_positions = initial_conditions[0])


def get_frame(obs):
    """Tile all splat-rendered camera views side by side into one RGB frame."""
    return np.concatenate(list(obs["splat"].values()), axis=1)


frames = [get_frame(obs)]
while True:
    action = torch.tensor(env.action_space.sample())
    obs, rew, term, trunc, info = env.step(action, expensive=True)

    frames.append(get_frame(obs))

    if term[0] or trunc[0]:
        break

print(f"Episode Finished. Success: {info['rubric']['success']}, Progress: {info['rubric']['progress']}")

video_path = "hello_world.mp4"
mediapy.write_video(video_path, frames, fps=5)
print(f"Saved video to {video_path}")
