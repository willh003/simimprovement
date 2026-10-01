from dataclasses import dataclass

import torch

from polaris.latent_rl.steered_policy import SteeredPolicy
from polaris.policy.droid_jointpos_client import DroidJointPosClient


@dataclass
class ChunkTransition:
    feat: object
    z: object
    next_feat: object
    reward: float
    done: bool  # true termination (success/term); cuts bootstrapping
    episode_over: bool  # done or time-out
    discount: float  # gamma^k
    info: dict


class ChunkEnv:
    """SMDP wrapper: one 'action' is the steering noise z; it executes one action chunk.

    Reward = progress delta (rubric) summed over the chunk + success bonus on first success.
    """

    def __init__(self, env, client: DroidJointPosClient, steered: SteeredPolicy, instruction: str, gamma: float = 0.99, success_bonus: float = 1.0):
        self.env, self.client, self.steered = env, client, steered
        self.instruction, self.gamma, self.success_bonus = instruction, gamma, success_bonus
        self.horizon = client.open_loop_horizon
        self.obs = None
        self.request = None
        self.feat = None
        self.prev_progress = 0.0

    def reset(self, **kwargs):
        self.obs, info = self.env.reset(**kwargs)
        self.client.reset()
        self.prev_progress = 0.0
        self.request, _ = self.client.build_request(self.obs, self.instruction)
        self.feat = self.steered.encoder(self.request)
        return self.obs

    def step(self) -> ChunkTransition:
        out = self.steered.step(self.request, self.feat)
        reward, over, k, info = 0.0, False, 0, {}
        term = trunc = [False]
        success = False
        for i in range(self.horizon):
            action = self.client.postprocess_action(out.chunk[i])
            # only the last step of a chunk needs the (expensive) splat render for the next query
            self.obs, _, term, trunc, info = self.env.step(
                torch.tensor(action).reshape(1, -1), expensive=(i == self.horizon - 1)
            )
            k += 1
            progress = float(info["rubric"]["progress"])
            reward += progress - self.prev_progress
            self.prev_progress = progress
            success = bool(info["rubric"]["success"])
            if success:
                reward += self.success_bonus
            over = bool(term[0]) or bool(trunc[0]) or success
            if over:
                break
        self.request, _ = self.client.build_request(self.obs, self.instruction)
        next_feat = self.feat = self.steered.encoder(self.request)
        terminated = success or bool(term[0])
        return ChunkTransition(out.feat, out.z, next_feat, reward, terminated, over, self.gamma**k, info)
