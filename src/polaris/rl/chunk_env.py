from dataclasses import dataclass, field

import numpy as np
import torch

from polaris.latent_rl.steered_policy import SteeredPolicy
from polaris.policy.droid_jointpos_client import DroidJointPosClient
from polaris.rl.tasks import RubricSplatTask


@dataclass
class ChunkTransition:
    feat: object
    z: object
    next_feat: object
    reward: float  # == sum(reward_terms.values())
    done: bool  # true termination (success/term); cuts bootstrapping
    episode_over: bool  # done or time-out
    discount: float  # gamma^k
    info: dict
    reward_terms: dict[str, float] = field(default_factory=dict)
    steps: int = 0  # env steps executed in this chunk (k)
    success: bool = False
    reached: dict[str, float] = field(default_factory=dict)  # per-term max-ever 0/1 flags
    frames: list[np.ndarray] = field(default_factory=list)  # model-view frames, only when recording


class ChunkEnv:
    """SMDP wrapper: one 'action' is the steering noise z; it executes one action chunk.

    Reward terms and success come from `task` (default `RubricSplatTask`: one sparse term per rubric
    criterion, 1/N the first time it is reached) + `success_bonus` on overall success.
    `reward_terms` keeps them apart.
    """

    def __init__(self, env, client: DroidJointPosClient, steered: SteeredPolicy, instruction: str, gamma: float = 0.99, success_bonus: float = 1.0, task=None):
        self.env, self.client, self.steered = env, client, steered
        self.task = task or RubricSplatTask()
        self.instruction, self.gamma, self.success_bonus = instruction, gamma, success_bonus
        self.horizon = client.open_loop_horizon
        self.obs = None
        self.request = None
        self.feat = None
        self.frame = None  # model view of self.obs (exterior | wrist)

    def reset(self, **kwargs):
        self.obs = self.task.reset(self.env, **kwargs)
        self.client.reset()
        self.request, self.frame = self.client.build_request(self.obs, self.instruction)
        self.feat = self.steered.encoder(self.request)
        return self.obs

    def step(self, record: bool = False) -> ChunkTransition:
        """Execute one chunk. With `record`, splat-render every step and return its frames."""
        out = self.steered.step(self.request, self.feat)
        terms = {"success_bonus": 0.0}
        frames = [self.frame] if record else []
        over, k = False, 0
        res = None
        for i in range(self.horizon):
            action = self.client.postprocess_action(out.chunk[i])
            # only the last step of a chunk needs the (expensive) splat render for the next query
            res = self.task.step(self.env, torch.tensor(action).reshape(1, -1), expensive=record or (i == self.horizon - 1))
            self.obs = res.obs
            k += 1
            for name, v in res.terms.items():
                terms[name] = terms.get(name, 0.0) + v
            if res.success:
                terms["success_bonus"] += self.success_bonus
            over = res.terminated or res.truncated or res.success
            if record and not over and i < self.horizon - 1:
                frames.append(self.client.build_request(self.obs, self.instruction)[1])
            if over:
                break
        self.request, self.frame = self.client.build_request(self.obs, self.instruction)
        if record and over:
            frames.append(self.frame)  # final frame of the episode
        next_feat = self.feat = self.steered.encoder(self.request)
        terminated = res.success or res.terminated
        return ChunkTransition(
            out.feat, out.z, next_feat, sum(terms.values()), terminated, over, self.gamma**k, res.info,
            reward_terms=terms, steps=k, frames=frames, success=res.success, reached=res.reached,
        )
