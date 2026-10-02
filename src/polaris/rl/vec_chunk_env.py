from dataclasses import dataclass, field

import numpy as np
import torch

from polaris.latent_rl.checkpoint import SteeringConfig
from polaris.latent_rl.encoders import encode_batch
from polaris.latent_rl.interfaces import BatchChunkPolicy, ObsEncoder
from polaris.latent_rl.steered_policy import apply_residual, build_noise
from polaris.rl.chunk_env import ChunkTransition


@dataclass
class VecChunkTransition:
    """One chunk in each of N envs; arrays are (N,). Per-env quantities stop at the env's episode end."""

    next_feat: np.ndarray  # (N, feat_dim); for envs whose episode ended: first feat of the new episode
    reward: np.ndarray  # == sum(reward_terms.values())
    reward_terms: dict[str, np.ndarray]
    terminated: np.ndarray  # bool; true termination (success / env terminated), cuts bootstrapping
    truncated: np.ndarray  # bool; time-out without termination
    episode_over: np.ndarray  # terminated | truncated
    discount: np.ndarray  # gamma^k
    steps: np.ndarray  # env steps of this env's episode executed in the chunk (k)
    success: np.ndarray  # bool
    reached: dict[str, np.ndarray]
    frames: list[list[np.ndarray]] = field(default_factory=list)  # model-view frames of the first `record` envs

    def unbatch(self, i: int) -> ChunkTransition:
        """Env i's transition as a single-env ChunkTransition (for EpisodeStats)."""
        return ChunkTransition(
            None, None, self.next_feat[i], float(self.reward[i]), bool(self.terminated[i]), bool(self.episode_over[i]),
            float(self.discount[i]), {}, reward_terms={k: float(v[i]) for k, v in self.reward_terms.items()},
            steps=int(self.steps[i]), success=bool(self.success[i]), reached={k: float(v[i]) for k, v in self.reached.items()},
            frames=self.frames[i] if i < len(self.frames) else [],
        )


class VecChunkEnv:
    """Vectorized SMDP wrapper over an N-env, auto-resetting env: action = steering noise z (N, steer_dim).

    All envs query the base policy together (one batched call) and run the chunk in lockstep. An env whose episode ends
    mid-chunk is masked for the rest of the chunk (its reward/k stop; it receives the task's hold action) and starts its
    next chunk in the fresh episode. Reward terms / success come from a vectorized `task` (e.g. `VecSimEvalsTask`)
    + `success_bonus` on success.
    """

    def __init__(self, env, client, base: BatchChunkPolicy, encoder: ObsEncoder, cfg: SteeringConfig, instruction: str,
                 gamma: float = 0.99, success_bonus: float = 0.0, task=None):
        if tuple(base.noise_shape) != tuple(cfg.noise_shape):
            raise ValueError(f"noise_shape mismatch: base {base.noise_shape} vs cfg {cfg.noise_shape}")
        self.env, self.client, self.base, self.encoder, self.cfg, self.task = env, client, base, encoder, cfg, task
        self.instruction, self.gamma, self.success_bonus = instruction, gamma, success_bonus
        self.horizon = client.open_loop_horizon
        self.device = getattr(env.unwrapped, "device", "cpu")
        self.obs = self.requests = self.views = self.feat = self.packed = None
        self.num_envs = 0

    def _observe(self):
        self.requests, self.views = self.client.build_requests(self.obs, self.instruction)
        if hasattr(self.encoder, "pack"):  # trainable image encoder: keep the packed (images, proprio) the feat came from
            self.packed = self.encoder.pack(self.requests)
            self.feat = self.encoder.encode_packed(*self.packed)
        else:
            self.feat = encode_batch(self.encoder, self.requests)
        self.num_envs = len(self.requests)
        return self.feat

    def reset(self) -> np.ndarray:
        """Reset all envs; returns feat (N, feat_dim)."""
        self.obs = self.task.reset(self.env)
        return self._observe()

    def step(self, z: np.ndarray, record: int = 0) -> VecChunkTransition:
        """Execute one chunk per env from the actor's action z (N, cfg.act_dim) = [steering noise | unit residual]. With
        `record`, keep every step's model view of the first `record` envs (the terminal frame of an episode is lost to
        the auto-reset)."""
        n, zeros = self.num_envs, np.zeros(self.num_envs)
        z = np.asarray(z, np.float32)
        noise = build_noise(z[:, : self.cfg.steer_dim], self.cfg.noise_shape, self.cfg.steer_horizon)
        chunks = self.base.query_batch(self.requests, noise)  # (N, H, A)
        chunks = apply_residual(chunks, z[:, self.cfg.steer_dim :], self.cfg)
        alive = np.ones(n, bool)
        k = np.zeros(n, int)
        terms = {"success_bonus": zeros.copy()}
        reached: dict[str, np.ndarray] = {}
        terminated, truncated, success = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
        frames = [[self.views[j]] for j in range(min(record, n))]
        ended = np.zeros(n, bool)
        for i in range(self.horizon):
            action = self.client.postprocess_actions(chunks[:, i])
            if not alive.all():
                action[~alive] = self.task.hold_action(self.obs)[~alive]
            res = self.task.step(self.env, torch.as_tensor(action, dtype=torch.float32, device=self.device))
            self.obs = res.obs
            if (res.success & ~res.terminated).any():
                raise RuntimeError("success without termination: the env needs a success termination term (make_parallel_env)")
            for name, v in res.terms.items():
                terms[name] = terms.get(name, zeros) + np.where(alive, v, 0.0)
            terms["success_bonus"] += self.success_bonus * (res.success & alive)
            for name, v in res.reached.items():
                reached[name] = np.where(alive, v, reached.get(name, zeros))
            k += alive
            ended = alive & (res.terminated | res.truncated)
            success |= ended & res.success
            terminated |= ended & res.terminated
            truncated |= ended & res.truncated & ~res.terminated
            alive &= ~ended
            if frames and i < self.horizon - 1 and alive[: len(frames)].any():
                _, views = self.client.build_requests(self.obs, self.instruction, env_ids=np.arange(len(frames)))
                for j, f in enumerate(frames):
                    if alive[j]:
                        f.append(views[j])
            if not alive.any():
                break
        if ended.any():  # reset on the last executed step: nothing rendered since, refresh the cameras
            self.obs = self.task.refresh_obs(self.env)
        self._observe()
        return VecChunkTransition(
            self.feat, sum(terms.values()), terms, terminated, truncated, terminated | truncated, self.gamma**k, k,
            success, reached, frames,
        )
