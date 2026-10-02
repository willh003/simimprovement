from dataclasses import dataclass

import numpy as np
import torch

from polaris.latent_rl.checkpoint import SteeringConfig, load_checkpoint
from polaris.latent_rl.encoders import make_encoder
from polaris.latent_rl.interfaces import ChunkPolicy, ObsEncoder
from polaris.latent_rl.networks import SteeringActor


def build_noise(z: np.ndarray, noise_shape: tuple[int, int], steer_horizon: int) -> np.ndarray:
    """Full model noise: z in the first `steer_horizon` rows, N(0, I) elsewhere.

    z: (steer_dim,) -> (action_horizon, action_dim), or batched (N, steer_dim) -> (N, action_horizon, action_dim).
    """
    ah, ad = noise_shape
    batch = z.shape[:-1]
    noise = np.random.randn(*batch, ah, ad).astype(np.float32)
    noise[..., :steer_horizon, :] = z.reshape(*batch, steer_horizon, ad)
    return noise


@dataclass
class SteerOutput:
    chunk: np.ndarray
    z: np.ndarray  # steered slice, flat (steer_dim,)
    feat: np.ndarray


class SteeredPolicy:
    """Steering actor + frozen base ChunkPolicy -> action chunk. Same class for train and eval."""

    def __init__(self, actor: SteeringActor, cfg: SteeringConfig, encoder: ObsEncoder, base: ChunkPolicy, deterministic: bool = True, device="cpu"):
        if tuple(base.noise_shape) != tuple(cfg.noise_shape):
            raise ValueError(f"noise_shape mismatch: base {base.noise_shape} vs ckpt {cfg.noise_shape}")
        self.actor, self.cfg, self.encoder, self.base = actor, cfg, encoder, base
        self.deterministic, self.device = deterministic, device

    @classmethod
    def from_checkpoint(cls, path, base: ChunkPolicy, deterministic: bool = True, device="cpu"):
        actor, cfg = load_checkpoint(path, device)
        encoder = make_encoder(cfg.encoder, **cfg.encoder_kwargs)
        return cls(actor, cfg, encoder, base, deterministic, device)

    def build_noise(self, z: np.ndarray) -> np.ndarray:
        return build_noise(z, self.cfg.noise_shape, self.cfg.steer_horizon)

    @torch.no_grad()
    def step(self, obs: dict, feat: np.ndarray | None = None) -> SteerOutput:
        if feat is None:
            feat = self.encoder(obs)
        z, _ = self.actor(torch.as_tensor(feat, dtype=torch.float32, device=self.device)[None], self.deterministic)
        z = z[0].cpu().numpy()
        return SteerOutput(self.base.query(obs, self.build_noise(z)), z, feat)
