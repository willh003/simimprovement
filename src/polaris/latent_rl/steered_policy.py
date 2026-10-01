from dataclasses import dataclass

import numpy as np
import torch

from polaris.latent_rl.checkpoint import SteeringConfig, load_checkpoint
from polaris.latent_rl.encoders import make_encoder
from polaris.latent_rl.interfaces import ChunkPolicy, ObsEncoder
from polaris.latent_rl.networks import SteeringActor


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
        ah, ad = self.cfg.noise_shape
        noise = np.random.randn(ah, ad).astype(np.float32)
        noise[: self.cfg.steer_horizon] = z.reshape(self.cfg.steer_horizon, ad)
        return noise

    @torch.no_grad()
    def step(self, obs: dict, feat: np.ndarray | None = None) -> SteerOutput:
        if feat is None:
            feat = self.encoder(obs)
        z, _ = self.actor(torch.as_tensor(feat, dtype=torch.float32, device=self.device)[None], self.deterministic)
        z = z[0].cpu().numpy()
        return SteerOutput(self.base.query(obs, self.build_noise(z)), z, feat)
