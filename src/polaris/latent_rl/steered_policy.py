from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from polaris.latent_rl.checkpoint import ARM_DOF, SteeringConfig, load_checkpoint
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


def apply_residual(chunk: np.ndarray, res: np.ndarray, cfg: SteeringConfig) -> np.ndarray:
    """chunk (..., H, 8) absolute joint targets + gripper; res (..., res_dim) unit residual in (-1, 1).
    Adds `cfg.residual_scale * res` (radians) to the arm joints of the first `residual_horizon` steps."""
    if cfg.res_dim == 0:
        return chunk
    chunk = np.array(chunk, dtype=np.float32, copy=True)
    h = cfg.residual_horizon
    chunk[..., :h, :ARM_DOF] += cfg.residual_scale * np.asarray(res, np.float32).reshape(*res.shape[:-1], h, ARM_DOF)
    return chunk


@dataclass
class SteerOutput:
    chunk: np.ndarray
    z: np.ndarray  # the actor's full action, flat (act_dim,) = [steering noise (steer_dim), unit residual (res_dim)]
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
        kwargs = dict(cfg.encoder_kwargs)
        weights = Path(path) / "encoder.pt"
        if weights.exists():  # finetuned encoder: skip the pretrained download, load the saved weights
            kwargs.update(pretrained=False, device=device)
        encoder = make_encoder(cfg.encoder, **kwargs)
        if weights.exists():
            encoder.load_state_dict(torch.load(weights, map_location=device))
        return cls(actor, cfg, encoder, base, deterministic, device)

    def build_noise(self, z: np.ndarray) -> np.ndarray:
        return build_noise(z, self.cfg.noise_shape, self.cfg.steer_horizon)

    @torch.no_grad()
    def step(self, obs: dict, feat: np.ndarray | None = None) -> SteerOutput:
        if feat is None:
            feat = self.encoder(obs)
        a, _ = self.actor(torch.as_tensor(feat, dtype=torch.float32, device=self.device)[None], self.deterministic)
        a = a[0].cpu().numpy()
        z, res = a[: self.cfg.steer_dim], a[self.cfg.steer_dim :]
        chunk = apply_residual(self.base.query(obs, self.build_noise(z)), res, self.cfg)
        return SteerOutput(chunk, a, feat)
