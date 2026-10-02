import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from polaris.latent_rl.networks import SteeringActor

ARM_DOF = 7


@dataclass
class SteeringConfig:
    noise_shape: tuple[int, int]  # full model noise (action_horizon, action_dim)
    steer_horizon: int  # leading timesteps that are steered (rest ~ N(0, I))
    feat_dim: int
    encoder: str
    z_bound: float = 3.0
    hidden: int = 256
    base_policy: str = ""
    encoder_kwargs: dict = field(default_factory=dict)
    residual_scale: float = 0.0  # radians; 0 disables the residual (plain DSRL)
    residual_horizon: int = 0  # leading chunk steps that get a residual on the 7 arm joints (absolute joint targets)

    @property
    def steer_dim(self) -> int:
        return self.steer_horizon * self.noise_shape[1]

    @property
    def res_dim(self) -> int:
        return self.residual_horizon * ARM_DOF if self.residual_scale > 0 else 0

    @property
    def act_dim(self) -> int:
        """Actor output: [steering noise z, unit residual r in (-1, 1)]."""
        return self.steer_dim + self.res_dim


def save_checkpoint(path, actor: SteeringActor, cfg: SteeringConfig, encoder=None):
    """`encoder`: pass a finetuned encoder module to save its weights (encoder.pt) next to the actor."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), path / "actor.pt")
    if encoder is not None:
        torch.save(encoder.state_dict(), path / "encoder.pt")
    (path / "config.json").write_text(json.dumps(asdict(cfg), indent=2))


def load_checkpoint(path, device="cpu") -> tuple[SteeringActor, SteeringConfig]:
    path = Path(path)
    d = json.loads((path / "config.json").read_text())
    d["noise_shape"] = tuple(d["noise_shape"])
    cfg = SteeringConfig(**d)
    actor = SteeringActor(cfg.feat_dim, cfg.steer_dim, cfg.z_bound, cfg.hidden, cfg.res_dim).to(device)
    actor.load_state_dict(torch.load(path / "actor.pt", map_location=device))
    return actor.eval(), cfg
