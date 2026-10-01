import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from polaris.latent_rl.networks import SteeringActor


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

    @property
    def steer_dim(self) -> int:
        return self.steer_horizon * self.noise_shape[1]


def save_checkpoint(path, actor: SteeringActor, cfg: SteeringConfig):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), path / "actor.pt")
    (path / "config.json").write_text(json.dumps(asdict(cfg), indent=2))


def load_checkpoint(path, device="cpu") -> tuple[SteeringActor, SteeringConfig]:
    path = Path(path)
    d = json.loads((path / "config.json").read_text())
    d["noise_shape"] = tuple(d["noise_shape"])
    cfg = SteeringConfig(**d)
    actor = SteeringActor(cfg.feat_dim, cfg.steer_dim, cfg.z_bound, cfg.hidden).to(device)
    actor.load_state_dict(torch.load(path / "actor.pt", map_location=device))
    return actor.eval(), cfg
