import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from polaris.latent_rl.networks import SteeringActor, TwinQ


@dataclass
class SACConfig:
    lr: float = 3e-4
    tau: float = 0.005
    init_alpha: float = 0.1
    target_entropy: float | None = None  # None -> fixed alpha; else auto-tune toward this value
    hidden: int = 256


class DSRLSAC:
    """SAC where the 'action' is the (steered slice of the) flow-matching noise z.

    Batches carry `discount` = gamma^k per transition (k = env steps in the chunk).
    """

    def __init__(self, feat_dim: int, steer_dim: int, bound: float = 3.0, cfg: SACConfig | None = None, device="cpu"):
        self.cfg = cfg or SACConfig()
        self.device = device
        self.actor = SteeringActor(feat_dim, steer_dim, bound, self.cfg.hidden).to(device)
        self.q = TwinQ(feat_dim, steer_dim, self.cfg.hidden).to(device)
        self.q_targ = copy.deepcopy(self.q).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=self.cfg.lr)
        self.log_alpha = torch.tensor(float(torch.log(torch.tensor(self.cfg.init_alpha))), device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().detach()

    @torch.no_grad()
    def act(self, feat: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        z, _ = self.actor(feat.to(self.device), deterministic)
        return z

    def update(self, batch: dict) -> dict:
        b = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            nz, nlogp = self.actor(b["next_feat"])
            tq = torch.min(*self.q_targ(b["next_feat"], nz)) - self.alpha * nlogp
            target = b["reward"] + b["discount"] * (1 - b["done"]) * tq
        q1, q2 = self.q(b["feat"], b["z"])
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        z, logp = self.actor(b["feat"])
        actor_loss = (self.alpha * logp - torch.min(*self.q(b["feat"], z))).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        if self.cfg.target_entropy is not None:
            alpha_loss = -(self.log_alpha * (logp.detach() + self.cfg.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.q_targ.parameters()):
                tp.lerp_(p, self.cfg.tau)
        return dict(q_loss=q_loss.item(), actor_loss=actor_loss.item(), alpha=self.alpha.item(), q_mean=q1.mean().item(), logp=logp.mean().item())
