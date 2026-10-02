from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from polaris.latent_rl.networks import SteeringActor, ValueNet
from polaris.latent_rl.rollout import RolloutBuffer


@dataclass
class PPOConfig:
    lr: float = 3e-4
    epochs: int = 5
    minibatches: int = 4
    clip: float = 0.2
    value_clip: float | None = None  # None: unclipped value loss
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02  # stop the epoch loop once approx KL exceeds this
    gae_lambda: float = 0.95
    norm_adv: bool = True
    hidden: int = 256


class DSRLPPO:
    """PPO where the 'action' is the (steered slice of the) flow-matching noise z.

    The actor is the same `SteeringActor` as SAC (zero-init last layer -> starts at ~N(0, I), the base policy's own noise),
    so checkpoints work with `SteeredPolicy` / the `Steered` client. Rollouts carry `discount` = gamma^k per transition.
    """

    def __init__(self, feat_dim: int, steer_dim: int, bound: float = 3.0, cfg: PPOConfig | None = None, device="cpu"):
        self.cfg = cfg or PPOConfig()
        self.device = device
        self.actor = SteeringActor(feat_dim, steer_dim, bound, self.cfg.hidden).to(device)
        self.critic = ValueNet(feat_dim, self.cfg.hidden).to(device)
        self.params = [*self.actor.parameters(), *self.critic.parameters()]
        self.opt = torch.optim.Adam(self.params, lr=self.cfg.lr, eps=1e-5)

    def _t(self, x) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def act(self, feat, deterministic: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(N, feat_dim) -> z (N, steer_dim), logp (N,), value (N,) as numpy."""
        feat = self._t(feat)
        z, logp = self.actor(feat, deterministic)
        return z.cpu().numpy(), logp.cpu().numpy(), self.critic(feat).cpu().numpy()

    @torch.no_grad()
    def value(self, feat) -> np.ndarray:
        return self.critic(self._t(feat)).cpu().numpy()

    def update(self, buf: RolloutBuffer) -> dict:
        """PPO epochs over `buf`; the caller must have run `buf.compute_returns(last_value, cfg.gae_lambda)`."""
        cfg = self.cfg
        stats: dict[str, list[float]] = {}
        epochs_run = 0
        for _ in range(cfg.epochs):
            kls = []
            for b in buf.minibatches(cfg.minibatches):
                logp, entropy = self.actor.evaluate(b["feat"], b["z"])
                log_ratio = logp - b["logp"]
                ratio = log_ratio.exp()
                adv = b["adv"]
                if cfg.norm_adv and adv.numel() > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg_loss = torch.max(-adv * ratio, -adv * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                v = self.critic(b["feat"])
                if cfg.value_clip is None:
                    v_loss = 0.5 * (v - b["ret"]).pow(2).mean()
                else:
                    v_clipped = b["value"] + (v - b["value"]).clamp(-cfg.value_clip, cfg.value_clip)
                    v_loss = 0.5 * torch.max((v - b["ret"]).pow(2), (v_clipped - b["ret"]).pow(2)).mean()
                ent = entropy.mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clipfrac = ((ratio - 1).abs() > cfg.clip).float().mean().item()
                kls.append(approx_kl)
                for k, val in dict(
                    pg_loss=pg_loss.item(), v_loss=v_loss.item(), entropy=ent.item(), approx_kl=approx_kl,
                    clipfrac=clipfrac, grad_norm=grad_norm.item(),
                ).items():
                    stats.setdefault(k, []).append(val)
            epochs_run += 1
            if cfg.target_kl is not None and np.mean(kls) > cfg.target_kl:
                break
        data = buf.flat()
        with torch.no_grad():
            _, log_std = self.actor.dist(data["feat"])
            var_ret = data["ret"].var()
            explained_var = (1 - (data["ret"] - data["value"]).var() / var_ret).item() if var_ret > 0 else float("nan")
        return dict(
            **{k: float(np.mean(v)) for k, v in stats.items()}, epochs=epochs_run, explained_var=explained_var,
            std_mean=log_std.exp().mean().item(), value_mean=data["value"].mean().item(), return_mean=data["ret"].mean().item(),
            adv_std=data["adv"].std().item(), z_abs_mean=data["z"].abs().mean().item(), reward_mean=data["reward"].mean().item(),
        )

    def state_dict(self) -> dict:
        return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(), opt=self.opt.state_dict())

    def load_state_dict(self, d: dict):
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])
        self.opt.load_state_dict(d["opt"])
