import math

import torch
import torch.nn as nn


def mlp(inp: int, hidden: int, out: int, depth: int = 2) -> nn.Sequential:
    layers, d = [], inp
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class SteeringActor(nn.Module):
    """pi(z | feat). u ~ N(mean, std); z = bound * tanh(u / bound).

    For |u| << bound, z ~ u, and the output layer is zero-initialised (mean 0, std 1),
    so the untrained actor samples ~N(0, I) -- the base policy's own noise distribution.

    With `res_dim > 0` the action is [z (steer_dim), r (res_dim)]: r is a squashed Gaussian with bound 1 (r in (-1, 1);
    the caller scales it, e.g. by 0.01 rad, and adds it to the base policy's action chunk -- residual flow steering).
    """

    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, feat_dim: int, steer_dim: int, bound: float = 3.0, hidden: int = 256, res_dim: int = 0):
        super().__init__()
        self.steer_dim, self.res_dim = steer_dim, res_dim
        self.act_dim = steer_dim + res_dim
        self.register_buffer("bound", torch.cat([torch.full((steer_dim,), float(bound)), torch.ones(res_dim)]), persistent=False)
        self.trunk = mlp(feat_dim, hidden, 2 * self.act_dim)
        nn.init.zeros_(self.trunk[-1].weight)
        nn.init.zeros_(self.trunk[-1].bias)

    def dist(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pre-tanh Gaussian (mean, log_std)."""
        mean, log_std = self.trunk(feat).chunk(2, dim=-1)
        return mean, log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)

    def _logp(self, u, mean, log_std):
        t = torch.tanh(u / self.bound)
        logp = (-0.5 * ((u - mean) / log_std.exp()) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        return t, logp - torch.log(1 - t**2 + 1e-6).sum(-1)  # dz/du = 1 - tanh^2

    def forward(self, feat: torch.Tensor, deterministic: bool = False):
        mean, log_std = self.dist(feat)
        if deterministic:
            u = mean
        else:
            u = mean + log_std.exp() * torch.randn_like(mean)
        t, logp = self._logp(u, mean, log_std)
        return self.bound * t, logp

    def evaluate(self, feat: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """log pi(z | feat) for given z (as returned by forward), and the pre-tanh Gaussian entropy (used by PPO)."""
        mean, log_std = self.dist(feat)
        u = self.bound * torch.atanh((z / self.bound).clamp(-1 + 1e-6, 1 - 1e-6))
        _, logp = self._logp(u, mean, log_std)
        entropy = (log_std + 0.5 * (1 + math.log(2 * math.pi))).sum(-1)
        return logp, entropy


class TwinQ(nn.Module):
    def __init__(self, feat_dim: int, steer_dim: int, hidden: int = 256):
        super().__init__()
        self.q1 = mlp(feat_dim + steer_dim, hidden, 1)
        self.q2 = mlp(feat_dim + steer_dim, hidden, 1)

    def forward(self, feat, z):
        x = torch.cat([feat, z], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class ValueNet(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 256):
        super().__init__()
        self.v = mlp(feat_dim, hidden, 1)

    def forward(self, feat):
        return self.v(feat).squeeze(-1)
