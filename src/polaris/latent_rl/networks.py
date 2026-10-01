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
    """

    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, feat_dim: int, steer_dim: int, bound: float = 3.0, hidden: int = 256):
        super().__init__()
        self.bound = bound
        self.steer_dim = steer_dim
        self.trunk = mlp(feat_dim, hidden, 2 * steer_dim)
        nn.init.zeros_(self.trunk[-1].weight)
        nn.init.zeros_(self.trunk[-1].bias)

    def forward(self, feat: torch.Tensor, deterministic: bool = False):
        mean, log_std = self.trunk(feat).chunk(2, dim=-1)
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        if deterministic:
            u = mean
        else:
            u = mean + log_std.exp() * torch.randn_like(mean)
        t = torch.tanh(u / self.bound)
        z = self.bound * t
        logp = (-0.5 * ((u - mean) / log_std.exp()) ** 2 - log_std - 0.5 * torch.log(torch.tensor(2 * torch.pi))).sum(-1)
        logp = logp - torch.log(1 - t**2 + 1e-6).sum(-1)  # dz/du = 1 - tanh^2
        return z, logp


class TwinQ(nn.Module):
    def __init__(self, feat_dim: int, steer_dim: int, hidden: int = 256):
        super().__init__()
        self.q1 = mlp(feat_dim + steer_dim, hidden, 1)
        self.q2 = mlp(feat_dim + steer_dim, hidden, 1)

    def forward(self, feat, z):
        x = torch.cat([feat, z], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)
