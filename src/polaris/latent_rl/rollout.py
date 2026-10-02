import torch


class RolloutBuffer:
    """On-policy storage for `num_steps` SMDP transitions from each of `num_envs` parallel envs (PPO).

    One step = one chunk per env; `discount` = gamma^k per transition (k = env steps in the chunk).
    `over` marks an episode boundary (terminated or truncated); `terminated` cuts bootstrapping, truncation
    bootstraps with the value of the chunk's start state (the terminal obs is lost to the env's auto-reset).
    """

    def __init__(self, num_steps: int, num_envs: int, feat_dim: int, steer_dim: int, device="cpu"):
        self.num_steps, self.num_envs, self.device = num_steps, num_envs, device
        z = lambda *s: torch.zeros(num_steps, num_envs, *s, device=device)
        self.d = dict(
            feat=z(feat_dim), z=z(steer_dim), logp=z(), value=z(), reward=z(), discount=z(), over=z(),
            adv=z(), ret=z(),
        )
        self.ptr = 0

    def add(self, feat, z, logp, value, reward, discount, terminated, over, truncated=None):
        t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
        reward, value, discount = t(reward), t(value), t(discount)
        if truncated is not None:  # time-out without termination: bootstrap with V(s_t) in place of V(terminal)
            reward = reward + discount * value * t(truncated) * (1 - t(terminated))
        vals = dict(feat=feat, z=z, logp=logp, value=value, reward=reward, discount=discount, over=over)
        for k, v in vals.items():
            self.d[k][self.ptr] = t(v)
        self.ptr += 1

    @property
    def full(self) -> bool:
        return self.ptr == self.num_steps

    def compute_returns(self, last_value, lam: float = 0.95):
        """SMDP GAE. delta_t = r_t + disc_t (1 - over_t) V_{t+1} - V_t;  A_t = delta_t + disc_t lam (1 - over_t) A_{t+1}."""
        d = self.d
        next_value = torch.as_tensor(last_value, dtype=torch.float32, device=self.device)
        adv = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.ptr)):
            nonterminal = 1 - d["over"][t]
            delta = d["reward"][t] + d["discount"][t] * nonterminal * next_value - d["value"][t]
            adv = delta + d["discount"][t] * lam * nonterminal * adv
            d["adv"][t] = adv
            next_value = d["value"][t]
        d["ret"][: self.ptr] = d["adv"][: self.ptr] + d["value"][: self.ptr]

    def flat(self) -> dict:
        return {k: v[: self.ptr].reshape(self.ptr * self.num_envs, *v.shape[2:]) for k, v in self.d.items()}

    def minibatches(self, n: int):
        data = self.flat()
        size = self.ptr * self.num_envs
        perm = torch.randperm(size, device=self.device)
        for idx in perm.chunk(n):
            yield {k: v[idx] for k, v in data.items()}

    def reset(self):
        self.ptr = 0

    def __len__(self):
        return self.ptr * self.num_envs
