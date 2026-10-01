import torch


class ReplayBuffer:
    """Stores SMDP transitions (feat, z, reward, next_feat, done, discount=gamma^k)."""

    def __init__(self, feat_dim: int, steer_dim: int, capacity: int = 100_000, device="cpu"):
        self.capacity, self.device = capacity, device
        self.size = self.ptr = 0
        z = lambda *s: torch.zeros(capacity, *s)
        self.d = dict(
            feat=z(feat_dim), z=z(steer_dim), reward=z(), next_feat=z(feat_dim), done=z(), discount=z()
        )

    def add(self, feat, z, reward, next_feat, done, discount):
        vals = dict(feat=feat, z=z, reward=reward, next_feat=next_feat, done=float(done), discount=discount)
        for k, v in vals.items():
            self.d[k][self.ptr] = torch.as_tensor(v, dtype=torch.float32)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        idx = torch.randint(0, self.size, (batch_size,))
        return {k: v[idx].to(self.device) for k, v in self.d.items()}

    def __len__(self):
        return self.size
