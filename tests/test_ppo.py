import numpy as np
import pytest
import torch

from polaris.latent_rl.algos import DSRLPPO, PPOConfig
from polaris.latent_rl.networks import SteeringActor
from polaris.latent_rl.rollout import RolloutBuffer
from polaris.latent_rl.steered_policy import build_noise


def test_evaluate_logp_matches_forward():
    torch.manual_seed(0)
    actor = SteeringActor(3, 4)
    torch.nn.init.normal_(actor.trunk[-1].weight, std=0.3)  # non-trivial mean / std
    feat = torch.randn(64, 3)
    z, logp = actor(feat)
    logp2, entropy = actor.evaluate(feat, z)
    assert torch.allclose(logp, logp2, atol=1e-3)
    assert entropy.shape == (64,)


def test_build_noise_single_and_batched():
    z = np.arange(6, dtype=np.float32)
    n = build_noise(z, (4, 3), 2)
    assert n.shape == (4, 3) and np.array_equal(n[:2], z.reshape(2, 3))
    zb = np.ones((5, 6), np.float32)
    nb = build_noise(zb, (4, 3), 2)
    assert nb.shape == (5, 4, 3) and (nb[:, :2] == 1).all() and not (nb[:, 2:] == 1).all()


def test_gae_smdp_discount_truncation_and_boundary():
    buf = RolloutBuffer(3, 1, 1, 1)
    lam = 0.5
    v = [1.0, 2.0, 3.0]
    disc = [0.9, 0.8, 0.7]
    r = [1.0, 0.0, 2.0]
    # step 0: normal; step 1: truncated (episode over, bootstrap with V(s_1)); step 2: normal, then bootstrap last_value
    flags = [(False, False, False), (False, True, True), (False, False, False)]
    for t in range(3):
        term, over, trunc = flags[t]
        buf.add([0.0], [0.0], 0.0, v[t], r[t], disc[t], term, over, trunc)
    buf.compute_returns(last_value=[4.0], lam=lam)
    r1 = r[1] + disc[1] * v[1]  # truncation bootstrap folded into the reward
    a2 = r[2] + disc[2] * 4.0 - v[2]
    a1 = r1 - v[1]  # episode boundary: no next value, no carry
    a0 = (r[0] + disc[0] * v[1] - v[0]) + disc[0] * lam * a1
    adv = buf.d["adv"][:, 0]
    assert torch.allclose(adv, torch.tensor([a0, a1, a2]), atol=1e-6)
    assert torch.allclose(buf.d["ret"][:, 0], adv + torch.tensor(v))


def test_terminated_does_not_bootstrap():
    buf = RolloutBuffer(1, 2, 1, 1)
    buf.add([[0.0], [0.0]], [[0.0], [0.0]], [0.0, 0.0], [5.0, 5.0], [1.0, 1.0], [0.9, 0.9], [True, False], [True, True], [False, True])
    buf.compute_returns(last_value=[100.0, 100.0], lam=0.95)
    assert buf.d["ret"][0].tolist() == pytest.approx([1.0, 1.0 + 0.9 * 5.0])


def test_ppo_improves_bandit():
    """Contextual bandit over 64 parallel 'envs': reward = -||z - target(feat)||^2, episode = one chunk."""
    torch.manual_seed(0)
    np.random.seed(0)
    n, steps = 64, 4
    ppo = DSRLPPO(1, 2, cfg=PPOConfig(lr=1e-3, epochs=4, minibatches=4, target_kl=None))
    target = np.array([1.5, -1.0], np.float32)

    def mean_reward():
        z, _, _ = ppo.act(np.zeros((500, 1), np.float32), deterministic=True)
        return -((z - target) ** 2).sum(-1).mean()

    before = mean_reward()
    buf = RolloutBuffer(steps, n, 1, 2)
    feat = np.zeros((n, 1), np.float32)
    ones = np.ones(n)
    for _ in range(60):
        buf.reset()
        for _ in range(steps):
            z, logp, value = ppo.act(feat)
            buf.add(feat, z, logp, value, -((z - target) ** 2).sum(-1), 0.9 * ones, ones, ones, 0 * ones)
        buf.compute_returns(ppo.value(feat), 0.95)
        metrics = ppo.update(buf)
    after = mean_reward()
    assert after > before + 1.0 and after > -0.5, (before, after)
    assert np.isfinite(metrics["approx_kl"]) and metrics["explained_var"] == metrics["explained_var"]


class _TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 3)

    def forward(self, img, proprio):
        return self.lin(img.flatten(1).float() / 255)


def _warmup_setup(encoder):
    torch.manual_seed(0)
    np.random.seed(0)
    n, steps = 8, 4
    ppo = DSRLPPO(3, 2, cfg=PPOConfig(epochs=2, minibatches=2), encoder=encoder)
    buf = RolloutBuffer(steps, n, 3, 2, img_shape=(2, 2) if encoder is not None else None)
    ones = np.ones(n)
    for _ in range(steps):
        feat = np.random.randn(n, 3).astype(np.float32)
        z, logp, value = ppo.act(feat)
        packed = (np.random.randint(0, 255, (n, 2, 2), dtype=np.uint8), np.zeros((n, 1), np.float32)) if encoder is not None else None
        buf.add(feat, z, logp, value, np.random.randn(n), 0.9 * ones, 0 * ones, 0 * ones, 0 * ones, packed)
    buf.compute_returns(ppo.value(feat), 0.95)
    return ppo, buf


def _snap(module):
    return [p.detach().clone() for p in module.parameters()]


def _same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


def test_critic_warmup_only_moves_critic():
    ppo, buf = _warmup_setup(None)
    actor0, critic0 = _snap(ppo.actor), _snap(ppo.critic)
    for _ in range(3):  # repeated updates: Adam state must not drift the actor either
        m = ppo.update(buf, train_policy=False)
    assert _same(actor0, _snap(ppo.actor))
    assert not _same(critic0, _snap(ppo.critic))
    assert m["warmup"] == 1 and not {"pg_loss", "approx_kl", "clipfrac", "entropy"} & m.keys() and "explained_var" in m
    actor1 = _snap(ppo.actor)
    m = ppo.update(buf)  # a normal update afterwards moves the actor and reports policy metrics
    assert not _same(actor1, _snap(ppo.actor)) and "pg_loss" in m and "warmup" not in m


@pytest.mark.parametrize("train_encoder", [False, True])
def test_critic_warmup_encoder_frozen_by_default(train_encoder):
    enc = _TinyEncoder()
    ppo, buf = _warmup_setup(enc)
    actor0, enc0, critic0 = _snap(ppo.actor), _snap(enc), _snap(ppo.critic)
    m = ppo.update(buf, train_policy=False, warmup_train_encoder=train_encoder)
    assert _same(actor0, _snap(ppo.actor)) and not _same(critic0, _snap(ppo.critic))
    assert _same(enc0, _snap(enc)) == (not train_encoder)
    assert ("enc_grad_norm" in m) == train_encoder


@pytest.mark.parametrize("encoder", [None, _TinyEncoder])
def test_privileged_state_reaches_actor_and_critic(encoder):
    torch.manual_seed(0)
    n, steps, pd = 8, 4, 5
    ppo = DSRLPPO(3, 2, cfg=PPOConfig(epochs=2, minibatches=2, target_kl=None), encoder=encoder() if encoder else None, priv_dim=pd)
    assert ppo.actor.trunk[0].in_features == 3 + pd and ppo.critic.v[0].in_features == 3 + pd
    buf = RolloutBuffer(steps, n, 3, 2, img_shape=(2, 2) if encoder else None, priv_dim=pd)
    ones = np.ones(n)
    for _ in range(steps):
        feat, priv = np.random.randn(n, 3).astype(np.float32), np.random.randn(n, pd).astype(np.float32)
        z, logp, value = ppo.act(feat, priv=priv)
        packed = (np.random.randint(0, 255, (n, 2, 2), dtype=np.uint8), np.zeros((n, 1), np.float32)) if encoder else None
        buf.add(feat, z, logp, value, np.random.randn(n), 0.9 * ones, 0 * ones, 0 * ones, 0 * ones, packed, priv)
    buf.compute_returns(ppo.value(feat, priv), 0.95)
    with pytest.raises(ValueError):
        ppo.act(feat)
    critic0 = _snap(ppo.critic)
    m = ppo.update(buf)
    assert not _same(critic0, _snap(ppo.critic)) and np.isfinite(m["pg_loss"])


def test_build_noise_broadcast():
    z = np.arange(3, dtype=np.float32)
    n = build_noise(z, (4, 3), 2, broadcast=True)
    assert n.shape == (4, 3) and (n[0] == z).all() and (n[1] == z).all() and not (n[2] == z).all()
    nb = build_noise(np.stack([z, z + 10]), (4, 3), 3, broadcast=True)
    assert nb.shape == (2, 4, 3) and (nb[1, :3] == z + 10).all()
