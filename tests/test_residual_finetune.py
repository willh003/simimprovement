import numpy as np
import torch
import torch.nn as nn

from polaris.latent_rl.algos import DSRLPPO, PPOConfig
from polaris.latent_rl.checkpoint import SteeringConfig
from polaris.latent_rl.networks import SteeringActor
from polaris.latent_rl.rollout import RolloutBuffer
from polaris.latent_rl.steered_policy import apply_residual


def test_residual_actor_bounds_and_logp():
    actor = SteeringActor(5, 4, bound=3.0, res_dim=6)
    torch.nn.init.normal_(actor.trunk[-1].weight, std=0.02)
    feat = torch.randn(10, 5)
    a, logp = actor(feat)
    assert a.shape == (10, 10) and (a[:, 4:].abs() < 1).all()
    logp2, _ = actor.evaluate(feat, a)
    assert torch.allclose(logp, logp2, atol=1e-3)


def test_apply_residual_scale():
    cfg = SteeringConfig((15, 32), 8, 8, "droid_proprio", residual_scale=0.01, residual_horizon=8)
    chunk = np.zeros((2, 15, 8), np.float32)
    out = apply_residual(chunk, np.ones((2, cfg.res_dim), np.float32), cfg)
    assert np.allclose(out[:, :8, :7], 0.01) and (out[:, :8, 7] == 0).all() and (out[:, 8:] == 0).all() and (chunk == 0).all()
    cfg0 = SteeringConfig((15, 32), 8, 8, "droid_proprio", residual_scale=0.0, residual_horizon=8)
    assert cfg0.res_dim == 0 and apply_residual(chunk, np.zeros((2, 0), np.float32), cfg0) is chunk


class TinyEnc(nn.Module):
    feat_dim = 3

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2 * 4 * 4 * 3, 3)

    def forward(self, imgs, proprio):
        return self.lin(imgs.float().reshape(len(imgs), -1) / 255)


def test_ppo_finetune_encoder_updates_encoder():
    enc = TinyEnc()
    ppo = DSRLPPO(3, 2, cfg=PPOConfig(encoder_microbatch=3, target_kl=None, epochs=1), res_dim=2, encoder=enc)
    buf = RolloutBuffer(2, 4, 3, 4, img_shape=(2, 4, 4, 3), proprio_dim=1)
    w0 = enc.lin.weight.clone()
    for _ in range(2):
        imgs = np.random.randint(0, 255, (4, 2, 4, 4, 3), np.uint8)
        feat = enc.eval()(torch.from_numpy(imgs), None).detach().numpy()
        z, logp, v = ppo.act(feat)
        buf.add(feat, z, logp, v, np.random.randn(4), np.full(4, 0.9), np.zeros(4), np.zeros(4), None, (imgs, np.zeros((4, 1))))
    buf.compute_returns(np.zeros(4))
    m = ppo.update(buf)
    assert not torch.equal(w0, enc.lin.weight) and "enc_grad_norm" in m and "res_abs_mean" in m
    assert np.isfinite(m["approx_kl"]) and abs(m["approx_kl"]) < 0.1
