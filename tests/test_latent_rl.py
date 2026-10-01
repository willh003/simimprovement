import numpy as np
import torch

from polaris.latent_rl.algos import DSRLSAC, SACConfig
from polaris.latent_rl.buffer import ReplayBuffer
from polaris.latent_rl.checkpoint import SteeringConfig, load_checkpoint, save_checkpoint
from polaris.latent_rl.steered_policy import SteeredPolicy


class ToyBase:
    """chunk = first noise row; reward is highest when chunk ~ target (depends on state)."""

    noise_shape = (4, 2)

    def query(self, obs, noise=None):
        return noise[:1].copy()


def test_actor_starts_near_standard_normal():
    algo = DSRLSAC(3, 2)
    z = algo.act(torch.zeros(10000, 3))
    assert abs(z.std().item() - 1.0) < 0.15 and abs(z.mean().item()) < 0.05


def test_sac_improves_bandit_and_ckpt_roundtrip(tmp_path):
    torch.manual_seed(0)
    np.random.seed(0)
    algo = DSRLSAC(1, 2, cfg=SACConfig(init_alpha=0.02, target_entropy=None))
    buf = ReplayBuffer(1, 2)
    target = torch.tensor([1.5, -1.0])

    def reward(z):
        return -((z - target) ** 2).sum().item()

    def mean_reward(n=200):
        z = algo.act(torch.zeros(n, 1))
        return -((z - target) ** 2).sum(-1).mean().item()

    before = mean_reward()
    feat = np.zeros(1, np.float32)
    for i in range(1500):
        z = algo.act(torch.as_tensor(feat)[None])[0]
        buf.add(feat, z, reward(z), feat, True, 1.0)
        if len(buf) >= 64:
            algo.update(buf.sample(64))
    after = mean_reward()
    assert after > before + 1.0 and after > -1.0, (before, after)

    cfg = SteeringConfig(noise_shape=(4, 2), steer_horizon=1, feat_dim=1, encoder="droid_proprio")
    save_checkpoint(tmp_path, algo.actor, cfg)
    actor, cfg2 = load_checkpoint(tmp_path)
    x = torch.zeros(1, 1)
    assert torch.allclose(actor(x, True)[0], algo.actor(x, True)[0])
    assert cfg2.noise_shape == (4, 2)


def test_steered_policy_builds_noise(tmp_path):
    algo = DSRLSAC(8, 4)
    cfg = SteeringConfig(noise_shape=(4, 2), steer_horizon=2, feat_dim=8, encoder="droid_proprio")
    save_checkpoint(tmp_path, algo.actor, cfg)
    sp = SteeredPolicy.from_checkpoint(tmp_path, ToyBase())
    obs = {"observation/joint_position": np.zeros(7), "observation/gripper_position": np.zeros(1)}
    out = sp.step(obs)
    assert out.chunk.shape == (1, 2) and out.z.shape == (4,)


def test_vision_encoder_shape_and_ckpt_kwargs(tmp_path):
    import pytest

    from polaris.latent_rl.encoders import make_encoder

    try:
        enc = make_encoder("droid_vision", device="cpu")
    except Exception as e:  # weights unavailable offline
        pytest.skip(f"backbone unavailable: {e}")
    obs = {
        "observation/exterior_image_1_left": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros(7),
        "observation/gripper_position": np.zeros(1),
    }
    f1, f2 = enc(obs), enc(obs)
    assert f1.shape == (enc.feat_dim,) and np.isfinite(f1).all() and np.allclose(f1, f2)

    algo = DSRLSAC(enc.feat_dim, 4)
    cfg = SteeringConfig(noise_shape=(4, 2), steer_horizon=2, feat_dim=enc.feat_dim, encoder="droid_vision", encoder_kwargs={"device": "cpu"})
    save_checkpoint(tmp_path, algo.actor, cfg)
    sp = SteeredPolicy.from_checkpoint(tmp_path, ToyBase())
    assert sp.encoder.feat_dim == enc.feat_dim
    assert sp.step(obs).chunk.shape == (1, 2)
