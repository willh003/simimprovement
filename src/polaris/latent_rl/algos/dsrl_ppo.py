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
    encoder_lr: float = 3e-5  # finetuned encoder only
    encoder_microbatch: int = 32  # transitions per encoder forward/backward chunk (2 images each) during the update


class DSRLPPO:
    """PPO where the 'action' is the (steered slice of the) flow-matching noise z.

    The actor is the same `SteeringActor` as SAC (zero-init last layer -> starts at ~N(0, I), the base policy's own noise),
    so checkpoints work with `SteeredPolicy` / the `Steered` client. Rollouts carry `discount` = gamma^k per transition.
    """

    def __init__(self, feat_dim: int, steer_dim: int, bound: float = 3.0, cfg: PPOConfig | None = None, device="cpu",
                 res_dim: int = 0, encoder: nn.Module | None = None, priv_dim: int = 0):
        """`encoder`: a finetuned image encoder (`encoder(img, proprio) -> feat`, buffer built with `img_shape`); the update
        re-encodes each minibatch with gradients. `priv_dim`: privileged state appended to the features for the actor and
        critic (they take `[feat, priv]`; the base policy never sees it). The actor action is [z (steer_dim), unit residual (res_dim)]."""
        self.cfg = cfg or PPOConfig()
        self.device = device
        self.steer_dim, self.encoder, self.priv_dim = steer_dim, encoder, priv_dim
        feat_dim += priv_dim
        self.actor = SteeringActor(feat_dim, steer_dim, bound, self.cfg.hidden, res_dim).to(device)
        self.critic = ValueNet(feat_dim, self.cfg.hidden).to(device)
        self.params = [*self.actor.parameters(), *self.critic.parameters()]
        groups = [dict(params=self.params, lr=self.cfg.lr)]
        self.enc_params = list(encoder.parameters()) if encoder is not None else []
        if self.enc_params:
            groups.append(dict(params=self.enc_params, lr=self.cfg.encoder_lr))
        self.opt = torch.optim.Adam(groups, eps=1e-5)

    def _encode(self, b: dict) -> torch.Tensor:
        """Encoder features of a minibatch without grad, in microbatches."""
        m = self.cfg.encoder_microbatch
        with torch.no_grad():
            return torch.cat([self.encoder(b["img"][s : s + m], b["proprio"][s : s + m]) for s in range(0, len(b["img"]), m)])

    def _backward_encoder(self, b: dict, feat_grad: torch.Tensor):
        """Backprop d loss / d feat into the encoder, recomputing each microbatch's forward with grad (bounded memory)."""
        m = self.cfg.encoder_microbatch
        for s in range(0, len(b["img"]), m):
            self.encoder(b["img"][s : s + m], b["proprio"][s : s + m]).backward(feat_grad[s : s + m])

    def _t(self, x) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _obs(self, feat, priv=None) -> torch.Tensor:
        """Actor/critic input [feat, priv] (N, feat_dim + priv_dim)."""
        feat = self._t(feat)
        if self.priv_dim:
            if priv is None:
                raise ValueError(f"this PPO was built with priv_dim={self.priv_dim}: pass the privileged state")
            feat = torch.cat([feat, self._t(priv)], dim=-1)
        return feat

    @torch.no_grad()
    def act(self, feat, deterministic: bool = False, priv=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(N, feat_dim) [+ priv (N, priv_dim)] -> z (N, steer_dim), logp (N,), value (N,) as numpy."""
        feat = self._obs(feat, priv)
        z, logp = self.actor(feat, deterministic)
        return z.cpu().numpy(), logp.cpu().numpy(), self.critic(feat).cpu().numpy()

    @torch.no_grad()
    def value(self, feat, priv=None) -> np.ndarray:
        return self.critic(self._obs(feat, priv)).cpu().numpy()

    def update(self, buf: RolloutBuffer, train_policy: bool = True, warmup_train_encoder: bool = False) -> dict:
        """PPO epochs over `buf`; the caller must have run `buf.compute_returns(last_value, cfg.gae_lambda)`.

        `train_policy=False` (critic warmup): loss = vf_coef * v_loss only; the actor gets no gradient (Adam skips params whose
        grad is None, so it does not move), there is no target_kl early stop, and policy metrics are not reported. The encoder
        is frozen (features detached) unless `warmup_train_encoder`."""
        cfg = self.cfg
        train_enc = self.encoder is not None and (train_policy or warmup_train_encoder)
        params = self.params if train_policy else list(self.critic.parameters())
        stats: dict[str, list[float]] = {}
        epochs_run = 0
        for _ in range(cfg.epochs):
            kls = []
            for b in buf.minibatches(cfg.minibatches):
                feat = b["feat"]
                if self.encoder is not None:  # fresh features from the current encoder; its grads are filled in after backward
                    feat = self._encode(b)
                    if train_enc:
                        feat.requires_grad_()
                obs = torch.cat([feat, b["priv"]], dim=-1) if self.priv_dim else feat  # grads flow back to feat -> encoder
                v = self.critic(obs)
                if cfg.value_clip is None:
                    v_loss = 0.5 * (v - b["ret"]).pow(2).mean()
                else:
                    v_clipped = b["value"] + (v - b["value"]).clamp(-cfg.value_clip, cfg.value_clip)
                    v_loss = 0.5 * torch.max((v - b["ret"]).pow(2), (v_clipped - b["ret"]).pow(2)).mean()
                if train_policy:
                    logp, entropy = self.actor.evaluate(obs, b["z"])
                    log_ratio = logp - b["logp"]
                    ratio = log_ratio.exp()
                    adv = b["adv"]
                    if cfg.norm_adv and adv.numel() > 1:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    pg_loss = torch.max(-adv * ratio, -adv * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                    ent = entropy.mean()
                    loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent
                else:
                    loss = cfg.vf_coef * v_loss
                self.opt.zero_grad(set_to_none=True)  # Adam skips grad-less params: the actor (and a frozen encoder) stay put
                loss.backward()
                extra = {}
                if train_enc:
                    self._backward_encoder(b, feat.grad)
                    extra["enc_grad_norm"] = nn.utils.clip_grad_norm_(self.enc_params, cfg.max_grad_norm).item()
                grad_norm = nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                self.opt.step()
                step_stats = dict(v_loss=v_loss.item(), grad_norm=grad_norm.item(), **extra)
                if train_policy:
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - log_ratio).mean().item()
                        clipfrac = ((ratio - 1).abs() > cfg.clip).float().mean().item()
                    kls.append(approx_kl)
                    step_stats.update(pg_loss=pg_loss.item(), entropy=ent.item(), approx_kl=approx_kl, clipfrac=clipfrac)
                for k, val in step_stats.items():
                    stats.setdefault(k, []).append(val)
            epochs_run += 1
            if train_policy and cfg.target_kl is not None and np.mean(kls) > cfg.target_kl:
                break
        data = buf.flat()
        with torch.no_grad():
            _, log_std = self.actor.dist(torch.cat([data["feat"], data["priv"]], dim=-1) if self.priv_dim else data["feat"])
            var_ret = data["ret"].var()
            explained_var = (1 - (data["ret"] - data["value"]).var() / var_ret).item() if var_ret > 0 else float("nan")
        return dict(
            **{k: float(np.mean(v)) for k, v in stats.items()}, epochs=epochs_run, explained_var=explained_var,
            **({} if train_policy else {"warmup": 1}),
            std_mean=log_std.exp().mean().item(), value_mean=data["value"].mean().item(), return_mean=data["ret"].mean().item(),
            adv_std=data["adv"].std().item(), z_abs_mean=data["z"][:, : self.steer_dim].abs().mean().item(),
            **({"res_abs_mean": data["z"][:, self.steer_dim :].abs().mean().item()} if self.actor.res_dim else {}), reward_mean=data["reward"].mean().item(),
        )

    def state_dict(self) -> dict:
        d = dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(), opt=self.opt.state_dict())
        if self.encoder is not None:
            d["encoder"] = self.encoder.state_dict()
        return d

    def load_state_dict(self, d: dict):
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])
        self.opt.load_state_dict(d["opt"])
        if self.encoder is not None:
            self.encoder.load_state_dict(d["encoder"])
