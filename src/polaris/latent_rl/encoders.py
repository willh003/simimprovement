import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ENCODERS: dict[str, type] = {}


def register_encoder(name: str):
    def deco(cls):
        ENCODERS[name] = cls
        cls.name = name
        return cls

    return deco


def make_encoder(name: str, **kwargs):
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder {name}. Available: {list(ENCODERS)}")
    return ENCODERS[name](**kwargs)


def encode_batch(encoder, requests: list[dict]) -> np.ndarray:
    """(N, feat_dim) features; uses the encoder's own `batch` method when it has one."""
    if hasattr(encoder, "batch"):
        return encoder.batch(requests)
    return np.stack([encoder(r) for r in requests]).astype(np.float32)


def _proprio(obs: dict) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["observation/joint_position"], np.float32).reshape(-1),
            np.asarray(obs["observation/gripper_position"], np.float32).reshape(-1),
        ]
    )


@register_encoder("droid_proprio")
class DroidProprioEncoder:
    """Joint position (7) + gripper position (1) from an openpi-DROID request dict."""

    feat_dim = 8

    def __init__(self, **_):
        pass

    def __call__(self, obs: dict) -> np.ndarray:
        return _proprio(obs)


class ImageEncoder(nn.Module):
    """Base for the camera encoders: one backbone shared by the exterior + wrist cameras -> per-camera pooled feature,
    layer-normed, + proprio. Output: [feat_exterior, feat_wrist, joint_position, gripper_position].

    `finetune=True` makes the backbone trainable (DSRLPPO re-encodes stored images with gradients via `forward`); otherwise
    it is frozen and features are cached in the rollout buffer. The module is always kept in eval mode (BatchNorm /
    dropout fixed) so rollout-time and update-time features agree. `batch` / `encode_packed` are the no-grad numpy paths
    used for rollouts and deployment. bf16 autocast on cuda. Subclasses set `self.backbone`, `self.img_dim` and implement
    `_pool(x) -> (B*2, img_dim)`.
    """

    def __init__(self, finetune: bool, device: str | None):
        super().__init__()
        self.finetune = finetune
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def _finish_init(self):
        self.feat_dim = 2 * self.img_dim + 8
        self.backbone.requires_grad_(self.finetune)
        self.to(self.dev).eval()

    def train(self, mode: bool = True):
        return super().train(False)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, imgs: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """imgs (B, 2, H, W, 3) uint8 [exterior, wrist], proprio (B, 8) -> feat (B, feat_dim); differentiable."""
        b = imgs.shape[0]
        x = imgs.reshape(b * 2, *imgs.shape[2:]).permute(0, 3, 1, 2).float() / 255.0
        x = (x - self.mean) / self.std
        with torch.autocast(self.dev.split(":")[0], dtype=torch.bfloat16, enabled=self.dev.startswith("cuda")):
            f = self._pool(x)
        f = F.layer_norm(f.float(), f.shape[-1:]).reshape(b, -1)
        return torch.cat([f, proprio.float()], dim=-1)

    def pack(self, requests: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """Request dicts (224x224 uint8 images) -> (images (N, 2, H, W, 3) uint8, proprio (N, 8) float32)."""
        imgs = np.stack([np.stack([r["observation/exterior_image_1_left"], r["observation/wrist_image_left"]]) for r in requests])
        return imgs, np.stack([_proprio(r) for r in requests])

    @torch.no_grad()
    def encode_packed(self, imgs: np.ndarray, proprio: np.ndarray, minibatch: int = 64) -> np.ndarray:
        out = []
        for s in range(0, len(imgs), minibatch):
            i = torch.from_numpy(imgs[s : s + minibatch]).to(self.dev)
            p = torch.from_numpy(proprio[s : s + minibatch]).to(self.dev)
            out.append(self(i, p).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def batch(self, requests: list[dict]) -> np.ndarray:
        return self.encode_packed(*self.pack(requests))

    def __call__(self, obs, proprio=None):
        if isinstance(obs, dict):  # numpy deployment path
            return self.batch([obs])[0]
        return super().__call__(obs, proprio)


@register_encoder("droid_vision")
class DroidVisionEncoder(ImageEncoder):
    """DINOv2 (HF) CLS embeddings of the exterior + wrist cameras, plus proprio; trainable with `finetune=True`."""

    def __init__(self, backbone: str = "facebook/dinov2-small", pretrained: bool = True, finetune: bool = True,
                 device: str | None = None, **_):
        super().__init__(finetune, device)
        from transformers import AutoConfig, AutoModel

        self.backbone = AutoModel.from_pretrained(backbone) if pretrained else AutoModel.from_config(AutoConfig.from_pretrained(backbone))
        self.img_dim = self.backbone.config.hidden_size
        self._finish_init()

    def _pool(self, x):
        return self.backbone(pixel_values=x).last_hidden_state[:, 0]


@register_encoder("droid_resnet50")
class DroidResNet50Encoder(ImageEncoder):
    """ImageNet ResNet50 pooled features of the exterior + wrist cameras, plus proprio; trainable with `finetune=True`."""

    img_dim = 2048

    def __init__(self, pretrained: bool = True, finetune: bool = True, device: str | None = None, **_):
        super().__init__(finetune, device)
        import torchvision

        net = torchvision.models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        net.fc = nn.Identity()
        self.backbone = net.to(memory_format=torch.channels_last)
        self._finish_init()

    def _pool(self, x):
        return self.backbone(x.contiguous(memory_format=torch.channels_last))
