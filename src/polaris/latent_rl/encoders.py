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


@register_encoder("droid_vision")
class DroidVisionEncoder:
    """Frozen pretrained ViT CLS embeddings of the exterior + wrist cameras, plus proprio.

    Input is the openpi-DROID request dict (224x224 uint8 images). Output:
    [cls_exterior, cls_wrist, joint_position, gripper_position].
    """

    def __init__(self, backbone: str = "facebook/dinov2-small", normalize: bool = True, device: str | None = None, **_):
        import torch
        from transformers import AutoModel

        self.backbone, self.normalize = backbone, normalize
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(backbone).to(self.device, self.dtype).eval().requires_grad_(False)
        self.img_dim = self.model.config.hidden_size
        self.feat_dim = 2 * self.img_dim + 8
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=self.dtype).view(1, 3, 1, 1)

    def __call__(self, obs: dict) -> np.ndarray:
        return self.batch([obs])[0]

    def batch(self, requests: list[dict], minibatch: int = 64) -> np.ndarray:
        import torch

        feats = []
        for s in range(0, len(requests), minibatch):
            reqs = requests[s : s + minibatch]
            imgs = np.stack([img for r in reqs for img in (r["observation/exterior_image_1_left"], r["observation/wrist_image_left"])])
            x = torch.from_numpy(imgs).to(self.device).permute(0, 3, 1, 2).to(self.dtype) / 255.0
            x = (x - self.mean) / self.std
            with torch.no_grad():
                cls = self.model(pixel_values=x).last_hidden_state[:, 0].float()
            if self.normalize:
                cls = torch.nn.functional.layer_norm(cls, cls.shape[-1:])
            cls = cls.reshape(len(reqs), -1).cpu().numpy()  # [exterior, wrist] per request
            feats.append(np.concatenate([cls, np.stack([_proprio(r) for r in reqs])], axis=1))
        return np.concatenate(feats).astype(np.float32)


@register_encoder("droid_resnet50")
class DroidResNet50Encoder(nn.Module):
    """ImageNet ResNet50 (shared by the exterior + wrist cameras) -> per-camera pooled 2048-d, layer-normed, + proprio.

    Output: [feat_exterior, feat_wrist, joint_position, gripper_position] (4104-d). `finetune=True` makes the backbone
    trainable (DSRLPPO re-encodes stored images with gradients: `forward(img, proprio)`); BatchNorm always stays in eval
    mode (running stats frozen) so rollout-time and update-time features agree. `batch` / `encode_packed` are the
    no-grad numpy paths used for rollouts and deployment. bf16 autocast on cuda.
    """

    img_dim = 2048

    def __init__(self, pretrained: bool = True, finetune: bool = True, device: str | None = None, **_):
        super().__init__()
        import torchvision

        net = torchvision.models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        net.fc = nn.Identity()
        self.net = net.to(memory_format=torch.channels_last)
        self.finetune = finetune
        self.feat_dim = 2 * self.img_dim + 8
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.net.requires_grad_(finetune)
        self.to(self.dev).eval()

    def train(self, mode: bool = True):
        return super().train(False)  # BatchNorm statistics stay frozen

    def forward(self, imgs: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """imgs (B, 2, H, W, 3) uint8 [exterior, wrist], proprio (B, 8) -> feat (B, feat_dim); differentiable."""
        b = imgs.shape[0]
        x = imgs.reshape(b * 2, *imgs.shape[2:]).permute(0, 3, 1, 2).float() / 255.0
        x = ((x - self.mean) / self.std).contiguous(memory_format=torch.channels_last)
        with torch.autocast(self.dev.split(":")[0], dtype=torch.bfloat16, enabled=self.dev.startswith("cuda")):
            f = self.net(x)
        f = F.layer_norm(f.float(), f.shape[-1:]).reshape(b, -1)
        return torch.cat([f, proprio.float()], dim=-1)

    def pack(self, requests: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """Request dicts -> (images (N, 2, H, W, 3) uint8, proprio (N, 8) float32)."""
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

    def __call__(self, obs, *args, **kwargs):
        if isinstance(obs, dict):  # numpy deployment path
            return self.batch([obs])[0]
        return super().__call__(obs, *args, **kwargs)
