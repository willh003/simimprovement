import numpy as np

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


@register_encoder("droid_proprio")
class DroidProprioEncoder:
    """Joint position (7) + gripper position (1) from an openpi-DROID request dict."""

    feat_dim = 8

    def __init__(self, **_):
        pass

    def __call__(self, obs: dict) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(obs["observation/joint_position"], np.float32).reshape(-1),
                np.asarray(obs["observation/gripper_position"], np.float32).reshape(-1),
            ]
        )


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
        import torch

        imgs = np.stack([obs["observation/exterior_image_1_left"], obs["observation/wrist_image_left"]])
        x = torch.from_numpy(imgs).to(self.device).permute(0, 3, 1, 2).to(self.dtype) / 255.0
        x = (x - self.mean) / self.std
        with torch.no_grad():
            cls = self.model(pixel_values=x).last_hidden_state[:, 0].float()
        if self.normalize:
            cls = torch.nn.functional.layer_norm(cls, cls.shape[-1:])
        proprio = np.concatenate(
            [
                np.asarray(obs["observation/joint_position"], np.float32).reshape(-1),
                np.asarray(obs["observation/gripper_position"], np.float32).reshape(-1),
            ]
        )
        return np.concatenate([cls.reshape(-1).cpu().numpy(), proprio]).astype(np.float32)
