from __future__ import annotations

import os

import torch
import torch.nn as nn
from torchvision.models import resnet18

from utils.checkpoint_io import remove_state_prefix, safe_torch_load, tensor_state_dict


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_path(path: str | os.PathLike[str] | None) -> str | None:
    if path is None:
        return None
    path = os.fspath(path)
    if not path:
        return None
    path = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Pretrained weight file not found: {path}")
    return path


def _load_pretrained(module: nn.Module, path: str | os.PathLike[str] | None, skip_prefixes: tuple[str, ...]) -> None:
    path = _resolve_path(path)
    if path is None:
        return

    target = module.state_dict()
    state = {}
    for key, value in tensor_state_dict(safe_torch_load(path, "cpu")).items():
        key = remove_state_prefix(key)
        if any(key.startswith(prefix) for prefix in skip_prefixes):
            continue
        if key in target and target[key].shape == value.shape:
            state[key] = value
    module.load_state_dict(state, strict=False)


class SEBlock(nn.Module):
    def __init__(self, channels: int = 512, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class ResNet18Encoder(nn.Module):
    def __init__(
        self,
        input_channel: int = 3,
        embedding_size: int = 256,
        pretrained_path: str | os.PathLike[str] | None = None,
        use_se: bool = False,
    ):
        super().__init__()
        if embedding_size % 2 != 0:
            raise ValueError("embedding_size must be divisible by 2")
        self.part_dim = embedding_size // 2
        self.backbone = resnet18(weights=None)
        if input_channel != 3:
            self.backbone.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        _load_pretrained(self.backbone, pretrained_path, skip_prefixes=("fc.",))
        self.backbone.fc = nn.Identity()
        self.se = SEBlock(512) if use_se else nn.Identity()
        self.project = nn.Sequential(
            nn.Conv2d(512, embedding_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_size),
            nn.ReLU(inplace=False),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(embedding_size)
        self.shared_head = nn.Sequential(nn.Linear(embedding_size, self.part_dim, bias=False))
        self.specific_head = nn.Sequential(nn.Linear(embedding_size, self.part_dim, bias=False))
        self._init_part_heads()
        self.local_dim = embedding_size

    def _init_part_heads(self):
        with torch.no_grad():
            self.shared_head[0].weight.zero_()
            self.specific_head[0].weight.zero_()
            self.shared_head[0].weight[:, : self.part_dim].copy_(torch.eye(self.part_dim))
            self.specific_head[0].weight[:, self.part_dim :].copy_(torch.eye(self.part_dim))

    def _backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        model = self.backbone
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        return self.se(model.layer4(x))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self._backbone_features(x))

    def embedding_from_features(self, feat_map: torch.Tensor) -> torch.Tensor:
        return self.bn(self.global_pool(feat_map).flatten(1))

    def split_embedding(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.shared_head(embedding), self.specific_head(embedding)

    def parts_from_features(self, feat_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.split_embedding(self.embedding_from_features(feat_map))

    def parts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw ``(shared, specific)`` head outputs for an image batch."""

        return self.parts_from_features(self.forward_features(x))

    def forward(self, x: torch.Tensor, return_spatial: bool = False) -> torch.Tensor:
        feat_map = self.forward_features(x)
        if return_spatial:
            return feat_map
        return self.embedding_from_features(feat_map)


def build_encoder(
    modality: str,
    input_channel: int = 3,
    embedding_size: int = 256,
    pretrained_path: str | os.PathLike[str] | None = None,
    **kwargs,
) -> nn.Module:
    name = modality.strip().lower()
    if name in {"palm", "vein"}:
        return ResNet18Encoder(
            input_channel=input_channel,
            embedding_size=embedding_size,
            pretrained_path=pretrained_path,
            use_se=name == "vein",
            **kwargs,
        )
    raise ValueError(f"Unsupported modality: {modality}")
