from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from torchvision.models import resnet18


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


def _safe_torch_load(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _state_dict_from_checkpoint(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Pretrained checkpoint must be a state dict or contain one")

    for key in ("state_dict", "model", "encoder"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            checkpoint = value
            break

    state = {str(key): value for key, value in checkpoint.items() if torch.is_tensor(value)}
    if not state:
        raise TypeError("No tensor state dict found in pretrained checkpoint")
    return state


def _clean_key(key: str) -> str:
    for prefix in ("module.", "model.", "backbone."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _load_pretrained(module: nn.Module, path: str | os.PathLike[str] | None, skip_prefixes: tuple[str, ...]) -> None:
    path = _resolve_path(path)
    if path is None:
        return

    target = module.state_dict()
    state = {}
    for key, value in _state_dict_from_checkpoint(_safe_torch_load(path)).items():
        key = _clean_key(key)
        if any(key.startswith(prefix) for prefix in skip_prefixes):
            continue
        if key in target and target[key].shape == value.shape:
            state[key] = value
    module.load_state_dict(state, strict=False)


class ResNet18Encoder(nn.Module):
    def __init__(
        self,
        input_channel: int = 3,
        input_size: int = 224,
        embedding_size: int = 256,
        pretrained_path: str | os.PathLike[str] | None = None,
    ):
        super().__init__()
        if embedding_size % 2 != 0:
            raise ValueError("embedding_size must be divisible by 2")
        self.input_size = input_size
        self.part_dim = embedding_size // 2
        self.backbone = resnet18(weights=None)
        if input_channel != 3:
            self.backbone.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        _load_pretrained(self.backbone, pretrained_path, skip_prefixes=("fc.",))
        self.backbone.fc = nn.Identity()
        self.project = nn.Sequential(
            nn.Conv2d(512, embedding_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_size),
            nn.ReLU(inplace=False),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(embedding_size)
        self.shared_head = nn.Sequential(nn.Linear(512, self.part_dim, bias=False), nn.BatchNorm1d(self.part_dim))
        self.specific_head = nn.Sequential(nn.Linear(512, self.part_dim, bias=False), nn.BatchNorm1d(self.part_dim))
        self.out_dim = embedding_size
        self.local_dim = embedding_size

    def _backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        model = self.backbone
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        return model.layer4(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self._backbone_features(x))

    def forward_parts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.global_pool(self._backbone_features(x)).flatten(1)
        return self.shared_head(pooled), self.specific_head(pooled)

    def forward(self, x: torch.Tensor, return_spatial: bool = False) -> torch.Tensor:
        feat_map = self.forward_features(x)
        if return_spatial:
            return feat_map
        pooled = self.global_pool(feat_map).flatten(1)
        return self.bn(pooled)


def build_encoder(
    modality: str,
    input_channel: int = 3,
    input_size: int = 224,
    embedding_size: int = 256,
    pretrained_path: str | os.PathLike[str] | None = None,
    **kwargs,
) -> nn.Module:
    name = modality.strip().lower()
    if name in {"palm", "vein"}:
        return ResNet18Encoder(input_channel, input_size, embedding_size, pretrained_path, **kwargs)
    raise ValueError(f"Unsupported modality: {modality}")
