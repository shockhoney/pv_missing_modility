from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


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


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class ResNet50PalmEncoder(nn.Module):
    def __init__(
        self,
        input_channel: int = 3,
        input_size: int = 224,
        embedding_size: int = 256,
        pretrained_path: str | os.PathLike[str] | None = None,
    ):
        super().__init__()
        if input_channel != 3:
            raise ValueError("ResNet50 pretrained encoder requires 3 input channels")
        self.input_size = input_size
        self.backbone = resnet50(weights=None)
        _load_pretrained(self.backbone, pretrained_path, skip_prefixes=("fc.",))
        self.backbone.fc = nn.Identity()
        self.project = nn.Sequential(
            conv1x1(2048, embedding_size),
            nn.BatchNorm2d(embedding_size),
            nn.ReLU(inplace=False),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(embedding_size)
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

    def forward(self, x: torch.Tensor, return_spatial: bool = False) -> torch.Tensor:
        feat_map = self.forward_features(x)
        if return_spatial:
            return feat_map
        pooled = self.global_pool(feat_map).flatten(1)
        return self.bn(pooled)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last"):
        super().__init__()
        if data_format not in {"channels_last", "channels_first"}:
            raise ValueError(f"Unsupported data_format: {data_format}")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class GRN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask.div(keep_prob)


class Block(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


class ConvNeXtV2(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        depths: list[int] | tuple[int, ...] = (3, 3, 9, 3),
        dims: list[int] | tuple[int, ...] = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
        head_init_scale: float = 1.0,
    ):
        super().__init__()
        self.depths = list(depths)
        self.dims = list(dims)
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            )
        )
        for index in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[index], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[index], dims[index + 1], kernel_size=2, stride=2),
                )
            )

        rates = [value.item() for value in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        cursor = 0
        for index in range(4):
            self.stages.append(nn.Sequential(*(Block(dims[index], rates[cursor + j]) for j in range(depths[index]))))
            cursor += depths[index]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        for index in range(4):
            x = self.downsample_layers[index](x)
            x = self.stages[index](x)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.forward_spatial(x).mean([-2, -1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def convnextv2_tiny(**kwargs) -> ConvNeXtV2:
    return ConvNeXtV2(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), **kwargs)


class ConvNeXtV2TinyVeinEncoder(nn.Module):
    def __init__(
        self,
        input_channel: int = 3,
        input_size: int = 224,
        embedding_size: int = 256,
        pretrained_path: str | os.PathLike[str] | None = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.backbone = convnextv2_tiny(in_chans=input_channel, num_classes=1000)
        _load_pretrained(self.backbone, pretrained_path, skip_prefixes=("head.",))
        self.backbone.head = nn.Identity()
        self.project = nn.Sequential(
            conv1x1(768, embedding_size),
            nn.BatchNorm2d(embedding_size),
            nn.ReLU(inplace=False),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(embedding_size)
        self.out_dim = embedding_size
        self.local_dim = embedding_size

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.backbone.forward_spatial(x))

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
    if name == "palm":
        return ResNet50PalmEncoder(input_channel, input_size, embedding_size, pretrained_path, **kwargs)
    if name == "vein":
        return ConvNeXtV2TinyVeinEncoder(input_channel, input_size, embedding_size, pretrained_path, **kwargs)
    raise ValueError(f"Unsupported modality: {modality}")
