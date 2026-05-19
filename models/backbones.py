from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_channels: int, out_channels: int, stride: int = 1, groups: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=groups,
        bias=False,
    )


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        width = planes
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride=stride)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


class ResNet50PalmEncoder(nn.Module):
    def __init__(
        self,
        input_channel: int = 3,
        input_size: int = 224,
        embedding_size: int = 256,
        layers: Sequence[int] = (3, 4, 6, 3),
        zero_init_residual: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.layers = tuple(layers)
        self.inplanes = 64

        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.project = nn.Sequential(
            conv1x1(512 * Bottleneck.expansion, embedding_size),
            nn.BatchNorm2d(embedding_size),
            nn.ReLU(inplace=False),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(embedding_size)
        self.out_dim = embedding_size
        self.local_dim = embedding_size

        self._init_weights(zero_init_residual)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        out_channels = planes * Bottleneck.expansion
        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(conv1x1(self.inplanes, out_channels, stride), nn.BatchNorm2d(out_channels))

        layers = [Bottleneck(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = out_channels
        layers.extend(Bottleneck(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _init_weights(self, zero_init_residual: bool) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, Bottleneck):
                    nn.init.zeros_(module.bn3.weight)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.project(x)

    def forward(self, x: torch.Tensor, return_spatial: bool = False) -> torch.Tensor:
        feat_map = self.forward_features(x)
        if return_spatial:
            return feat_map
        pooled = self.global_pool(feat_map).flatten(1)
        return self.bn(pooled)


class Stem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Embedding(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FactorizedLargeDW(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), padding=(0, pad), groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), padding=(pad, 0), groups=channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LaKBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        small_kernels: Iterable[int] = (3, 7),
        mlp_ratio: int = 2,
    ):
        super().__init__()
        hidden = channels * mlp_ratio
        self.pre = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
        )
        self.large_dw = FactorizedLargeDW(channels, kernel_size)
        self.dw5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False)
        self.dw_d2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=False)
        self.dw_d4 = nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4, groups=channels, bias=False)
        self.small_dws = nn.ModuleList(
            nn.Conv2d(channels, channels, kernel_size=k, padding=k // 2, groups=channels, bias=False)
            for k in small_kernels
        )
        self.gate = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1), nn.Sigmoid())
        self.pw = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pre(x)
        branches = self.large_dw(y) + self.dw5(y) + self.dw_d2(y) + self.dw_d4(y)
        count = 4
        for branch in self.small_dws:
            branches = branches + branch(y)
            count += 1
        y = (branches / count) * self.gate(y)
        return x + self.pw(y)


class Neck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StarLKNetEncoder(nn.Module):
    def __init__(
        self,
        input_channel: int = 3,
        input_size: int = 224,
        embedding_size: int = 256,
        depths: Sequence[int] = (2, 2, 18, 2),
        channels: Sequence[int] = (128, 256, 512, 1024),
        kernels: Sequence[int] = (31, 29, 27, 13),
    ):
        super().__init__()
        if not (len(depths) == len(channels) == len(kernels)):
            raise ValueError("depths, channels and kernels must have the same length")

        self.input_size = input_size
        self.depths = tuple(depths)
        self.channels = tuple(channels)
        self.kernels = tuple(kernels)
        stem_channels = 64
        self.stem = Stem(input_channel, stem_channels)
        self.stages = nn.ModuleList(
            nn.Sequential(*(LaKBlock(dim, kernel) for _ in range(depth)))
            for depth, dim, kernel in zip(depths, channels, kernels)
        )
        self.embeddings = nn.ModuleList(
            [Embedding(stem_channels, channels[0], stride=1)]
            + [Embedding(channels[i], channels[i + 1], stride=2) for i in range(len(channels) - 1)]
        )
        self.neck = Neck(channels[-1], embedding_size)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(embedding_size)
        self.out_dim = embedding_size
        self.local_dim = embedding_size

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for index, stage in enumerate(self.stages):
            x = self.embeddings[index](x)
            x = stage(x)
        return self.neck(x)

    def forward(self, x: torch.Tensor, return_spatial: bool = False) -> torch.Tensor:
        feat_map = self.forward_features(x)
        if return_spatial:
            return feat_map
        pooled = self.global_pool(feat_map).flatten(1)
        return self.bn(pooled)


class LaKNet(StarLKNetEncoder):
    pass


def build_encoder(
    modality: str,
    input_channel: int = 3,
    input_size: int = 224,
    embedding_size: int = 256,
    **kwargs,
) -> nn.Module:
    name = modality.strip().lower()
    if name == "palm":
        return ResNet50PalmEncoder(input_channel, input_size, embedding_size, **kwargs)
    if name == "vein":
        return StarLKNetEncoder(input_channel, input_size, embedding_size, **kwargs)
    raise ValueError(f"Unsupported modality: {modality}")
