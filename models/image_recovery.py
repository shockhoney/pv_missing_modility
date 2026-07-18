"""Image-domain recovery modules for missing-modality recognition."""

from __future__ import annotations

import torch
import torch.nn as nn


ARCHITECTURE_VERSION = "recovery_domain_residual_adapter_v1"


class ResidualImageBlock(nn.Module):
    """A small residual block that remains stable across acquisition sessions."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.net(inputs)


class ResidualImageAdapter(nn.Module):
    """Recover a target-style image while retaining available-modality structure."""

    def __init__(self, channels: int = 32, blocks: int = 3) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*(ResidualImageBlock(channels) for _ in range(blocks)))
        self.out = nn.Conv2d(channels, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.silu(self.stem(inputs))
        hidden = torch.nn.functional.silu(self.blocks(hidden))
        return inputs + self.out(hidden)


class BidirectionalImageRecovery(nn.Module):
    """Palm-to-vein and vein-to-palm residual image recoverers."""

    def __init__(self, channels: int = 32, blocks: int = 3) -> None:
        super().__init__()
        self.v2p = ResidualImageAdapter(channels=channels, blocks=blocks)
        self.p2v = ResidualImageAdapter(channels=channels, blocks=blocks)

    def forward(
        self,
        vein_as_palm: torch.Tensor,
        palm_as_vein: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "generated_palm": self.v2p(vein_as_palm),
            "generated_vein": self.p2v(palm_as_vein),
        }


def load_image_recovery_state(model: BidirectionalImageRecovery, checkpoint: dict) -> None:
    """Load either a versioned checkpoint or the early two-key experiment format."""
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        return
    if "v2p" not in checkpoint or "p2v" not in checkpoint:
        raise ValueError("Recovery checkpoint must contain both v2p and p2v weights")
    model.v2p.load_state_dict(checkpoint["v2p"])
    model.p2v.load_state_dict(checkpoint["p2v"])
