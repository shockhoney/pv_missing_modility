"""Image-level MMANet reproduction for two-modality biometric recognition.

This module retains the official three-network design: a complete-modality
teacher, a modality-dropout deployment network, and an auxiliary regularizer.
MAD is applied to layer-4 spatial relations and MAR is activated only after
the five-epoch weak-combination mining stage.
"""

from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock

from models.comparisons.mmanet import _presence_mask


ARCHITECTURE_VERSION: Final = "mmanet_image_teacher_deployment_mar_v3"
OFFICIAL_COMMIT: Final = "c3306896c0f2e569e27755aa6100399966d29fb5"


class SELayer(nn.Module):
    """The SE block used by the released MMANet classification backbone."""

    def __init__(self, channels: int = 128, reduction: int = 16) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = feature.shape
        weights = self.fc(self.pool(feature).reshape(batch, channels))
        return feature * weights.reshape(batch, channels, 1, 1)


class ModalityStem(nn.Module):
    """Official ResNet18 modality-specific trunk through layer2."""

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        # The official SURF/CeFA ResNet18 uses a 3x3 rather than torchvision's
        # 7x7 input convolution, followed by an SE block after layer2.
        backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=2, padding=1, bias=False
        )
        nn.init.kaiming_normal_(
            backbone.conv1.weight, mode="fan_out", nonlinearity="relu"
        )
        self.network = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
        )
        self.se_layer = SELayer(128, reduction=16)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.se_layer(self.network(images))


class FusionTail(nn.Module):
    """ResNet18 shared layer3/layer4 adapted to concatenated modality maps."""

    def __init__(self, in_channels: int = 256, num_classes: int = 432) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        # Official ``layer3_new`` is constructed with ``inplanes`` equal to
        # the concatenated stem width. An extra 1x1 bottleneck changes both
        # the tensor layout and the distillation target.
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(in_channels, 256, stride=2, downsample=downsample),
            BasicBlock(256, 256),
        )
        self.layer4 = backbone.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)
        for module in self.layer3.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, fused: torch.Tensor) -> dict[str, torch.Tensor]:
        layer3 = self.layer3(fused)
        layer4 = self.layer4(layer3)
        embedding = self.pool(layer4).flatten(1)
        # Classification in the official network uses the raw pooled feature.
        # L2 normalization is only our gallery/probe adaptation.
        logits = self.classifier(embedding)
        representation = F.normalize(embedding, dim=1)
        return {
            "layer3": layer3,
            "layer4": layer4,
            "representation": representation,
            "embedding": embedding,
            "logits": logits,
        }


class MultimodalNetwork(nn.Module):
    """Teacher or deployment network following official early concatenation."""

    def __init__(self, num_classes: int = 432) -> None:
        super().__init__()
        self.palm_stem = ModalityStem()
        self.vein_stem = ModalityStem()
        self.tail = FusionTail(256, num_classes)

    @staticmethod
    def _validate(images: torch.Tensor, name: str) -> None:
        if images.ndim != 4 or images.size(1) != 3:
            raise ValueError(f"{name} must have shape [B, 3, H, W]")
        if not images.is_floating_point():
            raise TypeError(f"{name} must be floating point")

    def forward(
        self,
        palm: torch.Tensor,
        vein: torch.Tensor,
        palm_present: bool | torch.Tensor = True,
        vein_present: bool | torch.Tensor = True,
    ) -> dict[str, torch.Tensor]:
        self._validate(palm, "palm")
        self._validate(vein, "vein")
        if palm.shape != vein.shape or palm.device != vein.device:
            raise ValueError("Palm and vein inputs must share shape and device")
        batch = palm.size(0)
        palm_mask = _presence_mask(
            palm_present, batch_size=batch, device=palm.device, name="palm_present"
        )
        vein_mask = _presence_mask(
            vein_present, batch_size=batch, device=palm.device, name="vein_present"
        )
        if torch.any(~palm_mask & ~vein_mask):
            raise ValueError("Every sample must contain at least one modality")
        palm_feature = self.palm_stem(palm)
        vein_feature = self.vein_stem(vein)
        palm_feature = palm_feature * palm_mask[:, None, None, None]
        vein_feature = vein_feature * vein_mask[:, None, None, None]
        fused_stem = torch.cat((palm_feature, vein_feature), dim=1)
        output = self.tail(fused_stem)
        output.update(
            {
                "fused_stem": fused_stem,
                "palm_feature": palm_feature,
                "vein_feature": vein_feature,
                "palm_present": palm_mask,
                "vein_present": vein_mask,
            }
        )
        return output


class MMANetImageModel(nn.Module):
    """Full teacher/deployment/regularization MMANet training system."""

    PALM_ONLY: Final = 0
    VEIN_ONLY: Final = 1
    COMPLETE: Final = 2

    def __init__(
        self,
        num_classes: int = 432,
        mad_weight: float = 30.0,
        mar_weight: float = 0.5,
        warmup_epochs: int = 5,
    ) -> None:
        super().__init__()
        if num_classes <= 1 or warmup_epochs < 1:
            raise ValueError("Invalid MMANet configuration")
        self.num_classes = int(num_classes)
        self.representation_dim = 512
        self.mad_weight = float(mad_weight)
        self.mar_weight = float(mar_weight)
        self.warmup_epochs = int(warmup_epochs)
        self.teacher = MultimodalNetwork(self.num_classes)
        self.deployment = MultimodalNetwork(self.num_classes)
        self.regularizer = FusionTail(256, self.num_classes)
        self.register_buffer(
            "mar_distance_memory",
            torch.zeros(self.warmup_epochs, 2, dtype=torch.float64),
        )
        self.register_buffer("mar_observed", torch.zeros(self.warmup_epochs, dtype=torch.bool))
        self.register_buffer("weak_combination", torch.tensor(-1, dtype=torch.long))

    def freeze_teacher(self) -> None:
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

    def teacher_loss(
        self, palm: torch.Tensor, vein: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        output = self.teacher(palm, vein, True, True)
        loss = F.cross_entropy(output["logits"], labels.long())
        return {"total": loss, **output}

    @staticmethod
    def margin_aware_distillation(
        student_map: torch.Tensor,
        teacher_map: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Official repository MAD: entropy-weighted Gram-matrix MSE."""

        # Layer-4 contains tens of thousands of values per item. FP16 Gram
        # accumulation can overflow although the feature maps are finite, so
        # retain the official equation and gradient but perform its reduction
        # in FP32 while the convolutional forward remains autocast.
        with torch.autocast(device_type=student_map.device.type, enabled=False):
            student = student_map.float().flatten(1)
            teacher = teacher_map.detach().float().flatten(1)
            student_relation = F.normalize(student @ student.t(), p=2, dim=1)
            teacher_relation = F.normalize(teacher @ teacher.t(), p=2, dim=1)
            discrepancy = F.mse_loss(
                student_relation, teacher_relation, reduction="none"
            )
            probability = F.softmax(teacher_logits.detach().float(), dim=1).clamp_min(1e-12)
            entropy = -(probability * probability.log()).sum(dim=1)
            weights = entropy / entropy.sum().clamp_min(1e-12)
            return torch.sum(discrepancy * weights)

    @staticmethod
    def _histogram_kl(singleton: torch.Tensor, complete: torch.Tensor) -> torch.Tensor:
        return F.kl_div(
            F.log_softmax(complete.unsqueeze(0), dim=1),
            F.softmax(singleton.unsqueeze(0), dim=1),
            reduction="batchmean",
        )

    @torch.no_grad()
    def record_mar_epoch(
        self,
        epoch: int,
        palm_histogram: torch.Tensor,
        vein_histogram: torch.Tensor,
        complete_histogram: torch.Tensor,
    ) -> None:
        """Record official train-set prediction distributions for Eq. 9-13."""

        if not 1 <= int(epoch) <= self.warmup_epochs:
            raise ValueError("MAR observations are restricted to warm-up epochs")
        histograms = []
        for value in (palm_histogram, vein_histogram, complete_histogram):
            value = value.to(device=self.mar_distance_memory.device, dtype=torch.float64)
            if value.ndim != 1 or value.numel() != self.num_classes or value.sum() <= 0:
                raise ValueError("Each MAR histogram must contain class prediction counts")
            histograms.append(value / value.sum())
        row = int(epoch) - 1
        self.mar_distance_memory[row, 0] = self._histogram_kl(histograms[0], histograms[2])
        self.mar_distance_memory[row, 1] = self._histogram_kl(histograms[1], histograms[2])
        self.mar_observed[row] = True
        if int(epoch) == self.warmup_epochs:
            if not bool(self.mar_observed.all()):
                raise RuntimeError("All warm-up epochs must be observed before locking MAR")
            mean_distance = self.mar_distance_memory.mean(dim=0)
            self.weak_combination.copy_(mean_distance.argmax().long())

    @property
    def weak_modality(self) -> str | None:
        return {self.PALM_ONLY: "palm", self.VEIN_ONLY: "vein"}.get(
            int(self.weak_combination.item())
        )

    def deployment_loss(
        self,
        palm: torch.Tensor,
        vein: torch.Tensor,
        labels: torch.Tensor,
        palm_present: torch.Tensor,
        vein_present: torch.Tensor,
        epoch: int,
    ) -> dict[str, torch.Tensor]:
        """Official Eq. 1 for a mini-batch of sampled modality combinations."""

        labels = labels.long()
        deployment = self.deployment(
            palm, vein, palm_present=palm_present, vein_present=vein_present
        )
        task = F.cross_entropy(deployment["logits"], labels)
        # Released classification code uses task-only optimization for the
        # N-epoch weak-combination mining warm-up, then enables MAD and MAR.
        auxiliary_active = int(epoch) > self.warmup_epochs
        mad = task.new_zeros(())
        if auxiliary_active:
            with torch.no_grad():
                teacher = self.teacher(palm, vein, True, True)
            mad = self.margin_aware_distillation(
                deployment["layer4"], teacher["layer4"], teacher["logits"]
            )
        mar = task.new_zeros(())
        mar_active = auxiliary_active and self.weak_modality is not None
        if mar_active:
            if int(self.weak_combination.item()) == self.PALM_ONLY:
                weak_mask = palm_present & ~vein_present
            else:
                weak_mask = ~palm_present & vein_present
            # The released code evaluates the auxiliary tail on the complete
            # batch, masks per-sample CE afterwards, and divides by the full
            # batch (so its BN statistics see every sampled combination).
            regularization = self.regularizer(deployment["fused_stem"])
            per_sample = F.cross_entropy(
                regularization["logits"], labels, reduction="none"
            )
            mar = (per_sample * weak_mask.to(per_sample.dtype)).sum() / labels.numel()
        total = task + self.mad_weight * mad + self.mar_weight * mar
        return {
            "task": task,
            "mad": mad,
            "mar": mar,
            "total": total,
            "mar_active": torch.tensor(mar_active, device=total.device),
            "logits": deployment["logits"],
            "representation": deployment["representation"],
        }

    def encode(
        self,
        palm: torch.Tensor,
        vein: torch.Tensor,
        palm_present: bool | torch.Tensor = True,
        vein_present: bool | torch.Tensor = True,
    ) -> dict[str, torch.Tensor]:
        return self.deployment(palm, vein, palm_present, vein_present)

    def representation(
        self,
        palm: torch.Tensor,
        vein: torch.Tensor,
        palm_present: bool | torch.Tensor = True,
        vein_present: bool | torch.Tensor = True,
    ) -> torch.Tensor:
        return self.encode(palm, vein, palm_present, vein_present)["representation"]


__all__ = [
    "ARCHITECTURE_VERSION",
    "FusionTail",
    "MMANetImageModel",
    "ModalityStem",
    "MultimodalNetwork",
    "OFFICIAL_COMMIT",
    "SELayer",
]
