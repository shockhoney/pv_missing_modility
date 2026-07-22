"""Gallery-conditioned selective prototype recovery for missing modalities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


ARCHITECTURE_VERSION = "gallery_conditioned_selective_prototype_recovery_v6"


def _inverse_sqrt(matrix: torch.Tensor, eigen_floor: torch.Tensor) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = eigenvalues.clamp_min(eigen_floor)
    return (eigenvectors * eigenvalues.rsqrt().unsqueeze(0)) @ eigenvectors.t()


class RegularizedSharedIdentityProjector(nn.Module):
    """Fit paired modalities to one canonical identity space using regularized CCA."""

    def __init__(self, dim: int = 256, unit_input: bool = False):
        super().__init__()
        self.dim = dim
        self.unit_input = unit_input
        self.register_buffer("palm_mean", torch.zeros(1, dim))
        self.register_buffer("vein_mean", torch.zeros(1, dim))
        self.register_buffer("palm_projection", torch.eye(dim))
        self.register_buffer("vein_projection", torch.eye(dim))
        self.register_buffer("canonical_correlations", torch.zeros(dim))
        self.register_buffer("fitted", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def fit(self, palm: torch.Tensor, vein: torch.Tensor, eigen_floor: float) -> None:
        if palm.ndim != 2 or vein.ndim != 2 or palm.shape != vein.shape:
            raise ValueError("Regularized CCA requires equal paired [N, D] tensors")
        if palm.size(1) != self.dim or palm.size(0) < 2:
            raise ValueError("Paired feature shape is incompatible with this projector")
        if eigen_floor <= 0:
            raise ValueError("eigen_floor must be positive")
        palm = palm.detach().double()
        vein = vein.detach().double()
        if self.unit_input:
            palm = F.normalize(palm, dim=1)
            vein = F.normalize(vein, dim=1)
        palm_mean = palm.mean(dim=0, keepdim=True)
        vein_mean = vein.mean(dim=0, keepdim=True)
        palm_centered = palm - palm_mean
        vein_centered = vein - vein_mean
        denominator = palm.size(0) - 1
        palm_covariance = palm_centered.t() @ palm_centered / denominator
        vein_covariance = vein_centered.t() @ vein_centered / denominator
        cross_covariance = palm_centered.t() @ vein_centered / denominator
        palm_floor = palm_covariance.diag().mean() * eigen_floor
        vein_floor = vein_covariance.diag().mean() * eigen_floor
        palm_whitener = _inverse_sqrt(palm_covariance, palm_floor)
        vein_whitener = _inverse_sqrt(vein_covariance, vein_floor)
        left, correlations, right_t = torch.linalg.svd(
            palm_whitener @ cross_covariance @ vein_whitener
        )
        self.palm_mean.copy_(palm_mean.float())
        self.vein_mean.copy_(vein_mean.float())
        self.palm_projection.copy_((palm_whitener @ left).float())
        self.vein_projection.copy_((vein_whitener @ right_t.t()).float())
        self.canonical_correlations.copy_(correlations.float())
        self.fitted.fill_(True)

    def transform(self, features: torch.Tensor, modality: str, dimensions: int) -> torch.Tensor:
        if not bool(self.fitted.item()):
            raise RuntimeError("Shared identity projector has not been fitted")
        if not 1 <= dimensions <= self.dim:
            raise ValueError(f"dimensions must be in [1, {self.dim}]")
        if self.unit_input:
            features = F.normalize(features, dim=1)
        if modality == "palm":
            mean, projection = self.palm_mean, self.palm_projection
        elif modality == "vein":
            mean, projection = self.vein_mean, self.vein_projection
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        return (features - mean) @ projection[:, :dimensions]


def _zero_linear(linear: nn.Linear) -> None:
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class TrainableSharedProjector(nn.Module):
    """CCA-initialized projection with a conservative trainable residual."""

    def __init__(
        self,
        input_dim: int = 256,
        shared_dim: int = 192,
        dropout: float = 0.1,
        unit_input: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.shared_dim = shared_dim
        self.unit_input = unit_input
        self.base = nn.Linear(input_dim, shared_dim)
        self.refiner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, shared_dim),
        )
        _zero_linear(self.refiner[-1])
        self.register_buffer("initial_weight", torch.zeros(shared_dim, input_dim))
        self.register_buffer("initial_bias", torch.zeros(shared_dim))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def initialize_from_cca(
        self,
        projector: RegularizedSharedIdentityProjector,
        modality: str,
    ) -> None:
        if not bool(projector.fitted.item()):
            raise RuntimeError("CCA projector must be fitted before initialization")
        if modality == "palm":
            mean = projector.palm_mean
            projection = projector.palm_projection
        elif modality == "vein":
            mean = projector.vein_mean
            projection = projector.vein_projection
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        projection = projection[:, : self.shared_dim]
        weight = projection.t().to(self.base.weight)
        bias = (-(mean.to(weight) @ projection.to(weight))).squeeze(0)
        self.base.weight.copy_(weight)
        self.base.bias.copy_(bias)
        self.initial_weight.copy_(weight)
        self.initial_bias.copy_(bias)
        self.initialized.fill_(True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.unit_input:
            features = F.normalize(features, dim=1)
        shared = self.base(features)
        return F.normalize(shared + self.refiner(shared), dim=1)

    def anchor_loss(self) -> torch.Tensor:
        if not bool(self.initialized.item()):
            raise RuntimeError("Trainable shared projector has not been initialized")
        return F.mse_loss(self.base.weight, self.initial_weight) + F.mse_loss(
            self.base.bias, self.initial_bias
        )


def _gallery_templates(
    features: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = labels.to(device=features.device, dtype=torch.long)
    identities = labels.unique(sorted=True)
    templates = torch.stack(
        [features[labels == identity].mean(dim=0) for identity in identities]
    )
    return F.normalize(templates, dim=1), identities


class TrainableSharedFeatureRecovery(nn.Module):
    """Recover a target-modality prototype from the enrolled closed-set gallery."""

    BRANCHES = ("available", "shared_same", "shared_cross", "recovered")

    def __init__(
        self,
        input_dim: int = 256,
        shared_dim: int = 192,
        dropout: float = 0.1,
        unit_input: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.shared_dim = shared_dim
        projector_kwargs = dict(
            input_dim=input_dim,
            shared_dim=shared_dim,
            dropout=dropout,
            unit_input=unit_input,
        )
        self.palm_projector = TrainableSharedProjector(**projector_kwargs)
        self.vein_projector = TrainableSharedProjector(**projector_kwargs)
        self.register_buffer("palm_refiner_enabled", torch.tensor(True, dtype=torch.bool))
        self.register_buffer("vein_refiner_enabled", torch.tensor(True, dtype=torch.bool))
        self.register_buffer("p2v_base_weights", torch.tensor([0.0, 1.0, 0.0]))
        self.register_buffer("v2p_base_weights", torch.tensor([0.0, 1.0, 0.0]))
        self.register_buffer("p2v_temperature", torch.tensor(0.05))
        self.register_buffer("v2p_temperature", torch.tensor(0.01))
        self.register_buffer("p2v_recovery_alpha", torch.tensor(0.29))
        self.register_buffer("v2p_recovery_alpha", torch.tensor(0.10))
        self.register_buffer("p2v_margin_floor", torch.tensor(0.0))
        self.register_buffer("v2p_margin_floor", torch.tensor(0.0))
        self.register_buffer("p2v_margin_ceiling", torch.tensor(0.2))
        self.register_buffer("v2p_margin_ceiling", torch.tensor(0.2))
        self.register_buffer("p2v_margin_slope", torch.tensor(0.01))
        self.register_buffer("v2p_margin_slope", torch.tensor(0.01))

    @torch.no_grad()
    def initialize_from_cca(
        self,
        projector: RegularizedSharedIdentityProjector,
    ) -> None:
        self.palm_projector.initialize_from_cca(projector, "palm")
        self.vein_projector.initialize_from_cca(projector, "vein")

    def project(self, features: torch.Tensor, modality: str) -> torch.Tensor:
        if modality == "palm":
            return self.palm_projector(features)
        if modality == "vein":
            return self.vein_projector(features)
        raise ValueError(f"Unsupported modality: {modality}")

    @torch.no_grad()
    def set_calibration(
        self,
        available_modality: str,
        base_weights: torch.Tensor,
        temperature: float,
        recovery_alpha: float,
        margin_floor: float = 0.0,
        margin_ceiling: float = 0.2,
        margin_slope: float = 0.01,
    ) -> None:
        if base_weights.shape != (3,) or torch.any(base_weights < 0):
            raise ValueError("base_weights must be a non-negative length-3 tensor")
        if not torch.isclose(base_weights.sum(), base_weights.new_tensor(1.0)):
            raise ValueError("base_weights must sum to one")
        if temperature <= 0 or not 0 <= recovery_alpha <= 1:
            raise ValueError("temperature and recovery_alpha are invalid")
        if margin_floor < 0 or margin_ceiling <= margin_floor or margin_slope <= 0:
            raise ValueError("margin band gate settings are invalid")
        if available_modality == "palm":
            weights, temp, alpha, floor, ceiling, slope = (
                self.p2v_base_weights,
                self.p2v_temperature,
                self.p2v_recovery_alpha,
                self.p2v_margin_floor,
                self.p2v_margin_ceiling,
                self.p2v_margin_slope,
            )
        elif available_modality == "vein":
            weights, temp, alpha, floor, ceiling, slope = (
                self.v2p_base_weights,
                self.v2p_temperature,
                self.v2p_recovery_alpha,
                self.v2p_margin_floor,
                self.v2p_margin_ceiling,
                self.v2p_margin_slope,
            )
        else:
            raise ValueError(f"Unsupported modality: {available_modality}")
        weights.copy_(base_weights.to(weights))
        temp.fill_(temperature)
        alpha.fill_(recovery_alpha)
        floor.fill_(margin_floor)
        ceiling.fill_(margin_ceiling)
        slope.fill_(margin_slope)

    def calibration(self, available_modality: str) -> dict[str, torch.Tensor]:
        if available_modality == "palm":
            return {
                "base_weights": self.p2v_base_weights,
                "temperature": self.p2v_temperature,
                "recovery_alpha": self.p2v_recovery_alpha,
                "margin_floor": self.p2v_margin_floor,
                "margin_ceiling": self.p2v_margin_ceiling,
                "margin_slope": self.p2v_margin_slope,
            }
        if available_modality == "vein":
            return {
                "base_weights": self.v2p_base_weights,
                "temperature": self.v2p_temperature,
                "recovery_alpha": self.v2p_recovery_alpha,
                "margin_floor": self.v2p_margin_floor,
                "margin_ceiling": self.v2p_margin_ceiling,
                "margin_slope": self.v2p_margin_slope,
            }
        raise ValueError(f"Unsupported modality: {available_modality}")

    def recover_with_gallery(
        self,
        available_features: torch.Tensor,
        available_modality: str,
        gallery_available: torch.Tensor,
        gallery_target: torch.Tensor,
        gallery_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor | str]:
        target_modality = "vein" if available_modality == "palm" else "palm"
        if available_modality not in ("palm", "vein"):
            raise ValueError(f"Unsupported modality: {available_modality}")
        available_templates, labels = _gallery_templates(
            gallery_available, gallery_labels
        )
        target_templates, target_labels = _gallery_templates(
            gallery_target, gallery_labels
        )
        if not torch.equal(labels, target_labels):
            raise ValueError("Gallery identity order differs across modalities")
        available_shared = self.project(available_features, available_modality)
        same_templates, same_labels = _gallery_templates(
            self.project(gallery_available, available_modality), gallery_labels
        )
        cross_templates, cross_labels = _gallery_templates(
            self.project(gallery_target, target_modality), gallery_labels
        )
        if not (
            torch.equal(labels, same_labels) and torch.equal(labels, cross_labels)
        ):
            raise ValueError("Gallery identity order differs across score branches")
        branches = torch.stack(
            [
                F.normalize(available_features, dim=1) @ available_templates.t(),
                available_shared @ same_templates.t(),
                available_shared @ cross_templates.t(),
            ],
            dim=2,
        )
        calibration = self.calibration(available_modality)
        base_scores = (
            branches * calibration["base_weights"].unsqueeze(0).unsqueeze(0)
        ).sum(dim=2)
        posterior = torch.softmax(
            base_scores / calibration["temperature"].clamp_min(1e-6), dim=1
        )
        recovered_feature = F.normalize(posterior @ target_templates, dim=1)
        recovered_scores = recovered_feature @ target_templates.t()
        posterior_confidence = posterior.max(dim=1).values
        top_two = base_scores.topk(k=2, dim=1).values
        recovery_reliability = top_two[:, 0] - top_two[:, 1]
        slope = calibration["margin_slope"].clamp_min(1e-6)
        above_floor = torch.sigmoid(
            (recovery_reliability - calibration["margin_floor"]) / slope
        )
        below_ceiling = torch.sigmoid(
            (calibration["margin_ceiling"] - recovery_reliability) / slope
        )
        recovery_gate = calibration["recovery_alpha"] * above_floor * below_ceiling
        fused_scores = (1.0 - recovery_gate.unsqueeze(1)) * base_scores + (
            recovery_gate.unsqueeze(1) * recovered_scores
        )
        return {
            "shared": available_shared,
            "mean": recovered_feature,
            "posterior": posterior,
            "posterior_confidence": posterior_confidence,
            "recovery_reliability": recovery_reliability,
            "recovery_gate": recovery_gate,
            "base_weights": calibration["base_weights"].expand(
                available_features.size(0), -1
            ),
            "base_branch_scores": branches,
            "base_scores": base_scores,
            "recovered_scores": recovered_scores,
            "fused_scores": fused_scores,
            "candidate_labels": labels,
            "target_modality": target_modality,
        }

    def anchor_loss(self) -> torch.Tensor:
        return self.palm_projector.anchor_loss() + self.vein_projector.anchor_loss()
