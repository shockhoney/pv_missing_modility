"""Distribution-aligned recovery of cross-modally shared identity features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


ARCHITECTURE_VERSION = "regularized_shared_identity_recovery_v1"


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


TRAINABLE_ARCHITECTURE_VERSION = "trainable_probabilistic_shared_recovery_v2"


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


class ProbabilisticFeatureRecoverer(nn.Module):
    """Predict a target-modality conditional mean and diagonal uncertainty."""

    def __init__(
        self,
        shared_dim: int = 192,
        target_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        min_logvar: float = -6.0,
        max_logvar: float = 2.0,
    ):
        super().__init__()
        self.shared_dim = shared_dim
        self.target_dim = target_dim
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.mean_base = nn.Linear(shared_dim, target_dim)
        self.mean_residual = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_dim),
        )
        _zero_linear(self.mean_residual[-1])
        self.logvar_residual = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_dim),
        )
        _zero_linear(self.logvar_residual[-1])
        self.logvar_bias = nn.Parameter(torch.zeros(target_dim))
        self.register_buffer("initial_mean_weight", torch.zeros(target_dim, shared_dim))
        self.register_buffer("initial_mean_bias", torch.zeros(target_dim))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def initialize_ridge(
        self,
        shared: torch.Tensor,
        target: torch.Tensor,
        ridge: float = 1e-3,
    ) -> None:
        if shared.ndim != 2 or target.ndim != 2:
            raise ValueError("Ridge initialization requires two rank-2 tensors")
        if shared.size(0) != target.size(0):
            raise ValueError("Ridge initialization requires paired samples")
        if shared.size(1) != self.shared_dim or target.size(1) != self.target_dim:
            raise ValueError("Ridge initialization dimensions are incompatible")
        shared64 = shared.detach().double()
        target64 = target.detach().double()
        ones = torch.ones(shared64.size(0), 1, dtype=shared64.dtype, device=shared64.device)
        design = torch.cat([shared64, ones], dim=1)
        regularizer = torch.eye(design.size(1), dtype=design.dtype, device=design.device)
        regularizer[-1, -1] = 0.0
        solution = torch.linalg.solve(
            design.t() @ design + ridge * regularizer,
            design.t() @ target64,
        )
        weight = solution[:-1].t().float().to(self.mean_base.weight)
        bias = solution[-1].float().to(self.mean_base.bias)
        self.mean_base.weight.copy_(weight)
        self.mean_base.bias.copy_(bias)
        self.initial_mean_weight.copy_(weight)
        self.initial_mean_bias.copy_(bias)
        prediction = F.linear(shared.to(weight), weight, bias)
        residual_variance = (target.to(prediction) - prediction).square().mean(dim=0)
        self.logvar_bias.copy_(residual_variance.clamp_min(1e-6).log())
        self.initialized.fill_(True)

    def forward(self, shared: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_base(shared) + self.mean_residual(shared)
        logvar = self.logvar_bias + self.logvar_residual(shared)
        return mean, logvar.clamp(self.min_logvar, self.max_logvar)

    def anchor_loss(self) -> torch.Tensor:
        if not bool(self.initialized.item()):
            raise RuntimeError("Feature recoverer has not been initialized")
        return F.mse_loss(
            self.mean_base.weight, self.initial_mean_weight
        ) + F.mse_loss(self.mean_base.bias, self.initial_mean_bias)


def _reliability_inputs(
    shared: torch.Tensor,
    recovered_mean: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    statistics = torch.stack(
        [
            logvar.mean(dim=1),
            logvar.std(dim=1, unbiased=False),
            recovered_mean.norm(dim=1) / recovered_mean.size(1) ** 0.5,
        ],
        dim=1,
    )
    return torch.cat(
        [shared, F.normalize(recovered_mean, dim=1), statistics],
        dim=1,
    )


class ReliabilityHead(nn.Module):
    def __init__(
        self,
        shared_dim: int = 192,
        target_dim: int = 256,
        hidden_dim: int = 128,
    ):
        super().__init__()
        input_dim = shared_dim + target_dim + 3
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 1.0)

    def forward(
        self,
        shared: torch.Tensor,
        recovered_mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.net(_reliability_inputs(shared, recovered_mean, logvar))
        ).squeeze(1)


class DynamicScoreGate(nn.Module):
    """Produce candidate-independent branch weights for one missing probe."""

    NUM_BRANCHES = 4

    def __init__(
        self,
        shared_dim: int = 192,
        target_dim: int = 256,
        hidden_dim: int = 128,
    ):
        super().__init__()
        input_dim = shared_dim + target_dim + 3
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.NUM_BRANCHES),
        )
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([0.0, 1.0, -1.0, -2.0]))

    def forward(
        self,
        shared: torch.Tensor,
        recovered_mean: torch.Tensor,
        logvar: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.net(_reliability_inputs(shared, recovered_mean, logvar))
        weights = logits.softmax(dim=1)
        reliability_scale = torch.cat(
            [torch.ones_like(weights[:, :3]), reliability.unsqueeze(1)], dim=1
        )
        scaled = weights * reliability_scale
        return scaled / scaled.sum(dim=1, keepdim=True).clamp_min(1e-8)


class TrainableSharedFeatureRecovery(nn.Module):
    """Bidirectional probabilistic feature recovery with dynamic score gating."""

    BRANCHES = ("available", "shared_same", "shared_cross", "recovered")

    def __init__(
        self,
        input_dim: int = 256,
        shared_dim: int = 192,
        hidden_dim: int = 256,
        gate_hidden_dim: int = 128,
        dropout: float = 0.1,
        unit_input: bool = False,
    ):
        super().__init__()
        projector_kwargs = dict(
            input_dim=input_dim,
            shared_dim=shared_dim,
            dropout=dropout,
            unit_input=unit_input,
        )
        self.input_dim = input_dim
        self.shared_dim = shared_dim
        self.register_buffer("palm_refiner_enabled", torch.tensor(True, dtype=torch.bool))
        self.register_buffer("vein_refiner_enabled", torch.tensor(True, dtype=torch.bool))
        self.palm_projector = TrainableSharedProjector(**projector_kwargs)
        self.vein_projector = TrainableSharedProjector(**projector_kwargs)
        recovery_kwargs = dict(
            shared_dim=shared_dim,
            target_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.p2v = ProbabilisticFeatureRecoverer(**recovery_kwargs)
        self.v2p = ProbabilisticFeatureRecoverer(**recovery_kwargs)
        head_kwargs = dict(
            shared_dim=shared_dim,
            target_dim=input_dim,
            hidden_dim=gate_hidden_dim,
        )
        self.p2v_reliability = ReliabilityHead(**head_kwargs)
        self.v2p_reliability = ReliabilityHead(**head_kwargs)
        self.p2v_gate = DynamicScoreGate(**head_kwargs)
        self.v2p_gate = DynamicScoreGate(**head_kwargs)

    @torch.no_grad()
    def initialize_from_cca(
        self,
        projector: RegularizedSharedIdentityProjector,
        palm_features: torch.Tensor,
        vein_features: torch.Tensor,
        ridge: float = 1e-3,
        reliability_temperature: float = 0.1,
    ) -> None:
        if reliability_temperature <= 0:
            raise ValueError("reliability_temperature must be positive")
        self.palm_projector.initialize_from_cca(projector, "palm")
        self.vein_projector.initialize_from_cca(projector, "vein")
        palm_shared = self.palm_projector(palm_features)
        vein_shared = self.vein_projector(vein_features)
        self.p2v.initialize_ridge(palm_shared, vein_features, ridge)
        self.v2p.initialize_ridge(vein_shared, palm_features, ridge)
        p2v_mean = self.p2v.mean_base(palm_shared)
        v2p_mean = self.v2p.mean_base(vein_shared)
        reliability_pairs = (
            (self.p2v_reliability, p2v_mean, vein_features),
            (self.v2p_reliability, v2p_mean, palm_features),
        )
        for head, recovered, target in reliability_pairs:
            cosine_error = 1.0 - F.cosine_similarity(recovered, target, dim=1)
            initial_reliability = torch.exp(
                -cosine_error / reliability_temperature
            ).mean().clamp(1e-4, 1.0 - 1e-4)
            head.net[-1].bias.fill_(torch.logit(initial_reliability).item())
        # Start from the reliable CCA branches.  The learned recovery branch
        # must earn weight through its sample-level reliability estimate.
        self.p2v_gate.net[-1].bias.copy_(
            torch.tensor([-4.0, 1.1, 0.0, -2.5], device=palm_features.device)
        )
        self.v2p_gate.net[-1].bias.copy_(
            torch.tensor([-4.0, 1.5, -4.0, -2.5], device=palm_features.device)
        )

    def project(self, features: torch.Tensor, modality: str) -> torch.Tensor:
        if modality == "palm":
            return self.palm_projector(features)
        if modality == "vein":
            return self.vein_projector(features)
        raise ValueError(f"Unsupported modality: {modality}")

    def recover(
        self,
        available_features: torch.Tensor,
        available_modality: str,
    ) -> dict[str, torch.Tensor]:
        shared = self.project(available_features, available_modality)
        if available_modality == "palm":
            mean, logvar = self.p2v(shared)
            reliability = self.p2v_reliability(shared, mean, logvar)
            weights = self.p2v_gate(shared, mean, logvar, reliability)
            target_modality = "vein"
        elif available_modality == "vein":
            mean, logvar = self.v2p(shared)
            reliability = self.v2p_reliability(shared, mean, logvar)
            weights = self.v2p_gate(shared, mean, logvar, reliability)
            target_modality = "palm"
        else:
            raise ValueError(f"Unsupported modality: {available_modality}")
        return {
            "shared": shared,
            "mean": mean,
            "logvar": logvar,
            "reliability": reliability,
            "weights": weights,
            "target_modality": target_modality,
        }

    def anchor_loss(self) -> torch.Tensor:
        return (
            self.palm_projector.anchor_loss()
            + self.vein_projector.anchor_loss()
            + self.p2v.anchor_loss()
            + self.v2p.anchor_loss()
        )
