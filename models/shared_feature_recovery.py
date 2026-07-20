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
