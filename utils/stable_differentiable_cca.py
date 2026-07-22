"""Cholesky-based differentiable CCA for rank-deficient mini-batches.

Repeated ridge eigenvalues make the eigenvector derivative of an exact
mini-batch whitening unstable when batch size is below feature dimension.  The
Cholesky form represents the same regularized whitening without differentiating
through an arbitrary basis of the null space.  Training maximizes the sum of
squared canonical correlations; singular values are computed only for detached
diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CCAStatistics:
    correlation: torch.Tensor
    minimum_palm_variance: torch.Tensor
    minimum_vein_variance: torch.Tensor
    effective_rank_palm: torch.Tensor
    effective_rank_vein: torch.Tensor


def _validate_pair(left: torch.Tensor, right: torch.Tensor) -> None:
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("CCA inputs must be equal-shaped [batch, dimension] tensors")
    if left.size(0) < 2 or not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("CCA inputs require at least two finite samples")


def _whitened_cross(
    left: torch.Tensor, right: torch.Tensor, ridge: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_pair(left, right)
    x = left.float() - left.float().mean(dim=0, keepdim=True)
    y = right.float() - right.float().mean(dim=0, keepdim=True)
    denominator = x.size(0) - 1
    identity = torch.eye(x.size(1), device=x.device, dtype=x.dtype)
    cxx = x.transpose(0, 1) @ x / denominator + float(ridge) * identity
    cyy = y.transpose(0, 1) @ y / denominator + float(ridge) * identity
    cxy = x.transpose(0, 1) @ y / denominator
    lx = torch.linalg.cholesky(cxx)
    ly = torch.linalg.cholesky(cyy)
    left_whitened = torch.linalg.solve_triangular(lx, cxy, upper=False)
    whitened = torch.linalg.solve_triangular(
        ly, left_whitened.transpose(0, 1), upper=False
    ).transpose(0, 1)
    return whitened, cxx, cyy


def squared_correlation(
    left: torch.Tensor, right: torch.Tensor, ridge: float = 1e-3
) -> tuple[torch.Tensor, CCAStatistics]:
    whitened, cxx, cyy = _whitened_cross(left, right, ridge)
    squared = whitened.square().sum()
    with torch.no_grad():
        singular = torch.linalg.svdvals(whitened)
        eigen_x = torch.linalg.eigvalsh(cxx).clamp_min(1e-12)
        eigen_y = torch.linalg.eigvalsh(cyy).clamp_min(1e-12)

        def effective_rank(values):
            probabilities = values / values.sum()
            return (-(probabilities * probabilities.clamp_min(1e-30).log()).sum()).exp()

        statistics = CCAStatistics(
            correlation=singular.sum().to(left.dtype),
            minimum_palm_variance=eigen_x.min().to(left.dtype),
            minimum_vein_variance=eigen_y.min().to(left.dtype),
            effective_rank_palm=effective_rank(eigen_x).to(left.dtype),
            effective_rank_vein=effective_rank(eigen_y).to(left.dtype),
        )
    return squared.to(left.dtype), statistics


def deep_cca_loss(
    palm: torch.Tensor,
    vein: torch.Tensor,
    ridge: float = 1e-3,
    eigen_floor: float = 1e-6,
    topk: int | None = None,
) -> tuple[torch.Tensor, CCAStatistics]:
    del eigen_floor
    squared, statistics = squared_correlation(palm, vein, ridge=ridge)
    maximum_rank = min(palm.size(0) - 1, palm.size(1))
    normalizer = maximum_rank if topk is None else min(int(topk), maximum_rank)
    return -squared / max(1, normalizer), statistics


def nr_dcca_loss(
    projector,
    features: torch.Tensor,
    noise_scale: float = 1.0,
    ridge: float = 1e-3,
    topk: int = 32,
) -> torch.Tensor:
    del topk
    noise = torch.randn_like(features) * features.std(dim=0, unbiased=False).clamp_min(1e-3)
    noise = noise * float(noise_scale) + features.mean(dim=0, keepdim=True)
    input_score, _ = squared_correlation(features, noise, ridge=ridge)
    output_features = projector.forward_raw(features)
    output_noise = projector.forward_raw(noise)
    output_score, _ = squared_correlation(output_features, output_noise, ridge=ridge)
    input_rank = min(features.size(0) - 1, features.size(1))
    output_rank = min(output_features.size(0) - 1, output_features.size(1))
    return (output_score / input_rank - (input_score / output_rank).detach()).abs()
