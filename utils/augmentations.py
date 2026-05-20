from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UAAAffineAugmenter(nn.Module):
    def __init__(
        self,
        max_translate: float = 0.2,
        max_rotate: float = 0.25,
        min_scale: float = 0.8,
        max_scale: float = 1.2,
    ) -> None:
        super().__init__()
        self.max_translate = max_translate
        self.max_rotate = max_rotate
        self.min_scale = min_scale
        self.max_scale = max_scale

    def project(self, params: torch.Tensor) -> torch.Tensor:
        if params.size(-1) != 4:
            raise ValueError("params must be [..., 4] in [tx, ty, theta, scale] order")
        if not torch.is_floating_point(params):
            params = params.float()
        offset = torch.nan_to_num(params[..., :3], nan=0.0, posinf=0.0, neginf=0.0)
        raw_scale = torch.nan_to_num(params[..., 3], nan=1.0, posinf=self.max_scale, neginf=self.min_scale)
        tx = offset[..., 0].clamp(-self.max_translate, self.max_translate)
        ty = offset[..., 1].clamp(-self.max_translate, self.max_translate)
        theta = offset[..., 2].clamp(-self.max_rotate, self.max_rotate)
        scale = raw_scale.clamp(self.min_scale + 1e-6, self.max_scale - 1e-6)
        return torch.stack([tx, ty, theta, scale], dim=-1)

    def forward(self, images: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        params = self.project(params.to(device=images.device, dtype=images.dtype))
        if params.dim() == 1:
            params = params.unsqueeze(0)
        if params.size(0) == 1 and images.size(0) != 1:
            params = params.expand(images.size(0), -1)
        tx, ty, theta, scale = params.unbind(dim=1)
        cos_t = torch.cos(theta) * scale
        sin_t = torch.sin(theta) * scale
        affine = torch.stack(
            [cos_t, -sin_t, tx, sin_t, cos_t, ty],
            dim=1,
        ).view(-1, 2, 3)
        grid = F.affine_grid(affine, images.size(), align_corners=False)
        return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)


def _init_uaa_params(
    batch_size: int,
    device,
    dtype,
    augmenter: UAAAffineAugmenter,
    prev_params: torch.Tensor | None,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype).expand(batch_size, -1)
    if prev_params is None or prev_params.numel() == 0:
        return base, base
    prev_params = augmenter.project(prev_params.detach().to(device=device, dtype=dtype))
    if prev_params.dim() == 1:
        prev_params = prev_params.unsqueeze(0)
    if prev_params.size(0) != batch_size:
        prev_params = prev_params.mean(dim=0, keepdim=True)
    prev_params = prev_params.expand(batch_size, -1)
    return augmenter.project(beta * prev_params + (1.0 - beta) * base), prev_params


def _classifier_forward(classifier: nn.Module, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    try:
        return classifier(feats, labels)
    except TypeError:
        return classifier(feats)


def optimize_uaa_params(
    encoder: nn.Module,
    classifier: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    augmenter: UAAAffineAugmenter,
    steps: int = 1,
    step_size: float = 0.1,
    beta: float = 0.5,
    prev_params: torch.Tensor | None = None,
    gamma: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = images.size(0)
    selected_count = max(1, min(batch_size, int(round(batch_size * gamma))))
    selected_idx = torch.randperm(batch_size, device=images.device)[:selected_count]
    selected_images = images[selected_idx]
    selected_labels = labels.to(images.device)[selected_idx]
    params, reference = _init_uaa_params(selected_count, images.device, images.dtype, augmenter, prev_params, beta)

    encoder_was_training = encoder.training
    classifier_was_training = classifier.training
    encoder.eval()
    classifier.eval()
    try:
        with torch.enable_grad():
            for _ in range(max(int(steps), 0)):
                params = params.detach().requires_grad_(True)
                logits = _classifier_forward(classifier, encoder(augmenter(selected_images, params)), selected_labels)
                loss = F.cross_entropy(logits, selected_labels)
                if not torch.isfinite(loss):
                    break
                grad = torch.autograd.grad(loss, params, only_inputs=True, allow_unused=True)[0]
                if grad is None:
                    grad = torch.zeros_like(params)
                else:
                    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                params = augmenter.project(params + step_size * grad.sign()).detach()
    finally:
        encoder.train(encoder_was_training)
        classifier.train(classifier_was_training)

    with torch.no_grad():
        augmented = augmenter(selected_images, params)
    return augmented.detach(), selected_labels, params.detach(), selected_idx.detach()


class StarMix:
    def __init__(self, alpha: float = 1.0, threshold: tuple[float, float] = (0.3, 0.7)) -> None:
        self.alpha = alpha
        self.threshold = threshold

    def _sample_lambda(self, device) -> torch.Tensor:
        if self.alpha <= 0:
            return torch.tensor(1.0, device=device)
        concentration = torch.tensor(self.alpha, device=device)
        return torch.distributions.Beta(concentration, concentration).sample()

    def _star_mask(self, height: int, width: int, lam: torch.Tensor, device, dtype) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype).view(height, 1)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype).view(1, width)
        dist2 = x.square() + y.square()
        sigma_center = lam.clamp_min(0.05)
        sigma_outer = (1.0 - lam).clamp_min(0.05)
        center = torch.exp(-dist2 / (2.0 * sigma_center.square()))
        vertical = torch.exp(-x.square() / (2.0 * sigma_outer.square()))
        horizontal = torch.exp(-y.square() / (2.0 * sigma_outer.square()))
        mask = torch.sigmoid((center + vertical + horizontal) / 3.0)
        mask = mask / mask.mean().clamp_min(1e-6) * lam
        return mask.clamp(0.0, 1.0).view(1, 1, height, width)

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        lam: float | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if lam is None:
            lam_t = self._sample_lambda(images.device)
        else:
            lam_t = torch.as_tensor(lam, device=images.device, dtype=images.dtype)
        perm = torch.randperm(images.size(0), device=images.device)
        target_b = labels[perm.to(labels.device)]

        if self.threshold[0] <= float(lam_t) <= self.threshold[1]:
            mask = self._star_mask(images.size(2), images.size(3), lam_t, images.device, images.dtype)
            mixed = mask * images + (1.0 - mask) * images[perm]
            lam_eff = mask.mean().to(dtype=images.dtype)
        else:
            mixed = lam_t * images + (1.0 - lam_t) * images[perm]
            lam_eff = lam_t.to(dtype=images.dtype)
        return mixed, labels, target_b, lam_eff.clamp(0.0, 1.0)
