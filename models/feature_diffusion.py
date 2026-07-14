from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(num_steps: int, offset: float = 0.008) -> torch.Tensor:
    steps = torch.linspace(0, num_steps, num_steps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((steps / num_steps) + offset) / (1.0 + offset) * math.pi * 0.5).pow(2)
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-4, 0.999).float()


def _extract(values: torch.Tensor, timesteps: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return values.gather(0, timesteps).reshape(timesteps.size(0), *((1,) * (reference.ndim - 1)))


def _group_count(channels: int, maximum: int = 8) -> int:
    groups = min(maximum, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim < 4 or dim % 2 != 0:
            raise ValueError("time embedding dimension must be an even integer >= 4")
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            -scale * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        return torch.cat([angles.sin(), angles.cos()], dim=1)


class TimeConditionedResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time_proj(time_embedding).unsqueeze(-1).unsqueeze(-1)
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return hidden + self.skip(x)


class ConditionalFeatureUNet(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        base_channels: int = 64,
        time_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input_proj = nn.Conv2d(feature_channels * 2, base_channels, kernel_size=3, padding=1)
        self.down_block = TimeConditionedResBlock(base_channels, base_channels, time_dim, dropout)
        self.downsample = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        self.mid_block1 = TimeConditionedResBlock(base_channels * 2, base_channels * 2, time_dim, dropout)
        self.mid_block2 = TimeConditionedResBlock(base_channels * 2, base_channels * 2, time_dim, dropout)
        self.upsample_proj = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        self.up_block = TimeConditionedResBlock(base_channels * 2, base_channels, time_dim, dropout)
        self.output_norm = nn.GroupNorm(_group_count(base_channels), base_channels)
        self.output_proj = nn.Conv2d(base_channels, feature_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        noisy_target: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_target.shape != condition.shape:
            raise ValueError(
                f"noisy target and condition must have identical shapes, got {noisy_target.shape} and {condition.shape}"
            )
        time_embedding = self.time_embedding(timesteps)
        skip = self.down_block(self.input_proj(torch.cat([noisy_target, condition], dim=1)), time_embedding)
        hidden = self.downsample(skip)
        hidden = self.mid_block2(self.mid_block1(hidden, time_embedding), time_embedding)
        hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.upsample_proj(hidden)
        hidden = self.up_block(torch.cat([hidden, skip], dim=1), time_embedding)
        return self.output_proj(F.silu(self.output_norm(hidden)))


class ConditionalFeatureDiffusion(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        num_steps: int = 100,
        ddim_steps: int = 20,
        base_channels: int = 64,
        time_dim: int = 128,
        dropout: float = 0.0,
        stats_momentum: float = 0.99,
        clip_value: float = 5.0,
        high_noise_min_ratio: float = 0.8,
        high_noise_max_ratio: float = 0.95,
    ):
        super().__init__()
        if num_steps <= 1:
            raise ValueError("num_steps must be greater than 1")
        if ddim_steps <= 0:
            raise ValueError("ddim_steps must be positive")
        if not 0.0 <= high_noise_min_ratio < high_noise_max_ratio <= 1.0:
            raise ValueError("high-noise ratios must satisfy 0 <= min < max <= 1")

        self.num_steps = num_steps
        self.ddim_steps = min(ddim_steps, num_steps)
        self.stats_momentum = stats_momentum
        self.clip_value = clip_value
        self.high_noise_min_ratio = high_noise_min_ratio
        self.high_noise_max_ratio = high_noise_max_ratio
        self.denoiser = ConditionalFeatureUNet(
            feature_channels=feature_channels,
            base_channels=base_channels,
            time_dim=time_dim,
            dropout=dropout,
        )

        betas = cosine_beta_schedule(num_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())

        stats_shape = (1, feature_channels, 1, 1)
        self.register_buffer("condition_mean", torch.zeros(stats_shape))
        self.register_buffer("condition_std", torch.ones(stats_shape))
        self.register_buffer("target_mean", torch.zeros(stats_shape))
        self.register_buffer("target_std", torch.ones(stats_shape))
        self.register_buffer("stats_initialized", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def _update_statistics(self, condition: torch.Tensor, target: torch.Tensor) -> None:
        dims = (0, 2, 3)
        condition_mean = condition.mean(dim=dims, keepdim=True)
        condition_std = condition.std(dim=dims, keepdim=True, unbiased=False).clamp_min(1e-4)
        target_mean = target.mean(dim=dims, keepdim=True)
        target_std = target.std(dim=dims, keepdim=True, unbiased=False).clamp_min(1e-4)

        if not bool(self.stats_initialized.item()):
            self.condition_mean.copy_(condition_mean)
            self.condition_std.copy_(condition_std)
            self.target_mean.copy_(target_mean)
            self.target_std.copy_(target_std)
            self.stats_initialized.fill_(True)
            return

        update_weight = 1.0 - self.stats_momentum
        self.condition_mean.lerp_(condition_mean, update_weight)
        self.condition_std.lerp_(condition_std, update_weight)
        self.target_mean.lerp_(target_mean, update_weight)
        self.target_std.lerp_(target_std, update_weight)

    def _normalize_condition(self, feature: torch.Tensor) -> torch.Tensor:
        return (feature - self.condition_mean) / self.condition_std.clamp_min(1e-4)

    def _normalize_target(self, feature: torch.Tensor) -> torch.Tensor:
        return (feature - self.target_mean) / self.target_std.clamp_min(1e-4)

    def _denormalize_target(self, feature: torch.Tensor) -> torch.Tensor:
        return feature * self.target_std + self.target_mean

    def q_sample(self, clean_target: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            _extract(self.sqrt_alpha_bars, timesteps, clean_target) * clean_target
            + _extract(self.sqrt_one_minus_alpha_bars, timesteps, clean_target) * noise
        )

    def _sample_timesteps(self, batch_size: int, device: torch.device, high_noise=False) -> torch.Tensor:
        if not high_noise:
            return torch.randint(0, self.num_steps, (batch_size,), device=device)
        lower = min(int(self.num_steps * self.high_noise_min_ratio), self.num_steps - 1)
        upper = min(max(int(self.num_steps * self.high_noise_max_ratio), lower + 1), self.num_steps)
        return torch.randint(lower, upper, (batch_size,), device=device)

    def predict_clean_target(
        self,
        noisy_target: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        clean = (
            noisy_target - _extract(self.sqrt_one_minus_alpha_bars, timesteps, noisy_target) * predicted_noise
        ) / _extract(self.sqrt_alpha_bars, timesteps, noisy_target).clamp_min(1e-6)
        return clean.clamp(-self.clip_value, self.clip_value)

    def training_recovery(self, condition: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        self._update_statistics(condition.detach(), target.detach())
        condition_normalized = self._normalize_condition(condition)
        target_normalized = self._normalize_target(target)

        timesteps = self._sample_timesteps(target.size(0), target.device)
        noise = torch.randn_like(target_normalized)
        noisy_target = self.q_sample(target_normalized, timesteps, noise)
        predicted_noise = self.denoiser(noisy_target, condition_normalized, timesteps)

        high_timesteps = self._sample_timesteps(target.size(0), target.device, high_noise=True)
        high_noise = torch.randn_like(target_normalized)
        high_noisy_target = self.q_sample(target_normalized, high_timesteps, high_noise)
        high_predicted_noise = self.denoiser(high_noisy_target, condition_normalized, high_timesteps)
        predicted_clean = self.predict_clean_target(high_noisy_target, high_timesteps, high_predicted_noise)

        return {
            "feature": self._denormalize_target(predicted_clean),
            "diffusion_loss": 0.5 * (
                F.mse_loss(predicted_noise, noise) + F.mse_loss(high_predicted_noise, high_noise)
            ),
            "reconstruction_loss": F.l1_loss(predicted_clean, target_normalized),
        }

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, ddim_steps: int | None = None) -> torch.Tensor:
        condition_normalized = self._normalize_condition(condition)
        sample = torch.randn_like(condition_normalized)
        num_sampling_steps = min(ddim_steps or self.ddim_steps, self.num_steps)
        schedule = torch.linspace(
            self.num_steps - 1,
            0,
            steps=num_sampling_steps,
            device=condition.device,
        ).round().long().unique_consecutive()

        for index, timestep_value in enumerate(schedule):
            timesteps = torch.full(
                (condition.size(0),),
                int(timestep_value.item()),
                device=condition.device,
                dtype=torch.long,
            )
            predicted_noise = self.denoiser(sample, condition_normalized, timesteps)
            predicted_clean = self.predict_clean_target(sample, timesteps, predicted_noise)
            if index == schedule.numel() - 1:
                sample = predicted_clean
                break

            next_timestep = schedule[index + 1]
            next_alpha_bar = self.alpha_bars[next_timestep].to(dtype=sample.dtype)
            sample = next_alpha_bar.sqrt() * predicted_clean + (1.0 - next_alpha_bar).sqrt() * predicted_noise

        return self._denormalize_target(sample)
