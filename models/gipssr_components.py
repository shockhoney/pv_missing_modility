"""Reusable IGDCA and stage-1 recovery components for GIPSSR-Net."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.analytic_cca import RegularizedSharedIdentityProjector




def _zero_linear(layer: nn.Linear) -> None:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _templates(features: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    labels = labels.to(device=features.device, dtype=torch.long)
    identities = labels.unique(sorted=True)
    templates = torch.stack(
        [features[labels == identity].mean(dim=0) for identity in identities]
    )
    return F.normalize(templates, dim=1), identities


class IdentityGuidedDeepCorrelationAlignment(nn.Module):
    """IGDCA: CCA-initialized, identity-guided trainable correlation alignment."""

    def __init__(self, input_dim: int, shared_dim: int, dropout: float):
        super().__init__()
        self.input_dim = int(input_dim)
        self.shared_dim = int(shared_dim)
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
        self, projector: RegularizedSharedIdentityProjector, modality: str
    ) -> None:
        if not bool(projector.fitted.item()):
            raise RuntimeError("Analytic CCA must be fitted before initialization")
        if modality == "palm":
            mean, projection = projector.palm_mean, projector.palm_projection
        elif modality == "vein":
            mean, projection = projector.vein_mean, projector.vein_projection
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        projection = projection[:, : self.shared_dim]
        weight = projection.transpose(0, 1).to(self.base.weight)
        bias = (-(mean.to(weight) @ projection.to(weight))).squeeze(0)
        self.base.weight.copy_(weight)
        self.base.bias.copy_(bias)
        self.initial_weight.copy_(weight)
        self.initial_bias.copy_(bias)
        self.initialized.fill_(True)

    def forward_raw(self, features: torch.Tensor) -> torch.Tensor:
        shared = self.base(features)
        return shared + self.refiner(shared)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.forward_raw(features), dim=1)

    def anchor_loss(self) -> torch.Tensor:
        if not bool(self.initialized.item()):
            raise RuntimeError("Projector has not been initialized")
        return F.mse_loss(self.base.weight, self.initial_weight) + F.mse_loss(
            self.base.bias, self.initial_bias
        )


class Stage1SpecificEncoder(nn.Module):
    """Encode the residual 7x7 spatial tokens unique to one modality."""

    def __init__(
        self,
        input_dim: int,
        shared_dim: int,
        model_dim: int,
        layers: int,
        heads: int,
        dropout: float,
        max_spatial_tokens: int = 49,
    ):
        super().__init__()
        self.model_dim = int(model_dim)
        self.max_spatial_tokens = int(max_spatial_tokens)
        self.token_projection = nn.Conv2d(input_dim, model_dim, kernel_size=1, bias=False)
        self.shared_projection = nn.Linear(shared_dim, model_dim)
        self.residual_norm = nn.LayerNorm(model_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, model_dim))
        self.position = nn.Parameter(torch.empty(1, max_spatial_tokens + 1, model_dim))
        block = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            block, num_layers=layers, norm=nn.LayerNorm(model_dim)
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(
        self, spatial: torch.Tensor, shared: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parameter_dtype = self.token_projection.weight.dtype
        spatial = spatial.to(dtype=parameter_dtype)
        tokens = self.token_projection(spatial).flatten(2).transpose(1, 2)
        if tokens.size(1) > self.max_spatial_tokens:
            raise ValueError("Spatial map has more tokens than the configured positional table")
        condition = self.shared_projection(shared).unsqueeze(1)
        tokens = self.residual_norm(tokens - condition)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        sequence = torch.cat([cls, tokens], dim=1)
        sequence = sequence + self.position[:, : sequence.size(1)]
        encoded = self.encoder(sequence)
        return F.normalize(encoded[:, 0], dim=1), encoded[:, 1:]


class Stage1RecoveryDecoder(nn.Module):
    """Recover a distribution over the missing target representation."""

    def __init__(
        self,
        embedding_dim: int,
        shared_dim: int,
        model_dim: int,
        layers: int,
        heads: int,
        dropout: float,
        query_tokens: int = 4,
    ):
        super().__init__()
        self.shared_projection = nn.Linear(shared_dim, model_dim)
        self.target_embedding_projection = nn.Linear(embedding_dim, model_dim)
        self.target_specific_projection = nn.Linear(model_dim, model_dim)
        self.direction_embedding = nn.Parameter(torch.empty(2, 1, model_dim))
        self.missing_token = nn.Parameter(torch.empty(1, 1, model_dim))
        self.queries = nn.Parameter(torch.empty(1, query_tokens, model_dim))
        block = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            block, num_layers=layers, norm=nn.LayerNorm(model_dim)
        )
        self.full_residual = nn.Linear(model_dim, embedding_dim)
        self.specific_mean = nn.Linear(model_dim, model_dim)
        self.log_variance = nn.Linear(model_dim, 1)
        self.residual_logit = nn.Parameter(torch.tensor(-2.944439))  # sigmoid = 0.05
        nn.init.trunc_normal_(self.direction_embedding, std=0.02)
        nn.init.trunc_normal_(self.missing_token, std=0.02)
        nn.init.trunc_normal_(self.queries, std=0.02)
        _zero_linear(self.full_residual)
        nn.init.constant_(self.log_variance.bias, -2.0)

    def forward(
        self,
        source_tokens: torch.Tensor,
        shared: torch.Tensor,
        retrieved_target: torch.Tensor,
        retrieved_specific: torch.Tensor,
        direction: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = source_tokens.size(0)
        shared_token = self.shared_projection(shared).unsqueeze(1)
        target_token = self.target_embedding_projection(retrieved_target).unsqueeze(1)
        specific_token = self.target_specific_projection(retrieved_specific).unsqueeze(1)
        memory = torch.cat(
            [source_tokens, shared_token, target_token, specific_token], dim=1
        )
        queries = self.queries.expand(batch, -1, -1)
        queries = queries + self.direction_embedding[direction].unsqueeze(0)
        queries = queries + self.missing_token
        decoded = self.decoder(queries, memory).mean(dim=1)
        residual = self.full_residual(decoded)
        scale = torch.sigmoid(self.residual_logit)
        recovered = F.normalize(retrieved_target + scale * residual, dim=1)
        predicted_specific = F.normalize(self.specific_mean(decoded), dim=1)
        log_variance = self.log_variance(decoded).squeeze(1).clamp(-6.0, 2.0)
        return recovered, predicted_specific, log_variance


class GIPSSRCore(nn.Module):
    BRANCHES = ("available", "shared_same", "shared_cross", "recovered")

    def __init__(
        self,
        input_dim: int = 256,
        shared_dim: int = 192,
        specific_dim: int = 128,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.1,
        ablation: str = "full",
    ):
        super().__init__()
        if ablation not in {"full", "without_igdca", "without_sgssd", "without_giprd", "without_sgssd_giprd", "without_cuef_calibration", "without_cuef_conflict", "without_cuef_uncertainty"}:
            raise ValueError(f"Unsupported ablation: {ablation}")
        self.ablation = ablation
        self.input_dim = int(input_dim)
        self.shared_dim = int(shared_dim)
        self.specific_dim = int(specific_dim)
        self.palm_igdca = IdentityGuidedDeepCorrelationAlignment(input_dim, shared_dim, dropout)
        self.vein_igdca = IdentityGuidedDeepCorrelationAlignment(input_dim, shared_dim, dropout)
        specific_kwargs = dict(
            input_dim=input_dim,
            shared_dim=shared_dim,
            model_dim=specific_dim,
            layers=transformer_layers,
            heads=transformer_heads,
            dropout=dropout,
        )
        self.palm_specific = Stage1SpecificEncoder(**specific_kwargs)
        self.vein_specific = Stage1SpecificEncoder(**specific_kwargs)
        self.recovery_decoder = Stage1RecoveryDecoder(
            embedding_dim=input_dim,
            shared_dim=shared_dim,
            model_dim=specific_dim,
            layers=transformer_layers,
            heads=transformer_heads,
            dropout=dropout,
        )

    @torch.no_grad()
    def initialize_from_cca(self, projector: RegularizedSharedIdentityProjector) -> None:
        if self.ablation == "without_igdca":
            return
        self.palm_igdca.initialize_from_cca(projector, "palm")
        self.vein_igdca.initialize_from_cca(projector, "vein")

    def igdca(self, modality: str) -> IdentityGuidedDeepCorrelationAlignment:
        if modality == "palm":
            return self.palm_igdca
        if modality == "vein":
            return self.vein_igdca
        raise ValueError(f"Unsupported modality: {modality}")

    def specific_encoder(self, modality: str) -> Stage1SpecificEncoder:
        if modality == "palm":
            return self.palm_specific
        if modality == "vein":
            return self.vein_specific
        raise ValueError(f"Unsupported modality: {modality}")

    def encode(
        self, features: torch.Tensor, spatial: torch.Tensor, modality: str
    ) -> dict[str, torch.Tensor]:
        if self.ablation == "without_igdca":
            shared_raw = features.new_zeros(features.size(0), self.shared_dim)
            shared = shared_raw
        else:
            igdca = self.igdca(modality)
            shared_raw = igdca.forward_raw(features)
            shared = F.normalize(shared_raw, dim=1)
        specific, tokens = self.specific_encoder(modality)(spatial, shared)
        return {
            "embedding": F.normalize(features, dim=1),
            "shared_raw": shared_raw,
            "shared": shared,
            "specific": specific,
            "tokens": tokens,
        }

    @torch.no_grad()
    def build_gallery_memory(
        self,
        gallery: dict[str, torch.Tensor],
        chunk_size: int = 256,
    ) -> dict[str, torch.Tensor]:
        labels = gallery["labels"].long()
        memory: dict[str, torch.Tensor] = {"labels": labels.unique(sorted=True)}
        for modality in ("palm", "vein"):
            encoded = {"embedding": [], "shared": [], "specific": []}
            for start in range(0, labels.numel(), chunk_size):
                stop = min(start + chunk_size, labels.numel())
                output = self.encode(
                    gallery[modality][start:stop],
                    gallery[f"{modality}_spatial"][start:stop],
                    modality,
                )
                for name in encoded:
                    encoded[name].append(output[name])
            for name, pieces in encoded.items():
                templates, template_labels = _templates(torch.cat(pieces), labels)
                if not torch.equal(template_labels, memory["labels"]):
                    raise ValueError("Gallery identity order differs across modalities")
                memory[f"{modality}_{name}"] = templates
        return memory

    @staticmethod
    def _direction(available_modality: str) -> tuple[str, int]:
        if available_modality == "palm":
            return "vein", 0
        if available_modality == "vein":
            return "palm", 1
        raise ValueError(f"Unsupported modality: {available_modality}")


    def recover_with_gallery(
        self,
        available_features: torch.Tensor,
        available_spatial: torch.Tensor,
        available_modality: str,
        memory: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor | str]:
        encoded = self.encode(available_features, available_spatial, available_modality)
        return self.score_from_encoding(encoded, available_modality, memory)

    def anchor_loss(self) -> torch.Tensor:
        if self.ablation == "without_igdca":
            return next(self.parameters()).new_zeros(())
        return self.palm_igdca.anchor_loss() + self.vein_igdca.anchor_loss()
