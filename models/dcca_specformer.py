"""HIASR-Net: hierarchical identity-prior attentive state-space recovery."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.recovery_backbone import StableRecoveryBackbone
from models.dcca_specformer_components import _templates


ARCHITECTURE_VERSION = "hiasr_identity_prior_state_space_v10"


def _zero_linear(layer: nn.Linear) -> None:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class GatedFeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int, dropout: float):
        super().__init__()
        hidden = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.input = nn.Linear(dim, hidden * 2)
        self.output = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        content, gate = self.input(self.norm(values)).chunk(2, dim=-1)
        return values + self.dropout(self.output(content * F.silu(gate)))


class ConditionedSelectiveScan2D(nn.Module):
    """Four-direction pure-PyTorch selective state-space mixer for 7x7 tokens."""

    def __init__(self, dim: int, condition_dim: int, dropout: float):
        super().__init__()
        self.dim = int(dim)
        self.norm = nn.LayerNorm(dim)
        self.local = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.input_projection = nn.Linear(dim, dim * 3)
        self.condition_projection = nn.Linear(condition_dim, dim * 2)
        self.direction_weights = nn.Linear(condition_dim, 4)
        self.output_projection = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = GatedFeedForward(dim, expansion=2, dropout=dropout)

    def _scan(
        self, sequence: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        value, decay, update = self.input_projection(sequence).chunk(3, dim=-1)
        condition_decay, condition_update = self.condition_projection(condition).chunk(
            2, dim=-1
        )
        decay = 0.5 + 0.49 * torch.sigmoid(decay + condition_decay.unsqueeze(1))
        update = torch.sigmoid(update + condition_update.unsqueeze(1))
        value = torch.tanh(value)
        prefix = torch.cumprod(decay, dim=1)
        innovations = (1.0 - decay) * value / prefix.clamp_min(1e-20)
        state = prefix * torch.cumsum(innovations, dim=1)
        return update * state + (1.0 - update) * value

    @staticmethod
    def _ordered_sequences(grid: torch.Tensor) -> list[torch.Tensor]:
        row = grid.flatten(1, 2)
        column = grid.transpose(1, 2).flatten(1, 2)
        return [row, row.flip(1), column, column.flip(1)]

    @staticmethod
    def _restore(
        sequence: torch.Tensor, direction: int, height: int, width: int
    ) -> torch.Tensor:
        if direction == 0:
            return sequence.reshape(sequence.size(0), height, width, -1)
        if direction == 1:
            return sequence.flip(1).reshape(sequence.size(0), height, width, -1)
        if direction == 2:
            return sequence.reshape(sequence.size(0), width, height, -1).transpose(1, 2)
        return (
            sequence.flip(1)
            .reshape(sequence.size(0), width, height, -1)
            .transpose(1, 2)
        )

    def forward(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor,
        height: int = 7,
        width: int = 7,
    ) -> torch.Tensor:
        if tokens.size(1) != height * width:
            raise ValueError("Selective scan expects a complete spatial token grid")
        normalized = self.norm(tokens)
        grid = normalized.reshape(tokens.size(0), height, width, self.dim)
        local = self.local(grid.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        grid = grid + local
        restored = [
            self._restore(self._scan(sequence, condition), index, height, width)
            for index, sequence in enumerate(self._ordered_sequences(grid))
        ]
        weights = torch.softmax(self.direction_weights(condition), dim=1)
        mixed = sum(
            weights[:, index, None, None, None] * value
            for index, value in enumerate(restored)
        )
        output = tokens + self.dropout(
            self.output_projection(mixed.flatten(1, 2))
        )
        return self.feed_forward(output)


class SharedGuidedStateSpaceDisentangler(nn.Module):
    """Shared-guided common/specific separation with role latents."""

    def __init__(
        self,
        input_dim: int,
        shared_dim: int,
        model_dim: int,
        heads: int,
        role_queries: int,
        dropout: float,
    ):
        super().__init__()
        self.role_queries_count = int(role_queries)
        self.spatial_projection = nn.Conv2d(input_dim, model_dim, kernel_size=1)
        self.local_three = nn.Conv2d(
            model_dim, model_dim, kernel_size=3, padding=1, groups=model_dim
        )
        self.local_dilated = nn.Conv2d(
            model_dim,
            model_dim,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=model_dim,
        )
        self.shared_projection = nn.Linear(shared_dim, model_dim)
        self.modality_embedding = nn.Parameter(torch.empty(2, model_dim))
        self.common_queries = nn.Parameter(torch.empty(1, role_queries, model_dim))
        self.specific_queries = nn.Parameter(torch.empty(1, role_queries, model_dim))
        self.shared_to_spatial = nn.MultiheadAttention(
            model_dim, heads, dropout=dropout, batch_first=True
        )
        self.spatial_to_common = nn.MultiheadAttention(
            model_dim, heads, dropout=dropout, batch_first=True
        )
        self.common_gate = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
            nn.Sigmoid(),
        )
        self.token_norm = nn.LayerNorm(model_dim)
        self.state_space = ConditionedSelectiveScan2D(
            model_dim, model_dim, dropout
        )
        self.role_attention = nn.MultiheadAttention(
            model_dim, heads, dropout=dropout, batch_first=True
        )
        role_block = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.role_mixer = nn.TransformerEncoder(
            role_block, num_layers=1, norm=nn.LayerNorm(model_dim)
        )
        self.specific_head = nn.Linear(model_dim, model_dim)
        nn.init.trunc_normal_(self.modality_embedding, std=0.02)
        nn.init.trunc_normal_(self.common_queries, std=0.02)
        nn.init.trunc_normal_(self.specific_queries, std=0.02)

    def forward(
        self, spatial: torch.Tensor, shared: torch.Tensor, modality_index: int
    ) -> dict[str, torch.Tensor]:
        projected = self.spatial_projection(
            spatial.to(dtype=self.spatial_projection.weight.dtype)
        )
        projected = projected + self.local_three(projected) + self.local_dilated(projected)
        tokens = projected.flatten(2).transpose(1, 2)
        condition = (
            self.shared_projection(shared)
            + self.modality_embedding[modality_index].unsqueeze(0)
        )
        common_queries = self.common_queries.expand(tokens.size(0), -1, -1)
        common_queries = common_queries + condition.unsqueeze(1)
        common_latents, _ = self.shared_to_spatial(
            common_queries, tokens, tokens, need_weights=False
        )
        common_context, _ = self.spatial_to_common(
            tokens, common_latents, common_latents, need_weights=False
        )
        common_strength = self.common_gate(torch.cat([tokens, common_context], dim=-1))
        specific_tokens = self.token_norm(tokens - common_strength * common_context)
        specific_tokens = self.state_space(specific_tokens, condition)
        role_queries = self.specific_queries.expand(tokens.size(0), -1, -1)
        role_queries = role_queries + condition.unsqueeze(1)
        role_latents, _ = self.role_attention(
            role_queries, specific_tokens, specific_tokens, need_weights=False
        )
        role_latents = self.role_mixer(role_latents)
        specific = F.normalize(self.specific_head(role_latents.mean(dim=1)), dim=1)
        return {
            "hierarchical_tokens": role_latents,
            "hierarchical_specific": specific,
            "orthogonal_reference": F.normalize(condition, dim=1),
            "common_latents": common_latents,
            "specific_spatial_tokens": specific_tokens,
        }


class HierarchicalIdentityPriorDecoder(nn.Module):
    """Top-K candidate interaction followed by specific and identity recovery."""

    def __init__(
        self,
        embedding_dim: int,
        shared_dim: int,
        model_dim: int,
        heads: int,
        role_queries: int,
        topk_candidates: int,
        dropout: float,
    ):
        super().__init__()
        self.model_dim = int(model_dim)
        self.role_queries_count = int(role_queries)
        self.topk_candidates = int(topk_candidates)
        self.shared_projection = nn.Linear(shared_dim, model_dim)
        self.available_projection = nn.Linear(embedding_dim, model_dim)
        self.embedding_projection = nn.Linear(embedding_dim, model_dim)
        self.specific_projection = nn.Linear(model_dim, model_dim)
        self.score_projection = nn.Linear(3, model_dim)
        self.rank_embedding = nn.Parameter(torch.empty(topk_candidates, model_dim))
        self.direction_embedding = nn.Parameter(torch.empty(2, model_dim))
        self.specific_queries = nn.Parameter(torch.empty(1, role_queries, model_dim))
        self.embedding_queries = nn.Parameter(torch.empty(1, role_queries, model_dim))
        self.stage_one_attention = nn.MultiheadAttention(
            model_dim, heads, dropout=dropout, batch_first=True
        )
        self.stage_two_attention = nn.MultiheadAttention(
            model_dim, heads, dropout=dropout, batch_first=True
        )
        stage_block = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.stage_one_mixer = nn.TransformerEncoder(
            stage_block, num_layers=1, norm=nn.LayerNorm(model_dim)
        )
        self.stage_two_mixer = nn.TransformerEncoder(
            stage_block, num_layers=1, norm=nn.LayerNorm(model_dim)
        )
        self.specific_head = nn.Linear(model_dim, model_dim)
        self.embedding_residual = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, embedding_dim),
        )
        self.variance_delta = nn.Linear(model_dim, 1)
        _zero_linear(self.embedding_residual[-1])
        _zero_linear(self.variance_delta)
        for parameter in (
            self.rank_embedding,
            self.direction_embedding,
            self.specific_queries,
            self.embedding_queries,
        ):
            nn.init.trunc_normal_(parameter, std=0.02)

    def _candidate_tokens(
        self,
        target_embeddings: torch.Tensor,
        target_specific: torch.Tensor,
        candidate_weights: torch.Tensor,
        direction: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, count, _ = target_embeddings.shape
        rank = torch.arange(count, device=target_embeddings.device)
        rank_fraction = rank.to(target_embeddings) / max(1, count - 1)
        score_features = torch.stack(
            [
                candidate_weights,
                candidate_weights.clamp_min(1e-8).log(),
                rank_fraction.unsqueeze(0).expand(batch, -1),
            ],
            dim=-1,
        )
        context = (
            self.score_projection(score_features)
            + self.rank_embedding[:count].unsqueeze(0)
            + self.direction_embedding[direction].view(1, 1, -1)
        )
        return (
            self.embedding_projection(target_embeddings) + context,
            self.specific_projection(target_specific) + context,
        )

    def forward(
        self,
        source_role_tokens: torch.Tensor,
        shared: torch.Tensor,
        available_embedding: torch.Tensor,
        target_embeddings: torch.Tensor,
        target_specific: torch.Tensor,
        candidate_weights: torch.Tensor,
        candidate_keep: torch.Tensor,
        direction: int,
    ) -> dict[str, torch.Tensor]:
        embedding_tokens, specific_tokens = self._candidate_tokens(
            target_embeddings, target_specific, candidate_weights, direction
        )
        shared_token = self.shared_projection(shared).unsqueeze(1)
        direction_token = self.direction_embedding[direction].view(1, 1, -1)
        batch = shared.size(0)
        stage_one_queries = self.specific_queries.expand(batch, -1, -1)
        stage_one_queries = stage_one_queries + shared_token + direction_token
        stage_one_memory = torch.cat(
            [source_role_tokens, shared_token, specific_tokens, embedding_tokens], dim=1
        )
        fixed = source_role_tokens.size(1) + 1
        candidate_padding = ~candidate_keep
        stage_one_padding = torch.cat(
            [
                torch.zeros(
                    batch,
                    fixed,
                    dtype=torch.bool,
                    device=shared.device,
                ),
                candidate_padding,
                candidate_padding,
            ],
            dim=1,
        )
        stage_one, _ = self.stage_one_attention(
            stage_one_queries,
            stage_one_memory,
            stage_one_memory,
            key_padding_mask=stage_one_padding,
            need_weights=False,
        )
        stage_one = self.stage_one_mixer(stage_one + stage_one_queries)
        predicted_specific = F.normalize(
            self.specific_head(stage_one.mean(dim=1)), dim=1
        )

        available_token = self.available_projection(available_embedding).unsqueeze(1)
        stage_two_queries = self.embedding_queries.expand(batch, -1, -1)
        stage_two_queries = stage_two_queries + predicted_specific.unsqueeze(1)
        stage_two_queries = stage_two_queries + direction_token
        stage_two_memory = torch.cat(
            [stage_one, shared_token, available_token, embedding_tokens], dim=1
        )
        stage_two_padding = torch.cat(
            [
                torch.zeros(
                    batch,
                    stage_one.size(1) + 2,
                    dtype=torch.bool,
                    device=shared.device,
                ),
                candidate_padding,
            ],
            dim=1,
        )
        stage_two, _ = self.stage_two_attention(
            stage_two_queries,
            stage_two_memory,
            stage_two_memory,
            key_padding_mask=stage_two_padding,
            need_weights=False,
        )
        stage_two = self.stage_two_mixer(stage_two + stage_two_queries)
        pooled = stage_two.mean(dim=1)
        return {
            "embedding_delta": self.embedding_residual(pooled),
            "predicted_specific": predicted_specific,
            "log_variance_delta": self.variance_delta(pooled).squeeze(1),
            "stage_one_tokens": stage_one,
            "stage_two_tokens": stage_two,
        }


class DCCASpecFormerRecovery(StableRecoveryBackbone):
    """Identity-preserving hierarchical recovery with a trainable state-space branch."""

    def __init__(
        self,
        input_dim: int = 256,
        shared_dim: int = 192,
        specific_dim: int = 128,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.1,
        max_gate: float = 0.75,
        min_recovery_weight: float = 0.15,
        retrieval_dropout: float = 0.10,
        branch_floor: float = 0.0,
        max_proxy_identities: int = 512,
        topk_candidates: int = 5,
        role_queries: int = 4,
        candidate_dropout: float = 0.20,
        max_refinement: float = 0.25,
    ):
        super().__init__(
            input_dim=input_dim,
            shared_dim=shared_dim,
            specific_dim=specific_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            dropout=dropout,
            max_gate=max_gate,
            min_recovery_weight=min_recovery_weight,
            retrieval_dropout=retrieval_dropout,
            branch_floor=branch_floor,
            max_proxy_identities=max_proxy_identities,
        )
        if topk_candidates < 2:
            raise ValueError("HIASR requires at least two independent candidates")
        if not 0.0 <= candidate_dropout < 1.0:
            raise ValueError("candidate_dropout must be in [0, 1)")
        if not 0.0 < max_refinement <= 1.0:
            raise ValueError("max_refinement must be in (0, 1]")
        self.topk_candidates = int(topk_candidates)
        self.role_queries = int(role_queries)
        self.candidate_dropout = float(candidate_dropout)
        self.max_refinement = float(max_refinement)
        self.shared_disentangler = SharedGuidedStateSpaceDisentangler(
            input_dim,
            shared_dim,
            specific_dim,
            transformer_heads,
            role_queries,
            dropout,
        )
        self.hierarchical_decoder = HierarchicalIdentityPriorDecoder(
            input_dim,
            shared_dim,
            specific_dim,
            transformer_heads,
            role_queries,
            topk_candidates,
            dropout,
        )
        self.refinement_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))

    def encode(
        self, features: torch.Tensor, spatial: torch.Tensor, modality: str
    ) -> dict[str, torch.Tensor]:
        output = super().encode(features, spatial, modality)
        modality_index = 0 if modality == "palm" else 1
        output.update(
            self.shared_disentangler(spatial, output["shared"], modality_index)
        )
        return output

    @torch.no_grad()
    def build_gallery_memory(
        self, gallery: dict[str, torch.Tensor], chunk_size: int = 256
    ) -> dict[str, torch.Tensor]:
        labels = gallery["labels"].long()
        memory: dict[str, torch.Tensor] = {"labels": labels.unique(sorted=True)}
        for modality in ("palm", "vein"):
            names = ("embedding", "shared", "specific", "hierarchical_specific")
            encoded = {name: [] for name in names}
            for start in range(0, labels.numel(), chunk_size):
                stop = min(start + chunk_size, labels.numel())
                output = self.encode(
                    gallery[modality][start:stop],
                    gallery[f"{modality}_spatial"][start:stop],
                    modality,
                )
                for name in names:
                    encoded[name].append(output[name])
            for name, pieces in encoded.items():
                templates, template_labels = _templates(torch.cat(pieces), labels)
                if not torch.equal(template_labels, memory["labels"]):
                    raise ValueError("Gallery identity order differs across modalities")
                memory[f"{modality}_{name}"] = templates
        return memory

    def _candidate_keep_mask(self, batch: int, count: int, device) -> torch.Tensor:
        keep = torch.ones(batch, count, dtype=torch.bool, device=device)
        if self.training and self.candidate_dropout > 0.0:
            keep = torch.rand(batch, count, device=device) >= self.candidate_dropout
            keep[:, 0] = True
        return keep

    def score_from_encoding(
        self,
        available: dict[str, torch.Tensor],
        available_modality: str,
        memory: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor | str]:
        backbone = super().score_from_encoding(
            available, available_modality, memory
        )
        target_modality, direction = self._direction(available_modality)
        count = min(self.topk_candidates, backbone["posterior"].size(1))
        top_values, top_indices = backbone["posterior"].topk(count, dim=1)
        target_embeddings = memory[f"{target_modality}_embedding"][top_indices]
        target_specific = memory[f"{target_modality}_hierarchical_specific"][
            top_indices
        ]
        candidate_keep = self._candidate_keep_mask(
            top_values.size(0), count, top_values.device
        )
        candidate_weights = top_values.masked_fill(~candidate_keep, 0.0)
        candidate_weights = candidate_weights / candidate_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        hierarchical = self.hierarchical_decoder(
            available["hierarchical_tokens"],
            available["shared"],
            available["embedding"],
            target_embeddings,
            target_specific,
            candidate_weights,
            candidate_keep,
            direction,
        )
        scale = self.max_refinement * torch.sigmoid(self.refinement_logit)
        proposed = F.normalize(
            backbone["mean"] + scale * hierarchical["embedding_delta"], dim=1
        )
        proposed_scores = proposed @ memory[f"{target_modality}_embedding"].t()
        backbone_top_two = backbone["recovered_scores"].topk(2, dim=1)
        proposed_top_two = proposed_scores.topk(2, dim=1)
        same_identity = backbone_top_two.indices[:, 0].eq(
            proposed_top_two.indices[:, 0]
        )
        backbone_margin = backbone_top_two.values[:, 0] - backbone_top_two.values[:, 1]
        proposed_margin = proposed_top_two.values[:, 0] - proposed_top_two.values[:, 1]
        hard_safety = same_identity & proposed_margin.ge(backbone_margin)
        soft_safety = same_identity.to(proposed) * torch.sigmoid(
            (proposed_margin - backbone_margin) / 0.02
        )
        safety = hard_safety.to(proposed) + soft_safety - soft_safety.detach()
        recovered = F.normalize(
            backbone["mean"]
            + safety.unsqueeze(1) * scale * hierarchical["embedding_delta"],
            dim=1,
        )
        recovered_scores = recovered @ memory[f"{target_modality}_embedding"].t()
        fused_scores = (
            backbone["shared_weight"].unsqueeze(1) * backbone["base_scores"]
            + backbone["recovery_weight"].unsqueeze(1) * recovered_scores
        )
        orthogonality = F.cosine_similarity(
            available["orthogonal_reference"],
            available["hierarchical_specific"],
            dim=1,
        ).square()
        backbone.update(
            {
                "mean": recovered,
                "predicted_specific": hierarchical["predicted_specific"],
                "cycle": self.cycle_reconstruct(recovered, direction),
                "log_variance": (
                    backbone["log_variance"]
                    + 0.1 * hierarchical["log_variance_delta"]
                ).clamp(-6.0, 2.0),
                "recovered_scores": recovered_scores,
                "fused_scores": fused_scores,
                "teacher_fused_scores": backbone["fused_scores"],
                "orthogonality": orthogonality,
                "candidate_indices": top_indices,
                "candidate_weights": candidate_weights,
                "candidate_keep_fraction": candidate_keep.float().mean(),
                "refinement_gate": safety * scale,
                "refinement_active_fraction": hard_safety.float().mean(),
                "hierarchical_stage_one": hierarchical["stage_one_tokens"],
                "hierarchical_stage_two": hierarchical["stage_two_tokens"],
            }
        )
        return backbone
