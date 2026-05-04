from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.stage1_mobileFacenet import MobileFaceNet
from utils.dymo_stats import compute_selection_rewards, score_embeddings
from utils.head import ArcFace


def resolve_selection_tau(selection_tau, selected_reward: torch.Tensor, missing_index: torch.Tensor) -> torch.Tensor:
    tau = torch.as_tensor(selection_tau, device=selected_reward.device, dtype=selected_reward.dtype)
    if tau.ndim == 0:
        return tau.expand_as(selected_reward)
    tau = tau.reshape(-1)
    if tau.numel() == 2:
        return tau[missing_index]
    if tau.numel() == selected_reward.numel():
        return tau.reshape_as(selected_reward)
    raise ValueError(
        "selection_tau must be a scalar, a length-2 tensor/list for [missing_palm, missing_vein], "
        "or one threshold per sample."
    )


class CrossModalRecoverer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        token_dim: int,
        num_tokens: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.token_dim = token_dim
        self.num_tokens = num_tokens

        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.global_head = nn.Linear(hidden_dim, feature_dim)
        self.token_head = nn.Linear(hidden_dim, token_dim * num_tokens)
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def forward(self, src_global: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.backbone(src_global)
        recovered_global = self.global_head(hidden)
        recovered_tokens = self.token_head(hidden).view(src_global.size(0), self.num_tokens, self.token_dim)
        recovered_confidence = torch.sigmoid(self.confidence_head(hidden)).squeeze(-1)
        return {
            "global": recovered_global,
            "tokens": recovered_tokens,
            "confidence": recovered_confidence,
        }


class CrossModalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor):
        batch_size = feat_a.size(0)

        q_a = self.q_proj(feat_a).view(batch_size, self.num_heads, self.head_dim)
        k_b = self.k_proj(feat_b).view(batch_size, self.num_heads, self.head_dim)
        v_b = self.v_proj(feat_b).view(batch_size, self.num_heads, self.head_dim)
        attn_a = (q_a * k_b).sum(dim=-1, keepdim=True) * self.scale
        attn_a = F.softmax(attn_a, dim=1)
        out_a = (attn_a * v_b).reshape(batch_size, -1)
        enhanced_a = self.norm(feat_a + self.out_proj(out_a))

        q_b = self.q_proj(feat_b).view(batch_size, self.num_heads, self.head_dim)
        k_a = self.k_proj(feat_a).view(batch_size, self.num_heads, self.head_dim)
        v_a = self.v_proj(feat_a).view(batch_size, self.num_heads, self.head_dim)
        attn_b = (q_b * k_a).sum(dim=-1, keepdim=True) * self.scale
        attn_b = F.softmax(attn_b, dim=1)
        out_b = (attn_b * v_a).reshape(batch_size, -1)
        enhanced_b = self.norm(feat_b + self.out_proj(out_b))

        return enhanced_a, enhanced_b


class ChannelAttentionFusion(nn.Module):
    def __init__(self, dim: int, reduction: int = 4) -> None:
        super().__init__()
        hidden_dim = max(dim // reduction, 16)
        self.attention = nn.Sequential(
            nn.Linear(2 * dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * dim),
        )
        for module in self.attention:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        logits = self.attention(torch.cat([feat_a, feat_b], dim=1))
        weights = F.softmax(logits.view(feat_a.size(0), 2, -1), dim=1)
        return weights[:, 0] * feat_a + weights[:, 1] * feat_b


class PalmVeinDynamicTransformer(nn.Module):
    """
    DyMo-style backbone for palmprint-palmvein missing-modality recognition.
    The current fusion core uses the teacher-style cross-modal attention and
    channel attention from the reference palm/vein fusion model.

    `missing_mask` uses DyMo semantics:
        False -> modality available
        True  -> modality missing
    """

    def __init__(
        self,
        num_classes: int,
        input_size: int = 224,
        encoder_dim: int = 256,
        token_grid: int = 4,
        transformer_dim: int = 256,
        transformer_heads: int = 8,
        transformer_layers: int = 2,
        transformer_mlp_ratio: float = 4.0,
        projection_dim: int = 256,
        dropout: float = 0.1,
        arcface_s: float = 64.0,
        arcface_m: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.encoder_dim = encoder_dim
        self.token_grid = token_grid
        self.num_tokens = token_grid * token_grid
        self.transformer_dim = transformer_dim
        self.projection_dim = projection_dim

        self.cnn_palm = MobileFaceNet(input_channel=3, input_size=input_size, embedding_size=encoder_dim)
        self.cnn_vein = MobileFaceNet(input_channel=3, input_size=input_size, embedding_size=encoder_dim)
        self.token_pool = nn.AdaptiveAvgPool2d((token_grid, token_grid))
        self.palm_global_proj = nn.Linear(encoder_dim, transformer_dim)
        self.vein_global_proj = nn.Linear(encoder_dim, transformer_dim)
        self.palm_global_norm = nn.LayerNorm(transformer_dim)
        self.vein_global_norm = nn.LayerNorm(transformer_dim)
        self.palm_token_proj = nn.Linear(encoder_dim, transformer_dim)
        self.vein_token_proj = nn.Linear(encoder_dim, transformer_dim)
        self.palm_token_norm = nn.LayerNorm(transformer_dim)
        self.vein_token_norm = nn.LayerNorm(transformer_dim)

        self.vein_from_palm = CrossModalRecoverer(
            feature_dim=encoder_dim,
            token_dim=encoder_dim,
            num_tokens=self.num_tokens,
            hidden_dim=max(transformer_dim * 2, 512),
            dropout=dropout,
        )
        self.palm_from_vein = CrossModalRecoverer(
            feature_dim=encoder_dim,
            token_dim=encoder_dim,
            num_tokens=self.num_tokens,
            hidden_dim=max(transformer_dim * 2, 512),
            dropout=dropout,
        )

        self.modality_embed = nn.Embedding(2, transformer_dim)
        self.source_embed = nn.Embedding(3, transformer_dim)  # 0 absent, 1 real, 2 recovered
        self.embed_dropout = nn.Dropout(dropout)
        self.global_cross_attn = CrossModalAttention(transformer_dim, num_heads=transformer_heads)
        self.global_channel_fusion = ChannelAttentionFusion(transformer_dim, reduction=4)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(transformer_dim, int(transformer_dim * transformer_mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(transformer_dim * transformer_mlp_ratio), transformer_dim),
        )
        self.out_norm = nn.LayerNorm(transformer_dim)

        self.classifier = ArcFace(transformer_dim, num_classes, s=arcface_s, m=arcface_m)
        self.projection = nn.Linear(transformer_dim, projection_dim)
        self.register_buffer("prototypes", F.normalize(torch.randn(num_classes, projection_dim), dim=1))

    def _extract_modality(self, encoder: MobileFaceNet, image: torch.Tensor):
        feat_map = encoder(image, return_spatial=True)
        pooled = encoder.global_pool(feat_map).reshape(feat_map.size(0), -1)
        global_feat = encoder.bn(pooled)
        tokens = self.token_pool(feat_map).flatten(2).transpose(1, 2)
        return global_feat, tokens

    def encode_modalities(self, palm: torch.Tensor, vein: torch.Tensor) -> Dict[str, torch.Tensor]:
        palm_global, palm_tokens = self._extract_modality(self.cnn_palm, palm)
        vein_global, vein_tokens = self._extract_modality(self.cnn_vein, vein)

        recovered_vein = self.vein_from_palm(palm_global)
        recovered_palm = self.palm_from_vein(vein_global)

        return {
            "palm_global": palm_global,
            "vein_global": vein_global,
            "palm_tokens": palm_tokens,
            "vein_tokens": vein_tokens,
            "recovered_palm_global": recovered_palm["global"],
            "recovered_palm_tokens": recovered_palm["tokens"],
            "recovered_palm_confidence": recovered_palm["confidence"],
            "recovered_vein_global": recovered_vein["global"],
            "recovered_vein_tokens": recovered_vein["tokens"],
            "recovered_vein_confidence": recovered_vein["confidence"],
        }

    def _select_global_features(
        self,
        encoded: Dict[str, torch.Tensor],
        missing_mask: torch.Tensor,
        use_recovered_mask: torch.Tensor,
    ):
        batch_size = missing_mask.size(0)
        device = missing_mask.device

        missing_mask = missing_mask.bool()
        use_recovered_mask = use_recovered_mask.bool() & missing_mask

        palm_active_real = ~missing_mask[:, 0]
        palm_active_recovered = use_recovered_mask[:, 0]
        vein_active_real = ~missing_mask[:, 1]
        vein_active_recovered = use_recovered_mask[:, 1]

        palm_global = torch.zeros_like(encoded["palm_global"])
        vein_global = torch.zeros_like(encoded["vein_global"])

        palm_global = torch.where(
            palm_active_real[:, None],
            encoded["palm_global"],
            palm_global,
        )
        palm_global = torch.where(
            palm_active_recovered[:, None],
            encoded["recovered_palm_global"],
            palm_global,
        )
        vein_global = torch.where(
            vein_active_real[:, None],
            encoded["vein_global"],
            vein_global,
        )
        vein_global = torch.where(
            vein_active_recovered[:, None],
            encoded["recovered_vein_global"],
            vein_global,
        )

        palm_source = torch.zeros(batch_size, dtype=torch.long, device=device)
        vein_source = torch.zeros(batch_size, dtype=torch.long, device=device)
        palm_source[palm_active_real] = 1
        palm_source[palm_active_recovered] = 2
        vein_source[vein_active_real] = 1
        vein_source[vein_active_recovered] = 2

        palm_feat = self.palm_global_norm(self.palm_global_proj(palm_global))
        vein_feat = self.vein_global_norm(self.vein_global_proj(vein_global))
        palm_feat = palm_feat + self.modality_embed(torch.zeros(batch_size, dtype=torch.long, device=device))
        vein_feat = vein_feat + self.modality_embed(torch.ones(batch_size, dtype=torch.long, device=device))
        palm_feat = palm_feat + self.source_embed(palm_source)
        vein_feat = vein_feat + self.source_embed(vein_source)
        return self.embed_dropout(palm_feat), self.embed_dropout(vein_feat), palm_source, vein_source

    def _select_projected_tokens(
        self,
        encoded: Dict[str, torch.Tensor],
        missing_mask: torch.Tensor,
        use_recovered_mask: torch.Tensor,
    ):
        missing_mask = missing_mask.bool()
        use_recovered_mask = use_recovered_mask.bool() & missing_mask

        palm_active_real = ~missing_mask[:, 0]
        palm_active_recovered = use_recovered_mask[:, 0]
        vein_active_real = ~missing_mask[:, 1]
        vein_active_recovered = use_recovered_mask[:, 1]

        palm_tokens = torch.zeros_like(encoded["palm_tokens"])
        vein_tokens = torch.zeros_like(encoded["vein_tokens"])
        palm_tokens = torch.where(palm_active_real[:, None, None], encoded["palm_tokens"], palm_tokens)
        palm_tokens = torch.where(
            palm_active_recovered[:, None, None],
            encoded["recovered_palm_tokens"],
            palm_tokens,
        )
        vein_tokens = torch.where(vein_active_real[:, None, None], encoded["vein_tokens"], vein_tokens)
        vein_tokens = torch.where(
            vein_active_recovered[:, None, None],
            encoded["recovered_vein_tokens"],
            vein_tokens,
        )
        return {
            "palm_tokens": self.palm_token_norm(self.palm_token_proj(palm_tokens)),
            "vein_tokens": self.vein_token_norm(self.vein_token_proj(vein_tokens)),
        }

    def forward_from_encoded(
        self,
        encoded: Dict[str, torch.Tensor],
        missing_mask: torch.Tensor,
        use_recovered_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if use_recovered_mask is None:
            use_recovered_mask = torch.zeros_like(missing_mask)

        palm_feat, vein_feat, palm_source, vein_source = self._select_global_features(
            encoded,
            missing_mask,
            use_recovered_mask,
        )
        palm_enhanced, vein_enhanced = self.global_cross_attn(palm_feat, vein_feat)
        fused = self.global_channel_fusion(palm_enhanced, vein_enhanced)
        cls_feat = self.out_norm(fused + self.fusion_mlp(fused))
        logits = self.classifier(cls_feat, labels)
        embedding = F.normalize(self.projection(cls_feat), dim=1)

        output = {
            "logits": logits,
            "embedding": embedding,
            "cls_feat": cls_feat,
            "missing_mask": missing_mask.bool(),
            "use_recovered_mask": use_recovered_mask.bool(),
            "palm_source": palm_source,
            "vein_source": vein_source,
            "recovery_confidence": torch.stack(
                [encoded["recovered_palm_confidence"], encoded["recovered_vein_confidence"]],
                dim=1,
            ),
        }
        if return_details:
            output["projected_tokens"] = self._select_projected_tokens(encoded, missing_mask, use_recovered_mask)
        return output

    def forward(
        self,
        palm: torch.Tensor,
        vein: torch.Tensor,
        missing_mask: torch.Tensor,
        use_recovered_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encode_modalities(palm, vein)
        return self.forward_from_encoded(
            encoded,
            missing_mask=missing_mask,
            use_recovered_mask=use_recovered_mask,
            labels=labels,
            return_details=return_details,
        )


class PalmVeinDyMoSelector(nn.Module):
    def __init__(self, backbone: PalmVeinDynamicTransformer) -> None:
        super().__init__()
        self.backbone = backbone

    @torch.no_grad()
    def forward(
        self,
        palm: torch.Tensor,
        vein: torch.Tensor,
        missing_mask: torch.Tensor,
        stats: Dict,
        selection_mode: str = "open",
        temperature: float = 0.1,
        selection_tau=0.0,
        quality_mode: str = "log_prob",
        max_shift: Optional[float] = None,
        min_density: Optional[float] = None,
        return_details: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if selection_mode not in {"open", "ics"}:
            raise ValueError(f"Unsupported selection mode: {selection_mode}")

        missing_mask = missing_mask.bool()
        encoded = self.backbone.encode_modalities(palm, vein)

        before = self.backbone.forward_from_encoded(
            encoded,
            missing_mask=missing_mask,
            use_recovered_mask=torch.zeros_like(missing_mask),
        )
        candidate_recovered = missing_mask.clone()
        after = self.backbone.forward_from_encoded(
            encoded,
            missing_mask=missing_mask,
            use_recovered_mask=candidate_recovered,
        )

        missing_has_candidate = missing_mask.any(dim=1)
        missing_index = missing_mask.float().argmax(dim=1).long()
        after_recovery_conf = torch.ones(missing_mask.size(0), device=missing_mask.device, dtype=after["embedding"].dtype)
        after_recovery_conf[missing_has_candidate] = after["recovery_confidence"][
            missing_has_candidate,
            missing_index[missing_has_candidate],
        ]

        before_scores = score_embeddings(
            before["embedding"],
            before["missing_mask"],
            stats,
            temperature=temperature,
            quality_mode=quality_mode,
        )
        after_scores = score_embeddings(
            after["embedding"],
            after["missing_mask"] & ~candidate_recovered,
            stats,
            recovery_confidence=after_recovery_conf,
            temperature=temperature,
            quality_mode=quality_mode,
        )
        rewards = compute_selection_rewards(before_scores, after_scores, quality_mode=quality_mode)

        selected_reward = rewards["open_reward"] if selection_mode == "open" else rewards["ics_reward"]
        selected_density = after_scores["open_gaussian"] if selection_mode == "open" else after_scores["ics_gaussian"]
        selected_tau = resolve_selection_tau(selection_tau, selected_reward, missing_index)
        use_recovered = (selected_reward > selected_tau)[:, None] & candidate_recovered
        embedding_shift = 1.0 - F.cosine_similarity(before["embedding"], after["embedding"], dim=1)
        if max_shift is not None:
            use_recovered = use_recovered & (embedding_shift <= max_shift)[:, None]
        if min_density is not None:
            use_recovered = use_recovered & (selected_density >= min_density)[:, None]

        final = self.backbone.forward_from_encoded(
            encoded,
            missing_mask=missing_mask,
            use_recovered_mask=use_recovered,
        )
        final["selected_recovered_mask"] = use_recovered
        if return_details:
            final["before"] = before
            final["after"] = after
            final["before_scores"] = before_scores
            final["after_scores"] = after_scores
            final["rewards"] = rewards
            final["embedding_shift"] = embedding_shift
            final["selected_density"] = selected_density
            final["selection_tau"] = selected_tau
        return final
