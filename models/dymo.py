from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.stage1_mobileFacenet import MobileFaceNet
from utils.dymo_stats import compute_selection_rewards, score_embeddings


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


class PalmVeinDynamicTransformer(nn.Module):
    """
    DyMo-style backbone for palmprint-palmvein missing-modality recognition.

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
        self.palm_token_proj = nn.Linear(encoder_dim, transformer_dim)
        self.vein_token_proj = nn.Linear(encoder_dim, transformer_dim)
        self.palm_token_norm = nn.LayerNorm(transformer_dim)
        self.vein_token_norm = nn.LayerNorm(transformer_dim)

        self.vein_from_palm = CrossModalRecoverer(
            feature_dim=encoder_dim,
            token_dim=transformer_dim,
            num_tokens=self.num_tokens,
            hidden_dim=max(transformer_dim * 2, 512),
            dropout=dropout,
        )
        self.palm_from_vein = CrossModalRecoverer(
            feature_dim=encoder_dim,
            token_dim=transformer_dim,
            num_tokens=self.num_tokens,
            hidden_dim=max(transformer_dim * 2, 512),
            dropout=dropout,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 2 * self.num_tokens, transformer_dim))
        self.modality_embed = nn.Embedding(2, transformer_dim)
        self.source_embed = nn.Embedding(3, transformer_dim)  # 0 absent, 1 real, 2 recovered
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=int(transformer_dim * transformer_mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.out_norm = nn.LayerNorm(transformer_dim)

        self.classifier = nn.Linear(transformer_dim, num_classes)
        self.projection = nn.Linear(transformer_dim, projection_dim)
        self.register_buffer("prototypes", F.normalize(torch.randn(num_classes, projection_dim), dim=1))

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _extract_modality(self, encoder: MobileFaceNet, image: torch.Tensor):
        feat_map = encoder(image, return_spatial=True)
        pooled = encoder.global_pool(feat_map).reshape(feat_map.size(0), -1)
        global_feat = encoder.bn(pooled)
        tokens = self.token_pool(feat_map).flatten(2).transpose(1, 2)
        return global_feat, tokens

    def encode_modalities(self, palm: torch.Tensor, vein: torch.Tensor) -> Dict[str, torch.Tensor]:
        palm_global, palm_tokens = self._extract_modality(self.cnn_palm, palm)
        vein_global, vein_tokens = self._extract_modality(self.cnn_vein, vein)

        palm_tokens = self.palm_token_norm(self.palm_token_proj(palm_tokens))
        vein_tokens = self.vein_token_norm(self.vein_token_proj(vein_tokens))

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

    def _assemble_sequence(
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

        palm_tokens = torch.zeros_like(encoded["palm_tokens"])
        vein_tokens = torch.zeros_like(encoded["vein_tokens"])

        palm_tokens = torch.where(
            palm_active_real[:, None, None],
            encoded["palm_tokens"],
            palm_tokens,
        )
        palm_tokens = torch.where(
            palm_active_recovered[:, None, None],
            encoded["recovered_palm_tokens"],
            palm_tokens,
        )

        vein_tokens = torch.where(
            vein_active_real[:, None, None],
            encoded["vein_tokens"],
            vein_tokens,
        )
        vein_tokens = torch.where(
            vein_active_recovered[:, None, None],
            encoded["recovered_vein_tokens"],
            vein_tokens,
        )

        palm_source = torch.zeros(batch_size, self.num_tokens, dtype=torch.long, device=device)
        vein_source = torch.zeros(batch_size, self.num_tokens, dtype=torch.long, device=device)
        palm_source[palm_active_real] = 1
        palm_source[palm_active_recovered] = 2
        vein_source[vein_active_real] = 1
        vein_source[vein_active_recovered] = 2

        palm_mod = self.modality_embed(torch.zeros(batch_size, self.num_tokens, dtype=torch.long, device=device))
        vein_mod = self.modality_embed(torch.ones(batch_size, self.num_tokens, dtype=torch.long, device=device))
        palm_tokens = palm_tokens + palm_mod + self.source_embed(palm_source)
        vein_tokens = vein_tokens + vein_mod + self.source_embed(vein_source)

        tokens = torch.cat([palm_tokens, vein_tokens], dim=1)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.embed_dropout(tokens + self.pos_embed[:, : tokens.size(1)])

        key_padding_mask = torch.zeros(batch_size, 1 + 2 * self.num_tokens, dtype=torch.bool, device=device)
        key_padding_mask[:, 1 : 1 + self.num_tokens] = ~(palm_active_real | palm_active_recovered)[:, None]
        key_padding_mask[:, 1 + self.num_tokens :] = ~(vein_active_real | vein_active_recovered)[:, None]
        return tokens, key_padding_mask

    def forward_from_encoded(
        self,
        encoded: Dict[str, torch.Tensor],
        missing_mask: torch.Tensor,
        use_recovered_mask: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if use_recovered_mask is None:
            use_recovered_mask = torch.zeros_like(missing_mask)

        tokens, key_padding_mask = self._assemble_sequence(encoded, missing_mask, use_recovered_mask)
        features = self.transformer(tokens, src_key_padding_mask=key_padding_mask)
        cls_feat = self.out_norm(features[:, 0])
        logits = self.classifier(cls_feat)
        embedding = F.normalize(self.projection(cls_feat), dim=1)

        output = {
            "logits": logits,
            "embedding": embedding,
            "cls_feat": cls_feat,
            "missing_mask": missing_mask.bool(),
            "use_recovered_mask": use_recovered_mask.bool(),
            "recovery_confidence": torch.stack(
                [encoded["recovered_palm_confidence"], encoded["recovered_vein_confidence"]],
                dim=1,
            ),
        }
        if return_details:
            output["key_padding_mask"] = key_padding_mask
        return output

    def forward(
        self,
        palm: torch.Tensor,
        vein: torch.Tensor,
        missing_mask: torch.Tensor,
        use_recovered_mask: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encode_modalities(palm, vein)
        return self.forward_from_encoded(
            encoded,
            missing_mask=missing_mask,
            use_recovered_mask=use_recovered_mask,
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
        rewards = compute_selection_rewards(before_scores, after_scores)

        selected_reward = rewards["open_reward"] if selection_mode == "open" else rewards["ics_reward"]
        selected_tau = resolve_selection_tau(selection_tau, selected_reward, missing_index)
        use_recovered = (selected_reward > selected_tau)[:, None] & candidate_recovered

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
            final["selection_tau"] = selected_tau
        return final
