import torch
import torch.nn as nn
import torch.nn.functional as F

from models.feature_diffusion import ConditionalFeatureDiffusion
from utils.head import ArcFace
from utils.scenarios import COMPLETE, PALMPRINT_MISSING, PALMVEIN_MISSING


ARCHITECTURE_VERSION = "spatial_cross_attention_v1"
LEGACY_GATE_KEYS = {"palm_missing_gate", "vein_missing_gate"}


def load_missing_model_state(model, state):
    if any(key.startswith("fusion.cross_attention.q.") for key in state):
        raise ValueError(
            "Checkpoint uses the removed vector head-gating attention. "
            "Reuse the encoder checkpoints and retrain the missing-model stages."
        )
    state = {key: value for key, value in state.items() if key not in LEGACY_GATE_KEYS}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Invalid missing-model checkpoint. Missing={missing}, unexpected={unexpected}")


class CrossModalAttention(nn.Module):
    def __init__(self, dim=256, heads=4):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.p2v_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.v2p_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.palm_norm = nn.LayerNorm(dim)
        self.vein_norm = nn.LayerNorm(dim)

    def forward(self, palm_map, vein_map):
        if palm_map.ndim != 4 or vein_map.ndim != 4:
            raise ValueError("Spatial cross-attention expects palm/vein feature maps shaped [B, C, H, W]")
        if palm_map.size(0) != vein_map.size(0):
            raise ValueError("Palm and vein feature maps must have the same batch size")
        if palm_map.size(1) != self.dim or vein_map.size(1) != self.dim:
            raise ValueError(f"Spatial cross-attention expects {self.dim} feature channels")

        palm_tokens = palm_map.flatten(2).transpose(1, 2)
        vein_tokens = vein_map.flatten(2).transpose(1, 2)
        palm_cross, _ = self.p2v_attn(
            palm_tokens,
            vein_tokens,
            vein_tokens,
            need_weights=False,
        )
        vein_cross, _ = self.v2p_attn(
            vein_tokens,
            palm_tokens,
            palm_tokens,
            need_weights=False,
        )
        palm_enhanced = self.palm_norm(palm_tokens + palm_cross)
        vein_enhanced = self.vein_norm(vein_tokens + vein_cross)
        return palm_enhanced.mean(dim=1), vein_enhanced.mean(dim=1)


class ChannelAttentionFusion(nn.Module):
    def __init__(self, dim=256, reduction=4):
        super().__init__()
        hidden = max(dim // reduction, 16)
        self.attention = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim * 2),
        )

    def forward(self, palm_feat, vein_feat):
        logits = self.attention(torch.cat([palm_feat, vein_feat], dim=1))
        weights = F.softmax(logits.view(palm_feat.size(0), 2, -1), dim=1)
        return weights[:, 0] * palm_feat + weights[:, 1] * vein_feat


class AvailableGuidedFusion(nn.Module):
    def __init__(self, dim=256, reduction=4):
        super().__init__()
        hidden = max(dim // reduction, 16)
        self.delta = nn.Sequential(
            nn.Linear(dim * 4, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
        )
        self.residual_scale = nn.Parameter(torch.zeros(1))

    @torch.no_grad()
    def reset_residual_scale(self):
        self.residual_scale.zero_()

    def forward(self, available_feat, restored_feat):
        available_feat = F.normalize(available_feat, dim=1)
        restored_feat = F.normalize(restored_feat, dim=1)
        cue = torch.cat(
            [available_feat, restored_feat, available_feat * restored_feat, (available_feat - restored_feat).abs()],
            dim=1,
        )
        residual = F.normalize(self.delta(cue), dim=1)
        scale = torch.tanh(self.residual_scale)
        return F.normalize(available_feat + scale * residual, dim=1)


class CrossChannelFusion(nn.Module):
    def __init__(self, dim=256, heads=4, reduction=4):
        super().__init__()
        self.cross_attention = CrossModalAttention(dim, heads)
        self.channel_attention = ChannelAttentionFusion(dim, reduction)
        self.available_fusion = AvailableGuidedFusion(dim, reduction)
        self.project = nn.Linear(dim, dim)

    def forward(self, palm_feat, vein_feat, scenario=COMPLETE):
        if scenario == PALMPRINT_MISSING:
            return self.available_fusion(vein_feat, palm_feat)
        if scenario == PALMVEIN_MISSING:
            return self.available_fusion(palm_feat, vein_feat)
        palm_feat, vein_feat = self.cross_attention(palm_feat, vein_feat)
        return F.normalize(self.project(self.channel_attention(palm_feat, vein_feat)), dim=1)


def cosine_alignment_loss(pred, target, detach_target=True):
    target = target.detach() if detach_target else target
    return (1.0 - F.cosine_similarity(pred, target, dim=1)).mean()


class MissingModalityRecognizer(nn.Module):
    def __init__(
        self,
        palm_encoder,
        vein_encoder,
        num_classes,
        dim=256,
        heads=4,
        reduction=4,
        arcface_s=32.0,
        arcface_m=0.25,
        palm_teacher=None,
        vein_teacher=None,
        diffusion_steps=100,
        ddim_steps=5,
        diffusion_base_channels=64,
        diffusion_time_dim=128,
        diffusion_dropout=0.0,
        diffusion_stats_momentum=0.99,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("dim must be divisible by 2")

        self.palm_encoder = palm_encoder
        self.vein_encoder = vein_encoder
        self.palm_teacher = palm_teacher
        self.vein_teacher = vein_teacher
        feature_channels = getattr(palm_encoder, "local_dim", None)
        if feature_channels is None or feature_channels != getattr(vein_encoder, "local_dim", None):
            raise ValueError("Both encoders must expose the same local_dim for feature diffusion")
        if feature_channels != dim:
            raise ValueError("Encoder local_dim must match the fusion embedding dimension")

        diffusion_kwargs = dict(
            feature_channels=feature_channels,
            num_steps=diffusion_steps,
            ddim_steps=ddim_steps,
            base_channels=diffusion_base_channels,
            time_dim=diffusion_time_dim,
            dropout=diffusion_dropout,
            stats_momentum=diffusion_stats_momentum,
        )
        self.p2v = ConditionalFeatureDiffusion(**diffusion_kwargs)
        self.v2p = ConditionalFeatureDiffusion(**diffusion_kwargs)
        self.fusion = CrossChannelFusion(dim, heads, reduction)
        self.classifier = ArcFace(dim, num_classes, arcface_s, arcface_m)

        for module in (self.palm_encoder, self.vein_encoder, self.palm_teacher, self.vein_teacher):
            self._freeze(module)

    @staticmethod
    def _freeze(module):
        if module is None:
            return
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        for module in (self.palm_encoder, self.vein_encoder, self.palm_teacher, self.vein_teacher):
            if module is not None:
                module.eval()
        return self

    @staticmethod
    def _encode_one(encoder, image):
        with torch.no_grad():
            feature_map = encoder(image, return_spatial=True)
        shared, specific = encoder.parts_from_features(feature_map)
        return feature_map, shared, specific

    def encode_modalities(self, palm, vein):
        palm_map, palm_shared, palm_specific = self._encode_one(self.palm_encoder, palm)
        vein_map, vein_shared, vein_specific = self._encode_one(self.vein_encoder, vein)
        return {
            "palm_map": palm_map,
            "vein_map": vein_map,
            "palm_shared": palm_shared,
            "palm_specific": palm_specific,
            "vein_shared": vein_shared,
            "vein_specific": vein_specific,
            "f_palm": self._compose(palm_shared, palm_specific),
            "f_vein": self._compose(vein_shared, vein_specific),
        }

    def _compose(self, shared, specific):
        return torch.cat([shared, specific], dim=1)

    def diffusion_loss(self, encoded):
        return 0.5 * (
            self.p2v.training_loss(encoded["palm_map"], encoded["vein_map"])
            + self.v2p.training_loss(encoded["vein_map"], encoded["palm_map"])
        )

    def recover_modalities(self, encoded, directions=("p2v", "v2p")):
        directions = set(directions)
        recovery = {}

        if "p2v" in directions:
            recovered_vein_map = self.p2v.sample(encoded["palm_map"])
        else:
            recovered_vein_map = encoded["vein_map"]

        if "v2p" in directions:
            recovered_palm_map = self.v2p.sample(encoded["vein_map"])
        else:
            recovered_palm_map = encoded["palm_map"]

        hat_vein_shared, hat_vein_specific = self.vein_encoder.parts_from_features(recovered_vein_map)
        hat_palm_shared, hat_palm_specific = self.palm_encoder.parts_from_features(recovered_palm_map)
        recovery.update(
            {
                "generated_palm_map": recovered_palm_map,
                "generated_vein_map": recovered_vein_map,
                "hat_palm_specific": hat_palm_specific,
                "hat_vein_specific": hat_vein_specific,
                "generated_palm": self._compose(hat_palm_shared, hat_palm_specific),
                "generated_vein": self._compose(hat_vein_shared, hat_vein_specific),
                "hat_palm": self._compose(encoded["vein_shared"], hat_palm_specific),
                "hat_vein": self._compose(encoded["palm_shared"], hat_vein_specific),
            }
        )
        return recovery

    def _scenario_from_mask(self, mask, scenario):
        if mask is None:
            return scenario
        if mask.ndim != 2 or mask.size(1) != 2:
            raise ValueError("mask must have shape [B, 2] for palm/vein availability")
        mask = mask.bool()
        palm_exists = mask[:, 0]
        vein_exists = mask[:, 1]
        if not torch.all(palm_exists | vein_exists):
            raise ValueError("Each sample must contain at least one available modality")
        if torch.all(palm_exists & vein_exists):
            return COMPLETE
        if torch.all(~palm_exists & vein_exists):
            return PALMPRINT_MISSING
        if torch.all(palm_exists & ~vein_exists):
            return PALMVEIN_MISSING
        raise ValueError("Mixed modality scenarios in one batch are not supported; use scenario-grouped batches")

    def _select_features(self, feats, scenario, mask):
        if mask is not None:
            mask = mask.to(feats["f_palm"].device).bool()
            palm_exists = mask[:, 0:1]
            vein_exists = mask[:, 1:2]
            return (
                torch.where(palm_exists, feats["f_palm"], feats["hat_palm"]),
                torch.where(vein_exists, feats["f_vein"], feats["hat_vein"]),
            )
        if scenario == COMPLETE:
            return feats["f_palm"], feats["f_vein"]
        if scenario == PALMPRINT_MISSING:
            return feats["hat_palm"], feats["f_vein"]
        if scenario == PALMVEIN_MISSING:
            return feats["f_palm"], feats["hat_vein"]
        raise ValueError(f"Unsupported scenario: {scenario}")

    def _teacher_logits(self, teacher, feature):
        with torch.no_grad():
            return teacher(feature)

    def _missing_teacher_logits(self, scenario, feats):
        if scenario == PALMPRINT_MISSING and self.vein_teacher is not None:
            return self._teacher_logits(self.vein_teacher, feats["f_vein"])
        if scenario == PALMVEIN_MISSING and self.palm_teacher is not None:
            return self._teacher_logits(self.palm_teacher, feats["f_palm"])
        return None

    def forward_from_encoded(self, encoded, recovery, labels=None, scenario=COMPLETE, mask=None):
        feats = {**encoded, **recovery}
        scenario = self._scenario_from_mask(mask, scenario)
        if scenario == COMPLETE:
            z = self.fusion(encoded["palm_map"], encoded["vein_map"], scenario=COMPLETE)
        else:
            use_palm, use_vein = self._select_features(feats, scenario, mask)
            z = self.fusion(use_palm, use_vein, scenario=scenario)

        fusion_logits_raw = self.classifier(z)
        fusion_logits = self.classifier(z, labels) if labels is not None else fusion_logits_raw
        output = {
            "logits": fusion_logits,
            "fusion_logits": fusion_logits,
            "fusion_logits_raw": fusion_logits_raw,
            "z": z,
            **feats,
        }
        if labels is not None:
            teacher_logits_raw = self._missing_teacher_logits(scenario, feats)
            if teacher_logits_raw is not None:
                output["teacher_logits_raw"] = teacher_logits_raw
        return output

    def _required_directions(self, scenario, mask):
        scenario = self._scenario_from_mask(mask, scenario)
        if scenario == PALMPRINT_MISSING:
            return ["v2p"]
        if scenario == PALMVEIN_MISSING:
            return ["p2v"]
        return []

    def forward(self, palm, vein, labels=None, scenario=COMPLETE, mask=None):
        encoded = self.encode_modalities(palm, vein)
        directions = self._required_directions(scenario, mask)
        recovery = self.recover_modalities(encoded, directions)
        return self.forward_from_encoded(encoded, recovery, labels, scenario, mask)
