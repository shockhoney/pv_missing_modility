import torch
import torch.nn as nn
import torch.nn.functional as F

from models.feature_diffusion import ConditionalFeatureDiffusion
from utils.head import ArcFace
from utils.scenarios import COMPLETE, PALMPRINT_MISSING, PALMVEIN_MISSING


class CrossModalAttention(nn.Module):
    def __init__(self, dim=256, heads=4):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def _attend(self, query, source):
        batch_size = query.size(0)
        q = self.q(query).view(batch_size, self.heads, self.head_dim)
        k = self.k(source).view(batch_size, self.heads, self.head_dim)
        v = self.v(source).view(batch_size, self.heads, self.head_dim)
        weights = F.softmax((q * k).sum(dim=-1, keepdim=True) * self.scale, dim=1)
        return (weights * v).reshape(batch_size, -1)

    def forward(self, palm_feat, vein_feat):
        palm_out = self._attend(palm_feat, vein_feat)
        vein_out = self._attend(vein_feat, palm_feat)
        return self.norm(palm_feat + self.out(palm_out)), self.norm(vein_feat + self.out(vein_out))


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

    def project_feature(self, feature):
        return F.normalize(self.project(feature), dim=1)

    def available_only(self, feature):
        return self.project_feature(F.normalize(feature, dim=1))

    def forward(self, palm_feat, vein_feat, scenario=COMPLETE):
        if scenario == PALMPRINT_MISSING:
            fused = self.available_fusion(vein_feat, palm_feat)
        elif scenario == PALMVEIN_MISSING:
            fused = self.available_fusion(palm_feat, vein_feat)
        else:
            palm_feat, vein_feat = self.cross_attention(palm_feat, vein_feat)
            fused = self.channel_attention(palm_feat, vein_feat)
        return self.project_feature(fused)


def cosine_alignment_loss(pred, target, detach_target=True):
    target = target.detach() if detach_target else target
    return (1.0 - F.cosine_similarity(pred, target, dim=1)).mean()


def shared_alignment_loss(palm_shared, vein_shared):
    return cosine_alignment_loss(palm_shared, vein_shared, detach_target=False)


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
        gate_init=0.0,
        diffusion_steps=100,
        ddim_steps=20,
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
        self.palm_missing_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.vein_missing_gate = nn.Parameter(torch.tensor(float(gate_init)))

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
        feature_map = encoder(image, return_spatial=True)
        shared, specific = encoder.parts_from_features(feature_map)
        return feature_map, shared, specific

    def encode_modalities(self, palm, vein):
        with torch.no_grad():
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

    def recover_modalities(self, encoded, training_recovery=False, directions=("p2v", "v2p")):
        directions = set(directions)
        zero = encoded["palm_map"].new_zeros(())
        recovery = {"diffusion_loss": zero, "reconstruction_loss": zero}

        if "p2v" in directions:
            result = (
                self.p2v.training_recovery(encoded["palm_map"], encoded["vein_map"])
                if training_recovery
                else {"feature": self.p2v.sample(encoded["palm_map"])}
            )
            recovered_vein_map = result["feature"]
            recovery["diffusion_loss"] = recovery["diffusion_loss"] + result.get("diffusion_loss", zero)
            recovery["reconstruction_loss"] = recovery["reconstruction_loss"] + result.get(
                "reconstruction_loss", zero
            )
        else:
            recovered_vein_map = encoded["vein_map"]

        if "v2p" in directions:
            result = (
                self.v2p.training_recovery(encoded["vein_map"], encoded["palm_map"])
                if training_recovery
                else {"feature": self.v2p.sample(encoded["vein_map"])}
            )
            recovered_palm_map = result["feature"]
            recovery["diffusion_loss"] = recovery["diffusion_loss"] + result.get("diffusion_loss", zero)
            recovery["reconstruction_loss"] = recovery["reconstruction_loss"] + result.get(
                "reconstruction_loss", zero
            )
        else:
            recovered_palm_map = encoded["palm_map"]

        if directions:
            recovery["diffusion_loss"] = recovery["diffusion_loss"] / len(directions)
            recovery["reconstruction_loss"] = recovery["reconstruction_loss"] / len(directions)

        hat_vein_shared, hat_vein_specific = self.vein_encoder.parts_from_features(recovered_vein_map)
        hat_palm_shared, hat_palm_specific = self.palm_encoder.parts_from_features(recovered_palm_map)
        recovery.update(
            {
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
        mask = mask.bool()
        palm_exists = mask[:, 0]
        vein_exists = mask[:, 1]
        if torch.all(palm_exists & vein_exists):
            return COMPLETE
        if torch.all(~palm_exists & vein_exists):
            return PALMPRINT_MISSING
        if torch.all(palm_exists & ~vein_exists):
            return PALMVEIN_MISSING
        return scenario

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

    def _teacher_logits(self, teacher, feature, labels):
        with torch.no_grad():
            raw_logits = teacher(feature)
            logits = teacher(feature, labels) if labels is not None else raw_logits
        return logits, raw_logits

    def _missing_logits(self, scenario, feats, fusion_logits, fusion_logits_raw, labels):
        if scenario == PALMPRINT_MISSING and self.vein_teacher is not None:
            teacher_logits, teacher_logits_raw = self._teacher_logits(self.vein_teacher, feats["f_vein"], labels)
            alpha = torch.sigmoid(self.palm_missing_gate)
        elif scenario == PALMVEIN_MISSING and self.palm_teacher is not None:
            teacher_logits, teacher_logits_raw = self._teacher_logits(self.palm_teacher, feats["f_palm"], labels)
            alpha = torch.sigmoid(self.vein_missing_gate)
        else:
            return fusion_logits, {}

        return (1.0 - alpha) * teacher_logits + alpha * fusion_logits, {
            "fusion_logits_raw": fusion_logits_raw,
            "teacher_logits_raw": teacher_logits_raw,
        }

    def forward_from_encoded(self, encoded, recovery, labels=None, scenario=COMPLETE, mask=None):
        feats = {**encoded, **recovery}
        scenario = self._scenario_from_mask(mask, scenario)
        use_palm, use_vein = self._select_features(feats, scenario, mask)
        z = self.fusion(use_palm, use_vein, scenario=scenario)
        fusion_logits = self.classifier(z, labels) if labels is not None else self.classifier(z)
        fusion_logits_raw = fusion_logits if labels is None else self.classifier(z)
        logits, logit_outputs = self._missing_logits(scenario, feats, fusion_logits, fusion_logits_raw, labels)
        return {
            "logits": logits,
            "z": z,
            **logit_outputs,
            **feats,
        }

    def _required_directions(self, scenario, mask):
        if mask is not None:
            mask = mask.bool()
            directions = []
            if torch.any(~mask[:, 0]):
                directions.append("v2p")
            if torch.any(~mask[:, 1]):
                directions.append("p2v")
            return directions
        if scenario == PALMPRINT_MISSING:
            return ["v2p"]
        if scenario == PALMVEIN_MISSING:
            return ["p2v"]
        return []

    def forward(self, palm, vein, labels=None, scenario=COMPLETE, mask=None):
        encoded = self.encode_modalities(palm, vein)
        if self.training:
            directions = ("p2v", "v2p")
            training_recovery = True
        else:
            directions = self._required_directions(scenario, mask)
            training_recovery = False
        recovery = self.recover_modalities(encoded, training_recovery, directions)
        return self.forward_from_encoded(encoded, recovery, labels, scenario, mask)
