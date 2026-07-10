import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.head import ArcFace


class CrossModalTransformation(nn.Module):
    def __init__(self, dim=128, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
        )

    def forward(self, feature):
        return self.net(feature)


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

    def forward(self, available_feat, restored_feat):
        cue = torch.cat(
            [available_feat, restored_feat, available_feat * restored_feat, (available_feat - restored_feat).abs()],
            dim=1,
        )
        gate = torch.sigmoid(self.residual_scale * self.delta(cue))
        return F.normalize(gate * available_feat + (1.0 - gate) * restored_feat, dim=1)


class CrossChannelFusion(nn.Module):
    def __init__(self, dim=256, heads=4, reduction=4):
        super().__init__()
        self.cross_attention = CrossModalAttention(dim, heads)
        self.channel_attention = ChannelAttentionFusion(dim, reduction)
        self.available_fusion = AvailableGuidedFusion(dim, reduction)
        self.project = nn.Linear(dim, dim)

    def forward(self, palm_feat, vein_feat, scenario="complete"):
        if scenario == "palmprint_missing":
            return self.available_fusion(vein_feat, palm_feat)
        if scenario == "palmvein_missing":
            return self.available_fusion(palm_feat, vein_feat)
        palm_feat, vein_feat = self.cross_attention(palm_feat, vein_feat)
        return F.normalize(self.project(self.channel_attention(palm_feat, vein_feat)), dim=1)


def transformation_loss(pred, target):
    return (1.0 - F.cosine_similarity(pred, target.detach(), dim=1)).mean()


def shared_alignment_loss(palm_shared, vein_shared):
    return (1.0 - F.cosine_similarity(palm_shared, vein_shared, dim=1)).mean()


def consistency_loss(pred, target):
    return (1.0 - F.cosine_similarity(pred, target.detach(), dim=1)).mean()


class MissingModalityRecognizer(nn.Module):
    def __init__(
        self,
        palm_encoder,
        vein_encoder,
        num_classes,
        dim=256,
        cmft_hidden=1024,
        heads=4,
        reduction=4,
        arcface_s=32.0,
        arcface_m=0.25,
        palm_teacher=None,
        vein_teacher=None,
        gate_init=0.0,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("dim must be divisible by 2")

        self.palm_encoder = palm_encoder
        self.vein_encoder = vein_encoder
        self.palm_teacher = palm_teacher
        self.vein_teacher = vein_teacher
        self.part_dim = dim // 2

        self.p2v = CrossModalTransformation(self.part_dim, cmft_hidden)
        self.v2p = CrossModalTransformation(self.part_dim, cmft_hidden)
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

    def _encode_one(self, encoder, image):
        if hasattr(encoder, "forward_parts"):
            return encoder.forward_parts(image)
        feature = encoder(image)
        return feature[:, : self.part_dim], feature[:, self.part_dim :]

    def _encode(self, palm, vein):
        with torch.no_grad():
            return self._encode_one(self.palm_encoder, palm), self._encode_one(self.vein_encoder, vein)

    def _compose(self, shared, specific):
        return torch.cat([shared, specific], dim=1)

    def _scenario_from_mask(self, mask, scenario):
        if mask is None:
            return scenario
        mask = mask.bool()
        palm_exists = mask[:, 0]
        vein_exists = mask[:, 1]
        if torch.all(palm_exists & vein_exists):
            return "complete"
        if torch.all(~palm_exists & vein_exists):
            return "palmprint_missing"
        if torch.all(palm_exists & ~vein_exists):
            return "palmvein_missing"
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
        if scenario == "complete":
            return feats["f_palm"], feats["f_vein"]
        if scenario == "palmprint_missing":
            return feats["hat_palm"], feats["f_vein"]
        if scenario == "palmvein_missing":
            return feats["f_palm"], feats["hat_vein"]
        raise ValueError(f"Unsupported scenario: {scenario}")

    def _teacher_logits(self, teacher, feature, labels):
        with torch.no_grad():
            raw_logits = teacher(feature)
            logits = teacher(feature, labels) if labels is not None else raw_logits
        return logits, raw_logits

    def _missing_logits(self, scenario, feats, fusion_logits, fusion_logits_raw, labels):
        if scenario == "palmprint_missing" and self.vein_teacher is not None:
            teacher_logits, teacher_logits_raw = self._teacher_logits(self.vein_teacher, feats["f_vein"], labels)
            alpha = torch.sigmoid(self.palm_missing_gate)
        elif scenario == "palmvein_missing" and self.palm_teacher is not None:
            teacher_logits, teacher_logits_raw = self._teacher_logits(self.palm_teacher, feats["f_palm"], labels)
            alpha = torch.sigmoid(self.vein_missing_gate)
        else:
            return fusion_logits, {"fusion_logits": fusion_logits}

        return (1.0 - alpha) * teacher_logits + alpha * fusion_logits, {
            "fusion_logits": fusion_logits,
            "fusion_logits_raw": fusion_logits_raw,
            "teacher_logits": teacher_logits,
            "teacher_logits_raw": teacher_logits_raw,
            "gate_alpha": alpha,
        }

    def forward(self, palm, vein, labels=None, scenario="complete", mask=None):
        (palm_shared, palm_specific), (vein_shared, vein_specific) = self._encode(palm, vein)
        hat_vein_specific = self.p2v(palm_specific)
        hat_palm_specific = self.v2p(vein_specific)
        feats = {
            "f_palm": self._compose(palm_shared, palm_specific),
            "f_vein": self._compose(vein_shared, vein_specific),
            "hat_palm": self._compose(vein_shared, hat_palm_specific),
            "hat_vein": self._compose(palm_shared, hat_vein_specific),
        }

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
            "palm_shared": palm_shared,
            "palm_specific": palm_specific,
            "vein_shared": vein_shared,
            "vein_specific": vein_specific,
            "hat_palm_specific": hat_palm_specific,
            "hat_vein_specific": hat_vein_specific,
            **feats,
        }
