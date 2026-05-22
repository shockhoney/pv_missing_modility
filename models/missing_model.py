import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.head import ArcFace


class SharedSpecificProjector(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("dim must be divisible by 2 for shared/specific split")
        part_dim = dim // 2
        self.shared = nn.Sequential(nn.Linear(dim, part_dim), nn.LayerNorm(part_dim))
        self.specific = nn.Sequential(nn.Linear(dim, part_dim), nn.LayerNorm(part_dim))

    def forward(self, x):
        return self.shared(x), self.specific(x)


class CrossModalTransformation(nn.Module):
    def __init__(self, dim=128, hidden=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
        )

    def forward(self, specific):
        return self.net(specific)


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


class CrossChannelFusion(nn.Module):
    def __init__(self, dim=256, heads=4, reduction=4):
        super().__init__()
        self.cross_attention = CrossModalAttention(dim, heads)
        self.channel_attention = ChannelAttentionFusion(dim, reduction)
        self.project = nn.Linear(dim, dim)

    def forward(self, palm_feat, vein_feat):
        palm_feat, vein_feat = self.cross_attention(palm_feat, vein_feat)
        fused = self.channel_attention(palm_feat, vein_feat)
        return F.normalize(self.project(fused), dim=1)


def transformation_loss(pred, target):
    return F.mse_loss(pred, target.detach())


def consistency_loss(pred, target):
    return (1.0 - F.cosine_similarity(pred, target.detach(), dim=1)).mean()


class MissingModalityRecognizer(nn.Module):
    def __init__(
        self,
        palm_encoder,
        vein_encoder,
        num_classes,
        dim=256,
        cmft_hidden=2048,
        heads=4,
        reduction=4,
        arcface_s=32.0,
        arcface_m=0.25,
        freeze_encoders=True,
    ):
        super().__init__()
        self.palm_encoder = palm_encoder
        self.vein_encoder = vein_encoder
        part_dim = dim // 2
        self.palm_projector = SharedSpecificProjector(dim)
        self.vein_projector = SharedSpecificProjector(dim)
        self.p2v = CrossModalTransformation(part_dim, cmft_hidden)
        self.v2p = CrossModalTransformation(part_dim, cmft_hidden)
        self.fusion = CrossChannelFusion(dim, heads, reduction)
        self.classifier = ArcFace(dim, num_classes, arcface_s, arcface_m)
        self.freeze_encoders = freeze_encoders
        if freeze_encoders:
            self._freeze_encoders()

    def _freeze_encoders(self):
        for encoder in (self.palm_encoder, self.vein_encoder):
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoders:
            self.palm_encoder.eval()
            self.vein_encoder.eval()
        return self

    def _encode(self, palm, vein):
        if self.freeze_encoders:
            with torch.no_grad():
                return self.palm_encoder(palm), self.vein_encoder(vein)
        return self.palm_encoder(palm), self.vein_encoder(vein)

    def _compose(self, shared, specific):
        return torch.cat([shared, specific], dim=1)

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

    def forward(self, palm, vein, labels=None, scenario="complete", mask=None):
        raw_palm, raw_vein = self._encode(palm, vein)
        palm_shared, palm_specific = self.palm_projector(raw_palm)
        vein_shared, vein_specific = self.vein_projector(raw_vein)
        hat_vein_specific = self.p2v(palm_specific)
        hat_palm_specific = self.v2p(vein_specific)
        feats = {
            "f_palm": self._compose(palm_shared, palm_specific),
            "f_vein": self._compose(vein_shared, vein_specific),
            "hat_palm": self._compose(vein_shared, hat_palm_specific),
            "hat_vein": self._compose(palm_shared, hat_vein_specific),
        }
        use_palm, use_vein = self._select_features(feats, scenario, mask)
        z = self.fusion(use_palm, use_vein)
        logits = self.classifier(z, labels) if labels is not None else self.classifier(z)
        return {
            "logits": logits,
            "z": z,
            "raw_palm": raw_palm,
            "raw_vein": raw_vein,
            "palm_shared": palm_shared,
            "palm_specific": palm_specific,
            "vein_shared": vein_shared,
            "vein_specific": vein_specific,
            "hat_palm_specific": hat_palm_specific,
            "hat_vein_specific": hat_vein_specific,
            **feats,
        }
