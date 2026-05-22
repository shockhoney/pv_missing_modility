import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.head import ArcFace


class FeatureRestorer(nn.Module):
    def __init__(self, dim=256, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)


class CrossChannelFusion(nn.Module):
    def __init__(self, dim=256, heads=4, reduction=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        fused_dim = dim * 2
        hidden = max(dim, fused_dim // reduction)
        self.channel_gate = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, fused_dim),
            nn.Sigmoid(),
        )
        self.project = nn.Sequential(
            nn.Linear(fused_dim, dim),
            nn.BatchNorm1d(dim),
        )

    def forward(self, palm_feat, vein_feat):
        tokens = torch.stack([palm_feat, vein_feat], dim=1)
        attended, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm(tokens + attended)
        fused = tokens.flatten(1)
        fused = fused * self.channel_gate(fused)
        return self.project(fused)


def recovery_loss(pred, target):
    return (1.0 - F.cosine_similarity(pred, target.detach(), dim=1)).mean()


def consistency_loss(pred, target):
    return (1.0 - F.cosine_similarity(pred, target.detach(), dim=1)).mean()


class MissingModalityRecognizer(nn.Module):
    def __init__(
        self,
        palm_encoder,
        vein_encoder,
        num_classes,
        dim=256,
        hidden=512,
        heads=4,
        reduction=4,
        arcface_s=32.0,
        arcface_m=0.25,
        freeze_encoders=True,
    ):
        super().__init__()
        self.palm_encoder = palm_encoder
        self.vein_encoder = vein_encoder
        self.p2v = FeatureRestorer(dim, hidden)
        self.v2p = FeatureRestorer(dim, hidden)
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

    def _select_features(self, f_palm, f_vein, hat_palm, hat_vein, scenario, mask):
        if mask is not None:
            mask = mask.to(f_palm.device).bool()
            palm_exists = mask[:, 0:1]
            vein_exists = mask[:, 1:2]
            return torch.where(palm_exists, f_palm, hat_palm), torch.where(vein_exists, f_vein, hat_vein)
        if scenario == "complete":
            return f_palm, f_vein
        if scenario == "palmprint_missing":
            return hat_palm, f_vein
        if scenario == "palmvein_missing":
            return f_palm, hat_vein
        raise ValueError(f"Unsupported scenario: {scenario}")

    def forward(self, palm, vein, labels=None, scenario="complete", mask=None):
        f_palm, f_vein = self._encode(palm, vein)
        hat_vein = self.p2v(f_palm)
        hat_palm = self.v2p(f_vein)
        use_palm, use_vein = self._select_features(f_palm, f_vein, hat_palm, hat_vein, scenario, mask)
        z = self.fusion(use_palm, use_vein)
        logits = self.classifier(z, labels) if labels is not None else self.classifier(z)
        return {
            "logits": logits,
            "z": z,
            "f_palm": f_palm,
            "f_vein": f_vein,
            "hat_palm": hat_palm,
            "hat_vein": hat_vein,
        }
