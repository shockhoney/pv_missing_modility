import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
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
    def __init__(self, dim: int, reduction: int = 4):
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

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor):
        logits = self.attention(torch.cat([feat_a, feat_b], dim=1))
        weights = F.softmax(logits.view(feat_a.size(0), 2, -1), dim=1)
        return weights[:, 0] * feat_a + weights[:, 1] * feat_b


class Stage2Fusion(nn.Module):
    def __init__(
        self,
        in_dim_global: int = 256,
        out_dim_final: int = 512,
        final_l2norm: bool = True,
    ):
        super().__init__()
        self.final_l2norm = final_l2norm
        self.global_cross_attn = CrossModalAttention(in_dim_global, num_heads=8)
        self.global_channel_fusion = ChannelAttentionFusion(in_dim_global, reduction=4)
        self.proj = nn.Linear(in_dim_global, out_dim_final)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.out_dim = out_dim_final

    def forward(self, palm_feat: torch.Tensor, vein_feat: torch.Tensor):
        palm_enhanced, vein_enhanced = self.global_cross_attn(palm_feat, vein_feat)
        fused_feat = self.global_channel_fusion(palm_enhanced, vein_enhanced)
        fused_feat = self.proj(fused_feat)
        if self.final_l2norm:
            fused_feat = F.normalize(fused_feat, dim=1)
        return fused_feat
