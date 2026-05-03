import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeLoss(nn.Module):
    """DyMo-style prototype loss on normalized embeddings."""

    def __init__(self, temperature: float = 0.1, metric_name: str = "cosine_similarity") -> None:
        super().__init__()
        if metric_name not in {"cosine_similarity", "squared_euclidean"}:
            raise ValueError(f"Unsupported metric: {metric_name}")
        self.temperature = temperature
        self.metric_name = metric_name

    def forward(self, labels_one_hot: torch.Tensor, prototypes: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        if self.metric_name == "cosine_similarity":
            logits = torch.mm(feat, prototypes.t()) / self.temperature
        else:
            feat = feat.unsqueeze(1)
            prototypes = prototypes.unsqueeze(0)
            logits = -((feat - prototypes).pow(2).sum(dim=2)) / self.temperature

        log_prob = F.log_softmax(logits, dim=1)
        loss = -(log_prob * labels_one_hot).sum(dim=1)
        return loss.mean()
