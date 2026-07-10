import torch
import torch.nn.functional as F


def recognition_rate(logits, labels):
    logits = torch.as_tensor(logits)
    labels = torch.as_tensor(labels)
    if labels.numel() == 0:
        return 0.0
    return (logits.argmax(1) == labels).float().mean().item()


def eer_from_embeddings(embeddings, labels):
    embeddings = F.normalize(torch.as_tensor(embeddings, dtype=torch.float32), dim=1)
    labels = torch.as_tensor(labels)
    if labels.numel() < 2:
        return float("nan")

    scores = embeddings @ embeddings.t()
    pair_mask = torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=1)
    same = labels[:, None].eq(labels[None, :])[pair_mask]
    scores = scores[pair_mask]
    positives = same.sum().item()
    negatives = same.numel() - positives
    if positives == 0 or negatives == 0:
        return float("nan")

    order = scores.argsort(descending=True)
    same = same[order]
    tp = same.float().cumsum(0)
    fp = (~same).float().cumsum(0)
    far = fp / negatives
    frr = 1.0 - tp / positives
    idx = (far - frr).abs().argmin()
    return ((far[idx] + frr[idx]) * 0.5).item()
