import torch


def recognition_rate(logits, labels):
    logits = torch.as_tensor(logits)
    labels = torch.as_tensor(labels)
    if labels.numel() == 0:
        return 0.0
    return (logits.argmax(1) == labels).float().mean().item()
