import torch

from models.backbones import build_encoder
from utils.head import ArcFace


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_encoder_from_checkpoint(ckpt_path, modality, input_size, embedding_size, device):
    ckpt = safe_torch_load(ckpt_path, device)
    encoder = build_encoder(modality, input_channel=3, input_size=input_size, embedding_size=embedding_size).to(device)
    state = ckpt.get("encoder", ckpt.get("model", ckpt))
    state = {key: value for key, value in state.items() if not key.startswith(("shared_head.", "specific_head."))}
    encoder.load_state_dict(state, strict=False)
    encoder.eval()
    return encoder


def load_arcface_from_checkpoint(ckpt_path, embedding_size, device):
    ckpt = safe_torch_load(ckpt_path, device)
    state = ckpt.get("classifier")
    if state is None:
        raise ValueError(f"Checkpoint does not contain an ArcFace classifier: {ckpt_path}")
    if state["weight"].shape[1] != embedding_size:
        raise ValueError(f"Classifier dim mismatch in {ckpt_path}")

    args = ckpt.get("args", {})
    classifier = ArcFace(
        embedding_size,
        ckpt.get("num_classes", state["weight"].shape[0]),
        args.get("arcface_s", 32.0),
        args.get("arcface_m", 0.25),
    ).to(device)
    classifier.load_state_dict(state)
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False
    return classifier
