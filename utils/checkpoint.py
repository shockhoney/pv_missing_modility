import torch

from models.backbones import build_encoder


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_encoder_from_checkpoint(ckpt_path, modality, input_size, embedding_size, device):
    ckpt = safe_torch_load(ckpt_path, device)
    encoder = build_encoder(modality, input_channel=3, input_size=input_size, embedding_size=embedding_size).to(device)
    encoder.load_state_dict(ckpt.get("encoder", ckpt.get("model", ckpt)), strict=False)
    encoder.eval()
    return encoder
