from models.backbones import build_encoder
from utils.checkpoint_io import safe_torch_load


def _encoder_from_checkpoint(ckpt, modality, embedding_size, device):
    encoder = build_encoder(modality, input_channel=3, embedding_size=embedding_size).to(device)
    state = ckpt.get("encoder", ckpt.get("model", ckpt))
    state = {key: value for key, value in state.items() if not key.startswith(("shared_head.", "specific_head."))}
    encoder.load_state_dict(state, strict=False)
    encoder.eval()
    return encoder


def load_encoder_from_checkpoint(ckpt_path, modality, embedding_size, device):
    checkpoint = safe_torch_load(ckpt_path, device)
    return _encoder_from_checkpoint(checkpoint, modality, embedding_size, device)
