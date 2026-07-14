from models.backbones import build_encoder
from utils.head import ArcFace
from utils.checkpoint_io import safe_torch_load, save_checkpoint


def load_encoder_from_checkpoint(ckpt_path, modality, embedding_size, device):
    ckpt = safe_torch_load(ckpt_path, device)
    encoder = build_encoder(modality, input_channel=3, embedding_size=embedding_size).to(device)
    state = ckpt.get("encoder", ckpt.get("model", ckpt))
    state = {key: value for key, value in state.items() if not key.startswith(("shared_head.", "specific_head."))}
    encoder.load_state_dict(state, strict=False)
    encoder.eval()
    return encoder


def _arcface_from_checkpoint(ckpt, embedding_size, device):
    state = ckpt.get("classifier")
    if state is None:
        raise ValueError("Checkpoint does not contain an ArcFace classifier.")
    if state["weight"].shape[1] != embedding_size:
        raise ValueError("Classifier embedding dimension does not match the requested encoder dimension.")

    args = ckpt.get("args", {})
    classifier = ArcFace(
        embedding_size,
        ckpt.get("num_classes", state["weight"].shape[0]),
        args.get("arcface_s", 32.0),
        args.get("arcface_m", 0.25),
    ).to(device)
    classifier.load_state_dict(state)
    classifier.eval()
    return classifier


def load_arcface_from_checkpoint(ckpt_path, embedding_size, device):
    ckpt = safe_torch_load(ckpt_path, device)
    classifier = _arcface_from_checkpoint(ckpt, embedding_size, device)
    for param in classifier.parameters():
        param.requires_grad = False
    return classifier
