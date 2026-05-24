# Repository Guidelines

## Core Direction

Keep the project focused on one reproducible route:

1. Train palm and vein single-modality encoders.
2. Freeze those encoders and ArcFace heads.
3. Train the missing-modality recognizer with cross-modal recovery, available-guided fusion, gated logit ensemble, and teacher distillation.

Do not add encoder fine-tuning, graph contrastive learning, image generation, or large architectural rewrites unless explicitly requested.

## Project Structure

- `train_encoder.py`: trains a `palm` or `vein` encoder with ArcFace.
- `test_encoder.py`: evaluates a saved single-modality checkpoint.
- `train_missing_model.py`: trains the missing-modality recognizer from frozen single-modality checkpoints.
- `test_missing_model.py`: evaluates `complete`, `palmprint_missing`, and `palmvein_missing`.
- `models/backbones.py`: ResNet18 encoder and identity shared/specific split heads.
- `models/missing_model.py`: CMFT, fusion, gated teacher ensemble, and loss helpers.
- `utils/`: dataset/protocol, preprocessing, checkpoint, ArcFace, and metric helpers.
- `tests/`: lightweight `unittest` coverage.

Ignored local folders include `data/`, `data_txt/`, `pretrained/`, `outputs/`, and `runs/`.

## Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate protocols:

```bash
python utils/datasets_txt.py --dataset cumt --root_dir data/CUMT --output_dir data_txt/cumt
```

Train:

```bash
python train_encoder.py --modality palm
python train_encoder.py --modality vein
python train_missing_model.py
```

Evaluate:

```bash
python test_encoder.py --modality palm --ckpt outputs/encoders/palm_best.pth
python test_encoder.py --modality vein --ckpt outputs/encoders/vein_best.pth
python test_missing_model.py --ckpt outputs/missing_model/best.pth
```

Test:

```bash
python -m unittest discover -s tests
```

## Coding Rules

- Write the smallest correct solution.
- Keep changes localized.
- Prefer existing helpers over new abstractions.
- Use clear `snake_case` names and 4-space indentation.
- Do not refactor unrelated code.
- Do not add dependencies unless necessary.
- Keep CLI options minimal and argparse-based.
- Do not commit datasets, pretrained weights, checkpoints, logs, or local paths.
