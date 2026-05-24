# Hetero-MMRNet

This project trains palmprint and palm-vein encoders, then uses them as fixed teachers for missing-modality recognition.

## Research Route

The current main route is intentionally simple:

1. Train a palm encoder with its ArcFace head.
2. Train a vein encoder with its ArcFace head.
3. Freeze both single-modality systems.
4. Train the missing-modality model with:
   - cross-modal feature recovery,
   - available-guided fusion,
   - logit-level gated ensemble,
   - distillation to the available single-modality teacher.

The missing model does not fine-tune encoders by default. This keeps the available-modality baseline stable.

## Protocol

Protocol format:

```text
palm_path vein_path label palm_exists vein_exists split
```

Generate protocol files:

```bash
python utils/datasets_txt.py --dataset cumt --root_dir data/CUMT --output_dir data_txt/cumt
```

Generated files:

```text
data_txt/<dataset_name>/ssfd_train_full.txt
data_txt/<dataset_name>/ssfd_test_protocol.txt
```

The test protocol contains `complete`, `palmprint_missing`, and `palmvein_missing`.

## Train

Download ResNet18 pretrained weights first:

```powershell
New-Item -ItemType Directory -Force pretrained
Invoke-WebRequest -Uri "https://download.pytorch.org/models/resnet18-f37072fd.pth" -OutFile "pretrained/resnet18_imagenet1k_v1.pth"
```

Train single-modality baselines:

```bash
python train_encoder.py --modality palm
python train_encoder.py --modality vein
```

Train the missing-modality recognizer:

```bash
python train_missing_model.py
```

## Test

```bash
python test_encoder.py --modality palm --ckpt outputs/encoders/palm_best.pth
python test_encoder.py --modality vein --ckpt outputs/encoders/vein_best.pth
python test_missing_model.py --ckpt outputs/missing_model/best.pth
```

Run unit tests:

```bash
python -m unittest discover -s tests
```

## File Map

| File | Role |
| --- | --- |
| `train_encoder.py` | Trains one single-modality encoder and ArcFace head. |
| `test_encoder.py` | Evaluates a saved palm or vein baseline checkpoint. |
| `train_missing_model.py` | Trains the missing-modality recognizer from frozen palm/vein checkpoints. |
| `test_missing_model.py` | Evaluates the missing-modality recognizer on all protocol splits. |
| `models/backbones.py` | Defines the ResNet18 encoder and identity shared/specific split heads. |
| `models/missing_model.py` | Defines CMFT, fusion, gated teacher ensemble, and missing-model losses. |
| `utils/datasets_txt.py` | Generates protocols and loads paired or single-modality samples. |
| `utils/preprocess.py` | Defines palm and vein image transforms. |
| `utils/head.py` | Defines the ArcFace classification head. |
| `utils/checkpoint.py` | Loads encoders and ArcFace heads from checkpoints. |
| `utils/evaluation.py` | Computes closed-set recognition rate. |
| `tests/` | Unit tests for protocol generation, transforms, schedules, shapes, and losses. |

Local data, generated protocols, pretrained weights, checkpoints, and logs stay under ignored directories such as `data/`, `data_txt/`, `pretrained/`, `outputs/`, and `runs/`.
