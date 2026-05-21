# Hetero-MMRNet

This project trains palmprint and palm-vein encoders for missing-modality recognition.

- Palm encoder baseline: TorchVision ResNet18
- Vein encoder baseline: TorchVision ResNet18
- Training head: ArcFace
- Test stage: normalized embedding with cosine similarity

This stage trains only strong single-modality baselines. Joint alignment is not used.


## Protocol

Protocol format:

```text
palm_path vein_path label palm_exists vein_exists split
```

Generate protocol files from a dataset root that contains palm and vein subfolders:

```bash
python utils/datasets_txt.py --protocol closed --root_dir data/PolyU --output_dir data_txt/polyu
```

Use `--palm_dir_name` and `--vein_dir_name` when a dataset uses different subfolder names.

Generated files:

```text
data_txt/<dataset_name>/closed_train_full.txt
data_txt/<dataset_name>/closed_val_full.txt
data_txt/<dataset_name>/closed_test_protocol.txt
```

## Train

Download pretrained weights first:

```powershell
New-Item -ItemType Directory -Force pretrained
Invoke-WebRequest -Uri "https://download.pytorch.org/models/resnet18-f37072fd.pth" -OutFile "pretrained/resnet18_imagenet1k_v1.pth"
```

Single-modality baselines:

```bash
python train_encoder.py --modality palm
python train_encoder.py --modality vein
```

## Test

```bash
python test_encoder.py --modality palm --ckpt outputs/encoders/palm_best.pth
python test_encoder.py --modality vein --ckpt outputs/encoders/vein_best.pth
```

## Main Files

- `models/backbones.py`: ResNet18 encoder
- `utils/datasets_txt.py`: protocol generation and datasets
- `train_encoder.py`: encoder training
- `test_encoder.py`: encoder evaluation
