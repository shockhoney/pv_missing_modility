# Hetero-MMRNet

This project trains palmprint and palm-vein encoders for missing-modality recognition.

- Palm encoder baseline: TorchVision ResNet18
- Vein encoder baseline: TorchVision ResNet18
- Training head: ArcFace
- Missing-modality model: SSFD-style cross-modal transformation + cross/channel attention fusion

Train single-modality baselines first, then train the missing-modality recognizer from their checkpoints.


## Protocol

Protocol format:

```text
palm_path vein_path label palm_exists vein_exists split
```

Generate SSFD-Net style protocol files:

```bash
python utils/datasets_txt.py --dataset casia --root_dir data/CASIA --output_dir data_txt/casia
python utils/datasets_txt.py --dataset cumt --root_dir data/CUMT --output_dir data_txt/cumt
python utils/datasets_txt.py --dataset tongji --root_dir data/tongji --output_dir data_txt/tongji
```

Generated files:

```text
data_txt/<dataset_name>/ssfd_train_full.txt
data_txt/<dataset_name>/ssfd_test_protocol.txt
```

The test protocol contains `complete`, `palmprint_missing`, and `palmvein_missing`.

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

Missing-modality recognizer:

```bash
python train_missing_model.py
```

Training saves `best.pth` by lowest training loss.

## Test

```bash
python test_encoder.py --modality palm --ckpt outputs/encoders/palm_best.pth
python test_encoder.py --modality vein --ckpt outputs/encoders/vein_best.pth
python test_missing_model.py --ckpt outputs/missing_model/best.pth
```

Reports closed-set recognition rate.

## Main Files

- `models/backbones.py`: ResNet18 encoder
- `utils/datasets_txt.py`: protocol generation and datasets
- `train_encoder.py`: encoder training
- `test_encoder.py`: encoder evaluation
- `train_missing_model.py`: missing-modality recognizer training
- `test_missing_model.py`: missing-modality recognizer evaluation
