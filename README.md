# Hetero-MMRNet

This project trains palmprint and palm-vein encoders for missing-modality recognition.

- Palm encoder baseline: TorchVision ResNet50
- Vein encoder baseline: ConvNeXt V2-Tiny
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
cd utils
python datasets_txt.py --root_dir data --output_dir data_txt/<dataset_name>
```

Use `--palm_dir_name` and `--vein_dir_name` when a dataset uses different subfolder names.

Generated files:

```text
data_txt/<dataset_name>/train_full.txt
data_txt/<dataset_name>/val_full.txt
data_txt/<dataset_name>/val_missing_fixed.txt
data_txt/<dataset_name>/test_missing_protocol.txt
```

## Train

Download pretrained weights first:

```powershell
New-Item -ItemType Directory -Force pretrained
Invoke-WebRequest -Uri "https://download.pytorch.org/models/resnet50-11ad3fa6.pth" -OutFile "pretrained/resnet50_imagenet1k_v2.pth"
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_tiny_22k_224_ema.pt" -OutFile "pretrained/convnextv2_tiny_22k_224_ema.pt"
```

Single-modality baselines:

```bash
python train_encoder.py --modality palm
python train_encoder.py --modality vein
```

## Test

```bash
python test_encoder.py ^
  --protocol_list data_txt/polyu/test_missing_protocol.txt ^
  --palm_ckpt outputs/encoders/palm_best.pth ^
  --vein_ckpt outputs/encoders/vein_best.pth
```

## Main Files

- `models/backbones.py`: ResNet50 and ConvNeXt V2-Tiny encoders
- `utils/datasets_txt.py`: protocol generation and datasets
- `train_encoder.py`: encoder training
- `test_encoder.py`: encoder evaluation
