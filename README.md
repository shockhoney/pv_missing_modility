# Hetero-MMRNet

This project trains palmprint and palm-vein encoders for missing-modality recognition.

- Palm encoder: ResNet50 + UAA geometric adversarial augmentation
- Vein encoder: StarLKNet/LaKNet + StarMix
- Training head: ArcFace
- Test stage: normalized embedding with cosine similarity


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

```bash
python train_encoder.py ^
  --modality joint ^
  --train_full_list data_txt/polyu/train_full.txt ^
  --val_full_list data_txt/polyu/val_full.txt ^
  --save_dir outputs/encoders
```

Single branch:

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

- `models/backbones.py`: palm and vein encoders
- `utils/augmentations.py`: UAA geometry and StarMix
- `utils/datasets_txt.py`: protocol generation and datasets
- `train_encoder.py`: encoder training
- `test_encoder.py`: encoder evaluation
