# Hetero-MMRNet

This project trains palmprint and palm-vein encoders, then uses them as fixed teachers for missing-modality recognition.

## Research Route

The current main route is intentionally simple:

1. Train a palm encoder with its ArcFace head.
2. Train a vein encoder with its ArcFace head.
3. Freeze both single-modality systems.
4. Train the missing-modality model with:
   - two conditional feature-diffusion recoverers (`palm -> vein` and `vein -> palm`),
   - available-guided fusion,
   - logit-level gated ensemble,
   - distillation to the available single-modality teacher.
5. Freeze the trained diffusion recoverers and fine-tune fusion with the same full DDIM samples used at test time.

The diffusion models operate on the frozen encoders' 2D feature maps. The first training stage uses the DDPM
noise-prediction objective plus feature reconstruction and identity-aware recognition losses. The second stage
freezes both recoverers, generates missing features with the same DDIM sampler used for inference, and aligns the
complete and missing-scenario embeddings. The missing model does not fine-tune encoders, so the available-modality
baseline stays fixed.

## Protocol

Protocol format:

```text
palm_path vein_path label palm_exists vein_exists split
```

Generate all four paired identity-disjoint closed-set protocols with the fixed seed `2026`:

```bash
python -m utils.datasets_txt
```

Palm identities are split between training and testing with the original per-dataset train/test ratio. All paired
samples from training identities are used to train the encoders and missing-modality model. Held-out test identities
are split into a complete-modality Gallery and Probes using the original per-identity ratio:

| Dataset | Train / test identities | Train pairs | Gallery / Probe pairs per test identity | Total Gallery / Probes |
| --- | ---: | ---: | ---: | ---: |
| Tongji Session 1 -> 2 | 480 / 120 | 4800 | 8 / 2 | 960 / 240 |
| CUMT | 232 / 58 | 2320 | 8 / 2 | 464 / 116 |
| PolyU | 417 / 83 | 5004 | 10 / 2 | 830 / 166 |
| CASIA | 133 / 67 | 798 | 4 / 2 | 268 / 134 |

Each palmprint-palm-vein pair stays in the same split, and training identities never occur in the Gallery or Probe
sets. Every Probe identity is enrolled in the held-out Gallery, so evaluation remains closed-set identification while
testing generalization to unseen identities. Tongji trains on Session 1, builds the held-out test-identity Gallery
from eight Session 1 pairs, and uses two Session 2 pairs as Probes. CUMT, PolyU, and CASIA keep their previous
within-dataset protocols and reserve one Probe pair from each acquisition half/session.

Generated files:

```text
data_txt/<dataset_name>/ssfd_train_full.txt
data_txt/<dataset_name>/ssfd_gallery_full.txt
data_txt/<dataset_name>/ssfd_test_protocol.txt
```

The Gallery file always contains complete modalities. The Probe protocol contains `complete`, `palmprint_missing`,
and `palmvein_missing`. Training and testing default to the Tongji files under `data_txt/tongji/`; use explicit paths
to run another dataset.

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

This runs 200 epochs of diffusion training and then 40 epochs of sampled-feature fusion fine-tuning. The first-stage
checkpoint is saved to `outputs/missing_model/diffusion_best.pth`; the final checkpoint is saved to
`outputs/missing_model/best.pth`. The default diffusion configuration uses 100 training steps and 20 DDIM sampling
steps. It can be changed with `--diffusion_steps` and `--ddim_steps`.

To fine-tune fusion directly from an existing diffusion checkpoint without retraining the recoverers:

```bash
python train_missing_model.py \
  --stage fusion \
  --diffusion_ckpt outputs/missing_model/best.pth \
  --save_path outputs/missing_model/best_sampled.pth
```

The fusion stage performs full DDIM sampling online under `no_grad`, so it is slower than ordinary classifier
fine-tuning but does not backpropagate through the sampling trajectory. Use a different output path when reusing an
existing checkpoint so that the source checkpoint remains available for comparison.

## Test

```bash
python test_encoder.py --modality palm --ckpt outputs/encoders/palm_best.pth
python test_encoder.py --modality vein --ckpt outputs/encoders/vein_best.pth
python test_missing_model.py --ckpt outputs/missing_model/best.pth
```

Evaluation builds one complete-modality Gallery template per held-out test identity by averaging its Gallery
embeddings. Probe samples are matched to the L2-normalized Gallery templates with cosine similarity. Each modality
condition reports:

- RR (Rank-1 recognition rate)
- EER
- TAR at FAR = `1e-4` and `1e-5`
- Top-1 and Top-5 Accuracy

Use `--top_k` or `--far_points` to change the reported operating points.

## File Map

| File | Role |
| --- | --- |
| `train_encoder.py` | Trains one single-modality encoder and ArcFace head. |
| `test_encoder.py` | Evaluates a saved palm or vein baseline checkpoint. |
| `train_missing_model.py` | Trains the missing-modality recognizer from frozen palm/vein checkpoints. |
| `test_missing_model.py` | Evaluates the missing-modality recognizer on all protocol splits. |
| `models/backbones.py` | Defines the ResNet18 encoder and identity shared/specific split heads. |
| `models/feature_diffusion.py` | Implements the shared conditional U-Net, DDPM training, and DDIM sampling. |
| `models/missing_model.py` | Connects bidirectional feature diffusion to fusion and the gated teacher ensemble. |
| `utils/datasets_txt.py` | Generates protocols and loads paired or single-modality samples. |
| `utils/preprocess.py` | Defines palm and vein image transforms. |
| `utils/head.py` | Defines the ArcFace classification head. |
| `utils/checkpoint.py` | Builds encoders and ArcFace heads from checkpoints. |
| `utils/checkpoint_io.py` | Provides the single checkpoint load/save implementation. |
| `utils/evaluation.py` | Builds Gallery templates and computes RR, EER, TAR@FAR, and Top-k Accuracy. |
| `utils/runtime.py` | Provides shared device, learning-rate, and DataLoader helpers. |
| `utils/scenarios.py` | Defines the shared missing-modality scenario names. |

Local data, generated protocols, pretrained weights, checkpoints, and logs stay under ignored directories such as `data/`, `data_txt/`, `pretrained/`, `outputs/`, and `runs/`.

The previous deterministic MLP recovery checkpoints are not compatible with the diffusion architecture and must
not be used for evaluation. Checkpoints trained with the previous identity-overlapping protocol must also not be
reused because they have already seen the new held-out test identities; regenerate the protocols and retrain both
encoders and the missing model.
