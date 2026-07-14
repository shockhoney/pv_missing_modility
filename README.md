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

The diffusion models operate on the frozen encoders' 2D feature maps. Training uses the DDPM noise-prediction
objective plus feature reconstruction and identity-aware recognition losses; inference uses DDIM sampling. The
missing model does not fine-tune encoders, so the available-modality baseline stays fixed.

## Protocol

Protocol format:

```text
palm_path vein_path label palm_exists vein_exists split
```

Generate all four paired closed-set protocols with the fixed seed `2026`:

```bash
python -m utils.datasets_txt
```

The generated splits are:

| Dataset | Modalities | Train pairs / identity | Test pairs / identity | Total train / test pairs |
| --- | --- | ---: | ---: | ---: |
| Tongji Session 1 | Palmprint / palm vein | 8 | 2 | 4800 / 1200 |
| CUMT | Palmprint / palm vein | 8 | 2 | 2320 / 580 |
| PolyU | Green / NIR | 10 | 2 | 5000 / 1000 |
| CASIA | VI / IR | 4 | 2 | 800 / 400 |

Each palmprint-palm-vein pair stays in the same split. Train and test identities overlap, so this is a closed-set
protocol. Tongji uses only Session 1 and must not be described as cross-session evaluation. CUMT, PolyU, and CASIA
reserve one test pair from each acquisition half/session.

Generated files:

```text
data_txt/<dataset_name>/ssfd_train_full.txt
data_txt/<dataset_name>/ssfd_test_protocol.txt
```

The test protocol contains `complete`, `palmprint_missing`, and `palmvein_missing`. Training and testing default to
the Tongji files under `data_txt/tongji/`; use explicit paths to run another dataset.

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

The default diffusion configuration uses 100 training steps and 20 DDIM sampling steps. It can be changed with
`--diffusion_steps` and `--ddim_steps`.

## Test

```bash
python test_encoder.py --modality palm --ckpt outputs/encoders/palm_best.pth
python test_encoder.py --modality vein --ckpt outputs/encoders/vein_best.pth
python test_missing_model.py --ckpt outputs/missing_model/best.pth
```

Evaluation builds one complete-modality Gallery template per identity by averaging its training embeddings. Test
samples are Probes and are matched to the L2-normalized Gallery templates with cosine similarity. Each modality
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
not be used for evaluation; train a new missing-model checkpoint after this change.
