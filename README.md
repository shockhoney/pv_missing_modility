# Hetero-MMRNet

This project trains palmprint and palm-vein encoders, then uses them as fixed teachers for missing-modality recognition.

## Residual Image-Recovery Route

The strongest missing-modality route keeps both pretrained single-modality encoders frozen.  Two lightweight residual
image adapters recover a palm-style image from an available vein image and a vein-style image from an available palm
image.  Gallery images are passed through the same adapters, so recovered probes are compared with recovered Gallery
templates rather than with real images from a different feature domain.  The final score is

```text
available-modality cosine score + alpha * recovery-domain cosine score
```

The two `alpha` values are selected on 48 validation identities held out from recovery learning, using disjoint
Session 1 Gallery and Probe pairs. Test identities are never used for training or weight selection. The evaluator
also reports available-only, recovery-only, real-missing oracle, and identity-shuffled recovery controls.  A useful
recovery model must beat available-only recognition and deteriorate when its recovered probes are assigned to the
wrong identities.

Train and evaluate this route with:

```bash
python train_image_recovery.py
python test_image_recovery.py --ckpt outputs/image_recovery/best.pth
```

Export generated images, target images, and signed residual visualizations with:

```bash
python export_recovery_images.py
```

The palm and vein checkpoints under `outputs/encoders/` are read-only teachers in this route.  The saved recovery
checkpoint records their SHA-256 fingerprints, the held-out validation identities, and the selected fusion weights.

## Feature-Diffusion Research Route

The training route is staged so that every later target space has already been trained:

1. Train the palm and vein encoders with their own ArcFace heads.
2. Freeze both backbones and train their existing shared/specific heads on paired samples. The loss combines
   modality-specific identity classification, shared-only identity supervision through the existing common ArcFace
   classifier, shared-feature cosine alignment, normalized shared/specific orthogonality, and cross-modal
   specific-feature separation. A relative norm-balance term prevents the shared branch from becoming negligible
   compared with the specific branch.
3. Freeze the encoders and train the complete-modality fusion path plus its ArcFace classifier.
4. Train the two conditional feature-diffusion recoverers (`palm -> vein` and `vein -> palm`) with DDPM noise
   prediction.
5. Fine-tune both recoverers through the exact DDIM sampling trajectory used at test time.
6. Freeze the complete fusion target, classifier, and recoverers, then train only the missing-modality
   available-guided fusion residual. Its main ArcFace loss is computed directly from the final fusion embedding;
   the available single-modality teacher is used only for auxiliary logit distillation.

Complete-modality fusion consumes the encoders' 2D feature maps. Each map is flattened to spatial tokens, followed by
bidirectional token-to-token `nn.MultiheadAttention`, residual normalization, token pooling, channel attention, and
projection. The diffusion models reuse the same frozen 2D maps. DDIM recovery training starts from Gaussian noise and
applies map reconstruction, embedding cosine, and target-teacher identity losses to the final samples. The final
missing-fusion stage calls the same frozen sampler without retaining its sampling graph. Training and inference share
the same sampling start distribution, timestep schedule, update equations, and number of DDIM steps.

## Protocol

Protocol format:

```text
palm_path vein_path label palm_exists vein_exists split
```

Generate all four paired identity-disjoint closed-set protocols with the fixed seed `2026`:

```bash
python -m utils.datasets_txt
```

Tongji uses only Session 1 and is split into identity-disjoint train/validation/test sets. All paired samples from
training identities are used to train the encoders and missing-modality model. Validation and test identities are
each split into a complete-modality Gallery and Probes. Other datasets retain their original train/test ratios:

| Dataset | Train / validation / test identities | Train pairs | Validation Gallery / Probes | Test Gallery / Probes |
| --- | ---: | ---: | ---: | ---: |
| Tongji Session 1 only | 432 / 48 / 120 | 4320 | 384 / 96 | 960 / 240 |
| CUMT | 232 / - / 58 | 2320 | - | 464 / 116 |
| PolyU | 417 / - / 83 | 5004 | - | 830 / 166 |
| CASIA | 133 / - / 67 | 798 | - | 268 / 134 |

Each palmprint-palm-vein pair stays in one identity split, and the Tongji training, validation, and test identity sets
are mutually disjoint. Every Probe identity is enrolled in the corresponding held-out Gallery, so validation and
testing are closed-set identification while measuring generalization to unseen identities. For Tongji, both Gallery
and Probe images come strictly from Session 1: eight pairs per identity form the Gallery and two disjoint pairs form
the Probe. CUMT, PolyU, and CASIA keep their previous within-dataset protocols.

Generated files:

```text
data_txt/<dataset_name>/ssfd_train_full.txt
data_txt/tongji/ssfd_val_gallery_full.txt
data_txt/tongji/ssfd_val_protocol.txt
data_txt/<dataset_name>/ssfd_gallery_full.txt
data_txt/<dataset_name>/ssfd_test_protocol.txt
```

Tongji additionally writes `train_identities.txt`, `val_identities.txt`, `test_identities.txt`, and `manifest.json`.
The generator validates the 432/48/120 counts, strict identity non-overlap, Session1-only paths, paired samples,
Gallery/Probe non-overlap, scenario completeness, and file hashes before reporting success.

The Gallery file always contains complete modalities. The Probe protocol contains `complete`, `palmprint_missing`,
and `palmvein_missing`. Training and testing default to the Tongji files under `data_txt/tongji/`; use explicit paths
to run another dataset.

## Train

The palmprint ResNet18 encoder is trained from random initialization. The palm-vein encoder loads ImageNet ResNet18
weights from `pretrained/resnet18_imagenet1k_v1.pth` by default. Train both baselines with seed `42`:

```bash
python train_encoder.py --modality palm --seed 42
python train_encoder.py --modality vein --seed 42
```

Train the missing-modality recognizer:

```bash
python train_missing_model.py
```

This runs 30 epochs of paired feature alignment, 40 epochs of complete-modality fusion, 200 epochs of DDPM
pretraining, 30 epochs of differentiable DDIM recovery fine-tuning, and 40 epochs of missing-modality fusion. The five
checkpoints are saved as `alignment_best.pth`, `complete_fusion_best.pth`, `diffusion_best.pth`, `recovery_best.pth`,
and `best.pth` under `outputs/missing_model/`. Each stage reloads the preceding stage's best checkpoint. The default
configuration uses 100 diffusion steps and 5 DDIM sampling steps; evaluation reads the saved value.

Old missing-model checkpoints use vector head-gating attention and Logit gates, so they are intentionally rejected.
Palm and vein encoder checkpoints produced by the current code can be reused, but all five missing-model stages must
be retrained. Encoder checkpoints made before protocol fingerprints were recorded are rejected by default; after
manually confirming that their label mapping still matches `--train_list`, they can be explicitly accepted with
`--allow_legacy_encoder_ckpt`.
Each new stage checkpoint records the training-protocol and encoder-checkpoint SHA-256 fingerprints. Resumed stages
also verify immutable spatial-attention, ArcFace, and diffusion/sampling settings, so a changed label mapping,
`attn_heads`, or `ddim_steps` cannot be loaded silently. Learning rates, batch sizes, and stage epoch counts remain
independently configurable.

To run only paired decomposition alignment from the two encoder checkpoints:

```bash
python train_missing_model.py --stage alignment
```

To train complete spatial fusion from the best alignment checkpoint:

```bash
python train_missing_model.py \
  --stage complete_fusion \
  --alignment_ckpt outputs/missing_model/alignment_best.pth
```

To train DDPM diffusion from the trained complete-fusion checkpoint:

```bash
python train_missing_model.py \
  --stage diffusion \
  --complete_fusion_ckpt outputs/missing_model/complete_fusion_best.pth
```

To resume only the DDIM recovery stage from a completed DDPM checkpoint:

```bash
python train_missing_model.py \
  --stage recovery \
  --diffusion_ckpt outputs/missing_model/diffusion_best.pth \
  --recovery_ckpt outputs/missing_model/recovery_best.pth
```

To train only fusion from a completed recovery checkpoint:

```bash
python train_missing_model.py \
  --stage fusion \
  --recovery_ckpt outputs/missing_model/recovery_best.pth \
  --save_path outputs/missing_model/best.pth
```

The recovery stage backpropagates through the full DDIM trajectory and is therefore the most memory-intensive stage;
the default batch size is 8. The final fusion stage freezes both recoverers, the complete-fusion path, and its
classifier. Gradients still pass through the fixed classifier into the final embedding while only the zero-initialized
available-guided residual is optimized. Training and testing use the fixed default seed `42`.

## Test

```bash
python test_encoder.py --modality palm --ckpt outputs/encoders/palm_best.pth
python test_encoder.py --modality vein --ckpt outputs/encoders/vein_best.pth
python test_missing_model.py --ckpt outputs/missing_model/best.pth
```

Run the missing-modality diagnostics with the same DDIM samples used by the normal evaluation:

```bash
python test_missing_model.py \
  --ckpt outputs/missing_model/best.pth \
  --seed 42 \
  --diagnostics
```

For each missing scenario, diagnostics report the real available modality against the complete-fusion Gallery and
the pure diffusion-recovered target modality against its corresponding real single-modality Gallery. This separates
cross-scenario embedding mismatch from poor diffusion recovery without repeating DDIM sampling.

Evaluation builds one complete-modality Gallery template per held-out test identity by averaging its Gallery
embeddings. Probe samples are matched to the L2-normalized Gallery templates with cosine similarity. Each modality
condition reports:

- EER
- TAR at FAR = `1e-3` and `1e-4`
- Top-1 and Top-5 Accuracy

Rank-1 recognition rate is identical to Top-1 in this closed-set protocol and is therefore not reported separately.
The evaluator prints both the count resolution (`1 / number_of_impostor_scores`) and the minimum positive FAR
supported by the actual threshold curve (which can be coarser when scores tie). It warns when a requested FAR is below
the latter and does not interpolate an unsupported low-FAR operating point. Use `--top_k` or `--far_points` to change
the reported operating points.

## File Map

| File | Role |
| --- | --- |
| `train_encoder.py` | Trains one single-modality encoder and ArcFace head. |
| `test_encoder.py` | Evaluates a saved palm or vein baseline checkpoint. |
| `train_image_recovery.py` | Trains residual image recoverers and selects recovery-domain fusion weights on held-out identities. |
| `test_image_recovery.py` | Evaluates missing-modality recovery with score fusion and shuffled/oracle ablations. |
| `models/image_recovery.py` | Defines bidirectional residual image adapters used in the recommended route. |
| `train_missing_model.py` | Trains the missing-modality recognizer from frozen palm/vein checkpoints. |
| `test_missing_model.py` | Evaluates the missing-modality recognizer on all protocol splits. |
| `models/backbones.py` | Defines the ResNet18 encoder and trainable shared/specific projection heads. |
| `models/feature_diffusion.py` | Implements the shared conditional U-Net, DDPM training, and DDIM sampling. |
| `models/missing_model.py` | Connects spatial complete fusion, bidirectional feature diffusion, missing fusion, and teacher distillation. |
| `utils/datasets_txt.py` | Generates protocols and loads paired or single-modality samples. |
| `utils/preprocess.py` | Defines palm and vein image transforms. |
| `utils/head.py` | Defines the ArcFace classification head. |
| `utils/checkpoint.py` | Builds encoders and ArcFace heads from checkpoints. |
| `utils/checkpoint_io.py` | Provides the single checkpoint load/save implementation. |
| `utils/evaluation.py` | Builds Gallery templates and computes EER, empirical TAR@FAR, and Top-k Accuracy. |
| `utils/runtime.py` | Provides shared device, learning-rate, and DataLoader helpers. |
| `utils/scenarios.py` | Defines the shared missing-modality scenario names. |

Local data, generated protocols, pretrained weights, checkpoints, and logs stay under ignored directories such as `data/`, `data_txt/`, `pretrained/`, `outputs/`, and `runs/`.

The previous deterministic MLP recovery checkpoints are not compatible with the diffusion architecture and must
not be used for evaluation. Checkpoints trained with the previous identity-overlapping protocol must also not be
reused because they have already seen the new held-out test identities; regenerate the protocols and retrain both
encoders and the missing model.
