# Hetero-MMRNet

Hetero-MMRNet performs palmprint/palm-vein recognition when one modality is unavailable. The two trained
single-modality encoders remain frozen. The current route is feature-level, bidirectional probabilistic recovery; it
does not generate palmprint or palm-vein images.

## Method

Each frozen encoder produces a 256-dimensional embedding. A regularized CCA model initializes two conservative
256-to-192 shared-identity projectors. Zero-initialized residual refiners can be trained by backpropagation, but a
direction keeps its refiner only when its identity-disjoint validation EER strictly improves over the CCA
initialization.

For each available modality, a probabilistic recoverer maps the 192-dimensional shared feature to a conditional
256-dimensional target-modality mean and diagonal log-variance. Training uses Gaussian NLL, cosine reconstruction,
supervised contrastive identity loss, all-training-identity hard negatives, shared-space cycle consistency,
reliability calibration, and a More-vs-Fewer safety loss. The frozen encoders are never updated.

At inference, four candidate-independent score branches are computed: available modality, same-modality shared
identity, cross-modality shared identity, and recovered target feature. A sample-level gate fuses them; the recovered
branch is additionally multiplied by predicted reliability and all weights are renormalized. Consequently, a
plausible-looking but unreliable 256-dimensional recovery can be suppressed instead of being forced into matching.
The recovered vector is a conditional estimate, not the unknowable true modality-specific feature.

Training is split into shared-space warmup, probabilistic recovery, and joint gating stages. Checkpoint selection uses
only the lexicographic pair of worst-direction and mean validation EER. The fixed test identities are absent from the
trainer. The legacy closed-form CCA checkpoint remains at `outputs/shared_feature_recovery/best.pth`; v2 uses a
separate directory.

## Results

The table below places the independently evaluated frozen single-modality encoders and the latest fixed 480-train /
120-test recovery protocol on the same unchanged Tongji Session-1 Gallery/Probe test set. Each row has 120 Gallery
identities, 240 Probes, 240 genuine scores, and 28,560 impostor scores.

| Missing modality | Available test input | Method / protocol | EER ↓ | TAR@FAR=1e-3 ↑ | TAR@FAR=1e-4 ↑ | Top-1 ↑ | Top-5 ↑ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Palmprint | Palm-vein only | Frozen palm-vein encoder (single-modality baseline) | 2.08% | 86.25% | 76.67% | 96.67% | 99.58% |
| Palmprint | Palm-vein only | **Latest probabilistic recovery + gated fusion (480/120)** | **0.71%** | **95.83%** | **90.00%** | **99.58%** | **100.00%** |
| Palm-vein | Palmprint only | Frozen palmprint encoder (single-modality baseline) | 1.83% | 89.17% | 76.25% | 96.67% | 98.75% |
| Palm-vein | Palmprint only | **Latest probabilistic recovery + gated fusion (480/120)** | **0.42%** | **98.75%** | **98.33%** | **99.58%** | **100.00%** |

Relative to the corresponding single-modality encoder, the latest method reduces EER by 1.38/1.41 percentage points
and improves TAR@FAR=1e-4 by 13.33/22.08 points for palmprint/palm-vein missing, respectively.

The single encoders remain the original 432-train/48-validation checkpoints; the 480-identity merge applies only to
the recovery module, which was trained for the predeclared 21 epochs without validation selection.

Auditable sources: `single_vein_test.log`, `single_palm_test.log`, and `test_metrics.json` under
`outputs/shared_feature_recovery/trainable_v2_trainval_480/`.

Research-integrity note: the direction-level rollback was introduced after an earlier fixed-test development run had
been inspected. Treat the table as development evidence, not a pristine confirmatory test. Publication claims should
be reconfirmed on a new identity-disjoint holdout or external dataset.

## Environment

The project is run in the `pvmd` Conda environment:

```bash
cd /root/autodl-tmp/pv_missing_modility
conda activate pvmd
pip install -r requirements.txt
```

Local data, generated protocols, pretrained weights, checkpoints, and logs are kept under ignored directories such
as `data/`, `data_txt/`, `pretrained/`, `outputs/`, and `runs/`.

## Protocol

Protocol rows have the following format:

```text
palm_path vein_path label palm_exists vein_exists split
```

Generate the paired, identity-disjoint protocols with the fixed seed `2026`:

```bash
python -m utils.datasets_txt
```

Tongji uses Session 1 only and splits identities into 432 training, 48 validation, and 120 test subjects. For each
validation or test identity, eight pairs form the complete-modality Gallery and two disjoint pairs form the Probe.
The Probe protocol contains `complete`, `palmprint_missing`, and `palmvein_missing` scenarios.

Generated files used by the feature-recovery route are:

```text
data_txt/tongji/ssfd_train_full.txt
data_txt/tongji/ssfd_trainval_full.txt
data_txt/tongji/ssfd_val_gallery_full.txt
data_txt/tongji/ssfd_val_protocol.txt
data_txt/tongji/ssfd_gallery_full.txt
data_txt/tongji/ssfd_test_protocol.txt
```

The protocol generator validates identity counts, split non-overlap, Session-1-only paths, paired samples,
Gallery/Probe non-overlap, scenario completeness, and file hashes.

## Train single-modality encoders

The existing single-modality implementation and checkpoints are independent of feature recovery. To train them from
scratch if needed:

```bash
python train_encoder.py --modality palm --seed 42
python train_encoder.py --modality vein --seed 42
```

Their default checkpoints are:

```text
outputs/encoders/palm_best.pth
outputs/encoders/vein_best.pth
```

## Train probabilistic feature recovery

Train with the frozen encoder feature caches and validation-EER-only selection:

```bash
python train_shared_feature_recovery.py --device cuda \
  --shared_dimensions 192 --epochs 120 \
  --shared_warmup_epochs 20 --recovery_end_epoch 70 \
  --batch_identities 32 --instances_per_identity 2 \
  --save_dir outputs/shared_feature_recovery/trainable_v2
```

The first run fingerprints the protocols and encoders and creates reusable caches under
`outputs/shared_feature_recovery/cache/`. Checkpoints are written atomically to:

```text
outputs/shared_feature_recovery/trainable_v2/best.pth
outputs/shared_feature_recovery/trainable_v2/last.pth
```

`best.pth` contains the validation-selected model; `last.pth` also contains optimizer state for `--resume`. Both
record architecture/configuration, validation history, gate statistics, rollback decisions, cache metadata, and
SHA-256 fingerprints. The trainer has no test-list argument.

For a fixed train+validation merge with no validation-based model selection:

```bash
python train_shared_feature_recovery.py --device cuda \
  --fixed_full_train --train_list data_txt/tongji/ssfd_trainval_full.txt \
  --epochs 21 \
  --save_dir outputs/shared_feature_recovery/trainable_v2_trainval_480 \
  --cache_dir outputs/shared_feature_recovery/cache_trainval_480
```

This mode retains the original stage-2 learning-rate schedule, applies the prevalidated directional policy after the
20-epoch warmup, and fixes epoch 21 as the checkpoint. Its checkpoint contains no validation results or fingerprints.

## Evaluate

Evaluate the unchanged single-modality baselines:

```bash
python test_encoder.py --modality palm --ckpt outputs/encoders/palm_best.pth
python test_encoder.py --modality vein --ckpt outputs/encoders/vein_best.pth
```

Evaluate both missing-modality directions using the validation-selected v2 checkpoint:

```bash
python test_shared_feature_recovery.py \
  --recovery_ckpt outputs/shared_feature_recovery/trainable_v2/best.pth \
  --output outputs/shared_feature_recovery/trainable_v2/test_metrics.json
```

Evaluate the fixed merged-training checkpoint:

```bash
python test_shared_feature_recovery.py \
  --recovery_ckpt outputs/shared_feature_recovery/trainable_v2_trainval_480/best.pth \
  --output outputs/shared_feature_recovery/trainable_v2_trainval_480/test_metrics.json
```

The evaluator verifies encoder fingerprints and architecture version, computes all four score branches and dynamic
fusion, and records checkpoint/test-protocol hashes. Each scenario reports EER, TAR at FAR `1e-3` and `1e-4`, Top-1,
Top-5, reliability distribution, predictive variance, and average branch weights.

## Generate publication figures

Generate the three legacy closed-form test-set figures used for analysis:

```bash
python visualize_shared_feature_recovery.py
```

The command reuses the frozen encoders, fitted projector, validation-selected fusion weights, and unchanged Tongji
test protocol. It exports 400-DPI PNG and vector PDF versions under:

```text
outputs/shared_feature_recovery/figures/figure2_low_far_roc.{png,pdf}
outputs/shared_feature_recovery/figures/figure3_hard_negative_margin.{png,pdf}
outputs/shared_feature_recovery/figures/figure4_shared_identity_space.{png,pdf}
```

Figure 2 contains empirical low-FAR verification curves. Figure 3 compares each Probe's genuine-minus-hardest-
impostor margin before and after score fusion. Figure 4 shows fitted and unseen-test canonical component
correlations together with identity margins before and after the shared projection.

The same directory contains `figure_source_data.npz` and `figure_manifest.json`, which record the plotted arrays,
metric summaries, source fingerprints, and figure hashes. Test features are used only for visualization and are
never used to refit the projector or select a fusion parameter.

An additional direct cross-modal template-alignment diagnostic can be exported with:

```bash
python visualize_shared_feature_recovery.py --include_alignment_diagnostic
```

This diagnostic is intentionally separate from the main figures because direct target-modality matching is not the
validation-selected recognition rule.

## File map

| File | Role |
| --- | --- |
| `train_encoder.py` | Trains one single-modality encoder and ArcFace head. |
| `test_encoder.py` | Evaluates a saved palmprint or palm-vein encoder. |
| `models/backbones.py` | Defines the ResNet18 encoders. |
| `models/shared_feature_recovery.py` | Implements CCA initialization, trainable residual projectors, probabilistic 192-to-256 recovery, reliability, and dynamic gating. |
| `train_shared_feature_recovery.py` | Runs cached backpropagation training with either validation selection or a fixed no-validation full-training policy. |
| `test_shared_feature_recovery.py` | Evaluates both missing-modality scenarios on the fixed test split. |
| `visualize_shared_feature_recovery.py` | Exports publication figures and their auditable source data. |
| `utils/feature_extraction.py` | Extracts frozen features and validates fingerprinted reusable caches. |
| `utils/datasets_txt.py` | Generates and loads the paired missing-modality protocols. |
| `utils/evaluation.py` | Builds Gallery templates and computes verification/identification metrics. |
| `utils/checkpoint.py` | Restores frozen encoders from their checkpoints. |
| `utils/checkpoint_io.py` | Provides checkpoint I/O and SHA-256 fingerprinting. |
| `utils/preprocess.py` | Defines palmprint and palm-vein transforms. |
| `utils/scenarios.py` | Defines the shared missing-modality scenario names. |
