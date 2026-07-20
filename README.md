# Hetero-MMRNet

Hetero-MMRNet performs palmprint/palm-vein recognition when one modality is unavailable. The repository keeps the
two trained single-modality encoders fixed and uses one feature-level recovery route: **Shared-Identity Feature
Recovery**.

The recovery route does not generate palmprint or palm-vein images, and it does not force the model to reconstruct a
complete target-modality feature. Instead, it learns the identity evidence statistically shared by the two
modalities and combines that evidence with the available-modality matching score.

## Method

The palmprint and palm-vein encoders each produce a 256-dimensional embedding. From paired training embeddings, a
regularized CCA projector fits a common canonical identity space. At inference time:

1. the available image is encoded by its unchanged single-modality encoder;
2. its embedding is projected into the shared identity space;
3. an available-modality cosine score and a shared-space cosine score are computed;
4. the two scores are fused using weights selected only on identity-disjoint validation subjects.

The search selects input normalization, covariance regularization, shared dimension, score weight, and cross-modal
Gallery contribution. The fixed Tongji test identities are not used for fitting or model selection. The current
checkpoint selects a 192-dimensional shared space.

## Results

With the current encoder checkpoints and the fixed Tongji Session-1 Gallery/Probe protocol:

| Missing scenario | Method | EER | TAR@FAR=1e-3 | TAR@FAR=1e-4 | Top-1 | Top-5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Palmprint missing (vein available) | Vein baseline | 2.08% | 86.25% | 76.67% | 96.67% | 99.58% |
| Palmprint missing (vein available) | Shared-feature fusion | **1.25%** | **95.00%** | **85.42%** | **97.50%** | **100.00%** |
| Palm-vein missing (palm available) | Palm baseline | 1.83% | 89.17% | 76.25% | 96.67% | 98.75% |
| Palm-vein missing (palm available) | Shared-feature fusion | **0.83%** | **98.33%** | **95.83%** | **99.17%** | **100.00%** |

The test evaluator also reports the shared-space-only result and the absolute improvement over the corresponding
single-modality baseline.

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

## Fit shared-identity feature recovery

Fit the projector and select its configuration on the validation identities:

```bash
python train_shared_feature_recovery.py
```

The command reads the two frozen encoder checkpoints and saves:

```text
outputs/shared_feature_recovery/best.pth
```

The checkpoint records the architecture version, fitted projections, canonical correlations, selected fusion
parameters, validation metrics, and SHA-256 fingerprints of both encoders and all fitting/validation protocols.

## Evaluate

Evaluate the unchanged single-modality baselines:

```bash
python test_encoder.py --modality palm --ckpt outputs/encoders/palm_best.pth
python test_encoder.py --modality vein --ckpt outputs/encoders/vein_best.pth
```

Evaluate both missing-modality directions using the fitted shared feature model:

```bash
python test_shared_feature_recovery.py
```

The evaluator verifies the encoder fingerprints and writes the full metrics plus checkpoint/protocol fingerprints to:

```text
outputs/shared_feature_recovery/test_metrics.json
```

Each scenario reports EER, TAR at FAR `1e-3` and `1e-4`, Top-1, and Top-5. Gallery templates are formed by averaging
the normalized embeddings of each held-out identity, and Probe samples are matched by cosine similarity.

## Generate publication figures

Generate the three test-set figures used for analysis:

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
| `models/shared_feature_recovery.py` | Implements regularized shared-identity projection. |
| `train_shared_feature_recovery.py` | Fits the projector and selects its configuration on validation identities. |
| `test_shared_feature_recovery.py` | Evaluates both missing-modality scenarios on the fixed test split. |
| `visualize_shared_feature_recovery.py` | Exports publication figures and their auditable source data. |
| `utils/feature_extraction.py` | Extracts paired or available-only embeddings with frozen encoders. |
| `utils/datasets_txt.py` | Generates and loads the paired missing-modality protocols. |
| `utils/evaluation.py` | Builds Gallery templates and computes verification/identification metrics. |
| `utils/checkpoint.py` | Restores frozen encoders from their checkpoints. |
| `utils/checkpoint_io.py` | Provides checkpoint I/O and SHA-256 fingerprinting. |
| `utils/preprocess.py` | Defines palmprint and palm-vein transforms. |
| `utils/scenarios.py` | Defines the shared missing-modality scenario names. |
