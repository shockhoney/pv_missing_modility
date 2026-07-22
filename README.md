# Hetero-MMRNet

Hetero-MMRNet performs feature-level palmprint/palm-vein recognition when one modality is unavailable. It does not generate images. The six retained single-modality checkpoints are frozen and are not modified by recovery training.

## Current recovery method

Each encoder outputs a 256-dimensional embedding. Two CCA-initialized 256-to-192 projectors preserve the identity evidence that is shared within each modality. The method does not attempt to hallucinate an unidentifiable 64-dimensional private component from the other modality.

The final method is `gallery_conditioned_selective_prototype_recovery_v6`:

1. The available 192-dimensional shared feature is compared with enrolled identities in the available-modality gallery.
2. A temperature-calibrated identity posterior weights the complete target-modality gallery templates, producing a recovered 256-dimensional target prototype.
3. The available-branch top-1/top-2 cosine margin `m` controls a band-pass gate:

   ```text
   q = alpha * sigmoid((m - floor) / slope) * sigmoid((ceiling - m) / slope)
   ```

   Very ambiguous samples are rejected by the lower boundary, moderately ambiguous samples receive recovered evidence, and already-easy samples are protected by the upper boundary.
4. Final scores are `(1-q) * base_score + q * recovered_score`. The evaluator always reports a strict `fusion without recovery` ablation beside `fusion with recovery`.

This is a closed-set method: recovery uses the enrolled complete gallery. It must not be described as open-set reconstruction of the true missing private feature.

## Final identity-disjoint test results

EER values are percentages. `Base` is the final shared branch with recovery disabled; `v6` adds the selectively gated recovered target-prototype evidence.

| Dataset | Missing modality | Base EER | v6 EER | Recovery gain | TAR@1e-3 | TAR@1e-4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Tongji | Palmprint | 0.7283 | **0.1155** | **0.6127 pp** | 99.58 | 95.42 |
| Tongji | Palm-vein | 0.4167 | **0.0210** | **0.3957 pp** | 100.00 | 98.75 |
| CUMT | Palmprint | 1.0133 | **0.6503** | **0.3630 pp** | 98.28 | 96.55* |
| CUMT | Palm-vein | 1.9208 | **1.4065** | **0.5142 pp** | 96.55 | 93.10* |
| PolyU | Palmprint | 0.5000 | **0.3737** | **0.1263 pp** | 96.00 | 61.00 |
| PolyU | Palm-vein | 0.3384 | **0.0859** | **0.2525 pp** | 100.00 | 99.00 |

`*` CUMT has only 6,612 impostor scores, so FAR=1e-4 is below the empirical count resolution; the reported value is the FAR=0 operating point.

All six recovery-enabled EERs are lower than the no-recovery ablation. PolyU palmprint-missing has a real tradeoff: EER and TAR@1e-3 improve, but TAR@1e-4 falls from 82.00% to 61.00%. Do not claim universal low-FAR improvement for that direction.

Auditable metrics:

```text
outputs/shared_feature_recovery/recovery_v6/tongji/test_metrics.json
outputs/shared_feature_recovery/recovery_v6/cumt/test_metrics.json
outputs/shared_feature_recovery/recovery_v6/polyu/test_metrics.json
```

## Environment

```bash
cd /root/autodl-tmp/pv_missing_modility
conda activate pvmd
pip install -r requirements.txt
```

## Protocols and frozen encoders

Protocol rows use:

```text
palm_path vein_path label palm_exists vein_exists split
```

Tongji uses the merged 480-identity training set and a disjoint 120-identity test set. CUMT and PolyU use identity-disjoint 8:2 train/test protocols. Recovery-only calibration splits are stored under `data_txt/<dataset>/`; official test identities are never used by the trainer.

Frozen checkpoints:

```text
outputs/encoders/palm_best.pth
outputs/encoders/vein_best.pth
outputs/encoders/identity_8_2/cumt/{palm,vein}_best.pth
outputs/encoders/identity_8_2/polyu/{palm,vein}_best.pth
```

## Train recovery

Training has two explicit stages. First calibrate on a training-identity holdout; then combine those calibration values with the shared projectors fitted on the complete training set. The frozen encoders are only used to build fingerprinted feature caches.

Tongji calibration example:

```bash
python train_shared_feature_recovery.py --device cuda \
  --train_list data_txt/tongji/ssfd_train_full.txt \
  --val_gallery_list data_txt/tongji/ssfd_val_gallery_full.txt \
  --val_protocol_list data_txt/tongji/ssfd_val_protocol.txt \
  --palm_ckpt outputs/encoders/palm_best.pth \
  --vein_ckpt outputs/encoders/vein_best.pth \
  --save_dir outputs/shared_feature_recovery/recovery_v6/tongji_validation
```

Final full-training checkpoint example:

```bash
python train_shared_feature_recovery.py --device cuda --fixed_full_train \
  --train_list data_txt/tongji/ssfd_trainval_full.txt \
  --palm_ckpt outputs/encoders/palm_best.pth \
  --vein_ckpt outputs/encoders/vein_best.pth \
  --calibration_ckpt outputs/shared_feature_recovery/recovery_v6/tongji_validation/best.pth \
  --save_dir outputs/shared_feature_recovery/recovery_v6/tongji
```

The retained final checkpoints are:

```text
outputs/shared_feature_recovery/recovery_v6/tongji/best.pth
outputs/shared_feature_recovery/recovery_v6/cumt/best.pth
outputs/shared_feature_recovery/recovery_v6/polyu/best.pth
```

## Evaluate

Tongji example:

```bash
python test_shared_feature_recovery.py \
  --gallery_list data_txt/tongji/ssfd_gallery_full.txt \
  --protocol_list data_txt/tongji/ssfd_test_protocol.txt \
  --palm_ckpt outputs/encoders/palm_best.pth \
  --vein_ckpt outputs/encoders/vein_best.pth \
  --recovery_ckpt outputs/shared_feature_recovery/recovery_v6/tongji/best.pth \
  --output outputs/shared_feature_recovery/recovery_v6/tongji/test_metrics.json
```

The evaluator verifies encoder/checkpoint fingerprints and reports every branch, recovery-disabled fusion, recovery-enabled fusion, posterior entropy, reliability margin, gate distribution, and gate activation fractions.

## File map

| File | Role |
| --- | --- |
| `models/shared_feature_recovery.py` | Shared projectors, closed-set target-prototype recovery, and band-pass sample gate. |
| `train_shared_feature_recovery.py` | Frozen-feature projector training and identity-holdout calibration. |
| `test_shared_feature_recovery.py` | Strict recovery/no-recovery evaluation for both missing directions. |
| `train_encoder.py`, `test_encoder.py` | Independent single-modality baseline code; unchanged by recovery work. |
| `utils/feature_extraction.py` | Fingerprinted frozen-feature caches. |
| `utils/evaluation.py` | Gallery templates and verification/identification metrics. |
| `utils/checkpoint_io.py` | Atomic checkpoint I/O and SHA-256 fingerprints. |
