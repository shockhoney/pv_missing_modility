# GIPSSR-Net: trainable missing-modality recovery and CUEF fusion

This repository retains the proposed method `gipssr_cuef_state_space_recovery_v3` (GIPSSR-Net, Gallery Identity-Prior State-Space Recovery Network) together with only the code required for its ablation studies, full comparison experiments, and hyperparameter experiments. Palmprint and palm-vein encoders are frozen; every shared-space, recovery, refinement, uncertainty, and score-fusion parameter is optimized by backpropagation.

## Method

The retained architecture contains four paper-level components:

- **IGDCA** — Identity-Guided Deep Correlation Alignment learns the shared cross-modal identity space with staged DCCA, identity proxies, and paired alignment.
- **SGSSD** — Shared-Guided State-Space Disentangler learns hierarchical modality-specific tokens without replacing the frozen encoders.
- **GIPRD** — Gallery Identity-Prior Recovery Decoder refines recovery from multiple gallery identity candidates.
- **CUEF** — Conflict-aware Uncertainty-Calibrated Evidential Fusion uses differentiable cohort/scale calibration, fixed-size evidence tokens, Transformer conflict interaction, predicted/external uncertainty, and bounded sample-wise evidence weights.

CUEF fuses four score branches: available-modality, same-modality shared, cross-modal shared, and recovered-modality scores. The recovered branch is constrained to 15%–75%; the three base branches retain nonzero mass. The weights are learned jointly with identity, ranking, cycle-consistency, recovery, and differentiable low-FAR pAUC objectives. There is no runtime fallback, best-of-two score choice, teacher comparison, deployment blend, or dataset-specific hyperparameter branch.

## Retained code

- `models/cuef.py`: CUEF calibration, evidence tokens, conflict interaction, uncertainty, and bounded fusion.
- `models/gipssr.py`: final GIPSSR-Net with SGSSD and GIPRD.
- `models/gipssr_components.py`: IGDCA and common recovery components.
- `models/gipssr_stage1.py`: trainable Stage-1 shared alignment and recovery backbone.
- `utils/gipssr_stage1_training.py`: staged shared/recovery optimization and ablations.
- `utils/gipssr_training.py`: validation selection, full-split replay, and checkpoint metadata.
- `train_gipssr.py`: final joint CUEF+SGSSD+GIPRD training entry point.
- `test_gipssr.py`: strict two-direction gallery/probe evaluation with CUEF diagnostics.
- `analyze_gipssr.py`: seed-42 result summarization and closed-form artifact audit.

Frozen encoder training/evaluation remain in `train_encoder.py` and `test_encoder.py` because they are required to reproduce the input embeddings.

The retained experiment entry points are:

- `run_missing_rate_experiments.py`: proposed-method missing-rate experiment.
- `train_full_comparison.py`, `test_full_comparison.py`, `run_tongji_full_comparisons.py`, and `run_tongji_full_missing_rate_experiments.py`: full image-level comparison experiments; see `FULL_COMPARISON_REPRODUCTION.md`.
- `run_tongji_hparams.py`, `summarize_tongji_validation_hparams.py`, and the `*_tongji_*report*` scripts: hyperparameter experiment execution and report generation.
- `visualizations/gipssr_paper_figures.py`: seed-42 paper figures.

## Retained artifacts

Final checkpoints and metrics are organized as:

    outputs/gipssr/ablations/checkpoints/{tongji,cumt,polyu}/seed_42/
    outputs/gipssr/ablations/results/{tongji,cumt,polyu}/seed_42/

The consolidated paper tables and machine-readable statistics are:

    outputs/gipssr/ablations/report.md
    outputs/gipssr/ablations/summary.json

Only final full/ablation replay checkpoints are retained. Validation-selection checkpoints, feature caches, smoke tests, rejected batch-global-pAUC probes, and pre-CUEF schemes are removed after audit.

## Formal protocol

All datasets use seed 42 and the same model/loss hyperparameters.

1. Train for at most 12 epochs on recovery-training identities.
2. Select the epoch on an identity-disjoint validation protocol.
3. Reinitialize and replay exactly the selected epoch count on the full training split.
4. Evaluate the resulting neural model once on the fixed test protocol.
5. Report each missing direction separately and macro-average the two directions for seed 42.

CUMT has 6,612 impostor scores per direction, so its smallest positive empirical FAR is 1.5124e-4; its TAR@FAR=1e-4 value is therefore the FAR=0 operating point.

## Ablation design

Ablations remove a claimed component without a learned replacement:

- `without_igdca`: remove trainable shared alignment and shared-space objectives.
- `without_sgssd`: remove SGSSD while retaining GIPRD.
- `without_giprd`: remove GIPRD while retaining the trained SGSSD.
- `without_sgssd_giprd`: remove both final hierarchical modules and retain the trainable Stage-1 recovery+CUEF backbone.
- `without_cuef_calibration`: remove differentiable cohort and scale calibration.
- `without_cuef_conflict`: remove conflict token interaction and conflict penalty.
- `without_cuef_uncertainty`: remove predicted/external uncertainty from evidence weighting.
- `full`: IGDCA + SGSSD + GIPRD + CUEF.

Two non-trained diagnostics are derived from each full run: `single` uses only the frozen available-modality score, and `without_recovery_fusion` removes the recovered-score branch without a replacement.

## Evaluate and audit

The evaluator defaults to the retained Tongji seed-42 full checkpoint:

    conda run -n pvmd python test_gipssr.py

Regenerate the consolidated report after all 24 seed-42 trained result/checkpoint pairs exist:

    conda run -n pvmd python analyze_gipssr.py

The analyzer fails closed on result/checkpoint SHA-256, architecture, ablation label, fixed-full replay status, selected epoch, CUEF/SGSSD/GIPRD parameter structure, unexpected fusion parameters, protocol fingerprints, required diagnostics, or recovery-weight bounds.

## Design references

- [Embracing Unimodal Aleatoric Uncertainty for Robust Multimodal Fusion, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Gao_Embracing_Unimodal_Aleatoric_Uncertainty_for_Robust_Multimodal_Fusion_CVPR_2024_paper.html)
- [Conflict-Guided Evidential Multimodal Fusion, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Deregnaucourt_A_Conflict-Guided_Evidential_Multimodal_Fusion_for_Semantic_Segmentation_WACV_2025_paper.html)
- [Fuzzy Multimodal Learning for Trusted Cross-modal Retrieval, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Duan_Fuzzy_Multimodal_Learning_for_Trusted_Cross-modal_Retrieval_CVPR_2025_paper.html)
- [SURE: Robust Multimodal Fusion with Missing Modalities and Distribution Shifts, 2025](https://arxiv.org/abs/2504.13465)
- [MambaIR, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02740.pdf)
- [MambaIRv2, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_MambaIRv2_Attentive_State_Space_Restoration_CVPR_2025_paper.html)
