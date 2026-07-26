# Trainable missing-modality recovery

The only retained model is `hiasr_identity_prior_state_space_v10` (HIASR). The
palmprint and palm-vein encoders stay frozen; every recovery component is optimized
by backpropagation.

HIASR combines a differentiable-DCCA shared space with shared-guided multi-scale
spatial disentanglement, a four-direction selective state-space mixer, independent
Top-5 gallery identity priors, and two-stage specific-to-identity reconstruction.
Training uses cross-modal identity proxies, gallery retrieval/candidate dropout,
cycle consistency, ranking, teacher-safe margin, orthogonality, and differentiable
low-FAR pAUC losses. Fusion is sample-adaptive and bounded: recovered evidence
always receives 15%-75% weight, so neither branch can structurally suppress the
other. Tongji, CUMT, and PolyU use one shared configuration.

The design follows recent restoration and missing-modality directions: MambaIR
(ECCV 2024), PLTrans (CVPR 2024), MambaIRv2 (CVPR 2025), and SimMLM (ICCV 2025).
The implemented model is repository-native PyTorch and does not depend on a custom
selective-scan extension.

## Main files

- `models/dcca_specformer.py`: final HIASR state-space recovery model.
- `models/recovery_backbone.py`: internal stage-1 initialization used by HIASR.
- `models/dcca_specformer_components.py`: shared recovery components.
- `train_dcca_specformer.py`: final stage-2 backpropagation training.
- `utils/recovery_backbone_training.py`: temporary stage-1 initialization training.
- `test_dcca_specformer.py`: strict two-direction gallery/probe evaluation.
- `analyze_end_to_end_dcca_specformer.py`: final result summary.
- `tests/`: gradient, Top-K, safety, fusion-bound, and low-FAR tests.

## Artifacts

Final checkpoints and metrics are under:

    outputs/dcca_specformer/hiasr_v10/{tongji,cumt,polyu}/

Identity-disjoint selection artifacts use the matching dataset_validation
directories. Frozen encoder checkpoints and reusable feature caches remain under
outputs/encoders and outputs/dcca_specformer/cache.

## Train

Create the temporary internal initialization on identity-disjoint validation:

    conda run -n pvmd python -m utils.recovery_backbone_training --device cuda \
      --train_list data_txt/tongji/ssfd_train_full.txt \
      --val_gallery_list data_txt/tongji/ssfd_val_gallery_full.txt \
      --val_protocol_list data_txt/tongji/ssfd_val_protocol.txt \
      --palm_ckpt outputs/encoders/palm_best.pth \
      --vein_ckpt outputs/encoders/vein_best.pth \
      --save_dir /tmp/hiasr/tongji_backbone_validation

Replay that initialization on all training identities:

    conda run -n pvmd python -m utils.recovery_backbone_training --device cuda \
      --fixed_full_train \
      --selection_ckpt /tmp/hiasr/tongji_backbone_validation/best.pth \
      --train_list data_txt/tongji/ssfd_train_full.txt \
      --palm_ckpt outputs/encoders/palm_best.pth \
      --vein_ckpt outputs/encoders/vein_best.pth \
      --save_dir /tmp/hiasr/tongji_backbone

Train HIASR and select its epoch count on the same disjoint validation split:

    conda run -n pvmd python train_dcca_specformer.py --device cuda \
      --warm_start_ckpt /tmp/hiasr/tongji_backbone_validation/best.pth \
      --train_list data_txt/tongji/ssfd_train_full.txt \
      --val_gallery_list data_txt/tongji/ssfd_val_gallery_full.txt \
      --val_protocol_list data_txt/tongji/ssfd_val_protocol.txt \
      --palm_ckpt outputs/encoders/palm_best.pth \
      --vein_ckpt outputs/encoders/vein_best.pth \
      --save_dir outputs/dcca_specformer/hiasr_v10/tongji_validation

Replay the selected HIASR epoch count on all training identities:

    conda run -n pvmd python train_dcca_specformer.py --device cuda \
      --warm_start_ckpt /tmp/hiasr/tongji_backbone/best.pth \
      --fixed_full_train \
      --selection_ckpt outputs/dcca_specformer/hiasr_v10/tongji_validation/best.pth \
      --train_list data_txt/tongji/ssfd_train_full.txt \
      --palm_ckpt outputs/encoders/palm_best.pth \
      --vein_ckpt outputs/encoders/vein_best.pth \
      --save_dir outputs/dcca_specformer/hiasr_v10/tongji

The internal stage-1 artifacts are temporary and are not retained after stage 2.

CUMT and PolyU use the same hyperparameters; only protocol, encoder, cache, and
output paths change.

## Evaluate and summarize

    conda run -n pvmd python test_dcca_specformer.py --device cuda \
      --gallery_list data_txt/tongji/ssfd_gallery_full.txt \
      --protocol_list data_txt/tongji/ssfd_test_protocol.txt \
      --recovery_ckpt outputs/dcca_specformer/hiasr_v10/tongji/best.pth \
      --output outputs/dcca_specformer/hiasr_v10/tongji/test_metrics.json

    conda run -n pvmd python analyze_end_to_end_dcca_specformer.py
    conda run -n pvmd python -m unittest discover -s tests -p 'test_*.py' -v

The consolidated result is outputs/dcca_specformer/hiasr_v10/report.md.

## Design references

- [MambaIR, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02740.pdf)
- [PLTrans, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Xie_Learning_Degradation-unaware_Representation_with_Prior-based_Latent_Transformations_for_Blind_Face_CVPR_2024_paper.html)
- [MambaIRv2, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_MambaIRv2_Attentive_State_Space_Restoration_CVPR_2025_paper.html)
- [SimMLM, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_SimMLM_A_Simple_Framework_for_Multi-modal_Learning_with_Missing_Modality_ICCV_2025_paper.pdf)
- [Missing-modality survey, 2024](https://arxiv.org/abs/2409.07825)
