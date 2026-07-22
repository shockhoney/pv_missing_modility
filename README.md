# Trainable missing-modality recovery

The retained model is balanced_staged_dcca_specformer_v9_1. Encoders stay frozen,
while the recovery model is updated by backpropagation.

The shared-space stage uses a CCA initialization followed by trainable nonlinear
refiners, differentiable DCCA, cross-modal identity proxies, supervised contrastive
alignment, and paired alignment. The recovery stage uses spatial Transformer
specific features, gallery retrieval dropout, cycle consistency, identity/ranking
losses, and differentiable low-FAR pAUC loss.

Final fusion is sample-adaptive and bounded: recovered evidence always receives
15%-75% weight, so neither shared nor recovered evidence can structurally suppress
the other. The same configuration is used for Tongji, CUMT, and PolyU.

## Main files

- models/dcca_specformer.py: final bounded-fusion model.
- models/dcca_specformer_components.py: reusable CCA and Transformer components.
- train_dcca_specformer.py: staged training and all trainable losses.
- test_dcca_specformer.py: strict two-direction gallery/probe evaluation.
- analyze_end_to_end_dcca_specformer.py: builds the final summary.
- tests/test_dcca_specformer.py: gradient and fusion-bound tests.

## Artifacts

Final checkpoints and metrics are under:

    outputs/dcca_specformer/v9_1/{tongji,cumt,polyu}/

Identity-disjoint selection artifacts use the matching dataset_validation
directories. Frozen encoder checkpoints and reusable feature caches remain under
outputs/encoders and outputs/dcca_specformer/cache.

## Train

First select the epoch count on identity-disjoint validation identities:

    conda run -n pvmd python train_dcca_specformer.py --device cuda \
      --train_list data_txt/tongji/ssfd_train_full.txt \
      --val_gallery_list data_txt/tongji/ssfd_val_gallery_full.txt \
      --val_protocol_list data_txt/tongji/ssfd_val_protocol.txt \
      --palm_ckpt outputs/encoders/palm_best.pth \
      --vein_ckpt outputs/encoders/vein_best.pth \
      --save_dir outputs/dcca_specformer/v9_1/tongji_validation

Then replay the selected epoch count on the complete training identities:

    conda run -n pvmd python train_dcca_specformer.py --device cuda \
      --fixed_full_train \
      --selection_ckpt outputs/dcca_specformer/v9_1/tongji_validation/best.pth \
      --train_list data_txt/tongji/ssfd_train_full.txt \
      --palm_ckpt outputs/encoders/palm_best.pth \
      --vein_ckpt outputs/encoders/vein_best.pth \
      --save_dir outputs/dcca_specformer/v9_1/tongji

CUMT and PolyU use the same hyperparameters; only protocol, encoder, cache, and
output paths change.

## Evaluate and summarize

    conda run -n pvmd python test_dcca_specformer.py --device cuda \
      --gallery_list data_txt/tongji/ssfd_gallery_full.txt \
      --protocol_list data_txt/tongji/ssfd_test_protocol.txt \
      --recovery_ckpt outputs/dcca_specformer/v9_1/tongji/best.pth \
      --output outputs/dcca_specformer/v9_1/tongji/test_metrics.json

    conda run -n pvmd python analyze_end_to_end_dcca_specformer.py
    conda run -n pvmd python -m unittest discover -s tests -p 'test_*.py' -v

The consolidated result is outputs/dcca_specformer/v9_1/report.md.
