# Five full missing-modality reproductions

This repository retains only the image-level full-reproduction pipeline:
`train_full_comparison.py`, `test_full_comparison.py`, and the resumable
`run_tongji_full_comparisons.py` runner.

## Implemented training systems

| Method | Full image-level path | Paper/official elements restored |
|---|---|---|
| SSFD-Net | `models/comparisons/ssfd.py`, `utils/full_ssfd_experiment.py` | independent VGG16 Dp/Dv pretraining; dual shared/specific VGG16 encoders; bidirectional CMFT; classification, triplet, transformation, inter- and intra-consistency losses |
| DMRNet | `models/comparisons/dmrnet.py`, `utils/full_dmrnet_experiment.py` | trainable dual encoders; random non-empty modality combinations; spatial mu/log-variance; reparameterization, KL, whole-warm-up variance mining and shared-predictor HCR |
| HCMIG | `models/comparisons/hcmig.py`, `utils/full_hcmig_experiment.py` | bidirectional ResNet-9 texture/structure generation, PatchGAN, cycle/adversarial/CMS/Fourier losses, followed by frozen-generation VGG16 MDSFF recognition |
| SimMLM | `models/comparisons/simmlm_full.py`, `utils/full_simmlm_experiment.py` | independent expert pretraining; mask-aware image router; logit-level DMoME; paired More-vs-Fewer training and MoFe ranking |
| MMANet | `models/comparisons/mmanet_full.py`, `utils/full_mmanet_experiment.py` | complete-modality teacher; deployment and regularization networks; modality dropout; entropy-weighted MAD and five-epoch full-training-set MAR mining |

SSFD-Net and HCMIG have no discoverable public author repository, so their
implementations are paper-derived.  DMRNet, SimMLM and MMANet record the exact
audited official repository commit in source/checkpoint metadata.

## Locked Tongji protocol

The shared comparison protocol is
`tongji_session1_id_disjoint_closed_set_v1`:

- 432 training identities, 48 validation identities and 120 test identities;
- identity sets are disjoint;
- validation/test use eight complete pairs per identity for gallery templates
  and two probes per identity for each missing-modality direction;
- the test set is read only after validation checkpoint selection;
- all methods share preprocessing, seed 42, protocol files and matcher metrics.

This deliberately differs from the SSFD-Net/HCMIG paper protocol.  Those
papers train on all 600 Tongji identities in session 1 and test the same 600
identities in session 2 as a closed-set identity classification/verification
task.  Their published recognition rates therefore are not numerically
comparable to this repository's unseen-identity verification results.

## Commands

Train and test all methods sequentially with the paper settings (100 epochs
for every configured stage):

```bash
python run_tongji_full_comparisons.py
```

Run or resume one method:

```bash
python train_full_comparison.py --method dmrnet
python test_full_comparison.py \
  --checkpoint outputs/gipssr/full_comparisons/tongji/seed_42/dmrnet/best.pth
```

Select a resumable subset:

```bash
python run_tongji_full_comparisons.py --methods ssfd,hcmig
```

The runner writes to
`outputs/gipssr/full_comparisons/tongji/seed_42/`:

- self-contained `best` and resumable `last` checkpoints;
- per-method training/test logs;
- test JSON with checkpoint and protocol SHA-256 fingerprints;
- `manifest.json`, `summary.csv`, and `summary.md`.

## Evaluation representation boundary

DMRNet uses pooled distribution mean at inference.  HCMIG uses the final MDSFF
normalized embedding.  SSFD-Net uses its reconstructed complete multimodal
feature.  SimMLM and MMANet were published for classification and do not define
an unseen-identity biometric matcher: SimMLM is evaluated with the normalized
concatenation of its gated expert embeddings, while MMANet uses its normalized
deployment-tail feature.  These two explicit extensions do not modify their
paper training losses and are recorded here to keep the comparison auditable.

The removed lightweight adapter runner is not part of the retained experiment
scope; all comparison artifacts use this full image-level pipeline.
