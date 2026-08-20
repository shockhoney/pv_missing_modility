# GIPSSR-Net paper visualizations

This directory contains the reproducible source for two static paper figures.
The figures use the retained full-model checkpoints and the fixed gallery/probe
protocols; no model is retrained.
Each dataset uses the frozen palmprint/palm-vein encoder pair whose SHA-256
fingerprints are embedded in its recovery checkpoint; generation stops on any
binding mismatch.

## Chart contracts

### Figure 1 — cross-modal alignment and feature recovery

- **Question:** How do IGDCA and GIPRD change cross-modal gallery-template
  alignment and missing-modality recovery geometry?
- **Takeaway tested:** IGDCA gallery templates should have greater cross-modal
  cosine than frozen-encoder templates. Coarse and final recovered embeddings
  are compared with the exact target-modality gallery identity template used by
  the verification model.
- **Data:** selected-dataset gallery templates and test probes, seed-42 full
  checkpoint. t-SNE is fitted on every identity/probe; five identities selected
  at evenly spaced label ranks are highlighted. All observations are used for
  the displayed cosine annotations.
- **Visual family:** faceted scatter (joint t-SNE within each panel).
- **Encoding:** identity uses a fixed five-color palette; modality/recovery
  status also uses marker shape and fill so the plot remains readable without
  color.
- **Output:** standalone Matplotlib PNG/PDF/SVG.

### Figure 2 — CUEF score separation and uncertainty-aware weighting

- **Question:** Does CUEF improve genuine/impostor score separation and assign
  recovery trust consistently with predicted recovery uncertainty?
- **Takeaway tested:** final CUEF scores should have less genuine/impostor
  overlap than the available-modality branch; predicted variance should track
  recovery error relative to the gallery identity template; branch weights
  should adapt across uncertainty quintiles.
- **Data:** selected-dataset palm-available / palm-vein-missing test direction,
  the seed-42 full checkpoint. Score densities and probe-level diagnostics
  are computed directly from that retained run before uncertainty binning.
- **Visual family:** density comparison, scatter with binned intervals, and
  100% stacked bars.
- **Encoding:** blue/orange for impostor/genuine distributions; four fixed
  branch colors plus hatch patterns for composition.
- **Output:** standalone Matplotlib PNG/PDF/SVG.

## Reproduce

```bash

Tongji-only render in the `pvmd` environment:

```bash
OMP_NUM_THREADS=1 conda run -n pvmd python visualizations/gipssr_paper_figures.py \
  --dataset tongji --output_dir outputs/gipssr/figures/tongji \
  --cache_dir outputs/gipssr/figures/tongji/cache
```
OMP_NUM_THREADS=1 conda run -n pvmd python visualizations/gipssr_paper_figures.py
```


Single-dataset renders default to DejaVu Serif, the closest installed match to
the Times-style serif typography in the reference figure. The environment does
not contain Times New Roman, so the match is stylistic rather than font-file exact.
Generated files are written to `outputs/gipssr/figures/`. The script also saves
`visualization_metrics.json` with the displayed statistics and source hashes.

