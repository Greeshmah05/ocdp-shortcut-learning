---
trigger: model_decision
description: Evaluation and model-selection protocol. Read before writing or changing any training loop, evaluation code, metric, config, checkpoint-selection rule, or results table.
---

# Evaluation protocol

## Splits
- Use the official Waterbirds train / val / test splits from `metadata.csv` (split column 0/1/2).
  Verify the column names and encodings from the actual file; do not assume them.
- Groups: g = 2*y + place (G0 landbird/land, G1 landbird/water, G2 waterbird/land, G3 waterbird/water).
  Confirm this mapping against the metadata before use and keep a pytest for it.

## Model selection (declare the tier in every run's config)
- T1 / T2: select checkpoints and hyperparameters on validation worst-group accuracy.
- T3 (no group labels): select on validation worst-class accuracy, as ShortcutProbe does.
- Never compare methods from different tiers in one table without a tier column.

## Metrics
- Always: overall accuracy, per-group accuracy + counts, mean group accuracy, worst-group accuracy.
- Report Waterbirds "average accuracy" two ways and say which: plain test accuracy, and
  accuracy re-weighted by training group proportions (the convention used by Group DRO / DFR papers).
- Signal-evaluation metrics (Candidate A), on the VALIDATION split only:
  AUROC and AUPRC for detecting minority-group membership (G1, G2), precision@k,
  and Spearman correlation between signal and per-example loss / error.
- Calibration: ECE (15 bins) and Brier score, overall and per group, where feasible.
- Cost: wall-clock training time, peak GPU memory, extra inference cost of computing a signal.

## Signals to compare (Candidate A)
Random; per-example loss; 1 - max softmax confidence; predictive entropy; JTT error set;
Grad-CAM / xGradCAM background-attribution fraction (mask from the model's own saliency);
background-only true-class confidence p(y|b); s = |p(y|x) - p(y|b)|; foreground-only confidence;
ExMap-style heatmap clusters; ShortcutProbe-style latent probe if implementable.
For every signal record which labels and masks it needs.

## Interventions
- Exact counterfactuals: re-composite the same CUB bird onto other Places backgrounds using the
  original Waterbirds generation procedure. Document the script, seeds and any artifacts.
- Model-derived masks (saliency thresholding) are the default. CUB ground-truth masks may be used
  only as an oracle/upper-bound condition, labelled as such.

## Statistics
- >= 3 seeds for main comparisons; report mean ± std and a 95% bootstrap CI for worst-group accuracy.
- Test-set numbers are produced once, at the end, by an explicit final-evaluation command.
