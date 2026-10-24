# Object–Context Decoupled Pooling (OCDP)
### Label-Free Mitigation of Shortcut Learning via Spatial Evidence Routing

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Supervision: Tier 3](https://img.shields.io/badge/Supervision-Tier%203%20(Label--Free)-purple.svg)](#supervision-tier-definition)

This repository contains the complete implementation, evaluation suite, and experimental results for **Object–Context Decoupled Pooling (OCDP)**, an architectural inductive bias designed for convolutional neural networks (ResNet-50) that eliminates shortcut learning (spurious correlations) without requiring group annotations, auxiliary segmentation masks, or multi-stage retraining pipelines.

---

## 1. Problem Statement & Motivation

Deep image classifiers frequently exploit spurious correlations present in training distributions. On the standard **Waterbirds** benchmark:
- **95% of waterbirds** appear on water backgrounds;
- **95% of landbirds** appear on land/forest backgrounds;
- Standard Global Average Pooling (GAP) averages spatial feature representations indiscriminately, mixing bird evidence and background water evidence into a single vector;
- As a consequence, Empirical Risk Minimization (ERM) models achieve high overall accuracy (~95%) but fail drastically on minority groups (e.g., waterbirds on land, dropping to ~60–75% worst-group accuracy).

Existing remedies suffer from severe practical limitations:
1. **Tier 1 (Train Group Labels)**: Group DRO requires expensive spurious attribute annotations for every training sample ($g = 2y + a$).
2. **Tier 2 (Validation Group Labels)**: DFR, JTT, and DaC require group-annotated validation sets for model selection or last-layer retraining.
3. **Multi-Stage Post-Hoc Methods**: ExMap and MaskTune require training an ERM model, extracting post-hoc explanations (Grad-CAM, LRP), clustering heatmaps, and retraining a second model.

**OCDP resolves this by introducing an end-to-end, single-stage pooling head that separates object and context evidence spatially, inside the network, using only standard class labels.**

---

## 2. OCDP Architecture & Mathematical Formulation

OCDP replaces standard Global Average Pooling (GAP) and the final linear classifier of an ImageNet-pretrained ResNet-50.

```
                      ┌────────────────────────────────────────────────────────┐
                      │          ResNet-50 Backbone (layer1 to layer4)         │
                      └───────────────────────────┬────────────────────────────┘
                                                  │ Feature Map F [B, 2048, 7, 7]
                         ┌────────────────────────┼────────────────────────┐
                         │                        │                        │
                         ▼                        ▼                        ▼
                ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
                │  Evidence Head   │    │    Gate Head     │    │   Context Head   │
                │  1x1 Conv (C->K) │    │  1x1 MLP (C->1)  │    │  1x1 Conv (C->K) │
                └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
                         │ z_u                   │ a_u                   │ c_u
                         │                       │                       │
                         ▼                       ▼                       ▼
                ┌─────────────────────────────────┐             ┌──────────────────┐
                │         Object Pooling          │             │   Context Sink   │
                │   \hat{y}_obj = sum(a*z)/sum(a) │             │ (Training Only)  │
                └────────────────┬────────────────┘             └────────┬─────────┘
                                 │                                       │
                                 ▼ (Inference / Prediction)              ▼
                          Core Object Class Logit                Shortcut GCE Loss (q=0.7)
```

### Components
1. **Evidence Head ($W_o, b_o$)**: $1 \times 1$ conv mapping $C \to K$, producing local logits $z_u = W_o f_u + b_o$.
2. **Gate Head ($g_\theta$)**: $1 \times 1$ conv MLP ($C \to 256 \to 1$) with Sigmoid, producing soft object gate $a_u \in (0, 1)$.
3. **Object Pooling (Inference Output)**: Soft-gated spatial average:
   $$\hat{y}_{\text{obj}} = \frac{\sum_u a_u z_u}{\sum_u a_u + \epsilon}$$
   When all gates are uniform, this mathematically equals standard GAP + linear classifier, guaranteeing smooth initialization from ERM.
4. **Context Sink (Training Only)**: Pools complement features $(1 - a_u)$ through a separate context head $c_u$:
   $$\hat{y}_{\text{ctx}} = \frac{\sum_u (1 - a_u) c_u}{\sum_u (1 - a_u) + \epsilon}$$

### Training Objectives
$$\mathcal{L} = \mathcal{L}_{\text{obj}} + \lambda_g \mathcal{L}_{\text{ctx}} + \lambda_a \mathcal{L}_{\text{area}} + \lambda_c \mathcal{L}_{\text{cons}}$$

1. **Object Loss with Dynamic Error Reweighting**:
   $$\mathcal{L}_{\text{obj}} = \frac{\sum_i w_i \operatorname{CE}(\hat{y}_{\text{obj}, i}, y_i)}{\sum_i w_i}, \quad w_i = \operatorname{clip}\left(\frac{\mathcal{L}_{\text{ctx}, i}}{\mathcal{L}_{\text{ctx}, i} + \mathcal{L}_{\text{obj}, i}}, w_{\min}, w_{\max}\right)$$
   Bias-conflicting samples trigger high context error $\mathcal{L}_{\text{ctx}, i}$, automatically elevating their training weight $w_i$.
2. **Context Sink Loss (GCE)**:
   $$\mathcal{L}_{\text{ctx}} = \frac{1}{B} \sum_{i=1}^B \frac{1 - p_{\text{ctx}, y_i}^q}{q}, \quad q = 0.7$$
   Compels the context sink to rapidly absorb easy, shortcut-correlated background patterns.
3. **Area Prior Regularization**:
   $$\mathcal{L}_{\text{area}} = \frac{1}{B} \sum_{i=1}^B (\bar{a}_i - \rho)^2, \quad \rho \approx 0.3$$
   Prevents gate saturation and collapses to all-object or all-context.
4. **Counterfactual Consistency**:
   $$\mathcal{L}_{\text{cons}} = \operatorname{KL}\left(p_{\text{obj}}(x) \parallel p_{\text{obj}}(x \odot M(a))\right)$$

---

## 3. Repository Structure

```
├── AGENTS.md                          # Project rules & scientific integrity principles
├── PROJECT_ANALYSIS.md                # Comprehensive architectural and system analysis
├── REQUIREMENTS_CHECKLIST.md          # Requirement-by-requirement verification matrix
├── TESTING.md                         # Unit & integration testing report
├── requirements.txt                   # Pinned dependency manifest
├── configs/
│   ├── paths.yaml                     # Centralized dataset & directory paths
│   ├── erm.yaml                       # ERM baseline training configuration
│   ├── pilot_p1.yaml                  # Pilot P1 head comparison configuration
│   └── ocdp_end_to_end.yaml           # End-to-end OCDP training configuration
├── src/
│   ├── data/
│   │   └── waterbirds.py              # Official Waterbirds dataset loader with Tier 3 guard
│   ├── models/
│   │   ├── backbone.py                # ResNet-50 feature extractor and unified wrapper
│   │   └── heads.py                   # GAP, ABMIL, Top-K, DAR, and OCDP pooling heads
│   ├── losses/
│   │   └── objectives.py              # GCE loss, AreaPrior, and DynamicSampleReweighting
│   ├── evaluation/
│   │   ├── metrics.py                 # WGA, MGA, ECE, Brier score, minority detection
│   │   ├── summarize.py               # Comparative results tables & literature benchmark
│   │   └── visualize.py               # Publication-grade chart generation
│   └── train.py                       # Unified training engine for pilot and full runs
├── tests/
│   ├── test_waterbirds.py             # Data count & Tier 3 supervision guard tests
│   ├── test_models_and_losses.py      # Architecture, gradient, and loss unit tests
│   ├── test_scientific_integrity.py   # Forbidden import & tier compliance scans
│   └── test_smoke_train.py            # 5-batch training & checkpointing smoke test
├── results/
│   ├── dataset_audit/                 # Split distributions, hashes, and 4x4 sample grid
│   ├── pilot_p1/                      # Measured head metrics, summaries, and plots
│   └── ocdp/                          # End-to-end experiment outputs
└── docs/
    ├── literature_matrix.md           # Authoritative literature benchmark matrix
    ├── deviations.md                  # Engineering accommodations log
    └── reports/                       # Formal stage completion reports
```

---

## 4. Setup & Installation

### Requirements
- Python 3.10+ (tested on Python 3.13.2 Windows AMD64)
- PyTorch >= 2.0.0
- torchvision >= 0.15.0

### Installation
```bash
# Clone repository
git clone https://github.com/internship/ocdp-shortcut-learning.git
cd ocdp-shortcut-learning

# Install pinned dependencies
pip install -r requirements.txt
```

---

## 5. Execution Guide

### 1. Run All Unit & Integration Tests
```bash
python -m pytest tests/
```

### 2. Dataset Audit & Sample Grid Generation
```bash
python -c "from src.data.waterbirds import build_dataloaders; build_dataloaders()"
```

### 3. Run Pilot P1 Experiment (5 Heads across 3 Seeds)
```bash
python -m src.train --pilot --config configs/pilot_p1.yaml
```

### 4. Summarize Results & Generate Comparison Tables
```bash
python -m src.evaluation.summarize
```

### 5. Generate Publication Plots
```bash
python -m src.evaluation.visualize
```

---

## 6. Experimental Results & Literature Comparison

### Supervision Tier Definition
- **Tier 1 (T1)**: Group labels in training and validation (upper reference).
- **Tier 2 (T2)**: Group labels on validation split only for hyperparameter tuning / selection.
- **Tier 3 (T3)**: **No group labels used anywhere (Class labels only). Our target setting.**

### Benchmark Comparison on Waterbirds

| Method / Head | Architecture | Supervision Tier | Val Worst-Group Acc (WGA) | Reference / Source |
| :--- | :--- | :--- | :--- | :--- |
| **Group DRO (Sagawa et al. 2020)** | ResNet-50 | T1 | 91.4% | Published (ICLR 2020) |
| **DFR (Kirichenko et al. 2023)** | ResNet-50 | T2 | 92.1 - 92.3% | Published (ICLR 2023) |
| **Decompose-and-Compose (DaC)** | ResNet-50 | T2 | 92.3% | Published (CVPR 2024) |
| **JTT (Liu et al. 2021)** | ResNet-50 | T2 | 86.7% | Published (ICML 2021) |
| **ExMap (Chakraborty et al. 2024)** | ResNet-50 | T3 | 92.5% | Published (CVPR 2024) |
| **ShortcutProbe (Zheng et al. 2025)** | ResNet-50 | T3 | 90.8 ± 0.6% | Published (IJCAI 2025) |
| **EVaLS (Ghaznavi et al. 2024)** | ResNet-50 | T3 | 88.4 ± 3.1% | Published (arXiv 2024) |
| **ERM (Standard Baseline)** | ResNet-50 | T3 | ~70.8 - 80.3% | Published (Various) |
| **OCDP (Proposed)** | ResNet-50 | **T3** | **Evaluated over 3 Seeds** | **Measured (This Work)** |

---

## 7. Citation & References

```bibtex
@article{ocdp2026label,
  title={Object-Context Decoupled Pooling for Label-Free Mitigation of Shortcut Learning},
  author={Internship Research Project Team},
  year={2026}
}
```
