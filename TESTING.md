# Comprehensive Verification & Testing Report (`TESTING.md`)
## Project: Object–Context Decoupled Pooling (OCDP) for Label-Free Mitigation of Shortcut Learning

This document records the testing strategy, test suites, verified test outcomes, and reproduction instructions for all components of the OCDP project.

---

### 1. Test Architecture Overview

The test harness is divided into three rigorous verification tiers:
1. **Data Integrity & Group Mapping Tests** ([`tests/test_waterbirds.py`](file:///d:/PROJECTS/internship/tests/test_waterbirds.py)):
   - Verifies split image counts (4,795 / 1,199 / 5,794).
   - Verifies group ID mapping formula $g = 2y + \text{place}$.
   - Verifies per-group sample distributions.
   - Verifies strict Tier 3 training isolation guard (`return_group=False`).
2. **Model Architecture & Loss Objective Tests** ([`tests/test_models_and_losses.py`](file:///d:/PROJECTS/internship/tests/test_models_and_losses.py)):
   - Verifies input/output tensor shapes and gradient backpropagation across all 5 pooling heads (`GAPHead`, `ABMILHead`, `TopKHead`, `DARHead`, `OCDPHead`).
   - Verifies parameter count overhead of the OCDP head (~0.53M parameters, ~2% of ResNet-50).
   - Verifies mathematical limits and stability of Generalized Cross-Entropy (GCE), Area Prior regularization, and Dynamic Sample Reweighting.
   - Verifies group metrics, Expected Calibration Error (ECE), and minority detection AUROC/AUPRC.
3. **Scientific Integrity & Import Guard Tests** ([`tests/test_scientific_integrity.py`](file:///d:/PROJECTS/internship/tests/test_scientific_integrity.py)):
   - Scans model and loss source files to verify zero forbidden imports from evaluation-only modules.
   - Verifies supervision tier declarations in experiment configuration files.
4. **End-to-End Smoke Training Loop Tests** ([`tests/test_smoke_train.py`](file:///d:/PROJECTS/internship/tests/test_smoke_train.py)):
   - Executes 5 training iterations with dynamic sample reweighting and GCE loss.
   - Verifies checkpoint saving and reloading.
   - Verifies validation evaluation.

---

### 2. Test Execution Log & Verified Results

All test suites were executed on the local environment (`Python 3.13.2`, `pytest 9.0.2`, `PyTorch 2.14.1+cpu` on Windows 11 AMD64).

| Test File | Test Name | Target Functionality | Result | Execution Time |
| :--- | :--- | :--- | :--- | :--- |
| `tests/test_waterbirds.py` | `test_group_mapping_formula` | $g = 2y + a$ mapping for all 4 groups | **PASSED** | < 0.01s |
| `tests/test_waterbirds.py` | `test_split_counts_and_groups` | Exact count match across train/val/test splits | **PASSED** | 9.42s |
| `tests/test_waterbirds.py` | `test_tier3_training_guard` | Enforces no group/place labels returned in training | **PASSED** | 0.15s |
| `tests/test_waterbirds.py` | `test_eval_dataset_returns_groups` | Exposes group labels for evaluation code only | **PASSED** | 0.14s |
| `tests/test_models_and_losses.py` | `test_head_shapes_and_gradients` | Forward/backward pass for all 5 pooling heads | **PASSED** | 1.82s |
| `tests/test_models_and_losses.py` | `test_ocdp_parameter_overhead` | Confirms parameter overhead is between 500k–600k (2.08%) | **PASSED** | < 0.01s |
| `tests/test_models_and_losses.py` | `test_faithful_evidence_map` | Evidence map computation ($a_u \cdot z_u[y]$) | **PASSED** | 0.02s |
| `tests/test_models_and_losses.py` | `test_generalized_cross_entropy_loss` | GCE loss formula and convergence to CE as $q \to 0$ | **PASSED** | 0.01s |
| `tests/test_models_and_losses.py` | `test_area_prior_loss` | Area regularization penalty at target $\rho = 0.3$ | **PASSED** | < 0.01s |
| `tests/test_models_and_losses.py` | `test_dynamic_sample_reweighting` | Assigns higher gradient weights to context sink errors | **PASSED** | < 0.01s |
| `tests/test_models_and_losses.py` | `test_evaluation_metrics_computation` | WGA, MGA, ECE, Brier score, minority detection AUROC | **PASSED** | 9.85s |
| `tests/test_scientific_integrity.py` | `test_no_forbidden_imports_in_models_or_losses` | Scans for forbidden imports in training code | **PASSED** | 0.12s |
| `tests/test_scientific_integrity.py` | `test_tier_declarations_in_configs` | Verifies supervision tier in experiment configs | **PASSED** | < 0.01s |
| `tests/test_smoke_train.py` | `test_smoke_training_loop` | 5-batch training step, checkpointing, validation | **PASSED** | 9.49s |

**Summary: 14 / 14 Unit & Integration Tests Passed (100% Pass Rate).**

---

### 3. How to Run the Test Suite

To reproduce all unit and integration tests:

```bash
# Run the entire test suite
python -m pytest tests/

# Run individual test modules
python -m pytest tests/test_waterbirds.py
python -m pytest tests/test_models_and_losses.py
python -m pytest tests/test_scientific_integrity.py
python -m pytest tests/test_smoke_train.py
```

---

### 4. Hardware & Environment Details

- **OS**: Windows 11 Home / Pro (10.0.26200 AMD64)
- **Python**: 3.13.2 (64-bit)
- **PyTorch**: 2.14.1+cpu
- **torchvision**: 0.29.1+cpu
- **PyArrow**: 26.0.0
- **scikit-learn**: 1.8.0
- **pandas**: 3.0.1
- **NumPy**: 2.4.2
- **pytest**: 9.0.2
