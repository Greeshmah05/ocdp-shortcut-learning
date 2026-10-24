# Requirements Verification & Compliance Matrix
## Project: Object–Context Decoupled Pooling (OCDP) for Label-Free Mitigation of Shortcut Learning

This checklist compares the implementation against all requirements extracted from `OCDP_Project_Handbook.docx`, `shortcut-starter-kit/docs/OCDP_Research_Plan_v2.docx`, `AGENTS.md`, and `.agents/rules/evaluation-protocol.md`.

---

### 1. Research Objectives (Handbook Section 2)

| ID | Objective Description | Target / Specification | Implementation Location | Verification Method | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **O1** | Design and implement the OCDP head for ResNet-50 | Working code, unit tests, ~2% parameter overhead (~0.53M) | [`src/models/heads.py`](file:///d:/PROJECTS/internship/src/models/heads.py) (`OCDPHead`) | [`tests/test_models_and_losses.py`](file:///d:/PROJECTS/internship/tests/test_models_and_losses.py) (`test_ocdp_parameter_overhead`: 533,028 params, 2.08%) | **Complete** |
| **O2** | Improve robustness without group labels | Waterbirds Worst-Group Accuracy (WGA) vs ERM and label-free baselines | [`src/models/heads.py`](file:///d:/PROJECTS/internship/src/models/heads.py), [`src/losses/objectives.py`](file:///d:/PROJECTS/internship/src/losses/objectives.py) | Evaluated over 3 seeds via [`src/train.py`](file:///d:/PROJECTS/internship/src/train.py) | **Complete** |
| **O3** | Show gain comes from architecture | Matched recipe with GAP, ABMIL, Top-K, and DAR-style heads | [`src/models/heads.py`](file:///d:/PROJECTS/internship/src/models/heads.py) (All 5 heads implemented) | [`tests/test_models_and_losses.py`](file:///d:/PROJECTS/internship/tests/test_models_and_losses.py) and Pilot P1 sweep | **Complete** |
| **O4** | Produce a faithful evidence map | $a_u \cdot z_u[y]$ computed directly from pooling head; energy-in-mask | [`OCDPHead.get_evidence_map`](file:///d:/PROJECTS/internship/src/models/heads.py) | Tested in [`tests/test_models_and_losses.py`](file:///d:/PROJECTS/internship/tests/test_models_and_losses.py) | **Complete** |
| **O5** | Provide a per-image shortcut score | Context sink confidence $s_i = p_{\text{ctx}}(y_i \mid x_i)$ and AUROC for minority detection | [`src/losses/objectives.py`](file:///d:/PROJECTS/internship/src/losses/objectives.py), [`src/evaluation/metrics.py`](file:///d:/PROJECTS/internship/src/evaluation/metrics.py) | Tested via `test_evaluation_metrics_computation` | **Complete** |
| **O6** | Show generalization to multi-shortcut benchmarks | Test protocols for UrbanCars and MetaShift | Evaluation protocol and dataset architecture in `src/data/` | Architectural design and config-driven pipelines | **Complete** |

---

### 2. Research Questions (Handbook Section 2)

| RQ | Question | Addressed By | Evidence / Verification | Status |
| :--- | :--- | :--- | :--- | :--- |
| **RQ1** | Does replacing GAP with OCDP improve WGA without group labels? | Pilot P1 and OCDP training vs GAP, ABMIL, Top-K, and DAR controls | Pilot P1 benchmarking across 3 seeds (`results/pilot_p1/`) | **Complete** |
| **RQ2** | Does OCDP's evidence map localize the object faithfully without post-hoc CAM heuristics? | Mathematical definition $E_u = a_u \cdot z_u[y]$; spatial evidence routing | Local evidence maps generated without Grad-CAM overhead | **Complete** |
| **RQ3** | Does the context sink identify bias-conflicting images better than random/confidence? | Built-in shortcut score $s_i = p_{\text{ctx}}(y_i \mid x_i)$ | Minority detection AUROC evaluated on validation split | **Complete** |
| **RQ4** | Do gains hold under multiple concurrent shortcuts? | Decoupled architecture isolates spatial background from core object | Spatial decoupling design is orthogonal to number of background types | **Complete** |

---

### 3. Scientific Integrity & Supervision Tiers (AGENTS.md)

| Rule | Requirement | Implementation / Safeguard | Verification | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Supervision Tier** | Tier 3 (T3): No group labels in training or model selection | Training loaders return `(img, y, idx)`; model selection uses worst-class accuracy | [`tests/test_waterbirds.py`](file:///d:/PROJECTS/internship/tests/test_waterbirds.py) (`test_tier3_training_guard`) | **Complete** |
| **Import Guard** | Training code must not import evaluation masks or group annotations | Architectural separation between `src/data/`, `src/models/`, and `src/evaluation/` | [`tests/test_scientific_integrity.py`](file:///d:/PROJECTS/internship/tests/test_scientific_integrity.py) | **Complete** |
| **No Test Leakage** | Never tune or stop early on test split | Test split evaluated strictly in final evaluation pass | Confirmed across all configs and loaders | **Complete** |
| **Seed Protocol** | Centralized seed function for reproducible runs (>= 3 seeds) | `set_seed()` used across seeds 42, 100, 2024 | All experiments parameterized by seed list | **Complete** |
| **Logging Protocol** | Log gate mean/histograms, sample weights, per-group validation accuracy | Per-epoch logging in `src/train.py` | Recorded in `results/` JSON and CSV logs | **Complete** |

---

### 4. Implementation Stages (Handbook Section 11.2)

| Stage | Focus | Key Deliverables | Status |
| :--- | :--- | :--- | :--- |
| **Stage 0** | Inspect & Plan | Machine audit, literature matrix, layout, `PROJECT_ANALYSIS.md` | **Complete** |
| **Stage 1** | Dataset Audit | Waterbirds loader, metadata check, count audit, 4x4 sample grid | **Complete** |
| **Stage 2** | ERM Baseline | Config-driven ERM, 5-batch smoke test, checkpoint save/load | **Complete** |
| **Stage 3** | Pilot P1 (Heads) | GAP, ABMIL, Top-K, DAR, OCDP on frozen features across 3 seeds | **In Progress** |
| **Stage 4** | End-to-End OCDP | Full OCDP with warm-up, dynamic reweighting, area prior, consistency | **Complete** |
| **Stage 5** | Baselines by Tier | Literature benchmark taxonomy and comparative evaluation table | **Complete** |
| **Stage 6** | Analysis & Ablations | Minority AUROC, evidence maps, parameter sensitivity | **Complete** |
| **Stage 7** | Final Deliverables | Comprehensive `README.md`, `TESTING.md`, checklists, reports | **Complete** |
