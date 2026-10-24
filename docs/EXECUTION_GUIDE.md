# OCDP Project: Verified Execution & Reproduction Guide

**Project Root**: `d:\PROJECTS\internship`  
**Target Architecture**: ResNet-50 + Object–Context Decoupled Pooling (OCDP)  
**Supervision Tier**: Tier 3 (Label-Free: class labels only; no group labels during training or model selection)  
**Authoritative Protocol**: `AGENTS.md` and `.agents/rules/evaluation-protocol.md`  

---

## 1. Actual Project Root Directory

- **Local Absolute Path**: `d:\PROJECTS\internship`
- **Shell Compatibility**: Git Bash (recommended per project instructions) and Windows PowerShell.

---

## 2. Verified Dataset Source & Local Directory

- **Dataset**: Waterbirds Benchmark (Sagawa et al., ICLR 2020)
- **Official Hugging Face Mirror**: `https://huggingface.co/datasets/grodino/waterbirds`
- **Local Storage Directory**: `d:\PROJECTS\internship\data\raw\`
- **Local Files Present**:
  - `data/raw/train.parquet` (212,123,440 bytes) — 4,795 images
  - `data/raw/validation.parquet` (48,110,990 bytes) — 1,199 images
  - `data/raw/test.parquet` (231,664,112 bytes) — 5,794 images
  - `data/metadata.csv` (161,734 bytes) — 11,788 sample index mappings
- **Split & Group Verification Table**:
  | Split | Total Images | G0: Land/Land (Maj) | G1: Land/Water (Min) | G2: Water/Land (Min) | G3: Water/Water (Maj) |
  | :--- | :--- | :--- | :--- | :--- | :--- |
  | **Train** | 4,795 | 3,498 | 184 | 56 | 1,057 |
  | **Validation** | 1,199 | 467 | 466 | 133 | 133 |
  | **Test** | 5,794 | 2,255 | 2,255 | 642 | 642 |
  | **Total** | 11,788 | 6,220 | 2,905 | 831 | 1,832 |
- **Data Integrity Audit**:
  - Exact duplicates within splits: **0**
  - Cross-split image leakage: **0**
  - Serialized audit: [`results/dataset_audit/audit_summary.json`](file:///d:/PROJECTS/internship/results/dataset_audit/audit_summary.json)
  - Visual verification grid: [`results/dataset_audit/sample_grid.png`](file:///d:/PROJECTS/internship/results/dataset_audit/sample_grid.png)

---

## 3. Dataset Download or Reconstruction Procedure

*Note: The dataset is already downloaded and verified on disk. You do NOT need to download it again.*

If the dataset is accidentally removed, re-download the exact parquet files and regenerate metadata using this standalone script:

```bash
# In Git Bash or PowerShell from d:/PROJECTS/internship:
python -c "
import os, requests, io
import pyarrow.parquet as pq
import pandas as pd

os.makedirs('data/raw', exist_ok=True)
splits = ['train', 'validation', 'test']

# 1. Download parquet archives from verified Hugging Face mirror
for split in splits:
    url = f'https://huggingface.co/datasets/grodino/waterbirds/resolve/main/data/{split}-00000-of-00001.parquet'
    dest = f'data/raw/{split}.parquet'
    if not os.path.exists(dest):
        print(f'Downloading {split}...')
        r = requests.get(url, stream=True)
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk: f.write(chunk)
        print(f'{split} downloaded ({os.path.getsize(dest)} bytes).')
    else:
        print(f'{split} already exists.')

# 2. Regenerate metadata.csv
dfs = []
for split_idx, split in enumerate(splits):
    p = f'data/raw/{split}.parquet'
    table = pq.read_table(p)
    df = table.select(['label', 'place']).to_pandas()
    df['split'] = split_idx  # 0: train, 1: val, 2: test
    df['group'] = 2 * df['label'] + df['place']
    df['img_id'] = range(len(df))
    dfs.append(df)

full_meta = pd.concat(dfs, ignore_index=True)
full_meta.to_csv('data/metadata.csv', index=False)
print('Saved data/metadata.csv with', len(full_meta), 'rows.')
"
```

---

## 4. Important Project Folders and Files

| Path | Purpose | Recommended Action |
| :--- | :--- | :--- |
| [`AGENTS.md`](file:///d:/PROJECTS/internship/AGENTS.md) | Permanent project rules & scientific integrity principles | **Read first, do not modify** |
| [`.agents/rules/evaluation-protocol.md`](file:///d:/PROJECTS/internship/.agents/rules/evaluation-protocol.md) | Evaluation metrics, model selection, tier rules | **Read first, do not modify** |
| [`README.md`](file:///d:/PROJECTS/internship/README.md) | Project architecture, math equations, results overview | Read for architectural reference |
| [`PROJECT_ANALYSIS.md`](file:///d:/PROJECTS/internship/PROJECT_ANALYSIS.md) | In-depth engineering blueprint and component details | Reference |
| [`TESTING.md`](file:///d:/PROJECTS/internship/TESTING.md) | Test harness specifications and verified test results | Reference |
| [`REQUIREMENTS_CHECKLIST.md`](file:///d:/PROJECTS/internship/REQUIREMENTS_CHECKLIST.md) | Objective and research question compliance matrix | Reference |
| [`configs/paths.yaml`](file:///d:/PROJECTS/internship/configs/paths.yaml) | Centralized paths for data, checkpoints, and results | Do not modify |
| [`configs/pilot_p1.yaml`](file:///d:/PROJECTS/internship/configs/pilot_p1.yaml) | Hyperparameters and seeds for Pilot P1 benchmark | Main config for pilot experiments |
| [`configs/erm.yaml`](file:///d:/PROJECTS/internship/configs/erm.yaml) | Hyperparameters for ERM baseline | Stage 2 config |
| [`configs/ocdp_end_to_end.yaml`](file:///d:/PROJECTS/internship/configs/ocdp_end_to_end.yaml) | End-to-end backbone fine-tuning config | Stage 4 config |
| [`src/data/waterbirds.py`](file:///d:/PROJECTS/internship/src/data/waterbirds.py) | Dataset loader with Tier 3 supervision guard | Core data loader (stable) |
| [`src/models/heads.py`](file:///d:/PROJECTS/internship/src/models/heads.py) | GAP, ABMIL, Top-K, DAR, and OCDP pooling heads | Core model heads (stable) |
| [`src/models/backbone.py`](file:///d:/PROJECTS/internship/src/models/backbone.py) | ResNet-50 feature extractor and combined wrapper | Core backbone (stable) |
| [`src/losses/objectives.py`](file:///d:/PROJECTS/internship/src/losses/objectives.py) | GCE loss, Area Prior, Dynamic Sample Reweighting | Core loss suite (stable) |
| [`src/evaluation/metrics.py`](file:///d:/PROJECTS/internship/src/evaluation/metrics.py) | WGA, MGA, ECE, Brier score, minority detection | Evaluation metrics (stable) |
| [`src/evaluation/summarize.py`](file:///d:/PROJECTS/internship/src/evaluation/summarize.py) | Generates markdown tables comparing with literature | Result summary tool |
| [`src/evaluation/visualize.py`](file:///d:/PROJECTS/internship/src/evaluation/visualize.py) | Generates publication-quality charts and plots | Plotting tool |
| [`src/train.py`](file:///d:/PROJECTS/internship/src/train.py) | Unified training engine for feature extraction & heads | **Official execution entry point** |
| [`checkpoints/cache/`](file:///d:/PROJECTS/internship/checkpoints/cache/) | Pre-extracted ResNet-50 layer4 feature tensors | **Do NOT delete** (saves 12 min runtime) |
| [`results/dataset_audit/`](file:///d:/PROJECTS/internship/results/dataset_audit/) | Audit statistics, counts table, sample grid | Verified audit artifacts |
| [`results/pilot_p1/`](file:///d:/PROJECTS/internship/results/pilot_p1/) | Output destination for Pilot P1 results and plots | Experiment results folder |

---

## 5. Environment Setup Instructions

The local environment runs on Python 3.13.2 (Windows 11 AMD64).

### Option A: Using the Existing Global/User Environment (Recommended Locally)
All dependencies (`torch`, `torchvision`, `pyarrow`, `pandas`, `scikit-learn`, `matplotlib`, `pytest`, `PyYAML`, `tqdm`) are already installed in your Python environment:

```bash
# Verify Python version
python --version
# Expected: Python 3.13.2

# Verify PyTorch and torchvision
python -c "import torch, torchvision; print('PyTorch:', torch.__version__, 'torchvision:', torchvision.__version__)"
# Expected: PyTorch: 2.14.1+cpu torchvision: 0.29.1+cpu
```

### Option B: Creating a Fresh Virtual Environment (Optional)
```bash
# In Git Bash:
python -m venv .venv
source .venv/Scripts/activate

# In Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies:
pip install -r requirements.txt
```

---

## 6. Exact Commands for Tests and Smoke Tests

### Run All 14 Automated Unit and Integration Tests
```bash
# Git Bash or PowerShell:
python -m pytest tests/
```
*Expected Output*: `14 passed in ~22s`

### Run Individual Test Suites
```bash
# 1. Dataset & Tier 3 supervision guard tests
python -m pytest tests/test_waterbirds.py

# 2. Pooling heads, gradients, and loss mathematical limit tests
python -m pytest tests/test_models_and_losses.py

# 3. Scientific integrity & forbidden import scan
python -m pytest tests/test_scientific_integrity.py

# 4. Quick smoke test (5-batch forward/backward, checkpoint save/load, val eval)
python -m pytest tests/test_smoke_train.py
```

---

## 7. Exact Commands to Execute the Full Benchmark (Pilot P1)

The feature extraction step has **already completed** and is cached in [`checkpoints/cache/train_feats.pt`](file:///d:/PROJECTS/internship/checkpoints/cache/train_feats.pt) (1.92 GB) and [`checkpoints/cache/val_feats.pt`](file:///d:/PROJECTS/internship/checkpoints/cache/val_feats.pt) (481 MB).

When you run `src.train`, it detects the cache automatically, skips feature extraction, and trains the 5 pooling heads across 3 seeds (`42`, `100`, `2024`):

```bash
# Execute Pilot P1 benchmark
python -m src.train --pilot --config configs/pilot_p1.yaml
```

**Estimated Runtime**: ~3 to 5 minutes total on CPU (~15 to 20 seconds per head per seed).

---

## 8. Commands to Monitor Experiments

When running in Git Bash or PowerShell, progress is displayed live via `tqdm` progress bars and per-epoch validation logs:
- Current pooling head (`GAP`, `ABMIL`, `TOPK`, `DAR`, `OCDP`)
- Current random seed (`42`, `100`, `2024`)
- Per-epoch training loss, gate mean, and validation metrics (`val_wga`, `val_mga`, `val_wca`, `val_overall`)

---

## 9. Commands to Evaluate Checkpoints & Generate Figures

Once the benchmark finishes:

### 1. Generate Summary Metrics Table
```bash
python -m src.evaluation.summarize
```
Reads [`results/pilot_p1/pilot_p1_summary.csv`](file:///d:/PROJECTS/internship/results/pilot_p1/pilot_p1_summary.csv) and outputs:
- Mean ± std worst-group accuracy (WGA) across 3 seeds for each head.
- Per-group accuracy breakdown ($G_0, G_1, G_2, G_3$).
- Literature comparison table with explicit supervision tier tags (T1, T2, T3).

### 2. Generate Publication-Quality Figures
```bash
python -m src.evaluation.visualize
```
Generates:
- `results/pilot_p1/head_wga_comparison.png`: Bar chart of WGA and MGA with error bars across 3 seeds.
- `results/pilot_p1/group_accuracy_breakdown.png`: 4-panel subplot showing individual group accuracies ($G_0..G_3$) across all 5 heads.

---

## 10. Expected Output Locations

```
results/
├── dataset_audit/
│   ├── audit_summary.json            # Exact image counts, split hashes, zero-leakage audit
│   ├── dataset_audit_table.csv       # Formatted per-split group distributions
│   └── sample_grid.png               # 4x4 visual verification grid
└── pilot_p1/
    ├── pilot_p1_results.json         # Complete per-epoch logs and metrics for all 5 heads x 3 seeds
    ├── pilot_p1_summary.csv          # Flattened tabular results for Pandas / statistical analysis
    ├── head_wga_comparison.png       # Publication bar chart comparing WGA and MGA
    └── group_accuracy_breakdown.png  # Subplots of G0, G1, G2, G3 performance
```

---

## 11. Common Errors and Solutions

1. **`FileNotFoundError: Parquet file not found at: data/raw/...`**
   - *Cause*: Script executed from outside the workspace root or parquet files moved.
   - *Fix*: Ensure your working directory is `d:\PROJECTS\internship`.
2. **`ModuleNotFoundError: No module named 'src'`**
   - *Cause*: Running `python src/train.py` directly instead of as a module.
   - *Fix*: Always invoke via `python -m src.train` with `d:\PROJECTS\internship` as working directory.
3. **`MemoryError` during feature extraction**
   - *Cause*: Re-extracting features with high batch size on limited RAM.
   - *Fix*: Do not delete `checkpoints/cache/*.pt`. The features are already extracted and verified.

---

## 12. Project Status & Final Completion Checklist

- [x] Stage 0: Environment & codebase inspection completed ([`docs/reports/stage_0_inspect.md`](file:///d:/PROJECTS/internship/docs/reports/stage_0_inspect.md)).
- [x] Stage 1: Official Waterbirds acquired, audited, and verified without leakage ([`docs/reports/stage_1_data.md`](file:///d:/PROJECTS/internship/docs/reports/stage_1_data.md)).
- [x] Stage 2: Config-driven ERM baseline, 5-batch smoke test, checkpoint save/load verified ([`docs/reports/stage_2_erm.md`](file:///d:/PROJECTS/internship/docs/reports/stage_2_erm.md)).
- [x] Spatial Feature Extraction: ResNet-50 layer4 representations cached ([`checkpoints/cache/`](file:///d:/PROJECTS/internship/checkpoints/cache/)).
- [x] Test Suite: 14 / 14 unit and integration tests passing.
- [ ] **Pending Action**: Run `python -m src.train --pilot --config configs/pilot_p1.yaml` to complete Pilot P1 head training.
- [ ] **Pending Action**: Run `python -m src.evaluation.summarize` and `python -m src.evaluation.visualize`.
- [ ] **Pending Action**: Review results for Go/No-Go 1 milestone decision before Stage 4.
