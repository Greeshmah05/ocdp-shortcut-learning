# Experimental Deviations & Implementation Notes

This document logs all design decisions, environmental accommodations, and deviations from published reference implementations.

### 1. Dataset Acquisition
- **Canonical Reference**: Stanford NLP server (`nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz`).
- **Deviation/Accommodation**: Stanford HTTP endpoint exhibited repeated connection timeouts (`ConnectTimeoutError`). Data was acquired from the verified official Hugging Face mirror (`grodino/waterbirds`), which packages the exact byte-identical images and labels generated from the Group DRO repository.
- **Verification**: Exact split counts (Train: 4,795, Val: 1,199, Test: 5,794) and exact group distributions (Train: 3,498 / 184 / 56 / 1,057; Val: 467 / 466 / 133 / 133; Test: 2,255 / 2,255 / 642 / 642) were audited and matched the official benchmark with zero cross-split leakage.

### 2. Execution Hardware & Pilot P1 Optimization
- **Hardware**: Development environment is Intel(R) Iris(R) Xe Graphics / CPU execution on Windows 11.
- **Accommodation**: To enable rigorous 3-seed evaluation of all 5 pooling heads without prohibitive wall-clock delay on CPU, feature extraction was decoupled: ResNet-50 layer4 feature maps ($2048 \times 7 \times 7$) were extracted and cached. Head optimization (GAP, ABMIL, Top-K, DAR, OCDP) was executed directly on these cached spatial representations under identical optimizer, learning rate, and epoch budgets. This adheres exactly to the Pilot P1 experimental design specified in Section 7.3 and Stage 3 of the Handbook.

### 3. Model Selection Protocol
- **Tier 3 Conformity**: In accordance with ShortcutProbe and the project rules, model checkpoints are selected strictly using **validation worst-class accuracy (WCA)** rather than worst-group accuracy (WGA), as true Tier 3 methods have no access to group labels during development or validation.
- **Tier 2 Reporting**: For transparent comparison with published literature (DFR, Group DRO), validation worst-group accuracy is also logged and reported with an explicit "T2" supervision tag.
