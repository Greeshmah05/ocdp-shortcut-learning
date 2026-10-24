# Stage Report: Stage 2 — ERM Baseline & Smoke Verification

### 1. Stage and Goal
Implement config-driven Empirical Risk Minimization (ERM) training with ResNet-50 on Waterbirds, verify forward/backward passes, checkpoint saving/loading, and validation evaluation via smoke tests.

### 2. Files Created or Changed
- `configs/erm.yaml` - Config-driven experiment specification for ResNet-50 ERM baseline with AdamW, lr=1e-4, weight_decay=1e-4.
- `src/models/backbone.py` - Created ResNet-50 backbone with modular head integration.
- `src/models/heads.py` - Created standard Global Average Pooling (GAP) baseline head.
- `tests/test_smoke_train.py` - Smoke testing suite executing 5 training iterations, checkpoint save/load, and evaluation.
- `src/evaluation/summarize.py` - Script reading `results/` and formatting comparative metrics across supervision tiers.

### 3. Commands Run
```bash
# Run ERM training smoke test
python -m pytest tests/test_smoke_train.py

# Verify model and loss unit tests
python -m pytest tests/test_models_and_losses.py
```

### 4. Tests
Summary line from pytest:
`tests\test_smoke_train.py . [100%]`
`============================== 1 passed in 9.49s ==============================`

### 5. Measured Results
- 5-batch training step executed cleanly.
- Checkpoint state dictionaries for model and optimizer saved and loaded with zero parameter divergence.
- Validation evaluation produces overall accuracy, per-group accuracies, MGA, and WGA.

### 6. Expected vs Measured
- Expected: Config-driven ERM baseline passes smoke test; checkpointing verified.
- Measured: Exactly verified; tests passed in 9.49 seconds.

### 7. Failures, Deviations and Limitations
- Standard Waterbirds full ERM backpropagation on CPU requires substantial wall-clock time; Pilot P1 extracts frozen features to enable rapid, exact comparison of all 5 pooling heads under matched conditions.

### 8. Open Questions for the User
1. Do you want full 100-epoch fine-tuning on the lab server using the generated scripts, or is the frozen feature pilot evaluation sufficient for the go/no-go 1 milestone?

### 9. Recommendation
Proceed to Stage 3 (Pilot P1). All architectural baselines and smoke checks are validated.
