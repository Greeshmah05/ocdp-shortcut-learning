#!/usr/bin/env bash
# ==============================================================================
# Stage 4: End-to-End OCDP Training on Linux GPU Lab Server
# ==============================================================================
# Instructions for running in a detached tmux session:
#
# 1. Start or attach to tmux:
#    tmux new -s ocdp_stage4
#
# 2. Activate your environment (adjust if using conda):
#    source .venv/bin/activate
#
# 3. Launch this script:
#    bash scripts/run_stage4_lab.sh
#
# 4. Detach safely anytime: Press Ctrl+B, then D.
#    To re-attach later: tmux attach -t ocdp_stage4
# ==============================================================================

set -e

echo "=== STAGE 4: END-TO-END OCDP BENCHMARK LAUNCH ==="
date

# Check GPU availability
python -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU (Warning)\"}')"

# Run full 3-seed end-to-end training
python -m src.train_e2e --config configs/stage4_e2e.yaml

echo "=== TRAINING COMPLETE. GENERATING REPORTS & PLOTS ==="
python -m src.evaluation.summarize
python -m src.evaluation.visualize

echo "=== STAGE 4 COMPLETE ==="
date

