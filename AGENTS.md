# Project rules: Explanation-Guided Detection of Shortcut Learning

These rules apply to every task in this repository. They do not change between stages.

## What this project is
- Main method: Object-Context Decoupled Pooling (OCDP), a head that replaces GAP + FC on an
  ImageNet-pretrained ResNet-50 (full spec: docs/OCDP_Research_Plan_v2.docx, Section 5).
  Per-location evidence head, gate head, object pooling (used at test), context sink trained
  with GCE (training only), example weights w_i = CE_ctx/(CE_ctx+CE_obj), area prior, consistency loss.
- Research question: does OCDP improve worst-group accuracy WITHOUT group labels, versus label-free
  methods and versus pooling controls (GAP, attention-MIL, top-k, DAR-style) trained with the same recipe?
- Secondary: evidence-map localisation (vs Grad-CAM on ERM) and the sink's confidence as a shortcut score.
- Training uses class labels only. Group labels and CUB masks may be read ONLY by src/evaluation/.
- Log per epoch: gate mean/histogram, weight distribution, sink accuracy, per-group val accuracy.
- Benchmarks: Waterbirds (official Group DRO splits) first; UrbanCars and MetaShift later.
- Stack: Python, PyTorch, torchvision, pandas, NumPy, scikit-learn, matplotlib, tqdm, pytest.

## How to work with me
- Work one stage at a time. The current stage's instructions come from the chat, taken from
  `docs/stage_prompts.md`. Do not start the next stage on your own.
- Inspect before you modify. Never delete or overwrite existing files without saying so in the plan first.
- Ask before downloading data, installing packages, or running anything longer than ~10 minutes.
- I work on Windows in Git Bash, so give local shell commands in bash syntax (not PowerShell).
  Long runs happen on a Linux lab server, launched by me inside tmux.
- Prefer existing, maintained code (official repos, SubpopBench, the DFR code) over rewriting baselines.
  Record every deviation from an official implementation in `docs/deviations.md`.

## Scientific integrity (non-negotiable)
- Never fabricate metrics, logs, citations, or completed experiments. Never create placeholder result files.
- Cite only papers listed in `docs/literature_matrix.md`. If you think a paper is missing, say so; do not invent one.
- Test split: never read test labels or test metrics for training, tuning, early stopping or model selection.
  Test evaluation runs only in a final-evaluation step that I explicitly request.
- Declare the supervision tier of every method run: (T1) train group labels, (T2) validation group labels only,
  (T3) no group labels. State whether CUB segmentation masks or any other annotation is used.
- Do not silently reduce epochs, seeds, data, or catch exceptions to make a run "pass". Report failures.
- Report negative and null results exactly like positive ones.

## Engineering conventions
- Config-driven experiments (YAML in `configs/`), one centralised seed function, deterministic flags where feasible.
- Every run writes to `results/<run_id>/`: the resolved config, git commit hash, package/CUDA/GPU versions,
  per-epoch logs, and final metrics as JSON/CSV.
- Always report: overall accuracy, all four group accuracies with group counts, mean group accuracy,
  worst-group accuracy. Main comparisons use at least 3 seeds (mean ± std).
- Never commit datasets, checkpoints, credentials or large files. Keep `.gitignore` up to date.
- Write a pytest for every data-loading and metric function before using it in a real run.

## At the end of every stage
Use the `stage-report` skill to produce the report, then stop and wait for me.
