# Stage prompts for Antigravity

Paste ONE stage at a time into a new conversation, starting with `/plan`.
Review the Implementation Plan, comment, then click Proceed. Finish with the `stage-report` skill.
The permanent rules live in AGENTS.md and .agents/rules/; do not repeat them here.

---

## Stage 0 - Inspect and plan (no code changes)

/plan Stage 0. Inspect only; do not create or modify any files except the stage report.
1. Inspect this repository and list what exists.
2. Report the machine: OS, Python version, PyTorch/torchvision versions, CUDA availability,
   GPU model and VRAM, free disk space. Give me the commands; I will run any that need my approval.
3. Read docs/literature_matrix.md and the research question in AGENTS.md.
4. Inspect (read-only, via the web) the official repositories for Group DRO/Waterbirds,
   DFR, SubpopBench, ExMap and Decompose-and-Compose. For each: licence, Python/PyTorch versions,
   how Waterbirds is loaded, whether it can be reused here. Do not clone anything yet.
5. Propose: repository layout, which existing codebase to build baselines on (and why),
   dependency list with pinned versions, and risks.
Produce the stage report and stop.

---

## Stage 1 - Environment and dataset audit

/plan Stage 1. Set up the environment and audit Waterbirds.
1. Create requirements.txt (pinned) and the agreed folder layout. Add .gitignore for data/, checkpoints/, results/*/checkpoints.
2. Tell me the exact download command and source for Waterbirds (official Group DRO instructions)
   and wait for me to run it. Data lives outside git, path set in configs/paths.yaml.
3. Write src/data/waterbirds.py and tests/test_waterbirds.py. Verify from metadata.csv: columns,
   label and place encodings, split encodings, group mapping g = 2*y + place.
4. Audit: counts per split x group, missing/corrupt files, exact-duplicate images (hash),
   any image appearing in two splits. Save tables to results/dataset_audit/.
5. Save a 4x4 grid of sample images per group to results/dataset_audit/.
Stage report, then stop.

---

## Stage 2 - ERM baseline

/plan Stage 2. ERM baseline on Waterbirds with ImageNet-pretrained ResNet-50.
1. Implement config-driven training (configs/erm.yaml) using the most widely used Waterbirds ERM recipe
   from the codebase chosen in Stage 0. Write the recipe (optimizer, lr, weight decay, batch size, epochs,
   augmentation, selection rule) into the config and docs/deviations.md if it differs.
2. Smoke test first: 5 batches, forward/backward, checkpoint save/load, val evaluation. Show it passing.
3. Then give me the command to run 3 seeds on the lab server inside tmux. Do not run full training yourself.
4. Model selection on VALIDATION only (T2: val worst-group acc; also log T3: val worst-class acc).
5. Write src/evaluation/summarize.py that reads results/ and produces the metrics table (mean ± std).
   Do not evaluate on test in this stage.
Stage report, then stop.

---

## Stage 3 - Pilot P1: pooling heads on frozen ERM features (go/no-go 1)

/plan Stage 3. Freeze the Stage 2 ERM backbones (3 seeds). Implement in src/models/heads.py:
gap (baseline), abmil (attention-MIL pooling), topk (top-k pooling), dar (DAR-style post-hoc spatial
attention), and ocdp (evidence head + gate head + object pooling + context sink, per plan Section 5).
1. Train each head on frozen layer4 features with the SAME recipe and T3 selection
   (worst-class validation accuracy). Class labels only.
2. For ocdp: GCE sink (q=0.7), warm-up then w_i = CE_ctx/(CE_ctx+CE_obj) (EMA, clipped), area prior rho=0.3.
   No consistency term in the pilot (frozen features).
3. Report validation WGA mean +/- std over 3 seeds per head, plus T2 selection as a labelled extra.
4. Evaluation only: energy-in-mask of each head's evidence map vs Grad-CAM on ERM, using CUB masks
   loaded by src/evaluation/ only.
5. Log gate histograms and weight distributions. Save to results/pilot_p1/.
Stage report, then stop. I will make the go/no-go 1 decision.

---

## Later stages (write the detailed prompt after go/no-go 1)
- Stage 4: end-to-end OCDP (backbone fine-tuned), consistency term, stop-gradient option, stability checks.
- Stage 5: baselines by tier: JTT, LfF, label-free DFR, EVaLS, ExMap, DaC, Group DRO; matched budgets.
- Stage 6: ablations; shortcut-score AUROC; localisation, deletion/insertion; background-swap sensitivity.
- Stage 7: UrbanCars, MetaShift; single final test evaluation; statistics and reproducibility audit.
