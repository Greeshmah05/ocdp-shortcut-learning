---
name: stage-report
description: Write the end-of-stage report for this research project. Use when a stage from docs/stage_prompts.md is finished, or when the user asks for a stage report, status, or summary of what was done.
---

# Stage report

Write the report to `docs/reports/stage_<N>_<short-name>.md` and also show it in chat.
Use exactly these sections, in this order. Keep it factual; no adjectives about results.

1. **Stage and goal** - one line.
2. **Files created or changed** - a list of paths with one line each on what changed.
3. **Commands run** - the exact commands, in order, so the user can reproduce them in Git Bash
   (local) or bash (lab server). Include environment setup commands if any.
4. **Tests** - which pytest tests or checks ran and whether each passed. Paste the summary line.
5. **Measured results** - only numbers read from files in `results/` or from logs produced in this stage.
   For each number give the file it came from. Use a table: run id, seed, supervision tier,
   overall acc, G0-G3 acc (with counts), mean group acc, worst-group acc.
   If no runs were completed, write "No measured results in this stage."
6. **Expected vs measured** - where an expectation existed, state both, clearly labelled.
7. **Failures, deviations and limitations** - everything that failed, was skipped, was shortened,
   or deviates from official implementations (also append deviations to `docs/deviations.md`).
8. **Open questions for the user** - numbered, each answerable in one line.
9. **Recommendation** - proceed / repeat / revise, with the one reason that matters most.

Then suggest a git commit command (bash syntax) with a clear message, and stop.
Do not start the next stage.
