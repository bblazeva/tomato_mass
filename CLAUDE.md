# CLAUDE.md — context for AI sessions

Master thesis: tomato cluster mass from RGB-D (Detectron2 Mask R-CNN + mass head predicting
`log(mass_kg)`, one cluster per image). Read `README.md` (layout, commands), `TODO.md` (current
plan), `runs.csv` (every run) and `docs/FINDINGS.md` (what is known) before doing anything.
Full pre-October history: `docs/history/AGENTS.md`.

## Environment
- Always run Python via `./run.sh <script> [args]` (env `tomato_thesis`, Python 3.6, torch 1.7).
  No `from __future__ import annotations`; f-strings are fine.
- detectron2 is vendored in `./detectron2` (late fusion removed: `mass_fcn1.weight` is (256,256,3,3)).
- 2× RTX 2080 Ti. Use `CUDA_VISIBLE_DEVICES`.

## Rules
- `output/` links to `/mnt/md0/blagicab/tomato_thesis_runs/`, which links to every OLD run folder.
  Never write into an existing run folder; never reuse a run name; never delete checkpoints.
- Inference/audit input construction must match training exactly: BGR + depth_mm, invalid depth
  (0 mm) filled with 915.043, PIXEL_MEAN [103.530, 116.280, 123.675, 915.043],
  PIXEL_STD [57.375, 57.120, 58.395, 411.683] (runs named `*_notNorm` / 4.1 use [1,1,1,411.683]).
- Mass bins / val set: `data_s0713` val = 4 clusters (208, 490, 670, 940 g), 190 frames.
  Near-duplicate frames: treat clusters, not frames, as the unit of evidence.
- Same training command gives 10–18% val MAPE (no seed yet): do not interpret single-run
  differences of a few points.
- Training scripts save no config yet; the exact command of old runs is in
  `docs/history/train_commands.txt` (from `~/.bash_history`).
- Ask before launching training runs (they take ~1 h of GPU each).
