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

## This machine (moved out of README.md, which is written for the supervisor)
- This repo started 2026-10-22 as a clean copy of `~/master_tomato/tomato_multitask_RGB-D`, kept as an archive.
  Don't delete or move that folder: `data/` and `data_s0713/depth` link into it, as do some old runs.
- Env `tomato_thesis` is a clone of `tomato_mass`, with detectron2 loaded from `./detectron2`.
- `data/` = the original mixed-session splits (links `images/`, `gt_json/`, `depth/`). The depth folder is
  still called `iter_4_depth_matched` in the old repo.
- `weights/mask_rcnn_R50_FPN_3x_RGBD.pth` = initial weights, with a link at the repo root (`--weights` default).
- Rebuilding splits needs `data/`. `data_prep/make_s0713_split.py` DELETES `data_s0713/` first: read its docstring.
- Other commands: `audit/audit_uncertainty.py <per_image.csv>`, `audit/summarize_audit_unified.py`,
  `data_prep/inspect_zoom_aug.py`.

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
- `train.py` = former `train_new_clean.py`. It writes `run_info.json` + `config.yaml` per run, takes
  `--seed`, and refuses existing run names. Runs before 2026-10-22 have no such record: their exact
  command is in `docs/history/train_commands.txt` (from `~/.bash_history`).
- Depth maps live in `<dataset>/depth/<image_name>.npy` (mm). In the old repo and in `docs/history` this
  folder is called `iter_4_depth_matched` — same files.
- Layout: core pipeline at the root (`train.py`, `evaluate.py`, `utils_detection.py`, `mass_evaluator.py`);
  `data_prep/` (split builders, `inspect_zoom_aug.py`), `audit/` (gradient audit), `analysis/` (tabular
  baselines), `legacy/`. Scripts in subfolders import root modules via `run.sh` (repo root on PYTHONPATH,
  cwd = repo root), so always run them as `./run.sh audit/<script>.py`.
- Shared modules: `utils_detection.py` (dataset dicts; mass parsed from the file name) and
  `mass_evaluator.py` (former `mass_evaluator_mean.py`; scores only detections matched to GT at IoU >= 0.5).
- Evaluation: `evaluate.py` (replaces `inference_rgbd_viz.py`, which was deleted 2026-10-22; verified
  frame-for-frame on model 5). Input modes via `--input_modes`, never by editing the input function.
- `legacy/` holds scripts of older pipelines (incl. the depth-only `legacy/train.py` and its
  `legacy/inference_depth_only.py`, which still has `depth_mm[:] = DEPTH_MEAN_MM` switched on, line 73). Don't point
  `legacy/` scripts at the root `train.py`: same module name, different pipeline.
- The user commits to git themselves — don't commit or add git logic to scripts.
- Ask before launching training runs (they take ~1 h of GPU each).
