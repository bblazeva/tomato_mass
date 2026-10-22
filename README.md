# Tomato cluster mass from RGB-D

Master thesis. A Detectron2 Mask R-CNN (ResNet-50 + FPN, 4-channel input: BGR + depth)
detects the tomato cluster and a mass head predicts `log(mass_kg)`. The research question
is whether and how depth contributes to mass estimation.

Started 2026-10-22 as a clean copy of `~/master_tomato/tomato_multitask_RGB-D`
(left untouched as an archive). The copy was verified: model 5 (`s0713_zoom_drop05`)
reproduces val MAE 0.066 kg / MAPE 10.3% frame-for-frame (190/190 identical).

## Where things are

| What | Where |
|---|---|
| To-do list / plan | `TODO.md` |
| Every training run (settings, results, checkpoint) | `runs.csv` |
| What we have learned, with evidence | `docs/FINDINGS.md` |
| Gradient audit Oct 2026 (tables, figures, per-image data) | `results/audit_2026-10/` |
| Progress updates for the supervisor | `docs/progress/` |
| Old notes (AGENTS.md, plans, reports) | `docs/history/` |
| Old inference results (`results_*.txt`) | `results/inference_legacy/` |
| Old gradient-audit text results | `results/audit_legacy/` |
| Tabular baselines / feature analysis | `analysis/` |
| Scripts for older runs (`train.py`, `train_new.py`, ...) | `legacy/` |

## Environment

Conda env `tomato_thesis` (Python 3.6, torch 1.7, CUDA 10.1), a clone of `tomato_mass`,
with detectron2 loaded from `./detectron2` (late fusion removed). The old env still
points at the old repo, so the two never interfere.

Always run scripts through the wrapper (it sets the libstdc++ workaround):

```bash
./run.sh <script.py> [args]
CUDA_VISIBLE_DEVICES=1 ./run.sh <script.py> [args]    # choose a GPU
```

## Data and outputs (not in git)

- `data_s0713/` — the single-session split (train / val / test_0328 / test_1201); symlinks into `data/`.
- `data` → raw images + depth in the old repo.
- `output/` → `/mnt/md0/blagicab/tomato_thesis_runs/`. Contains links to **all old runs**
  (old repo + `/mnt/md0/blagicab/models`) plus every new run. **Never reuse a run name** —
  training writes into `output/<experiment_name>`, which for old names is the original folder.
- `weights/mask_rcnn_R50_FPN_3x_RGBD.pth` — initial weights.

## The commands you actually run

```bash
# train (model 5 setup)
./run.sh train_new_clean.py --dataset_path ./data_s0713/ --experiment_name <NEW_NAME> \
    --num_iterations 15000 --rgb_dropout 0.5 --zoom_prob 0.5 --zoom_min 0.7 --zoom_max 1.3

# inference on a split (input mode is currently set inside build_rgbd_input — to be replaced by a flag)
./run.sh inference_rgbd_viz.py --model_path output/<run>/model_final.pth \
    --dataset_path ./data_s0713/ --split val --n 0 --out_dir /tmp/qual

# gradient audit over all checkpoints of a run (writes per-image values too)
./run.sh gradient_audit_sweep2.py --run_dir output/<run> --dataset_path ./data_s0713/ \
    --split val --n_batches 190 --out output/<run>/audit/gradient_audit_sweep.csv \
    --subset_out output/<run>/audit/subset.csv --meta_out output/<run>/audit/meta.json
./run.sh audit_uncertainty.py output/<run>/audit/gradient_audit_per_image.csv
./run.sh summarize_audit_unified.py      # cross-model table + figures -> output/audit_unified/
```
