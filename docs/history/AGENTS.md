# AGENTS.md — RGB-D Tomato Cluster Mass Estimation

Context handoff for agents picking up this project mid-stream. Read this first.

## What the project is

Estimate the **mass (kg) of a tomato cluster** from an RGB-D image, using a Detectron2
Mask R-CNN that detects/segments the cluster and a custom regression head that predicts
`log(mass_kg)`. There is **exactly one tomato cluster per image**, so per-image mass =
per-cluster mass.

Goal beyond accuracy: demonstrate whether/how **depth** contributes to mass estimation.
**Resolved (May 2026, see Key findings 8–10):** depth is *redundant same-session* (RGB recovers scale monocularly) but *more robust cross-session* (lighting-invariant); a naive RGB-D model defaults to RGB and must be forced/redesigned to use depth's robustness. Earlier "depth ignored" was an optimizer artifact (no RGB-dropout), not true depth-blindness.

## Directory layout

```
/home/blagicab/master_tomato/
├── AGENTS.md                     ← this file
├── plan.md                       ← tabular-baseline experiment plan
├── thesis_progress_update.md     ← written summary for the thesis supervisor
├── ARCHITECTURE_AGENT_BRIEF.md   ← proposed PointNet redirect (not yet executed)
├── analysis/
│   ├── extract_features.py       ← builds features.csv from GT masks + depth + intrinsics
│   ├── features.csv              ← 1190 rows × 18 engineered geometric features
│   ├── baseline_area_m2.py       ← area_m2 log-log linear baseline
│   ├── baseline_tabular.py       ← Ridge/ElasticNet/RF/HistGBM/TinyMLP tabular baselines
│   └── *.png, *.csv              ← analysis outputs
└── tomato_multitask_RGB-D/       ← the actual model code + data
    ├── train.py                  ← DEPTH-ONLY / experimental training (depth replicated into BGR)
    ├── train_new.py              ← RGB-D training (the "clean_baseline" lineage)
    ├── inference_rgbd_viz.py     ← RGB-D inference + viz
    ├── inference_depth_only.py   ← depth-only inference (MUST match train.py normalization)
    ├── inspect_model_input.py    ← prints per-channel stats of what the model actually receives
    ├── utils_detection.py        ← dataset dict builder (mass parsed from filename prefix)
    ├── make_s0713_split.py       ← builds data_s0713/ train+val (2025-07-13, split by mass)
    ├── make_s0713_test_session.py ← adds cross-session test_<MMDD> splits to data_s0713/
    ├── detectron2/               ← vendored detectron2 (LF ON variant of roi_heads)
    ├── detectron2_LF/, detectron2_no_late_fusion/  ← alternate trees (see "Late fusion")
    ├── data/                     ← images/, depth (.npy), gt_json/, camera_intrinsics.json
    ├── data_s0713/               ← single-session split (symlinks; see "Data facts")
    └── output/<run_name>/        ← checkpoints + metrics.json per run
```

## Environment (important — there are traps)

- Python: **`~/miniconda3/envs/tomato_mass/bin/python3`** (Python 3.6, sklearn 0.24.2, torch, detectron2).
- **PIL/libstdc++ error workaround** (needed for many scripts):
  ```bash
  LD_PRELOAD=$HOME/miniconda3/envs/tomato_mass/lib/libstdc++.so.6 \
    ~/miniconda3/envs/tomato_mass/bin/python3 <script>
  ```
- `xgboost` is NOT installed (don't `pip install` without asking; HistGBM is the substitute).
- `HistGradientBoostingRegressor` requires `from sklearn.experimental import enable_hist_gradient_boosting` first (sklearn 0.24).
- Do NOT use `from __future__ import annotations` (Py 3.6).

## Data facts

- `data/images/{train,val,test}/`: **875 / 165 / 150** images.
- Mass GT is the **filename prefix in grams**: `<mass_g>_<YYYY-MM-DD-HH-MM-SS>.<jpg|png>`. There is no separate mass label file.
- GT segmentation masks: COCO JSON (`data/gt_json/<split>/<split>.json`, also `data/json/`).
- Depth: per-image `.npy` in mm.
- **Three capture sessions — SAME CAMERA, different day/lighting + standoff** (this is the crux of generalization). The camera body/intrinsics are constant across all three; only the working *distance* (how close it was positioned) and lighting/appearance differ. So cross-session shift is **photometric (lighting/appearance) + standoff**, NOT optics.
  | Session | Format | train / test / val | Mean in-scene depth | Mass range |
  |---|---|---|---|---|
  | 2025-07-13 | `.jpg` | 476 / 39 / 58 | ~0.62 m | 164–1160 g |
  | 2025-03-28 | `.png` | 267 / 61 / 88 | ~0.60 m | 186–1029 g |
  | 2023-12-01 | `.png` | 132 / 50 / 19 | **~0.48 m (closest!)** | **46–430 g (below 0713 floor)** |
  - ⚠️ A previous version of this file wrongly called 2023-12-01 the "far ~1.4-1.9 m" session. Per `features.csv` it is actually the **nearest** (~0.48 m) and its masses (46–430 g) are mostly **below** the 2025-07-13 training floor (164 g) → any 0713→1201 eval is a heavy **extrapolation/OOD** test, not a clean domain-shift test.
- **Splits are distribution-shifted**: val has masses up to 1160 g (train max 1010 g); 7-9 unique mass values per eval split. **Cross-session generalization is the dominant failure mode.** Camera intrinsics + point clouds exist (for the PointNet direction).

### Single-session split `data_s0713/` (built for the decisive depth experiment)
Isolated, leakage-free **2025-07-13-only** dataset, split **by mass value** (not by frame — see below). Built by `tomato_multitask_RGB-D/make_s0713_split.py` (train/val) and `make_s0713_test_session.py` (cross-session test splits). It's a self-contained root of COCO JSONs + symlinks (images + the flat depth folder); nothing in `data/` is mutated, and `train.py`/`train_new.py` need no edits — just `--dataset_path ./data_s0713/`.
- `train`: 383 frames, 10 masses (incl. extremes 164 & 1160 g).
- `val`: 190 frames, 4 interior masses (208, 490, 670, 940 g) — pure interpolation.
- `test_0328`: 416 frames, 2025-03-28 (cross-session, in-range masses → **the clean cross-session test**).
- `test_1201`: 201 frames, 2023-12-01 (cross-session **OOD stress** — below-floor masses).
- **Why split by mass value:** the 573 frames are only ~31 time-bursts / 14 unique masses (30–60 near-duplicate frames per cluster). A random *frame* split leaks near-duplicates across train/val → false metrics. Splitting on mass value holds out whole clusters and tests generalization to unseen masses.

## Architecture

- Meta-arch `RGBDGeneralizedRCNN` (detectron2/.../meta_arch/rcnn.py): ResNet-50+FPN, conv1 expanded 3→4 channels.
- **Two depth pathways**:
  1. **Early fusion**: depth as conv1's 4th input channel.
  2. **Late fusion**: depth concatenated as channel 256 to mass-head ROI features (`roi_heads._concat_cached_depth_to_features`, fed by `RGBDGeneralizedRCNN._cache_depth`).
- **Mass head** (`MassRCNNConvUpsampleHead`): `5× Conv3x3+ReLU` (the 5th replaced an old ConvTranspose2d deconv) → `AdaptiveAvgPool2d((4,4))` → `Linear(4096→128)+ReLU` → `Linear(128→1)`, outputs `log(mass_kg)`. `fc2.bias` init = `0.0` (better than `-0.8064`).
- `PIXEL_MEAN = [103.530, 116.280, 123.675, 915.043]`. `PIXEL_STD`: originally `[1,1,1,411.683]` (Detectron2 BGR convention), changed to `[57.375, 57.120, 58.395, 411.683]` in the `_norm` experiments.

### Quick architecture check from a checkpoint
`roi_heads.mass_head.mass_fcn1.weight` shape `(256, 257, 3, 3)` ⇒ **LF ON**; `(256, 256, 3, 3)` ⇒ **LF OFF**.

## Run-naming conventions (and a gotcha)

- `LF` in the name ⇒ late fusion ON; absent ⇒ OFF. **Exception:** `clean_baseline_v1_5conv_aug_fix_DEPTH` has LF **ON** despite no "LF" in the name (naming inconsistency — verify via the checkpoint).
- `_DEPTH` ⇒ depth replicated into BGR channels (data change, not architecture). Channels 0-2 carry depth (global `[200,2000]mm`→0-255), channel 3 carries raw mm depth.
- `_noLF` ⇒ late fusion off.
- `_norm` ⇒ `PIXEL_STD` BGR changed to ~57 so all channels have std ≈ 1 after normalization.
- `_0` / `ch3=0` ⇒ channel-3 depth set to 915.043 (→ 0 after normalization) so the early-fusion depth channel carries no info.
- `_stip` ⇒ the `train_new.py` RGB-D lineage on `data_s0713`; `_drop05` ⇒ trained with `--rgb_dropout 0.5` (BGR zeroed on 50% of samples to *force* the depth pathway to develop).

## CRITICAL recurring bug: train/inference normalization mismatch

Several false conclusions came from the inference script not matching training. **Always verify the inference script's `PIXEL_MEAN`/`PIXEL_STD` AND its 4-channel construction match `train.py` for the checkpoint being evaluated.** Past instances:
- `inference_rgbd_viz.py` fed `depth/1000` (meters) while training used mm → garbage depth.
- `inference_depth_only.py` used `PIXEL_STD=[1,1,1,411]` while the `_norm` model trained with `[57,57,58,411]` → test MAPE looked like 35% instead of ~17%.
- (May 2026) `inference_rgbd_viz.py` had the **invalid-depth fill commented out** while training (`load_rgbd_image`) does `depth_mm[depth_mm==0] = 915.043`. Invalid (0 mm) pixels became −2.22σ at inference vs neutral 0 in training → real-depth eval looked far worse than constant-depth and the nearest cluster's detections collapsed. Fixed by mirroring the fill. **Any 4-ch construction must replicate the 0→mean fill.**

Use `inspect_model_input.py` (and its `--simulate_ch3_zero` flag) to verify exactly what the model receives before training or inferring.

## Key findings so far (the narrative)

1. **`aug_fix` is the single biggest win.** Removing multi-scale `ResizeShortestEdge` from the train mapper (the val mapper had no resize → train/val scale mismatch) dropped val MAPE **~50% → ~15%**. For mass = depth-scaled apparent area, random rescaling destroys the size↔mass signal.
2. **Architecture ablations are minor (<2pp).** 5-conv vs 4-conv+deconv, larger MLP, fc2 bias init. **Late fusion contributes nothing** — confirmed via a full 2×2 grid (RGB/depth × LF on/off); all land ~15% val MAPE, LF only adds train-set overfit.
3. **Best RGB-D model: `clean_baseline_v1LF_5conv_aug_fix`** — val MAPE **14.4%** / R² **0.86**, test MAPE **19.0%**. Overfits train (5.5%) and fails cross-session (`.png` 765 g group: 46.8% MAPE).
4. **The RGB-D model was depth-blind (spatially).** 4-scenario depth ablation (real / /1000 / zeros / 5000 mm) — scenarios 1-3 identical to ~1 g/image; only a constant 5000 mm shifts predictions, and only as a uniform ~16 g scalar offset. Root cause: gradient ratio `grad/depth ÷ grad/rgb ≈ 0.01 ≈ 1/52` = the inverse of the BGR-vs-depth input-scale ratio.
5. **Depth-only model (`clean_baseline_v1_5conv_aug_fix_DEPTH`)** (depth replicated into BGR): matches RGB-D on val (13.6% MAPE) and **beats it on test (14.1% vs 19.0%)** because depth is session-invariant while RGB texture isn't. Detection slightly worse (bbox 88 vs 92, segm 87 vs 98). ⇒ RGB contributes ~nothing to mass; the signal is geometric.
6. **Tabular oracle baselines** (engineered GT-mask features, `analysis/`): `area_m2` log-log linear val R² 0.66; best tabular (TinyMLP/HistGBM) val R² ~0.83-0.88. `area_m2` (depth-scaled mask area) is the dominant feature.
7. **Normalization fix (`PIXEL_STD` BGR 1→57)** equalized input scales → gradient ratio **0.01 → 0.95**: channel 3 became an *active, contributing* channel.
   - Training from scratch with **ch3 = 0** (`..._DEPTH_noLF_norm_0`) was **much worse**: detection collapsed (bbox AP 88→76, segm 85→76), val MAPE 16→24%, train loss 5× worse. ⇒ channel 3 (unbounded normalization) carries **complementary** depth that BGR's `[200,2000]` clip throws away (far background). It is NOT redundant once properly scaled.
   - Caveat: this is depth-vs-second-depth (both channels are depth in the depth-only model), **not** depth-vs-RGB.

### The decisive experiment — DONE (May 2026, on `data_s0713`, real RGB-D + `_norm`)

8. **Same-session: RGB and depth are each independently sufficient and mutually redundant.** On the dropout-forced model `clean_baseline_v1_5conv_aug_fix_noLF_norm_stip_drop05`, the full 2×2 on `val` (all ~14.5% MAPE):
   | | depth = real | depth = neutral const |
   |---|---|---|
   | RGB real | **14.4%** (fusion) | **14.5%** (RGB-only) |
   | RGB off  | **14.8%** (depth-only) | **no detections** (null floor) |
   - Depth-only 14.8% (not a mean-collapse) **proves depth is readable** — the earlier "depth-blind" results were just the optimizer skipping depth when RGB was always present (no dropout). With `--rgb_dropout 0.5` the depth pathway learns fine.
   - The null cell (both neutralized → 0 detections) proves the other three are genuine input-driven signal, not a learned prior.
   - **Why redundant:** predictions are ~distance-invariant across a **3.5× standoff range** within a cluster (`pred(far)/pred(near) ≈ 0.8–1.06` vs naive apparent-area `≈0.25–0.4`) ⇒ the backbone recovers metric scale from **RGB monocular cues** (known-fruit-size ruler, blur, context), so explicit depth adds nothing same-session.
   - (The no-dropout sibling `..._stip` hits val 8.3% / R² 0.95 but is effectively RGB-only — its ch3 ablation is a no-op.)

9. **Cross-session: depth wins, and naive fusion squanders it.** Train 2025-07-13 → test 2025-03-28 (`test_0328`, in-range masses):
   | config | MAPE |
   |---|---|
   | depth-only | **37.5%** |
   | fusion (RGB+depth) | 70.7% |
   | RGB-only | 70.1% |
   - RGB collapses (14.5%→70%); depth degrades far less (14.8%→37.5%). **Mechanism (same camera!):** the gap is **photometric/appearance + standoff shift, not optics** — RGB features are entangled with the training day's lighting/appearance; metric depth is lighting-invariant.
   - **fusion ≈ RGB-only, NOT depth-only**: with both present the model rides RGB off the cliff and the good depth signal doesn't rescue it. ⇒ you only get depth's robustness by **deploying depth-only** or building a fusion that down-weights RGB when it's OOD. Confirmed 3× (0328 + 1201 + the same-session 2×2).

10. **2023-12-01 (`test_1201`) is a triple-OOD stress test, not a fair benchmark.** Masses 46–430 g (mostly below the 164 g training floor) + closest standoff (~0.48 m) → systematic over-prediction (bias +0.3…+0.5 kg). Use **MAE, not MAPE** (MAPE explodes on tiny denominators: 257% depth-only vs 432% RGB). Ordering still holds: depth-only MAE 0.342 vs RGB/fusion 0.543 kg. Don't quote its accuracy; report only the modality gap.

**Resolved thesis answer:** same-session RGB≈depth≈fusion (redundant); cross-session depth ≫ RGB but **naive RGB-D defaults to RGB**. Depth's contribution is *generalization/robustness*, not same-session accuracy.

## Candidate next steps

- **Augmentation (to be discussed):** RGB's cross-session failure is appearance-overfitting (same optics) → stronger photometric aug (color/brightness/contrast/white-balance) + multi-session lighting diversity should narrow the RGB gap; depth needs no such crutch.
- **Fusion that exploits depth's robustness** (the open architecture problem): modality-confidence / RGB-dropout-at-test / a dedicated geometry branch so the model can fall back to depth when RGB is OOD.
- **7-fold by mass value** on `data_s0713` to put a confidence interval on the same-session ~14.5% (currently only 4 held-out masses).
- Hybrid mass head (CNN ROI + `area_m2`/depth scalars); PointNet-lite on per-instance metric clouds (intrinsics + clouds exist).
- To make 1201 a usable test: training data covering small-mass / close-standoff regime.

## Working conventions for this project

- Compare runs via `output/<run>/metrics.json` (training-time val metrics, unaffected by inference bugs). Per-image inference results are in `results_*.txt` files.
- When writing/modifying inference, **mirror the train.py mapper exactly** (channel construction + normalization).
- Mass is trained/predicted in **log space**; convert with `exp()` to kg for MAE/MAPE/R².
- Metrics reported: MAE (g), MAPE (%), R², bias (g). MAPE is the headline; report R² alongside (small clusters dominate MAPE). **Use MAE not MAPE when GT masses are tiny / OOD (e.g. `test_1201`)** — MAPE explodes on small denominators.
- Prefer editing existing scripts; the user iterates by toggling clearly-commented blocks. `inference_rgbd_viz.py` modality ablations are run by editing `build_rgbd_input` (BGR→means = RGB off; ch3 = `DEPTH_MEAN_MM` = depth neutral). **Always keep the invalid-depth `0→915.043` fill** (see recurring-bug section) — consider importing `load_rgbd_image` from `train_new.py` to avoid drift.
- Single-session work uses `data_s0713/` (`--dataset_path ./data_s0713/`, splits `train`/`val`/`test_0328`/`test_1201`); rebuild via `make_s0713_split.py` / `make_s0713_test_session.py`.
