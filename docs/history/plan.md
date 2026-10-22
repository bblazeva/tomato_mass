# Plan: Tabular ML baselines for tomato cluster mass

## Purpose

The current Mask R-CNN with a custom 2D mass head (`clean_baseline_v1LF_5conv_aug_fix`)
achieves val MAPE 14.4% / R² 0.86 and test MAPE 19.0% on the tomato-cluster-mass task.

Two prior findings frame this experiment:

1. **The CNN is depth-blind to spatial information.** A 4-scenario depth ablation
   (real depth, depth/1000, depth=0, depth=5000 mm) showed scenarios 1, 2, 3 produce
   near-identical predictions (mean diff ~1 g per image). Only the depth=5000 mm
   constant changes predictions, and only as a uniform scalar offset of ~16 g.
   The model has reduced the spatial 14×14 late-fusion depth channel into a single
   learned scalar bias.

2. **The single-feature `area_m2` linear baseline reaches val R² 0.66 / MAPE 23.7%.**
   The CNN beats this comfortably (R² 0.86 vs 0.66) on val, confirming the CNN extracts
   something beyond pixel-area + depth-correction. But it remains unclear how much
   of that "extra" is residual signal that simpler tabular models could also capture.

Before committing to an expensive architectural redirect (PointNet-based geometry head), we need a decisive answer to:

> **Can a properly-fit tabular model on the existing 18 engineered features
> match or beat the CNN?**

If yes, the CNN is overengineered and a simpler pipeline should replace it. If no,
the CNN extracts signal the engineered features don't capture (likely RGB context),
which informs whether a geometry-only PointNet is appropriate or whether a hybrid
is needed.

This is a 1-hour experiment with a decisive outcome.

---

## Deliverable

A single self-contained Python script at:

    /home/blagicab/master_tomato/analysis/baseline_tabular.py

that reads `features.csv`, trains 5 models, reports a unified comparison table,
prints a per-session breakdown for the best model, and saves a feature-importance
plot.

---

## Input data

**File:** `/home/blagicab/master_tomato/analysis/features.csv`
**Rows:** 1190 total — 875 train, 165 val, 150 test (split via the `split` column).

**Columns:**
- `split` (str): 'train' | 'val' | 'test'
- `mass_g` (int): target, in grams
- `file_name` (str): e.g. `1010_2025-07-13-21-47-45.jpg` — encodes the capture session
- `area_px` (int): mask area in pixels
- `area_m2` (float): mask area in real m², computed as `(Z² .sum()) / (fx · fy)`
- `mean_depth_m`, `median_depth_m`, `std_depth_m`, `min_depth_m`, `max_depth_m`,
  `depth_extent_m` (float): depth statistics inside the mask, in meters
- `bbox_w_px`, `bbox_h_px` (int), `bbox_aspect` (float)
- `solidity` (float): mask area / convex hull area
- `perimeter_px` (float)
- `n_polygons` (int): number of separate polygons in the segmentation
- `proxy_area` (float): duplicates `area_m2` — drop to avoid collinearity
- `proxy_vol_extent`, `proxy_vol_std`, `proxy_pseudo_vol` (float): area × depth-derived thickness

**Verification step before fitting:** load the csv and assert `len(df[df.split=='train'])==875`,
`val==165`, `test==150`.

---

## Reference numbers (include as fixed rows in the output table)

| Reference | Train MAE (g) | Train MAPE | Val MAE (g) | Val MAPE | Val R² | Test MAE (g) | Test MAPE | Test R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean predictor (train mean) | 261 | 81.5% | 448 | 87.1% | -0.82 | 249 | 86.9% | -0.04 |
| `area_m2` log-log linear | 102 | 28.2% | 140 | 23.7% | 0.66 | 132 | 33.2% | 0.59 |
| CNN (current best) | 31 | 5.5% | 119 | 14.4% | 0.86 | 114 | 19.0% | — |

The CNN's test R² wasn't logged; report whatever the new tabular models give for direct comparison
and leave the CNN test R² cell as "—" in the output.

---

## Models to fit (5)

All 5 models train on `log(mass_kg)` (i.e. `np.log(mass_g / 1000.0)`), predict in log space,
then `np.exp(...)` back to kg for metric computation.

### 1. Ridge
- `sklearn.linear_model.RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10, 100], cv=5)`
- Fit on train. Apply selected alpha.

### 2. ElasticNet
- `sklearn.linear_model.ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], n_alphas=20, cv=5, max_iter=5000, random_state=0)`

### 3. RandomForest
- `sklearn.ensemble.RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_leaf=5, n_jobs=-1, random_state=0)`
- No CV. Defaults are sensible at this data size.

### 4. HistGradientBoostingRegressor
- `sklearn.ensemble.HistGradientBoostingRegressor(max_iter=500, max_depth=4, learning_rate=0.05, early_stopping=True, validation_fraction=0.15, random_state=0)`
- Internal early stopping uses a slice of train — fine, val/test still untouched.

### 5. Tiny MLP (PyTorch)
- 2 hidden layers: input → 64 → 32 → 1
- ReLU activations, Dropout(0.3) after each hidden layer
- AdamW: `lr=1e-3, weight_decay=1e-4`
- Loss: MSE on log-target
- Early stopping: hold out 15% of train as internal val (random_state=0); stop when internal val MSE doesn't improve for 20 epochs
- Max 500 epochs, batch size 64

### Optional: XGBoost (skip if not installed)
- If `import xgboost` succeeds, also fit:
- `xgboost.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, random_state=0, n_jobs=-1)`
- Otherwise, skip silently.

---

## Preprocessing decisions (fixed — do not deviate)

Common to all models: drop columns `split`, `file_name`, `mass_g`, and `proxy_area`
(duplicate of `area_m2`). 17 features remain.

### Linear models (Ridge, ElasticNet) and MLP
- `log1p` these heavy-tailed features: `area_px`, `area_m2`, `proxy_vol_extent`,
  `proxy_vol_std`, `proxy_pseudo_vol`, `perimeter_px`.
- Then `StandardScaler` everything (fit on train only).

### Tree models (RF, HistGBM, XGBoost)
- Raw features, no scaling, no log transform. Trees are scale-invariant.

### Target (all models)
- `y_train = np.log(mass_g / 1000.0)` (mass in log-kg)
- After predict: `pred_kg = np.exp(pred_log)`

---

## Execution steps

1. **Environment**: use `~/miniconda3/envs/tomato_mass/bin/python3`. Verify with:
   ```
   ~/miniconda3/envs/tomato_mass/bin/python3 -c "import sklearn, pandas, numpy, matplotlib, torch; print('ok')"
   ```
   Optionally check `import xgboost`; if missing, skip the XGBoost block (do not install).

2. **Write the script** at `/home/blagicab/master_tomato/analysis/baseline_tabular.py`.
   Self-contained, ~250 lines, no side imports from other project files.

3. **Run it** (should finish in under a minute):
   ```
   ~/miniconda3/envs/tomato_mass/bin/python3 /home/blagicab/master_tomato/analysis/baseline_tabular.py
   ```

4. **Verify outputs**:
   - Console: unified comparison table.
   - Console: per-session breakdown for the best model.
   - File: `/home/blagicab/master_tomato/analysis/feature_importance.png` (RF importances, top 15 features).
   - File: `/home/blagicab/master_tomato/analysis/baseline_tabular_results.csv` (full table as csv).

---

## Output specification

### Console table (must include all 8 rows, in this order)

```
================================================================================
TABULAR BASELINE COMPARISON
================================================================================
MODEL              |   TRAIN              |   VAL                |   TEST
                   |  MAE   MAPE     R²   |  MAE   MAPE     R²   |  MAE   MAPE     R²
--------------------------------------------------------------------------------
Mean predictor     |  261   81.5%   -0.12 |  448   87.1%  -0.82  |  249   86.9%  -0.04
area_m2 linear     |  102   28.2%    0.82 |  140   23.7%   0.66  |  132   33.2%   0.59
CNN (reference)    |   31    5.5%    —    |  119   14.4%   0.86  |  114   19.0%   —
--------------------------------------------------------------------------------
Ridge              | ...
ElasticNet         | ...
RandomForest       | ...
HistGBM            | ...
TinyMLP            | ...
(XGBoost           | ... if installed)
================================================================================
```

MAE in grams. MAPE as percentage with `%` suffix. R² to 2 decimal places.

### Per-session breakdown (printed after the table)

For the model with the highest **val R²** (excluding reference rows), break down
val and test predictions by capture session, parsed from the filename:

- Session 1: `*.jpg` with date `2025-07-13-*`
- Session 2: `*.png` with date `2025-03-28-*`
- Session 3: `*.png` with date `2023-12-01-*`

Print a small table:
```
PER-SESSION BREAKDOWN — best model: <model_name>
Session              | Split | n  | MAE (g) | MAPE   | R²
2025-07-13 (.jpg)    | val   | .. | ..      | ..%    | ..
2025-07-13 (.jpg)    | test  | .. | ..      | ..%    | ..
2025-03-28 (.png)    | val   | .. | ..      | ..%    | ..
2025-03-28 (.png)    | test  | .. | ..      | ..%    | ..
2023-12-01 (.png)    | val   | .. | ..      | ..%    | ..
2023-12-01 (.png)    | test  | .. | ..      | ..%    | ..
```

### Feature importance plot

Save `analysis/feature_importance.png`: horizontal bar chart of RandomForest's
`feature_importances_`, top 15 features sorted descending. Title:
"RandomForest feature importances (train)". X-axis: importance score. Y-axis: feature name.

### Results CSV

Save `analysis/baseline_tabular_results.csv` with one row per model and columns:
`model, split, n, mae_g, mape_pct, r2, bias_g`. Include reference rows.

---

## Decision rules — what the results mean

After the script runs, classify the result into one of three buckets based on the
**val R² of the best tabular model**:

### Bucket A — Tabular beats CNN (val R² ≥ 0.90)
The engineered features alone capture more signal than the CNN extracts. CNN is overengineered.
Recommended next steps:
- Adopt the tabular model + Mask R-CNN detection as a production pipeline.
- Drop PointNet from the roadmap.
- Re-examine whether full Mask R-CNN segmentation is even necessary, or if a simpler detector
  (just bounding box + a fixed crop) suffices.

### Bucket B — Tabular matches CNN (val R² 0.83 – 0.89)
The tabular features are competitive with the CNN at the same data scale. Implications:
- Tabular pipeline can be a drop-in replacement for the CNN's mass head (much simpler, faster).
- The architecture isn't the bottleneck; the data is (likely cross-session shift).
- PointNet would need to clear both the CNN and the tabular baseline — a higher bar than
  previously framed. PointNet is still defensible but the upside is smaller.

### Bucket C — Tabular misses (val R² < 0.83)
The CNN extracts signal beyond engineered features. PointNet alone (geometry-only) would
lose that signal. Recommended:
- If pursuing PointNet, design as a **hybrid**: PointNet for geometry + per-ROI RGB CNN
  features fused at the mass head.
- The engineered feature set may be too narrow; consider adding more (texture, color
  histograms, mask shape descriptors).

---

## Mandatory final summary

At the end of the script's stdout, print a 4-line summary in this exact format:

```
SUMMARY
Best tabular model: <name> — val MAPE <X.X%>, val R² <X.XX>
Bucket: <A | B | C>
Top features (RF importance): <feat1>, <feat2>, <feat3>
Per-session note: <one-sentence description of cross-session pattern>
Recommended next step: <one-sentence recommendation aligned with bucket>
```

This summary is what the human will read first.

---

## Notes for the executing agent

- **Don't tune hyperparameters beyond what's specified.** Defaults are intentional;
  fancy tuning rarely matters at ~1000 train rows and adds days of work.
- **Train on train only.** Never use val or test for model selection, early stopping
  (use internal train slice for internal validation), or feature scaling.
- **Set seeds** (`random_state=0` where supported) for reproducibility.
- **Don't optimize MAPE directly.** Train on log-MSE; compute MAPE post-hoc.
- **Skip XGBoost if missing.** Don't pip install — HistGBM is the primary.
- **Don't add models** outside the 5 specified without asking.
- **Don't modify any other files** in the project. Only create:
  - `/home/blagicab/master_tomato/analysis/baseline_tabular.py`
  - `/home/blagicab/master_tomato/analysis/feature_importance.png`
  - `/home/blagicab/master_tomato/analysis/baseline_tabular_results.csv`

---

## Definition of done

The task is complete when:
1. The script exists, runs without error in under a minute.
2. The console table prints with all 8 rows (or 9 with XGBoost) and correct reference numbers.
3. The per-session breakdown is printed.
4. `feature_importance.png` and `baseline_tabular_results.csv` exist on disk.
5. The 4-line SUMMARY is printed at the end.
