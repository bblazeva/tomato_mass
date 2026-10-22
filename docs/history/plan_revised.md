# Plan: Oracle tabular ML baselines for tomato cluster mass

## Purpose

The current Mask R-CNN with a custom 2D mass head (`clean_baseline_v1LF_5conv_aug_fix`) achieves val MAPE **14.4%** / R² **0.86** and test MAPE **19.0%** on the tomato-cluster-mass task.

Two prior findings frame this experiment:

1. **The CNN is depth-blind to spatial information.** A 4-scenario depth ablation (`real depth`, `depth/1000`, `depth=0`, `depth=5000 mm`) showed scenarios 1, 2, and 3 produce near-identical predictions (mean diff around 1 g per image). Only the `depth=5000 mm` constant changes predictions, and only as a uniform scalar offset of about 16 g. The model appears to have reduced the spatial 14×14 late-fusion depth channel into a weak learned scalar bias rather than using object-specific depth geometry.

2. **The single-feature `area_m2` log-log linear baseline reaches val R² 0.66 / MAPE 23.7%.** The CNN beats this comfortably on validation (R² 0.86 vs 0.66), confirming that the CNN extracts signal beyond one simple pixel-area + depth-correction feature. However, it is still unclear whether a richer set of engineered geometry/depth/shape features can match the CNN.

This experiment answers the following diagnostic question:

> **Can a properly fit tabular model on the existing engineered features match or beat the CNN?**

This does **not** directly fix the depth-blind CNN. It tells us whether explicitly computed geometry/depth features contain enough information to compete with the current deep mass head. That result will determine whether the next architectural move should be a geometry-scalar branch, a hybrid RGB + geometry branch, or a point/XYZ branch.

This is a low-risk, high-value diagnostic experiment that should take about one hour to implement and run.

---

## Critical interpretation scope

The features in `features.csv` are assumed to be computed from **GT masks**. Therefore this experiment is an **oracle GT-mask engineered-feature baseline**, not a deployable production baseline yet.

Interpret the result as:

```text
GT-mask tabular model performs well
= explicit geometry/depth/shape signal exists when computed cleanly.
```

Do **not** immediately interpret it as:

```text
The whole CNN/Mask R-CNN pipeline should be replaced.
```

A realistic replacement pipeline requires a follow-up experiment using features computed from **predicted masks**.

---

## Deliverable

A single self-contained Python script at:

```text
/home/blagicab/master_tomato/analysis/baseline_tabular.py
```

that reads `features.csv`, trains the specified tabular models, reports a unified comparison table, prints per-session and per-mass-group breakdowns for the best model, saves a feature-importance plot, and saves a results CSV.

---

## Input data

**File:** `/home/blagicab/master_tomato/analysis/features.csv`  
**Rows:** 1190 total — 875 train, 165 val, 150 test, split via the `split` column.

**Columns:**

- `split` (str): `train` | `val` | `test`
- `mass_g` (int): target, in grams
- `file_name` (str): e.g. `1010_2025-07-13-21-47-45.jpg`; encodes mass, capture date/session, and file extension
- `area_px` (int): GT mask area in pixels
- `area_m2` (float): GT mask area in real m², computed as `(Z².sum()) / (fx · fy)`
- `mean_depth_m`, `median_depth_m`, `std_depth_m`, `min_depth_m`, `max_depth_m`, `depth_extent_m` (float): depth statistics inside the GT mask, in meters
- `bbox_w_px`, `bbox_h_px` (int), `bbox_aspect` (float)
- `solidity` (float): mask area / convex hull area
- `perimeter_px` (float)
- `n_polygons` (int): number of separate polygons in the segmentation
- `proxy_area` (float): duplicate of `area_m2`; drop to avoid collinearity
- `proxy_vol_extent`, `proxy_vol_std`, `proxy_pseudo_vol` (float): area × depth-derived thickness proxies

**Verification step before fitting:** load the CSV and assert:

```python
len(df[df.split == "train"]) == 875
len(df[df.split == "val"]) == 165
len(df[df.split == "test"]) == 150
```

---

## Safety checks before fitting

1. **Do not use `file_name` as a model feature.** It leaks mass/session information.
2. **Do not use `mass_g` or the mass prefix from the filename as input.** The filename contains the mass value, so any parsing must be for session/date/extension diagnostics only.
3. **Call this an oracle GT-mask baseline in all comments/output text.** Avoid overclaiming.
4. **Train only on train.** Do not use val/test for scaling, hyperparameter selection, early stopping, model selection, or feature engineering decisions.
5. **Use val and test together for interpretation.** Val alone has few unique mass values; do not classify success only by val R².

---

## Reference numbers to include as fixed rows

| Reference | Train MAE (g) | Train MAPE | Train R² | Val MAE (g) | Val MAPE | Val R² | Test MAE (g) | Test MAPE | Test R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean predictor (train mean) | 261 | 81.5% | -0.12 | 448 | 87.1% | -0.82 | 249 | 86.9% | -0.04 |
| `area_m2` log-log linear | 102 | 28.2% | 0.82 | 140 | 23.7% | 0.66 | 132 | 33.2% | 0.59 |
| CNN (current best) | 31 | 5.5% | — | 119 | 14.4% | 0.86 | 114 | 19.0% | — |

The CNN's test R² was not logged; report `—` for the CNN test R² cell.

---

## Models to fit

All models train on:

```python
y_train = np.log(mass_g / 1000.0)
```

and predict in log-space. Convert predictions back to kg with:

```python
pred_kg = np.exp(pred_log)
```

Then compute metrics in grams/kg as appropriate.

### 1. Ridge

- `sklearn.linear_model.RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10, 100], cv=5)`
- Fit on train only.

### 2. ElasticNet

- `sklearn.linear_model.ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], n_alphas=20, cv=5, max_iter=5000, random_state=0)`

### 3. RandomForest

- `sklearn.ensemble.RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_leaf=5, n_jobs=-1, random_state=0)`
- No CV. Defaults are acceptable at this data size.

### 4. HistGradientBoostingRegressor

- `sklearn.ensemble.HistGradientBoostingRegressor(max_iter=500, max_depth=4, learning_rate=0.05, early_stopping=True, validation_fraction=0.15, random_state=0)`
- Internal early stopping uses a slice of train only.

### 5. Tiny MLP (PyTorch)

- 2 hidden layers: input → 64 → 32 → 1
- ReLU activations
- `Dropout(0.3)` after each hidden layer
- AdamW: `lr=1e-3`, `weight_decay=1e-4`
- Loss: MSE on log-target
- Early stopping: hold out 15% of train as internal validation (`random_state=0`); stop when internal val MSE does not improve for 20 epochs
- Max 500 epochs
- Batch size 64

### Optional: XGBoost

If `import xgboost` succeeds, also fit:

```python
xgboost.XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    random_state=0,
    n_jobs=-1,
)
```

If XGBoost is not installed, skip silently. Do not install anything.

---

## Preprocessing decisions

Common to all models: drop columns:

```text
split, file_name, mass_g, proxy_area
```

`proxy_area` duplicates `area_m2` and should be dropped to avoid collinearity. Approximately 17 features should remain.

### Linear models and MLP

Apply `log1p` to these heavy-tailed features:

```text
area_px, area_m2, proxy_vol_extent, proxy_vol_std, proxy_pseudo_vol, perimeter_px
```

Then apply `StandardScaler` to all features. Fit scaler on train only.

### Tree models

Use raw features, no scaling, no log transform. Trees are scale-insensitive.

---

## Execution steps

1. **Environment:** use:

```bash
~/miniconda3/envs/tomato_mass/bin/python3
```

Verify with:

```bash
~/miniconda3/envs/tomato_mass/bin/python3 -c "import sklearn, pandas, numpy, matplotlib, torch; print('ok')"
```

Optionally check `import xgboost`; if missing, skip the XGBoost block.

2. **Write the script** at:

```text
/home/blagicab/master_tomato/analysis/baseline_tabular.py
```

The script should be self-contained, roughly 250–350 lines, and must not import other project files.

3. **Run it:**

```bash
~/miniconda3/envs/tomato_mass/bin/python3 /home/blagicab/master_tomato/analysis/baseline_tabular.py
```

4. **Verify outputs:**

- Console: unified comparison table
- Console: per-session breakdown for the best model
- Console: per-mass-group breakdown for the best model on val/test
- File: `/home/blagicab/master_tomato/analysis/feature_importance.png`
- File: `/home/blagicab/master_tomato/analysis/baseline_tabular_results.csv`

---

## Output specification

### Console table

Must include fixed reference rows plus fitted models:

```text
================================================================================
ORACLE GT-MASK TABULAR BASELINE COMPARISON
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

### Per-session breakdown

For the best fitted tabular model, break down val and test predictions by capture session parsed from filename:

- Session 1: `*.jpg` with date `2025-07-13-*`
- Session 2: `*.png` with date `2025-03-28-*`
- Session 3: `*.png` with date `2023-12-01-*`

Choose the best fitted model by **primary criterion: highest average of val R² and test R²**, with val R² as a tiebreaker. Do not choose using train metrics.

Print:

```text
PER-SESSION BREAKDOWN — best fitted tabular model: <model_name>
Session              | Split | n  | MAE (g) | MAPE   | R²
2025-07-13 (.jpg)    | val   | .. | ..      | ..%    | ..
2025-07-13 (.jpg)    | test  | .. | ..      | ..%    | ..
2025-03-28 (.png)    | val   | .. | ..      | ..%    | ..
2025-03-28 (.png)    | test  | .. | ..      | ..%    | ..
2023-12-01 (.png)    | val   | .. | ..      | ..%    | ..
2023-12-01 (.png)    | test  | .. | ..      | ..%    | ..
```

### Per-mass-group breakdown

For the best fitted tabular model, print per-mass-group breakdown for val and test:

```text
PER-MASS-GROUP BREAKDOWN — best fitted tabular model: <model_name>
Split | GT mass (kg) | n | mean_pred (kg) | MAE (g) | MAPE | bias (g)
```

This is important because val/test have very few unique mass groups; one failed group can dominate interpretation.

### Feature importance plot

Save:

```text
/home/blagicab/master_tomato/analysis/feature_importance.png
```

Use RandomForest `feature_importances_`, top 15 features sorted descending.

Title:

```text
RandomForest feature importances (train)
```

X-axis: importance score.  
Y-axis: feature name.

Caveat: RandomForest impurity-based importance is only a clue. It can be biased toward continuous/high-variance features. Do not use it as final proof that a feature is causally important.

### Results CSV

Save:

```text
/home/blagicab/master_tomato/analysis/baseline_tabular_results.csv
```

with one row per model and split:

```text
model, split, n, mae_g, mape_pct, r2, bias_g
```

Include reference rows.

---

## Decision rules — what the results mean

Classify using **both val and test**, not val alone. Use the best fitted tabular model according to the average of val R² and test R².

Let:

```text
val_r2 = best fitted tabular model validation R²
test_r2 = best fitted tabular model test R²
val_mape = best fitted tabular model validation MAPE
test_mape = best fitted tabular model test MAPE
```

### Bucket A — Tabular oracle beats CNN on val and test

Criteria:

```text
val_r2 >= 0.90 OR val_mape <= CNN_val_mape
AND
test_mape <= CNN_test_mape
```

Interpretation:

- Explicit GT-mask geometry/depth/shape features contain enough signal to beat the current CNN mass head.
- The current CNN is likely using a weaker representation than the engineered geometry pipeline.
- This does **not** automatically mean the full system should be replaced, because these are GT-mask oracle features.

Recommended next steps:

1. Compute the same features from **predicted masks** and re-run the tabular model.
2. Add a **geometry scalar branch** to the Detectron2 mass head: ROI visual embedding + selected geometry features → mass MLP.
3. Defer PointNet until predicted-mask features and scalar-branch experiments are evaluated.

### Bucket B — Tabular oracle matches CNN

Criteria:

```text
val_r2 between 0.83 and 0.89 OR val_mape close to CNN_val_mape
AND
test_mape close to CNN_test_mape
```

Interpretation:

- Engineered features are competitive with the CNN at this data scale.
- The CNN is not obviously overengineered, but it is not extracting uniquely deep visual information either.
- The binding constraint may be data split/session shift, representation, or the lack of an explicit geometry path.

Recommended next steps:

1. Add a **hybrid mass head**: CNN ROI features + scalar geometry features.
2. Run session-stratified sampling/retraining if per-session breakdown shows strong domain imbalance.
3. Consider PointNet only as a hybrid geometry branch, not as geometry-only replacement.

### Bucket C — Tabular oracle misses CNN

Criteria:

```text
val_r2 < 0.83 AND val_mape clearly worse than CNN_val_mape
OR
test_mape clearly worse than CNN_test_mape
```

Interpretation:

- The CNN extracts signal beyond the current engineered features, likely RGB/texture/context/shape details not represented in the table.
- A geometry-only PointNet would risk losing useful RGB signal.

Recommended next steps:

1. If pursuing PointNet, design it as **hybrid**: PointNet/XYZ geometry branch + per-ROI RGB CNN branch fused at the mass head.
2. Consider adding richer engineered features: RGB color histograms, texture features, mask compactness, tomato-count proxy, more shape descriptors.
3. Still attack depth-blindness directly through training pressure and architecture changes.

---

## Mandatory final summary

At the end of the script's stdout, print a summary in this exact format:

```text
SUMMARY
Best fitted tabular model: <name> — val MAPE <X.X%>, val R² <X.XX>, test MAPE <Y.Y%>, test R² <Y.YY>
Bucket: <A | B | C>
Top features (RF importance, clue only): <feat1>, <feat2>, <feat3>
Per-session note: <one-sentence description of cross-session pattern>
Recommended next step: <one-sentence recommendation aligned with bucket>
```

---

## Notes for the executing agent

- **Do not tune hyperparameters beyond what is specified.** Defaults are intentional; fancy tuning rarely matters at about 1000 train rows and adds days of work.
- **Train on train only.** Never use val or test for model selection, early stopping, feature scaling, or hyperparameter tuning.
- **Set seeds** (`random_state=0` where supported) for reproducibility.
- **Do not optimize MAPE directly.** Train on log-MSE; compute MAPE post-hoc.
- **Skip XGBoost if missing.** Do not pip install.
- **Do not add models** outside the specified set without asking.
- **Do not modify any other files** in the project. Only create:
  - `/home/blagicab/master_tomato/analysis/baseline_tabular.py`
  - `/home/blagicab/master_tomato/analysis/feature_importance.png`
  - `/home/blagicab/master_tomato/analysis/baseline_tabular_results.csv`

---

## Definition of done

The task is complete when:

1. The script exists and runs without error in under a few minutes.
2. The console table prints with all fixed reference rows and all fitted model rows.
3. The per-session breakdown is printed.
4. The per-mass-group breakdown is printed.
5. `feature_importance.png` and `baseline_tabular_results.csv` exist on disk.
6. The final `SUMMARY` block is printed at the end.
7. The stdout clearly states that this is an **oracle GT-mask engineered-feature diagnostic**, not proof that the CNN uses depth.
