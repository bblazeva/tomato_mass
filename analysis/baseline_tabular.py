"""
Oracle GT-mask engineered-feature tabular baseline for tomato cluster mass.

Trains Ridge, ElasticNet, RandomForest, HistGradientBoosting (and optionally XGBoost),
and a Tiny MLP on the engineered features from features.csv. Compares against fixed
reference rows (Mean predictor, area_m2 linear, CNN). Reports per-session and
per-mass-group breakdowns plus feature importance from RandomForest.

IMPORTANT: features.csv is computed from GT masks. This is therefore an ORACLE
GT-mask engineered-feature diagnostic, not a production-ready baseline. A
production baseline would need to recompute features from predicted masks.
"""

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn


# ---------- paths / config ----------

CSV_PATH    = Path("/home/blagicab/master_tomato/analysis/features.csv")
PLOT_PATH   = Path("/home/blagicab/master_tomato/analysis/feature_importance.png")
RESULTS_CSV = Path("/home/blagicab/master_tomato/analysis/baseline_tabular_results.csv")

LOG1P_FEATURES = [
    "area_px", "area_m2", "proxy_vol_extent",
    "proxy_vol_std", "proxy_pseudo_vol", "perimeter_px",
]
DROP_COLS = ["split", "file_name", "mass_g", "proxy_area"]

# fixed reference rows shown in the output table
REFERENCE_ROWS = [
    ("Mean predictor", "train", 875, 261, 81.5, -0.12, None),
    ("Mean predictor", "val",   165, 448, 87.1, -0.82, None),
    ("Mean predictor", "test",  150, 249, 86.9, -0.04, None),
    ("area_m2 linear", "train", 875, 102, 28.2,  0.82, None),
    ("area_m2 linear", "val",   165, 140, 23.7,  0.66, None),
    ("area_m2 linear", "test",  150, 132, 33.2,  0.59, None),
    ("CNN (reference)","train", 875,  31,  5.5,  None, None),
    ("CNN (reference)","val",   165, 119, 14.4,  0.86, None),
    ("CNN (reference)","test",  150, 114, 19.0,  None, None),
]

CNN_VAL_MAPE  = 14.4
CNN_TEST_MAPE = 19.0
CNN_VAL_R2    = 0.86


# ---------- IO + utility ----------

def load_data():
    df = pd.read_csv(CSV_PATH)
    n_tr = (df.split == "train").sum()
    n_va = (df.split == "val").sum()
    n_te = (df.split == "test").sum()
    assert n_tr == 875, f"train count: {n_tr}, expected 875"
    assert n_va == 165, f"val count: {n_va}, expected 165"
    assert n_te == 150, f"test count: {n_te}, expected 150"
    return df


def parse_session(file_name):
    """Returns (ext, date_prefix) parsed from filename."""
    fn = file_name.lower()
    ext = "jpg" if fn.endswith(".jpg") else ("png" if fn.endswith(".png") else "?")
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", file_name)
    date = m.group(1) if m else "?"
    return ext, date


def metrics(y_true_kg, y_pred_kg):
    """Compute MAE (g), MAPE (%), R², bias (g)."""
    y_true_kg = np.asarray(y_true_kg, dtype=float)
    y_pred_kg = np.asarray(y_pred_kg, dtype=float)
    mae_g  = mean_absolute_error(y_true_kg, y_pred_kg) * 1000.0
    mape   = float(np.mean(np.abs((y_true_kg - y_pred_kg) / y_true_kg)) * 100.0)
    r2     = r2_score(y_true_kg, y_pred_kg) if len(y_true_kg) > 1 else float("nan")
    bias_g = float(np.mean(y_pred_kg - y_true_kg)) * 1000.0
    return dict(mae_g=mae_g, mape_pct=mape, r2=r2, bias_g=bias_g)


# ---------- preprocessing ----------

def prep_linear(df_split, scaler=None, fit=False):
    """log1p heavy-tailed columns then StandardScaler."""
    X = df_split.drop(columns=DROP_COLS).copy()
    for col in LOG1P_FEATURES:
        if col in X.columns:
            X[col] = np.log1p(X[col].astype(float))
    feature_names = list(X.columns)
    if fit:
        scaler = StandardScaler()
        Xa = scaler.fit_transform(X.values)
    else:
        Xa = scaler.transform(X.values)
    return Xa.astype(np.float64), scaler, feature_names


def prep_tree(df_split):
    X = df_split.drop(columns=DROP_COLS).copy()
    return X.values.astype(np.float64), list(X.columns)


# ---------- per-model fit/predict ----------

def fit_predict_linear(model, train_df, val_df, test_df):
    Xtr, scaler, feat_names = prep_linear(train_df, fit=True)
    Xva, _, _ = prep_linear(val_df, scaler=scaler, fit=False)
    Xte, _, _ = prep_linear(test_df, scaler=scaler, fit=False)
    ytr = np.log(train_df.mass_g.values / 1000.0)
    model.fit(Xtr, ytr)
    preds = {}
    for name, X, df in [("train", Xtr, train_df), ("val", Xva, val_df), ("test", Xte, test_df)]:
        pred_kg = np.exp(model.predict(X))
        y_kg = df.mass_g.values / 1000.0
        preds[name] = (y_kg, pred_kg)
    return preds, feat_names, model


def fit_predict_tree(model, train_df, val_df, test_df):
    Xtr, feat_names = prep_tree(train_df)
    Xva, _ = prep_tree(val_df)
    Xte, _ = prep_tree(test_df)
    ytr = np.log(train_df.mass_g.values / 1000.0)
    model.fit(Xtr, ytr)
    preds = {}
    for name, X, df in [("train", Xtr, train_df), ("val", Xva, val_df), ("test", Xte, test_df)]:
        pred_kg = np.exp(model.predict(X))
        y_kg = df.mass_g.values / 1000.0
        preds[name] = (y_kg, pred_kg)
    return preds, feat_names, model


class TinyMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_predict_mlp(train_df, val_df, test_df, random_state=0):
    Xtr_full, scaler, feat_names = prep_linear(train_df, fit=True)
    Xva, _, _ = prep_linear(val_df, scaler=scaler, fit=False)
    Xte, _, _ = prep_linear(test_df, scaler=scaler, fit=False)

    ytr_full = np.log(train_df.mass_g.values / 1000.0).astype(np.float32)
    Xtr_full = Xtr_full.astype(np.float32)
    Xva = Xva.astype(np.float32)
    Xte = Xte.astype(np.float32)

    # 15% of train as internal validation for early stopping
    rng = np.random.RandomState(random_state)
    n = len(Xtr_full)
    idx = rng.permutation(n)
    n_int_val = int(0.15 * n)
    int_val_idx = idx[:n_int_val]
    int_tr_idx  = idx[n_int_val:]

    X_inttr = torch.from_numpy(Xtr_full[int_tr_idx])
    y_inttr = torch.from_numpy(ytr_full[int_tr_idx])
    X_intva = torch.from_numpy(Xtr_full[int_val_idx])
    y_intva = torch.from_numpy(ytr_full[int_val_idx])

    torch.manual_seed(random_state)
    model = TinyMLP(Xtr_full.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    patience = 20
    bad = 0
    batch_size = 64

    for _epoch in range(500):
        model.train()
        perm = torch.randperm(len(X_inttr))
        for i in range(0, len(X_inttr), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            pred = model(X_inttr[b])
            loss = loss_fn(pred, y_inttr[b])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_intva), y_intva).item()
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    preds = {}
    for name, X, df in [("train", Xtr_full, train_df), ("val", Xva, val_df), ("test", Xte, test_df)]:
        with torch.no_grad():
            pred_log = model(torch.from_numpy(X)).numpy()
        pred_kg = np.exp(pred_log)
        y_kg = df.mass_g.values / 1000.0
        preds[name] = (y_kg, pred_kg)

    return preds, feat_names, model


# ---------- output ----------

def fmt_cell(mae_g, mape_pct, r2):
    mae_s  = f"{mae_g:4.0f}" if mae_g is not None else "  — "
    mape_s = f"{mape_pct:5.1f}%" if mape_pct is not None else "  — "
    r2_s   = f"{r2:6.2f}" if r2 is not None else "    — "
    return f"{mae_s} {mape_s} {r2_s}"


def print_main_table(rows):
    """rows: list of dicts with keys model, split, mae_g, mape_pct, r2"""
    # group by model
    models_order = []
    by_model = {}
    for r in rows:
        m = r["model"]
        if m not in by_model:
            by_model[m] = {}
            models_order.append(m)
        by_model[m][r["split"]] = r

    print()
    print("=" * 95)
    print("ORACLE GT-MASK TABULAR BASELINE COMPARISON")
    print("=" * 95)
    print(f"{'MODEL':<18} |  {'TRAIN':<22} |  {'VAL':<22} |  {'TEST':<22}")
    print(f"{'':<18} |  {'MAE  MAPE    R²':<22} |  {'MAE  MAPE    R²':<22} |  {'MAE  MAPE    R²':<22}")
    print("-" * 95)
    # references first
    ref_names = ("Mean predictor", "area_m2 linear", "CNN (reference)")
    for m in ref_names:
        if m not in by_model:
            continue
        row_cells = []
        for split in ("train", "val", "test"):
            r = by_model[m].get(split)
            if r:
                row_cells.append(fmt_cell(r.get("mae_g"), r.get("mape_pct"), r.get("r2")))
            else:
                row_cells.append(fmt_cell(None, None, None))
        print(f"{m:<18} |  {row_cells[0]} |  {row_cells[1]} |  {row_cells[2]}")
    print("-" * 95)
    # fitted models
    for m in models_order:
        if m in ref_names:
            continue
        row_cells = []
        for split in ("train", "val", "test"):
            r = by_model[m].get(split)
            if r:
                row_cells.append(fmt_cell(r.get("mae_g"), r.get("mape_pct"), r.get("r2")))
            else:
                row_cells.append(fmt_cell(None, None, None))
        print(f"{m:<18} |  {row_cells[0]} |  {row_cells[1]} |  {row_cells[2]}")
    print("=" * 95)


def print_session_breakdown(best_name, fitted, val_df, test_df):
    """For best fitted model, break down val + test by session."""
    sessions = [
        ("2025-07-13 (.jpg)", "jpg", "2025-07-13"),
        ("2025-03-28 (.png)", "png", "2025-03-28"),
        ("2023-12-01 (.png)", "png", "2023-12-01"),
    ]
    preds = fitted[best_name][0]
    print()
    print(f"PER-SESSION BREAKDOWN — best fitted tabular model: {best_name}")
    print(f"{'Session':<22} | {'Split':<5} | {'n':>4} | {'MAE (g)':>8} | {'MAPE':>6} | {'R²':>6}")
    print("-" * 70)
    for split_name, df in [("val", val_df), ("test", test_df)]:
        y, p = preds[split_name]
        for label, ext, date in sessions:
            mask = df.apply(lambda r: parse_session(r.file_name) == (ext, date), axis=1).values
            if mask.sum() < 2:
                print(f"{label:<22} | {split_name:<5} | {int(mask.sum()):>4} | {'  —':>8} | {'  —':>6} | {'  —':>6}")
                continue
            sub_y = y[mask]
            sub_p = p[mask]
            m = metrics(sub_y, sub_p)
            print(f"{label:<22} | {split_name:<5} | {int(mask.sum()):>4} | {m['mae_g']:8.1f} | {m['mape_pct']:5.1f}% | {m['r2']:6.2f}")


def print_mass_breakdown(best_name, fitted, val_df, test_df):
    """For best fitted model, break down val + test by GT mass group."""
    preds = fitted[best_name][0]
    print()
    print(f"PER-MASS-GROUP BREAKDOWN — best fitted tabular model: {best_name}")
    print(f"{'Split':<5} | {'GT (kg)':>8} | {'n':>4} | {'mean_pred (kg)':>14} | {'MAE (g)':>8} | {'MAPE':>6} | {'bias (g)':>9}")
    print("-" * 80)
    for split_name, df in [("val", val_df), ("test", test_df)]:
        y, p = preds[split_name]
        groups = sorted(df.mass_g.unique())
        for g in groups:
            mask = (df.mass_g.values == g)
            sub_y = y[mask]
            sub_p = p[mask]
            mean_pred = float(np.mean(sub_p))
            m = metrics(sub_y, sub_p)
            print(f"{split_name:<5} | {g/1000.0:8.3f} | {int(mask.sum()):>4} | {mean_pred:14.3f} | {m['mae_g']:8.1f} | {m['mape_pct']:5.1f}% | {m['bias_g']:+9.1f}")


def plot_feature_importance(feat_names, rf_model):
    importances = rf_model.feature_importances_
    order = np.argsort(importances)[::-1][:15]
    feats = [feat_names[i] for i in order]
    vals  = [importances[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(feats)), vals[::-1])
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats[::-1])
    ax.set_xlabel("importance score (RandomForest, impurity-based — clue only)")
    ax.set_title("RandomForest feature importances (train)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"\nsaved {PLOT_PATH}")


def save_results_csv(rows):
    fieldnames = ["model", "split", "n", "mae_g", "mape_pct", "r2", "bias_g"]
    with RESULTS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    print(f"saved {RESULTS_CSV}")


def classify_bucket(val_r2, val_mape, test_mape):
    """
    Bucket A: val_r2 - 0.86 >= 0.04 AND test_mape - 19.0 <= -3
    Bucket B: |val_r2 - 0.86| <= 0.03 AND |test_mape - 19.0| <= 3
    Bucket C: val_r2 - 0.86 <= -0.03 OR test_mape - 19.0 >= 3
    """
    dr2  = val_r2 - CNN_VAL_R2
    dmt  = test_mape - CNN_TEST_MAPE
    if dr2 >= 0.04 and dmt <= -3.0:
        return "A"
    if abs(dr2) <= 0.03 and abs(dmt) <= 3.0:
        return "B"
    if dr2 <= -0.03 or dmt >= 3.0:
        return "C"
    # in between regions → call it B if closer to CNN, otherwise nearest bucket
    return "B"


def print_summary(best_name, results_rows, fitted, val_df, test_df, top_feats):
    by_ms = {(r["model"], r["split"]): r for r in results_rows}
    val_r2   = by_ms[(best_name, "val")]["r2"]
    val_mape = by_ms[(best_name, "val")]["mape_pct"]
    test_r2  = by_ms[(best_name, "test")]["r2"]
    test_mape= by_ms[(best_name, "test")]["mape_pct"]

    bucket = classify_bucket(val_r2, val_mape, test_mape)

    # Per-session pattern note: compare 2025-07-13 jpg vs older png groups for the best model
    preds = fitted[best_name][0]
    def get_session_mape(split_df, ext, date, split_name):
        y, p = preds[split_name]
        mask = split_df.apply(lambda r: parse_session(r.file_name) == (ext, date), axis=1).values
        if mask.sum() < 2:
            return None
        return float(np.mean(np.abs((y[mask] - p[mask]) / y[mask])) * 100)
    jpg_test  = get_session_mape(test_df, "jpg", "2025-07-13", "test")
    png_03_test = get_session_mape(test_df, "png", "2025-03-28", "test")
    png_12_test = get_session_mape(test_df, "png", "2023-12-01", "test")
    parts = []
    if jpg_test is not None: parts.append(f"2025-07-13 .jpg test MAPE={jpg_test:.1f}%")
    if png_03_test is not None: parts.append(f"2025-03-28 .png test MAPE={png_03_test:.1f}%")
    if png_12_test is not None: parts.append(f"2023-12-01 .png test MAPE={png_12_test:.1f}%")
    session_note = "; ".join(parts) if parts else "n/a"

    recs = {
        "A": "tabular oracle features beat CNN — next: recompute features from predicted masks to test production viability.",
        "B": "tabular oracle ties CNN — next: try hybrid CNN+geometry-scalar head and confirm with predicted-mask features.",
        "C": "tabular oracle misses CNN — next: keep CNN, consider hybrid geometry+RGB architectures (e.g. PointNet + RGB fusion).",
    }

    print()
    print("SUMMARY")
    print(f"Best fitted tabular model: {best_name} — "
          f"val MAPE {val_mape:.1f}%, val R² {val_r2:.2f}, "
          f"test MAPE {test_mape:.1f}%, test R² {test_r2:.2f}")
    print(f"Bucket: {bucket}")
    print(f"Top features (RF importance, clue only): {', '.join(top_feats[:3])}")
    print(f"Per-session note: {session_note}")
    print(f"Recommended next step: {recs[bucket]}")
    print()
    print("Reminder: this is an ORACLE GT-mask engineered-feature diagnostic. "
          "A production baseline requires recomputing features from predicted masks.")


# ---------- main ----------

def main():
    print("Loading data ...", flush=True)
    df = load_data()
    train_df = df[df.split == "train"].reset_index(drop=True)
    val_df   = df[df.split == "val"].reset_index(drop=True)
    test_df  = df[df.split == "test"].reset_index(drop=True)
    print(f"  train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    results = []
    for (m, s, n, mae, mape, r2, bias) in REFERENCE_ROWS:
        results.append(dict(model=m, split=s, n=n, mae_g=mae, mape_pct=mape, r2=r2, bias_g=bias))

    fitted = {}

    # Linear models
    print("\nFitting Ridge ...", flush=True)
    preds, feat_names, ridge = fit_predict_linear(
        RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10, 100], cv=5),
        train_df, val_df, test_df)
    fitted["Ridge"] = (preds, feat_names, ridge)
    print(f"  selected alpha = {ridge.alpha_}")
    for sp, (y, p) in preds.items():
        m = metrics(y, p)
        results.append(dict(model="Ridge", split=sp, n=len(y), **m))

    print("\nFitting ElasticNet ...", flush=True)
    preds, feat_names, en = fit_predict_linear(
        ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], n_alphas=20, cv=5,
                     max_iter=5000, random_state=0),
        train_df, val_df, test_df)
    fitted["ElasticNet"] = (preds, feat_names, en)
    print(f"  selected alpha = {en.alpha_:.5f}  l1_ratio = {en.l1_ratio_}")
    for sp, (y, p) in preds.items():
        m = metrics(y, p)
        results.append(dict(model="ElasticNet", split=sp, n=len(y), **m))

    # Tree models
    print("\nFitting RandomForest ...", flush=True)
    preds, feat_names, rf = fit_predict_tree(
        RandomForestRegressor(n_estimators=500, max_depth=None,
                              min_samples_leaf=5, n_jobs=-1, random_state=0),
        train_df, val_df, test_df)
    fitted["RandomForest"] = (preds, feat_names, rf)
    for sp, (y, p) in preds.items():
        m = metrics(y, p)
        results.append(dict(model="RandomForest", split=sp, n=len(y), **m))

    print("\nFitting HistGBM ...", flush=True)
    preds, feat_names, hgb = fit_predict_tree(
        HistGradientBoostingRegressor(max_iter=500, max_depth=4,
                                      learning_rate=0.05, early_stopping=True,
                                      validation_fraction=0.15, random_state=0),
        train_df, val_df, test_df)
    fitted["HistGBM"] = (preds, feat_names, hgb)
    for sp, (y, p) in preds.items():
        m = metrics(y, p)
        results.append(dict(model="HistGBM", split=sp, n=len(y), **m))

    # XGBoost (optional)
    try:
        import xgboost as xgb
        print("\nFitting XGBoost ...", flush=True)
        preds, feat_names, xgbm = fit_predict_tree(
            xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                             random_state=0, n_jobs=-1),
            train_df, val_df, test_df)
        fitted["XGBoost"] = (preds, feat_names, xgbm)
        for sp, (y, p) in preds.items():
            m = metrics(y, p)
            results.append(dict(model="XGBoost", split=sp, n=len(y), **m))
    except ImportError:
        print("\nXGBoost not installed, skipping.", flush=True)

    # Tiny MLP
    print("\nFitting TinyMLP ...", flush=True)
    preds, feat_names, mlp = fit_predict_mlp(train_df, val_df, test_df)
    fitted["TinyMLP"] = (preds, feat_names, mlp)
    for sp, (y, p) in preds.items():
        m = metrics(y, p)
        results.append(dict(model="TinyMLP", split=sp, n=len(y), **m))

    # main table
    print_main_table(results)

    # pick best fitted model: avg(val R², test R²), val R² tiebreaker
    by_ms = {(r["model"], r["split"]): r for r in results}
    best = None
    for name in fitted:
        val_r2  = by_ms[(name, "val")]["r2"]
        test_r2 = by_ms[(name, "test")]["r2"]
        if val_r2 is None or test_r2 is None:
            continue
        avg = (val_r2 + test_r2) / 2.0
        key = (avg, val_r2)
        if best is None or key > best[1]:
            best = (name, key)
    best_name = best[0]
    print(f"\nbest fitted model = {best_name}  "
          f"(avg val+test R² = {best[1][0]:.4f}, val R² = {best[1][1]:.4f})")

    # per-session breakdown
    print_session_breakdown(best_name, fitted, val_df, test_df)

    # per-mass-group breakdown
    print_mass_breakdown(best_name, fitted, val_df, test_df)

    # feature importance from RF
    feat_names = fitted["RandomForest"][1]
    rf_model = fitted["RandomForest"][2]
    importances = rf_model.feature_importances_
    order = np.argsort(importances)[::-1]
    top_feats = [feat_names[i] for i in order]
    plot_feature_importance(feat_names, rf_model)

    # results csv
    save_results_csv(results)

    # final summary
    print_summary(best_name, results, fitted, val_df, test_df, top_feats)


if __name__ == "__main__":
    main()
