"""
area_m2 linear baseline for tomato cluster mass.

Fits log(mass_g) ~ a * log(area_m2) + b on the train split,
evaluates on val and test, reports MAE/MAPE/R^2 in original units.

This is the BAR: any deep model must clear this to be worth the cost.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


CSV_PATH = "/home/blagicab/master_tomato/analysis/features.csv"
FEATURE = "area_m2"


def metrics(y_true_kg, y_pred_kg, name):
    mae_kg  = mean_absolute_error(y_true_kg, y_pred_kg)
    mape    = float(np.mean(np.abs((y_true_kg - y_pred_kg) / y_true_kg)) * 100.0)
    rmse_kg = float(np.sqrt(np.mean((y_true_kg - y_pred_kg) ** 2)))
    r2      = r2_score(y_true_kg, y_pred_kg)
    bias    = float(np.mean(y_pred_kg - y_true_kg))
    print(f"  {name:<8}  n={len(y_true_kg):4d}   "
          f"MAE={mae_kg*1000:6.1f} g   "
          f"MAPE={mape:5.2f}%   "
          f"RMSE={rmse_kg*1000:6.1f} g   "
          f"R2={r2:6.3f}   "
          f"bias={bias*1000:+6.1f} g")


def main():
    df = pd.read_csv(CSV_PATH)
    df = df[df[FEATURE] > 0].copy()           # log-safety
    df["mass_kg"] = df["mass_g"] / 1000.0

    tr   = df[df["split"] == "train"]
    va   = df[df["split"] == "val"]
    te   = df[df["split"] == "test"]

    Xtr = np.log(tr[FEATURE].values).reshape(-1, 1)
    ytr = np.log(tr["mass_kg"].values)

    model = LinearRegression().fit(Xtr, ytr)
    a, b  = model.coef_[0], model.intercept_

    # Implied power law: mass_kg = exp(b) * area_m2^a
    print("=" * 80)
    print(f"area_m2 linear baseline")
    print(f"  fit on train:  log(mass_kg) = {a:.4f} * log({FEATURE}) + {b:.4f}")
    print(f"  power form  :  mass_kg     = {np.exp(b):.4f} * area_m2^{a:.4f}")
    print("=" * 80)

    for name, split in [("TRAIN", tr), ("VAL", va), ("TEST", te)]:
        X = np.log(split[FEATURE].values).reshape(-1, 1)
        y_kg = split["mass_kg"].values
        y_pred_kg = np.exp(model.predict(X))
        print(f"\n[{name}]")
        metrics(y_kg, y_pred_kg, FEATURE)

        # Trivial baseline: predict the train mean (in log space) for everyone
        y_pred_mean = np.full_like(y_kg, np.exp(np.mean(ytr)))
        metrics(y_kg, y_pred_mean, "MEAN")


if __name__ == "__main__":
    main()
