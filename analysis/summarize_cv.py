"""
Combine the evaluate.py results of all cross-validation folds. Every image is in the val split of
exactly one fold, so together the folds give one prediction per image, from a model that never saw
that cluster.

Reads results/eval/<run>/val/per_image.csv for each fold and writes to results/eval/<name>_summary/:
  report.txt    per fold, overall, per session, per mass range, per cluster
  clusters.csv  fold, mass_g, session, n_images, and MAPE + bias + misses per input mode
  per_image.csv all folds together, with the fold number added

Metrics: MAPE per image, and MAPE per cluster (mean over clusters, so that clusters with many
near-duplicate images do not dominate). Misses (no detection) are counted, not scored. The "same frames"
numbers only use images where every input mode has a prediction. The interval on the difference
between two inputs is a bootstrap over clusters, since clusters are the unit of evidence.

  ./run.sh analysis/summarize_cv.py --name cv5_20k --run_template cv5_fold{}_20k --k 5
"""
import argparse
import csv
import os
import re
from collections import defaultdict

import numpy as np

MODES = ["rgbd", "rgb_only", "depth_only"]
MASS_RANGES = [(0, 250), (250, 500), (500, 800), (800, 10 ** 6)]
FILE_NAME = re.compile(r"^(\d+)_(\d{4}-\d{2}-\d{2})-")
BOOTSTRAP_SEED = 0
BOOTSTRAP_SAMPLES = 5000


def load(run_template, k):
    """-> list of rows, one per image x input mode, with fold / mass_g / session added."""
    rows = []
    for fold in range(k):
        with open(os.path.join("results", "eval", run_template.format(fold), "val", "per_image.csv")) as fh:
            for row in csv.DictReader(fh):
                match = FILE_NAME.match(row["file"])
                row.update(fold=fold, mass_g=int(match.group(1)), session=match.group(2),
                           err=float(row["abs_pct_err"]) if row["pred_kg"] else None,
                           signed_g=float(row["err_kg"]) * 1000 if row["pred_kg"] else None)
                rows.append(row)
    return rows


def mape(rows):
    errors = [r["err"] for r in rows if r["err"] is not None]
    return float(np.mean(errors)) if errors else float("nan")


def misses(rows):
    return sum(r["err"] is None for r in rows)


def cluster_mean(rows):
    """Mean of the per-cluster MAPEs (a cluster = one mass value in one session)."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["mass_g"], r["session"])].append(r)
    return float(np.nanmean([mape(g) for g in groups.values()]))


def table_row(label, rows):
    by_mode = {m: [r for r in rows if r["input_mode"] == m] for m in MODES}
    n_images = len(by_mode[MODES[0]])
    cells = "".join(f"{mape(by_mode[m]):>8.1f}{cluster_mean(by_mode[m]):>8.1f}{misses(by_mode[m]):>6}" for m in MODES)
    return f"{label:<20}{n_images:>6}{cells}"


def table_header():
    groups = "".join(f"{m:^22}" for m in MODES)
    columns = "".join(f"{'image':>8}{'cluster':>8}{'miss':>6}" for _ in MODES)
    return [f"{'':<26}{groups}", f"{'':<20}{'images':>6}{columns}"]


def same_frames(rows):
    predicted = [{r["file"] for r in rows if r["input_mode"] == m and r["err"] is not None} for m in MODES]
    common = set.intersection(*predicted)
    return common, [r for r in rows if r["file"] in common]


def difference_interval(rows, mode_a, mode_b):
    """Mean of per-cluster MAPE(a) - MAPE(b) on the same frames, with a 95% bootstrap over clusters."""
    per_cluster = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per_cluster[(r["mass_g"], r["session"])][r["input_mode"]].append(r["err"])
    diffs = np.array([np.mean(c[mode_a]) - np.mean(c[mode_b]) for c in per_cluster.values()])
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    samples = [rng.choice(diffs, len(diffs)).mean() for _ in range(BOOTSTRAP_SAMPLES)]
    low, high = np.percentile(samples, [2.5, 97.5])
    return diffs.mean(), low, high


def main():
    parser = argparse.ArgumentParser(description="Combine the evaluate.py results of all cross-validation folds")
    parser.add_argument("--name", required=True, help="Summary folder: results/eval/<name>_summary/")
    parser.add_argument("--run_template", required=True, help="Run name with {} for the fold, e.g. cv5_fold{}_20k")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    rows = load(args.run_template, args.k)
    out = os.path.join("results", "eval", f"{args.name}_summary")
    os.makedirs(out, exist_ok=True)

    lines = [f"{args.name}: {args.k} folds, {args.run_template}", "MAPE % per image and per cluster; miss = no detection", ""]
    lines += ["Per fold"] + table_header()
    per_fold = {m: [] for m in MODES}
    for fold in range(args.k):
        fold_rows = [r for r in rows if r["fold"] == fold]
        lines.append(table_row(f"fold {fold}", fold_rows))
        for m in MODES:
            per_fold[m].append(mape([r for r in fold_rows if r["input_mode"] == m]))
    lines.append(table_row("all folds", rows))
    lines.append("mean +- std over folds (per image): " + ", ".join(
        f"{m} {np.mean(per_fold[m]):.1f} +- {np.std(per_fold[m], ddof=1):.1f}" for m in MODES))

    common, common_rows = same_frames(rows)
    lines += ["", f"Same frames ({len(common)} images where every input has a prediction)"] + table_header()
    lines.append(table_row("all folds", common_rows))
    for a, b in [("rgbd", "rgb_only"), ("rgbd", "depth_only")]:
        mean, low, high = difference_interval(common_rows, a, b)
        lines.append(f"{a} - {b}, per cluster: {mean:+.1f} points (95% over clusters {low:+.1f} to {high:+.1f})")

    lines += ["", "Per session"] + table_header()
    for session in sorted({r["session"] for r in rows}, reverse=True):
        lines.append(table_row(session, [r for r in rows if r["session"] == session]))

    lines += ["", "Per mass range"] + table_header()
    for low, high in MASS_RANGES:
        label = f"{low}-{high} g" if high < 10 ** 6 else f">= {low} g"
        lines.append(table_row(label, [r for r in rows if low <= r["mass_g"] < high]))

    clusters = defaultdict(list)
    for r in rows:
        clusters[(r["mass_g"], r["session"], r["fold"])].append(r)
    fields = ["fold", "mass_g", "session", "n_images"] + [f"{m}_{c}" for m in MODES for c in ("mape", "bias_g", "misses")]
    lines += ["", "Per cluster, by mass (RGB-D / RGB only / depth only: MAPE %, RGB-D bias in g)"]
    with open(os.path.join(out, "clusters.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for (mass, session, fold), group in sorted(clusters.items()):
            record = {"fold": fold, "mass_g": mass, "session": session,
                      "n_images": len([r for r in group if r["input_mode"] == MODES[0]])}
            for m in MODES:
                mode_rows = [r for r in group if r["input_mode"] == m]
                signed = [r["signed_g"] for r in mode_rows if r["signed_g"] is not None]
                record[f"{m}_mape"] = round(mape(mode_rows), 2)
                record[f"{m}_bias_g"] = round(float(np.mean(signed)), 1) if signed else ""
                record[f"{m}_misses"] = misses(mode_rows)
            writer.writerow(record)
            lines.append(f"{mass:>5} g  {session}  fold {fold}  n={record['n_images']:>3}  "
                         f"{record['rgbd_mape']:>6.1f} {record['rgb_only_mape']:>6.1f} {record['depth_only_mape']:>6.1f}  "
                         f"bias {record['rgbd_bias_g'] if record['rgbd_bias_g'] != '' else 0:>+7.1f} g  "
                         f"depth misses {record['depth_only_misses']}")

    with open(os.path.join(out, "per_image.csv"), "w", newline="") as fh:
        columns = ["fold", "file", "mass_g", "session", "input_mode", "gt_kg", "pred_kg", "err_kg",
                   "abs_pct_err", "score", "n_detections", "distance_mm"]
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    report = "\n".join(lines) + "\n"
    with open(os.path.join(out, "report.txt"), "w") as fh:
        fh.write(report)
    print(report)
    print(f"-> {out}/")


if __name__ == "__main__":
    main()
