"""
Report the leave-one-session-out results: each model is scored on the session it never saw.
Each session is reported separately, since the three tests are not equally hard.

Reads results/eval/loso_<session>_20k/val/per_image.csv, the cross-validation summary for comparison,
the home-only models on the lab, and the hand counts (via tomato_count_analysis.py).
Writes results/eval/loso_20k_summary/report.txt

Reaction to count (a) and tomato size (b): see tomato_count_analysis.py. Perfect = +1.00.
Differences between inputs are per cluster, on the same frames, with a 95% bootstrap over clusters.

  ./run.sh analysis/summarize_loso.py
"""
import csv
import os
from collections import defaultdict

import numpy as np

from tomato_count_analysis import arrays, cluster_key, fit_with_interval, fmt, load_counts

MODES = ["rgbd", "rgb_only", "depth_only"]
RUNS = [("lab", "2025-03-28"), ("home", "2025-07-13"), ("2023", "2023-12-01")]
HOME_ONLY = [("model 5", "s0713_zoom_drop05"), ("7c", "s0713_zoom_drop_05_00"), ("7a", "s0713_zoom_drop065")]
MIN_TRAIN_MASS_2023_TEST = 164
BOOTSTRAP_SEED = 0
BOOTSTRAP_SAMPLES = 5000


def load(path, session=None):
    with open(path) as fh:
        rows = [r for r in csv.DictReader(fh) if session is None or session in r["file"]]
    for r in rows:
        r["key"] = cluster_key(r["file"])
        r["gt"] = float(r["gt_kg"]) * 1000
        r["pred"] = float(r["pred_kg"]) * 1000 if r["pred_kg"] else None
    return rows


def stats(rows):
    """MAPE per image, MAPE per cluster, MAE g, R2, bias g, misses."""
    hit = [r for r in rows if r["pred"] is not None]
    gt = np.array([r["gt"] for r in hit])
    pred = np.array([r["pred"] for r in hit])
    per_cluster = defaultdict(list)
    for r in hit:
        per_cluster[r["key"]].append(abs(r["pred"] - r["gt"]) / r["gt"] * 100)
    r2 = 1 - ((pred - gt) ** 2).sum() / ((gt - gt.mean()) ** 2).sum()
    return (float((np.abs(pred - gt) / gt).mean() * 100), float(np.mean([np.mean(v) for v in per_cluster.values()])),
            float(np.abs(pred - gt).mean()), float(r2), float((pred - gt).mean()), len(rows) - len(hit))


def stats_line(label, rows):
    s = stats(rows)
    return f"  {label:<22} MAPE {s[0]:5.1f}%  per cluster {s[1]:5.1f}%  MAE {s[2]:4.0f} g  R2 {s[3]:+.2f}  bias {s[4]:+5.0f} g  misses {s[5]}"


def difference(rows, mode_a, mode_b):
    by_file = defaultdict(dict)
    for r in rows:
        by_file[r["file"]][r["input_mode"]] = r
    per_cluster = defaultdict(lambda: ([], []))
    for modes in by_file.values():
        if all(modes[m]["pred"] is not None for m in MODES):
            for side, mode in enumerate((mode_a, mode_b)):
                r = modes[mode]
                per_cluster[r["key"]][side].append(abs(r["pred"] - r["gt"]) / r["gt"] * 100)
    diffs = np.array([np.mean(a) - np.mean(b) for a, b in per_cluster.values()])
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    samples = [rng.choice(diffs, len(diffs)).mean() for _ in range(BOOTSTRAP_SAMPLES)]
    low, high = np.percentile(samples, [2.5, 97.5])
    return f"{mode_a} - {mode_b}: {diffs.mean():+.1f} points ({low:+.1f} to {high:+.1f})"


def main():
    clusters = load_counts()
    lines = []
    for name, session in RUNS:
        rows = load(os.path.join("results", "eval", f"loso_{name}_20k", "val", "per_image.csv"))
        lines.append(f"=== Held out: {name} ({session}), {len(rows) // len(MODES)} images")
        for mode in MODES:
            lines.append(stats_line(mode, [r for r in rows if r["input_mode"] == mode]))
        for a, b in (("rgbd", "rgb_only"), ("rgbd", "depth_only")):
            lines.append("  " + difference(rows, a, b))

        cv = load("results/eval/cv5_20k_summary/per_image.csv", session)
        lines.append("  For comparison, same images, cross-validation (session in training):")
        for mode in MODES:
            lines.append(stats_line(mode, [r for r in cv if r["input_mode"] == mode]))
        if name == "lab":
            lines.append("  For comparison, same images, home-only models (RGB-D, then depth only):")
            for label, run in HOME_ONLY:
                old = load(os.path.join("results", "eval", run, "test_0328", "per_image.csv"))
                for mode in ("rgbd", "depth_only"):
                    lines.append(stats_line(f"{label} {mode}", [r for r in old if r["input_mode"] == mode]))
        if name == "2023":
            rgbd = [r for r in rows if r["input_mode"] == "rgbd"]
            lines.append("  RGB-D by whether the mass is inside the training range:")
            lines.append(stats_line(f"under {MIN_TRAIN_MASS_2023_TEST} g (outside)",
                                    [r for r in rgbd if r["gt"] < MIN_TRAIN_MASS_2023_TEST]))
            lines.append(stats_line(f"{MIN_TRAIN_MASS_2023_TEST} g and up (inside)",
                                    [r for r in rgbd if r["gt"] >= MIN_TRAIN_MASS_2023_TEST]))

        lines.append("  Reaction to tomato count (a) and size (b), perfect = +1.00:")
        for mode in MODES:
            logs = defaultdict(list)
            for r in rows:
                if r["input_mode"] == mode and r["pred"] is not None:
                    logs[r["key"]].append(np.log(r["pred"]))
            keys = sorted(logs)
            y = np.array([np.mean(logs[k]) for k in keys])
            count, fruit, _ = arrays(clusters, keys)
            a, b = fit_with_interval(y, [count, fruit])
            lines.append(f"    {mode:<11} a {fmt(a)}   b {fmt(b)}")

        lines.append("  Per cluster, by mass: tomatoes, g per tomato, mean prediction g (RGB-D / RGB only / depth only)")
        preds = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r["pred"] is not None:
                preds[r["key"]][r["input_mode"]].append(r["pred"])
        for key in sorted(clusters, key=lambda k: (clusters[k]["mass"], k)):
            if clusters[key]["session"] != session:
                continue
            c = clusters[key]
            means = " / ".join(f"{np.mean(preds[key][m]):5.0f}" if preds[key][m] else "  n/a" for m in MODES)
            lines.append(f"    {key:>5} g  {c['count']:>2} tomatoes  {c['fruit']:>4.0f} g each  {means}")
        lines.append("")

    report = "\n".join(lines)
    out = os.path.join("results", "eval", "loso_20k_summary")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "report.txt"), "w") as fh:
        fh.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
