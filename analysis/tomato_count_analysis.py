"""
What do the models follow: the number of tomatoes, or their size?

Uses the corrected hand labels in analysis/cluster_inventory.csv (earlier counts: analysis/tomato_counts.csv). mass = n_tomatoes x mean tomato mass, so per cluster
    log(prediction) = c + a * log(n_tomatoes) + b * log(mean tomato mass)
A perfect model has a = b = 1. A value below 1 means the model under-reacts to that factor.
Clusters are the unit of evidence (each counts once, whatever its number of images), and the
intervals are 95% bootstraps over clusters. Counts flagged with occlusion in the notes are less certain,
so everything that matters is repeated without them.

The two 136 g clusters are separated by capture time (vine at 16:45, rounder cluster at 17:23).

Sections:
  A. tomato size per session (is there a size the home-only models never saw?)
  B. what the cross-validation models follow, per input
  C. does removing the background shift small-tomato clusters more? (analysis/background_test.py)
  D. home-only models on the lab session: does the error depend on tomato size?

Writes results/eval/<name>_summary/tomato_counts.txt

  ./run.sh analysis/tomato_count_analysis.py --name cv5_20k
"""
import argparse
import csv
import os
import re
from collections import defaultdict

import numpy as np

COUNTS = "analysis/cluster_inventory.csv"
SESSION_DATES = {"home": "2025-07-13", "lab1": "2023-12-01", "lab2": "2025-03-28"}
SPLIT_136 = {"before": ("136a", 7), "after": ("136b", 9), "at": "17-00-00"}
FILE_NAME = re.compile(r"^(\d+)_(\d{4}-\d{2}-\d{2})-(\d{2}-\d{2}-\d{2})")
HOME_ONLY_MODELS = [("model 5", "s0713_zoom_drop05"), ("7c", "s0713_zoom_drop_05_00"), ("7a", "s0713_zoom_drop065")]
HOME, LAB = "2025-07-13", "2025-03-28"
BOOTSTRAP_SEED = 0
BOOTSTRAP_SAMPLES = 5000


def load_counts():
    """-> {cluster key: dict(mass, session, count, fruit, occluded)}, from the corrected labels."""
    clusters = {}
    with open(COUNTS) as fh:
        for row in csv.DictReader(fh):
            mass, count = int(row["mass_g"]), int(row["n_tomatoes"])
            occluded = re.search(r"oc+lus|hidd", row["note"], re.IGNORECASE) is not None
            clusters[row["key"]] = dict(mass=mass, session=SESSION_DATES[row["session"]], count=count,
                                        fruit=mass / count, occluded=occluded)
    return clusters


def cluster_key(file_name):
    match = FILE_NAME.match(file_name)
    if int(match.group(1)) != 136:
        return match.group(1)
    side = "before" if match.group(3) < SPLIT_136["at"] else "after"
    return SPLIT_136[side][0]


def read_predictions(paths, mode_column):
    """-> {mode: {cluster key: mean log(pred_kg)}} (frames with a prediction only)."""
    values = defaultdict(lambda: defaultdict(list))
    for path in paths:
        with open(path) as fh:
            for row in csv.DictReader(fh):
                if row["pred_kg"]:
                    values[row[mode_column]][cluster_key(row["file"])].append(np.log(float(row["pred_kg"])))
    return {mode: {k: float(np.mean(v)) for k, v in by_cluster.items()} for mode, by_cluster in values.items()}


def fit(y, predictors):
    """Least squares with an intercept -> coefficients without the intercept."""
    X = np.column_stack([np.ones(len(y))] + predictors)
    return np.linalg.lstsq(X, y, rcond=None)[0][1:]


def fit_with_interval(y, predictors, session_ids=None):
    """-> [(coefficient, low, high), ...] for each predictor. session_ids adds session offsets."""
    extra = []
    if session_ids is not None:
        for session in sorted(set(session_ids))[1:]:
            extra.append(np.array([s == session for s in session_ids], float))
    coefficients = fit(y, predictors + extra)[:len(predictors)]
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.randint(0, len(y), len(y))
        try:
            samples.append(fit(y[idx], [p[idx] for p in predictors] + [e[idx] for e in extra])[:len(predictors)])
        except np.linalg.LinAlgError:
            continue
    low, high = np.percentile(np.array(samples), [2.5, 97.5], axis=0)
    return list(zip(coefficients, low, high))


def fmt(entry):
    return f"{entry[0]:+.2f} ({entry[1]:+.2f} to {entry[2]:+.2f})"


def arrays(clusters, keys):
    count = np.log([clusters[k]["count"] for k in keys])
    fruit = np.log([clusters[k]["fruit"] for k in keys])
    mass = np.log([clusters[k]["mass"] for k in keys])
    return count, fruit, mass


def section_sizes(clusters, lines):
    lines.append("A. Mean tomato mass per cluster (g), per session")
    for session in (HOME, LAB, "2023-12-01"):
        group = sorted((c["fruit"], c["count"], c["mass"]) for c in clusters.values() if c["session"] == session)
        lines.append(f"  {session}: {len(group)} clusters, " + ", ".join(f"{f:.0f}" for f, _, _ in group))
        count, fruit = np.log([g[1] for g in group]), np.log([g[0] for g in group])
        mass = np.log([g[2] for g in group])
        lines.append(f"    correlation of log mass with log count {np.corrcoef(mass, count)[0, 1]:+.2f}, "
                     f"with log tomato mass {np.corrcoef(mass, fruit)[0, 1]:+.2f}")
    home = sorted(c["fruit"] for c in clusters.values() if c["session"] == HOME)
    gaps = [(b - a, a, b) for a, b in zip(home, home[1:])]
    _, low, high = max(gaps)
    inside = [k for k, c in clusters.items() if c["session"] == LAB and low < c["fruit"] < high]
    n_lab = sum(c["session"] == LAB for c in clusters.values())
    lines.append(f"  Largest gap in home tomato sizes: {low:.0f} to {high:.0f} g. "
                 f"Lab clusters inside that gap: {len(inside)} of {n_lab}")


def section_follow(clusters, predictions, lines):
    lines.extend(["", "B. Cross-validation models: reaction to tomato count (a) and tomato size (b). Perfect = +1.00"])
    for label, keep in [("all clusters", lambda c: True), ("without occlusion-flagged counts", lambda c: not c["occluded"])]:
        lines.append(f"  {label}")
        for mode in ("rgbd", "rgb_only", "depth_only"):
            keys = [k for k in predictions[mode] if keep(clusters[k])]
            y = np.array([predictions[mode][k] for k in keys])
            count, fruit, mass = arrays(clusters, keys)
            a, b = fit_with_interval(y, [count, fruit])
            residual = y - (y.mean() + mass - mass.mean())
            lines.append(f"    {mode:<11} n={len(keys)}  a {fmt(a)}   b {fmt(b)}   "
                         f"typical log error {np.sqrt(np.mean(residual ** 2)):.3f}")
    lines.append("  within each session (session offsets added, all clusters)")
    for mode in ("rgbd", "rgb_only", "depth_only"):
        keys = list(predictions[mode])
        y = np.array([predictions[mode][k] for k in keys])
        count, fruit, _ = arrays(clusters, keys)
        a, b = fit_with_interval(y, [count, fruit], [clusters[k]["session"] for k in keys])
        lines.append(f"    {mode:<11} n={len(keys)}  a {fmt(a)}   b {fmt(b)}")


def section_background(clusters, shifted, lines):
    lines.extend(["", "C. Prediction shift when the background is removed: log(altered / normal), per cluster",
                  "   Coefficient on log tomato mass at equal total mass. Negative = small tomatoes shift up more."])
    for mode in ("rgb_blur_bg", "rgb_blank_bg"):
        keys = [k for k in shifted["rgb_only"] if k in shifted[mode]]
        y = np.array([shifted[mode][k] - shifted["rgb_only"][k] for k in keys])
        _, fruit, mass = arrays(clusters, keys)
        result = fit_with_interval(y, [fruit, mass])[0]
        without = [k for k in keys if k != "208"]
        y2 = np.array([shifted[mode][k] - shifted["rgb_only"][k] for k in without])
        _, fruit2, mass2 = arrays(clusters, without)
        lines.append(f"  {mode:<13} tomato mass {fmt(result)}   without 208 g: {fmt(fit_with_interval(y2, [fruit2, mass2])[0])}")
    small = [k for k in shifted["rgb_only"] if clusters[k]["fruit"] < 40]
    large = [k for k in shifted["rgb_only"] if clusters[k]["fruit"] >= 40 and clusters[k]["mass"] < 500]
    for mode in ("rgb_blur_bg", "rgb_blank_bg"):
        lines.append(f"  {mode:<13} median factor: tomatoes under 40 g (n={len(small)}) "
                     f"x{np.exp(np.median([shifted[mode][k] - shifted['rgb_only'][k] for k in small])):.2f}, "
                     f"larger tomatoes in clusters under 500 g (n={len(large)}) "
                     f"x{np.exp(np.median([shifted[mode][k] - shifted['rgb_only'][k] for k in large])):.2f}")


def section_home_only(clusters, lines):
    lines.extend(["", "D. Home-only models on the lab session: log(pred / true) vs tomato count and size, per cluster"])
    for label, run in HOME_ONLY_MODELS:
        path = os.path.join("results", "eval", run, "test_0328", "per_image.csv")
        predictions = read_predictions([path], "input_mode")
        for mode in ("rgbd", "depth_only"):
            keys = [k for k in predictions[mode] if clusters[k]["session"] == LAB]
            count, fruit, mass = arrays(clusters, keys)
            y = np.array([predictions[mode][k] for k in keys]) + np.log(1000.0) - mass   # pred kg vs true g
            a, b = fit_with_interval(y, [count, fruit])
            lines.append(f"  {label:<8} {mode:<11} n={len(keys)}  count {fmt(a)}   size {fmt(b)}   "
                         f"mean over-estimate x{np.exp(y.mean()):.2f}")
    predictions = read_predictions([os.path.join("results", "eval", "s0713_zoom_drop05", "test_0328", "per_image.csv")],
                                   "input_mode")["rgbd"]
    lines.append("  Model 5 RGB-D, lab clusters by tomato count (mean log error as a factor):")
    by_count = defaultdict(list)
    for key, value in predictions.items():
        if clusters[key]["session"] == LAB:
            by_count[clusters[key]["count"]].append(value - np.log(clusters[key]["mass"] / 1000))
    for count in sorted(by_count):
        masses = sorted(clusters[k]["mass"] for k in predictions
                        if clusters[k]["session"] == LAB and clusters[k]["count"] == count)
        lines.append(f"    {count} tomatoes: x{np.exp(np.mean(by_count[count])):.2f}  masses {masses}")


def main():
    parser = argparse.ArgumentParser(description="Relate model predictions to the hand-counted tomatoes")
    parser.add_argument("--name", default="cv5_20k")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    clusters = load_counts()
    summary = os.path.join("results", "eval", f"{args.name}_summary")
    predictions = read_predictions([os.path.join(summary, "per_image.csv")], "input_mode")
    background_paths = [os.path.join("results", "eval", f"{args.name}_background", f"fold_{f}.csv") for f in range(args.k)]
    shifted = read_predictions(background_paths, "mode")

    lines =[f"{len(clusters)} clusters ({sum(c['occluded'] for c in clusters.values())} with occlusion-flagged counts)", ""]
    section_sizes(clusters, lines)
    section_follow(clusters, predictions, lines)
    section_background(clusters, shifted, lines)
    section_home_only(clusters, lines)

    report = "\n".join(lines) + "\n"
    with open(os.path.join(summary, "tomato_counts.txt"), "w") as fh:
        fh.write(report)
    print(report)


if __name__ == "__main__":
    main()
