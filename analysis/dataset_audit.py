"""
Audit the dataset by capture session and tomato type, and relate it to model behaviour.

Uses analysis/cluster_inventory.csv (hand-labelled type and tomato count per cluster, from
clusters_analysis.pdf), analysis/features.csv (mask area and distance per image, from the GT masks), the images,
and the saved predictions. Sessions: home = 2025-07-13, lab1 = 2023-12-01, lab2 = 2025-03-28. Clusters >= 120 g.

Baselines predict log(mass) from one measurement, fitted per image on two sessions and tested on the
third, the same test as the leave-one-session-out models. They use GT masks and hand counts, so they
show what those measurements could do across sessions, not what a deployed system would get.

Writes results/eval/dataset_audit.txt

  ./run.sh analysis/dataset_audit.py
"""
import csv
import json
import os
from collections import defaultdict

import cv2
import numpy as np

from tomato_count_analysis import cluster_key

SESSIONS = {"home": "2025-07-13", "lab1": "2023-12-01", "lab2": "2025-03-28"}
HOLD_OUT_RUN = {"home": "loso_home_20k", "lab1": "loso_2023_20k", "lab2": "loso_lab_20k"}
TYPES = ["cherry", "normal"]


def spearman(x, y):
    rank = lambda v: np.argsort(np.argsort(v))
    return float(np.corrcoef(rank(np.array(x)), rank(np.array(y)))[0, 1]) if len(x) > 2 else float("nan")


def load_inventory():
    with open("analysis/cluster_inventory.csv") as fh:
        rows = list(csv.DictReader(fh))
    return {r["key"]: dict(mass=int(r["mass_g"]), session=r["session"], type=r["type"], count=int(r["n_tomatoes"]),
                           fruit=int(r["mass_g"]) / int(r["n_tomatoes"])) for r in rows}


def load_images(clusters):
    """-> list of per-image dicts from features.csv (>= 120 g), with the cluster key and session."""
    session_of = {v: k for k, v in SESSIONS.items()}
    images = []
    with open("analysis/features.csv") as fh:
        for row in csv.DictReader(fh):
            if int(float(row["mass_g"])) < 120:
                continue
            key = cluster_key(row["file_name"])
            images.append(dict(file=row["file_name"], key=key, session=clusters[key]["session"],
                               type=clusters[key]["type"], mass=clusters[key]["mass"], count=clusters[key]["count"],
                               area_px=float(row["area_px"]), area_m2=float(row["area_m2"]),
                               depth_m=float(row["mean_depth_m"])))
            assert session_of[row["file_name"].split("_")[1][:10]] == images[-1]["session"]
    return images


def add_colours(images):
    """Mean colour inside and outside the GT mask, from data_min120."""
    with open("data_min120/gt_json/all/all.json") as fh:
        coco = json.load(fh)
    by_id = {a["image_id"]: a for a in coco["annotations"]}
    by_name = {i["file_name"]: by_id[i["id"]] for i in coco["images"]}
    for im in images:
        bgr = cv2.imread(os.path.join("data_min120/images/all", im["file"]))
        mask = np.zeros(bgr.shape[:2], np.uint8)
        for polygon in by_name[im["file"]]["segmentation"]:
            cv2.fillPoly(mask, [np.array(polygon, np.float32).reshape(-1, 2).round().astype(np.int32)], 1)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        inside, outside = mask.astype(bool), ~mask.astype(bool)
        im["sat_in"], im["val_in"] = float(hsv[:, :, 1][inside].mean()), float(hsv[:, :, 2][inside].mean())
        im["val_out"] = float(hsv[:, :, 2][outside].mean())


def section_inventory(clusters, images, lines):
    lines.append("A. Inventory")
    lines.append(f"  {'session':<7}{'type':<8}{'clusters':>9}{'images':>7}{'img/cluster':>13}{'mass g':>11}"
                 f"{'tomatoes':>10}{'g per tomato':>14}{'distance mm':>13}")
    for session in SESSIONS:
        for kind in TYPES:
            keys = [k for k, c in clusters.items() if c["session"] == session and c["type"] == kind]
            if not keys:
                lines.append(f"  {session:<7}{kind:<8}{0:>9}")
                continue
            ims = [i for i in images if i["key"] in keys]
            per = [sum(i["key"] == k for i in ims) for k in keys]
            mass = [clusters[k]["mass"] for k in keys]
            count = [clusters[k]["count"] for k in keys]
            fruit = [clusters[k]["fruit"] for k in keys]
            depth = [i["depth_m"] * 1000 for i in ims]
            lines.append(f"  {session:<7}{kind:<8}{len(keys):>9}{len(ims):>7}{f'{min(per)}-{max(per)}':>13}"
                         f"{f'{min(mass)}-{max(mass)}':>11}{f'{min(count)}-{max(count)}':>10}"
                         f"{f'{min(fruit):.0f}-{max(fruit):.0f}':>14}{f'{min(depth):.0f}-{max(depth):.0f}':>13}")


def section_count(clusters, lines):
    lines.extend(["", "B. More tomatoes = more mass? Rank correlation of count with mass, over clusters"])
    groups = [("all clusters mixed", lambda c: True)]
    for session in SESSIONS:
        groups.append((f"{session} mixed", lambda c, s=session: c["session"] == s))
        for kind in TYPES:
            groups.append((f"{session} {kind}", lambda c, s=session, t=kind: c["session"] == s and c["type"] == t))
    groups.append(("all normal mixed", lambda c: c["type"] == "normal"))
    groups.append(("all cherry mixed", lambda c: c["type"] == "cherry"))
    for label, keep in groups:
        chosen = [c for c in clusters.values() if keep(c)]
        if len(chosen) > 2:
            lines.append(f"  {label:<22} n={len(chosen):>2}  {spearman([c['count'] for c in chosen], [c['mass'] for c in chosen]):+.2f}")
    lines.extend(["", "  Normal tomatoes: mass range (g) by tomato count and session"])
    lines.append(f"  {'count':<7}" + "".join(f"{s:>20}" for s in SESSIONS))
    for count in sorted({c["count"] for c in clusters.values() if c["type"] == "normal"}):
        cells = ""
        for session in SESSIONS:
            mass = sorted(c["mass"] for c in clusters.values()
                          if c["count"] == count and c["session"] == session and c["type"] == "normal")
            cells += f"{(f'{mass[0]}-{mass[-1]}' if len(mass) > 1 else (str(mass[0]) if mass else '-')):>20}"
        lines.append(f"  {count:<7}{cells}")


def section_confounds(images, lines):
    lines.extend(["", "C. What else differs per session (per image; colours measured with the GT mask)"])
    lines.append(f"  {'session':<7}{'type':<8}{'tomato saturation':>19}{'tomato brightness':>19}{'bg brightness':>15}"
                 f"{'cluster px share %':>20}")
    for session in SESSIONS:
        for kind in TYPES:
            ims = [i for i in images if i["session"] == session and i["type"] == kind]
            if ims:
                lines.append(f"  {session:<7}{kind:<8}{np.mean([i['sat_in'] for i in ims]):>19.0f}"
                             f"{np.mean([i['val_in'] for i in ims]):>19.0f}{np.mean([i['val_out'] for i in ims]):>15.0f}"
                             f"{np.mean([i['area_px'] for i in ims]) / (1280 * 720) * 100:>20.1f}")
    lines.append("  Mean distance per cluster vs mass (rank correlation over clusters):")
    for session in SESSIONS:
        by = defaultdict(list)
        for i in images:
            if i["session"] == session:
                by[(i["key"], i["mass"])].append(i["depth_m"])
        keys = list(by)
        spread = np.median([max(v) / min(v) for v in by.values()])
        lines.append(f"    {session:<6}{spearman([k[1] for k in keys], [np.mean(by[k]) for k in keys]):+.2f}   "
                     f"median far/near distance ratio within a cluster: x{spread:.2f}")


def fit_predict(train_x, train_y, test_x):
    slope, intercept = np.polyfit(train_x, train_y, 1)
    return slope * test_x + intercept


def section_baselines(images, cnn, lines):
    lines.extend(["", "D. Leave-one-session-out: one-measurement baselines vs the CNN (MAPE %, per image; "
                      "per-type offset where stated)"])
    features = [("tomato count", lambda i: np.log(i["count"])), ("pixel area", lambda i: np.log(i["area_px"])),
                ("real area (mask + depth)", lambda i: np.log(i["area_m2"]))]
    lines.append(f"  {'measurement':<32}" + "".join(f"{s:>9}" for s in SESSIONS))
    for label, feature in features:
        cells = ""
        for session in SESSIONS:
            train = [i for i in images if i["session"] != session]
            test = [i for i in images if i["session"] == session]
            pred = np.exp(fit_predict(np.array([feature(i) for i in train]), np.log([i["mass"] for i in train]),
                                      np.array([feature(i) for i in test])))
            gt = np.array([i["mass"] for i in test], float)
            cells += f"{np.mean(np.abs(pred - gt) / gt) * 100:>9.1f}"
        lines.append(f"  {label:<32}{cells}")
    for mode in ("rgbd", "rgb_only", "depth_only"):
        lines.append(f"  {'CNN ' + mode:<32}" + "".join(f"{cnn[s][mode]['mape']:>9.1f}" for s in SESSIONS))
    lines.append("  Same, within normal tomatoes only:")
    for label, feature in features:
        cells = ""
        for session in SESSIONS:
            train = [i for i in images if i["session"] != session and i["type"] == "normal"]
            test = [i for i in images if i["session"] == session and i["type"] == "normal"]
            pred = np.exp(fit_predict(np.array([feature(i) for i in train]), np.log([i["mass"] for i in train]),
                                      np.array([feature(i) for i in test])))
            gt = np.array([i["mass"] for i in test], float)
            cells += f"{np.mean(np.abs(pred - gt) / gt) * 100:>9.1f}"
        lines.append(f"  {label:<32}{cells}")
    for mode in ("rgbd", "rgb_only", "depth_only"):
        lines.append(f"  {'CNN ' + mode:<32}" + "".join(f"{cnn[s][mode]['mape_normal']:>9.1f}" for s in SESSIONS))
    lines.append("  Mass per unit of real area (g per cm2 of visible mask), per session and type:")
    for session in SESSIONS:
        for kind in TYPES:
            ims = [i for i in images if i["session"] == session and i["type"] == kind]
            if ims:
                v = [i["mass"] / (i["area_m2"] * 1e4) for i in ims]
                lines.append(f"    {session:<6}{kind:<8}median {np.median(v):.1f}  (10-90%: {np.percentile(v, 10):.1f}-{np.percentile(v, 90):.1f})")


def load_predictions(path):
    rows = defaultdict(lambda: defaultdict(list))
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["pred_kg"]:
                rows[r["input_mode"]][cluster_key(r["file"])].append(float(r["pred_kg"]) * 1000)
    return rows


def group_lines(label, preds, clusters, keys, lines):
    keys = [k for k in keys if k in preds]
    if len(keys) < 2:
        return
    ratio = [np.exp(np.mean(np.log(preds[k]))) / clusters[k]["mass"] for k in keys]
    pred = [float(np.exp(np.mean(np.log(preds[k])))) for k in keys]
    mass = [clusters[k]["mass"] for k in keys]
    per_tomato = [p / clusters[k]["count"] for p, k in zip(pred, keys)]
    slope = np.polyfit(np.log(mass), np.log(pred), 1)[0]
    lines.append(f"    {label:<30} n={len(keys):>2}  pred/true x{np.exp(np.mean(np.log(ratio))):.2f}  "
                 f"pred range {min(pred):4.0f}-{max(pred):4.0f} g (true {min(mass)}-{max(mass)})  "
                 f"follows mass: rank {spearman(mass, pred):+.2f}, slope {slope:+.2f}  "
                 f"pred g/tomato {np.median(per_tomato):.0f} (true {np.median([clusters[k]['fruit'] for k in keys]):.0f})")


def section_behaviour(clusters, lines):
    lines.extend(["", "E. Model behaviour per cluster group, RGB-D (slope of log pred on log mass: 1 = follows, 0 = flat)"])
    cv = load_predictions("results/eval/cv5_20k_summary/per_image.csv")
    for session in SESSIONS:
        lines.append(f"  {session}")
        hold = load_predictions(os.path.join("results", "eval", HOLD_OUT_RUN[session], "val", "per_image.csv"))
        for kind in TYPES:
            keys = [k for k, c in clusters.items() if c["session"] == session and c["type"] == kind]
            group_lines(f"{kind}: session in training", cv["rgbd"], clusters, keys, lines)
            group_lines(f"{kind}: session unseen", hold["rgbd"], clusters, keys, lines)
            group_lines(f"{kind}: unseen, depth only", hold["depth_only"], clusters, keys, lines)
    home_only = load_predictions("results/eval/s0713_zoom_drop05/test_0328/per_image.csv")
    keys = [k for k, c in clusters.items() if c["session"] == "lab2"]
    lines.append("  lab2, model 5 (trained on home only)")
    group_lines("normal: RGB-D", home_only["rgbd"], clusters, keys, lines)
    group_lines("normal: depth only", home_only["depth_only"], clusters, keys, lines)
    train_fruit = lambda s: [c["fruit"] for c in clusters.values() if c["session"] != s and c["type"] == "normal"]
    lines.append("  Normal tomato size in training when a session is held out vs that session (g per tomato):")
    for session in SESSIONS:
        test = [c["fruit"] for c in clusters.values() if c["session"] == session and c["type"] == "normal"]
        lines.append(f"    {session:<6} train {min(train_fruit(session)):.0f}-{max(train_fruit(session)):.0f}, test {min(test):.0f}-{max(test):.0f}")


def cnn_scores(images):
    types = {i["file"]: i["type"] for i in images}
    scores = {}
    for session, run in HOLD_OUT_RUN.items():
        scores[session] = defaultdict(dict)
        with open(os.path.join("results", "eval", run, "val", "per_image.csv")) as fh:
            rows = list(csv.DictReader(fh))
        for mode in ("rgbd", "rgb_only", "depth_only"):
            hit = [r for r in rows if r["input_mode"] == mode and r["pred_kg"]]
            scores[session][mode]["mape"] = float(np.mean([float(r["abs_pct_err"]) for r in hit]))
            normal = [r for r in hit if types[r["file"]] == "normal"]
            scores[session][mode]["mape_normal"] = float(np.mean([float(r["abs_pct_err"]) for r in normal]))
    return scores


def main():
    clusters = load_inventory()
    images = load_images(clusters)
    add_colours(images)
    lines = [f"{len(clusters)} clusters, {len(images)} images"]
    section_inventory(clusters, images, lines)
    section_count(clusters, lines)
    section_confounds(images, lines)
    section_baselines(images, cnn_scores(images), lines)
    section_behaviour(clusters, lines)
    report = "\n".join(lines) + "\n"
    with open("results/eval/dataset_audit.txt", "w") as fh:
        fh.write(report)
    print(report)


if __name__ == "__main__":
    main()
