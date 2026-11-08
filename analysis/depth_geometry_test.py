"""
Does the mass head use depth as geometry? Change the input in physically defined ways and see what happens.

A model that computes real size (real area = pixels x Z^2, mass ~ real area ^ FIT_SLOPE) should react like this:
    depth x 1.25         same image, all depth values larger    mass x 1.25 ^ (2 * FIT_SLOPE)
    depth x 0.8          same image, all depth values smaller   mass x 0.8  ^ (2 * FIT_SLOPE)
    image zoom 1.25      cluster looks larger, depth unchanged  mass x 1.25 ^ (2 * FIT_SLOPE)
    consistent zoom 1.25 larger, depth / 1.25 (camera moved)    mass unchanged
Only "consistent zoom" happens in training (train.py: depth_consistent_zoom). rgb_only has no depth, so only the
zoom applies. Each cross-validation model runs on up to MAX_PER_CLUSTER evenly spaced images of each held-out cluster.
The reported factor is prediction(changed) / prediction(unchanged), summarized per cluster.

Writes results/eval/depth_geometry_test/{fold_<i>.csv, report.txt}

  CUDA_VISIBLE_DEVICES=0 ./run.sh analysis/depth_geometry_test.py --folds 0 1 2 3 4
  # one model on one split (results in results/eval/depth_geometry_test/<run>__<split>.*):
  ./run.sh analysis/depth_geometry_test.py --run s0713_zoom_drop05 --dataset ./data_s0713/ --split test_0328
"""
import argparse
import csv
import os
from argparse import Namespace
from collections import defaultdict

import cv2
import numpy as np
import torch

import utils_detection
from evaluate import build_eval_model, input_normalization
from tomato_count_analysis import cluster_key
from train import BGR_MEAN, DEPTH_MEAN_MM, depth_consistent_zoom

OUT = "results/eval/depth_geometry_test"
MAX_PER_CLUSTER = 6
FIT_SLOPE = 1.3     # log mass vs log real area, typical fit (analysis/real_area_predicted_masks.py)
ZOOM = 1.25
CONDITIONS = ["unchanged", "depth x1.25", "depth x0.8", "image zoom", "consistent zoom"]
INPUTS = {"rgbd": CONDITIONS, "depth_only": CONDITIONS, "rgb_only": ["unchanged", "image zoom"]}
EXPECTED = {"depth x1.25": 1.25 ** (2 * FIT_SLOPE), "depth x0.8": 0.8 ** (2 * FIT_SLOPE),
            "image zoom": ZOOM ** (2 * FIT_SLOPE), "consistent zoom": 1.0}


def base_image(bgr, depth):
    """4-channel [B, G, R, depth_mm], with invalid depth still 0; -> (image, invalid)."""
    H, W = bgr.shape[:2]
    depth_mm = cv2.resize(depth.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    image = np.zeros((H, W, 4), np.float32)
    image[:, :, :3] = bgr
    image[:, :, 3] = depth_mm
    return image, depth_mm == 0


def apply_condition(image, invalid, annotations, condition):
    out, inv = image.copy(), invalid.copy()
    if condition == "depth x1.25":
        out[:, :, 3] *= 1.25
    elif condition == "depth x0.8":
        out[:, :, 3] *= 0.8
    elif condition in ("image zoom", "consistent zoom"):
        out, inv, _ = depth_consistent_zoom(image, invalid, annotations, ZOOM)
        if condition == "image zoom":
            out[:, :, 3] *= ZOOM     # undo the 1/s: depth values stay as they were
    return out, inv


def apply_input(image, invalid, mode):
    out = image.copy()
    out[:, :, 3][invalid] = DEPTH_MEAN_MM   # invalid depth -> neutral, as in training
    if mode == "rgb_only":
        out[:, :, 3] = DEPTH_MEAN_MM
    elif mode == "depth_only":
        out[:, :, :3] = BGR_MEAN
    return out


def run_model(run_dir, dataset, split, fold):
    mean, std, _ = input_normalization(Namespace(pixel_std=None), run_dir)
    model = build_eval_model(os.path.join(run_dir, "model_final.pth"), mean, std, 0.7, dataset)

    groups = defaultdict(list)
    for record in utils_detection.get_tomato_dataset_dicts(dataset, split):
        groups[cluster_key(os.path.basename(record["file_name"]))].append(record)

    rows = []
    for key, records in sorted(groups.items()):
        records = sorted(records, key=lambda r: r["file_name"])
        picks = np.unique(np.linspace(0, len(records) - 1, MAX_PER_CLUSTER).round().astype(int))
        for index in picks:
            record = records[index]
            bgr = cv2.imread(record["file_name"])
            image, invalid = base_image(bgr, np.load(record["depth_file"]))
            for mode, conditions in INPUTS.items():
                for condition in conditions:
                    changed, inv = apply_condition(image, invalid, record["annotations"], condition)
                    x = apply_input(changed, inv, mode)
                    with torch.no_grad():
                        inst = model([{"image": torch.from_numpy(x.transpose(2, 0, 1)).float(),
                                       "height": x.shape[0], "width": x.shape[1]}])[0]["instances"].to("cpu")
                    pred = float(np.exp(inst.pred_mass[int(inst.scores.argmax())].item())) if len(inst) else ""
                    rows.append({"fold": fold, "key": key, "file": os.path.basename(record["file_name"]),
                                 "input": mode, "condition": condition, "pred_kg": pred})
        print(f"fold {fold}: {key} done", flush=True)
    return rows


def write_report(paths, report_path):
    rows = []
    for path in paths:
        with open(path) as fh:
            rows += list(csv.DictReader(fh))
    base = {(r["file"], r["input"]): float(r["pred_kg"]) for r in rows if r["condition"] == "unchanged" and r["pred_kg"]}
    factors = defaultdict(lambda: defaultdict(list))
    misses = defaultdict(int)
    for r in rows:
        if r["condition"] == "unchanged":
            continue
        if not r["pred_kg"]:
            misses[(r["input"], r["condition"])] += 1
        elif (r["file"], r["input"]) in base:
            factors[(r["input"], r["condition"])][r["key"]].append(np.log(float(r["pred_kg"]) / base[(r["file"], r["input"])]))

    lines = [f"{len({r['file'] for r in rows})} images, {len({r['key'] for r in rows})} clusters. "
             "Prediction factor: median over clusters (middle 50% of clusters)", ""]
    lines.append(f"{'condition':<20}{'if it computed size':>21}" + "".join(f"{m:>26}" for m in INPUTS))
    for condition in CONDITIONS[1:]:
        cells = ""
        for mode in INPUTS:
            data = factors.get((mode, condition))
            if not data:
                cells += f"{'-':>26}"
                continue
            values = np.exp([np.mean(v) for v in data.values()])
            low, mid, high = np.percentile(values, [25, 50, 75])
            cells += f"{f'x{mid:.2f} ({low:.2f}-{high:.2f})':>26}"
        lines.append(f"{condition:<20}{f'x{EXPECTED[condition]:.2f}':>21}{cells}")
    lines.append("")
    lines.append("No detection after the change: " + ", ".join(f"{m} / {c}: {n}" for (m, c), n in sorted(misses.items())) if misses else "No detection after the change: none")
    report = "\n".join(lines)
    with open(report_path, "w") as fh:
        fh.write(report + "\n")
    print(report)


def save_examples(path):
    """One image per session: the RGB and depth the model receives under each change (same depth colour scale per row)."""
    records = utils_detection.get_tomato_dataset_dicts("./data_min120/", "all")
    panels = [("unchanged", "unchanged"), ("image zoom", "image zoom"), ("consistent zoom", "consistent zoom"), ("depth x1.25", "depth x1.25")]
    rows = []
    for session in ("2025-07-13", "2023-12-01", "2025-03-28"):
        chosen = [r for r in records if session in r["file_name"]]
        record = sorted(chosen, key=lambda r: r["file_name"])[len(chosen) // 2]
        image, invalid = base_image(cv2.imread(record["file_name"]), np.load(record["depth_file"]))
        low, high = np.percentile(image[:, :, 3][~invalid], [2, 98])
        rgb_row, depth_row = [], []
        for condition, label in panels:
            changed, inv = apply_condition(image, invalid, record["annotations"], condition)
            rgb = cv2.resize(changed[:, :, :3].astype(np.uint8), (480, 270))
            scaled = np.clip((changed[:, :, 3] - low) / (high - low) * 255, 0, 255).astype(np.uint8)
            depth = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
            depth[inv] = 0
            depth = cv2.resize(depth, (480, 270), interpolation=cv2.INTER_NEAREST)
            for tile, text in ((rgb, f"{label}: RGB"), (depth, f"{label}: depth")):
                cv2.putText(tile, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(tile, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            rgb_row.append(rgb)
            depth_row.append(depth)
        rows += [np.hstack(rgb_row), np.hstack(depth_row)]
    cv2.imwrite(path, np.vstack(rows))
    print(f"-> {path}")


def main():
    parser = argparse.ArgumentParser(description="Test whether the mass head uses depth as geometry")
    parser.add_argument("--save_examples", action="store_true", help="Only save example inputs, then stop")
    parser.add_argument("--folds", type=int, nargs="+", help="Cross-validation folds to run")
    parser.add_argument("--run", help="One run under output/, instead of --folds")
    parser.add_argument("--dataset", help="With --run: dataset path")
    parser.add_argument("--split", default="val", help="With --run: split")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.save_examples:
        save_examples(os.path.join(OUT, "examples.jpg"))
        return
    if args.run:
        rows = run_model(os.path.join("output", args.run), args.dataset, args.split, 0)
        path = os.path.join(OUT, f"{args.run}__{args.split}.csv")
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_report([path], path.replace(".csv", "_report.txt"))
        return
    for fold in args.folds:
        rows = run_model(f"output/cv5_fold{fold}_20k", f"./data_min120/folds/fold_{fold}/", "val", fold)
        with open(os.path.join(OUT, f"fold_{fold}.csv"), "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if all(os.path.exists(os.path.join(OUT, f"fold_{f}.csv")) for f in range(args.k)):
        write_report([os.path.join(OUT, f"fold_{f}.csv") for f in range(args.k)], os.path.join(OUT, "report.txt"))


if __name__ == "__main__":
    main()
