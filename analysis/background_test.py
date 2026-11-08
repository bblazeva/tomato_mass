"""
Does RGB estimate mass from the cluster itself, or also from the scene around it?

Each cross-validation model is run on its own held-out images, with depth replaced by its mean
(the same as rgb_only in evaluate.py), so only RGB can be used. Three versions of the RGB image:
    rgb_only      unchanged (baseline; should reproduce evaluate.py)
    rgb_blur_bg   background blurred: no sharp objects, rough colours and lighting stay
    rgb_blank_bg  background replaced by the mean colour: only the cluster is left

"Background" = everything outside the annotated mask, dilated by --mask_margin px to keep the natural edge
of the cluster. The blur ignores the cluster's pixels, so no red halo spreads into the background.

The mask's outline stays visible, so shape and pixel size remain available. Altered images are also
unlike anything in training, so a higher error does not by itself prove that the model uses the background.
Frames where the top detection misses the cluster (IoU with the GT box < 0.5) are counted separately, as
those say nothing about mass, and the comparison uses frames that are valid in every mode.

Writes to results/eval/<name>_background/:
    fold_<i>.csv    per image x mode
    report.txt      per session and mass range, once all folds are there
    examples.jpg    the three inputs for one image per session

    ./run.sh analysis/background_test.py --name cv5_20k --run_template cv5_fold{}_20k \
        --dataset_template ./data_min120/folds/fold_{}/ --folds 0 1 2 3 4
"""
import argparse
import csv
import os
import re
import sys
from argparse import Namespace
from collections import defaultdict

import cv2
import numpy as np
import torch

import utils_detection
from evaluate import build_eval_model, build_input, input_normalization
from train import BGR_MEAN

MODES = ["rgb_only", "rgb_blur_bg", "rgb_blank_bg"]
MASS_RANGES = [(0, 250), (250, 500), (500, 800), (800, 10 ** 6)]
FILE_NAME = re.compile(r"^(\d+)_(\d{4}-\d{2}-\d{2})-")
IOU_VALID = 0.5
BOOTSTRAP_SEED = 0
BOOTSTRAP_SAMPLES = 5000


# ====================================================================================
# INPUT
# ====================================================================================

def cluster_mask(annotations, height, width, margin):
    mask = np.zeros((height, width), np.uint8)
    for annotation in annotations:
        for polygon in annotation["segmentation"]:
            points = np.array(polygon, np.float32).reshape(-1, 2).round().astype(np.int32)
            cv2.fillPoly(mask, [points], 1)
    if margin > 0:
        size = 2 * margin + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    return mask.astype(bool)


def blur_background(bgr, mask, sigma):
    """Gaussian blur that only averages background pixels (normalized convolution)."""
    weight = (~mask).astype(np.float32)
    blurred = cv2.GaussianBlur(bgr * weight[:, :, None], (0, 0), sigma)
    norm = cv2.GaussianBlur(weight, (0, 0), sigma)
    return blurred / np.maximum(norm, 1e-3)[:, :, None]


def change_background(image, mode, mask, sigma):
    """image: 4-channel input from build_input(..., "rgb_only"). Only the BGR background changes."""
    if mode == "rgb_only":
        return image
    out = image.copy()
    bgr = out[:, :, :3]
    if mode == "rgb_blur_bg":
        bgr[~mask] = blur_background(bgr, mask, sigma)[~mask]
    else:
        bgr[~mask] = BGR_MEAN
    return out


# ====================================================================================
# RUN ONE FOLD
# ====================================================================================

def box_iou(a, b):
    """a, b: [x0, y0, x1, y1]."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / union if union > 0 else 0.0


def run_fold(fold, args, examples):
    run_dir = os.path.join("output", args.run_template.format(fold))
    ckpt = os.path.join(run_dir, "model_final.pth")
    dataset_path = args.dataset_template.format(fold)
    mean, std, _ = input_normalization(Namespace(pixel_std=None), run_dir)
    model = build_eval_model(ckpt, mean, std, args.conf_thr, dataset_path)

    rows = []
    records = utils_detection.get_tomato_dataset_dicts(dataset_path, "val")
    for i, record in enumerate(records):
        bgr = cv2.imread(record["file_name"])
        depth = np.load(record["depth_file"])
        base = build_input(bgr, depth, "rgb_only")
        mask = cluster_mask(record["annotations"], bgr.shape[0], bgr.shape[1], args.mask_margin)
        x, y, w, h = record["annotations"][0]["bbox"]
        gt_box = [x, y, x + w, y + h]
        name = os.path.basename(record["file_name"])
        match = FILE_NAME.match(name)

        for mode in MODES:
            image = change_background(base, mode, mask, args.blur_sigma)
            if match.group(2) not in examples.setdefault(mode, {}):
                examples[mode][match.group(2)] = image[:, :, :3].astype(np.uint8).copy()
            with torch.no_grad():
                inst = model([{"image": torch.from_numpy(image.transpose(2, 0, 1)).float(),
                               "height": image.shape[0], "width": image.shape[1]}])[0]["instances"].to("cpu")
            row = {"fold": fold, "file": name, "mode": mode, "mass_g": int(match.group(1)), "session": match.group(2),
                   "gt_kg": record["cluster_mass"], "n_detections": len(inst), "score": "", "iou_gt": "",
                   "pred_kg": "", "abs_pct_err": ""}
            if len(inst) > 0:
                top = int(inst.scores.argmax())
                row["score"] = round(float(inst.scores[top]), 4)
                row["iou_gt"] = round(box_iou(inst.pred_boxes.tensor[top].tolist(), gt_box), 3)
                row["pred_kg"] = round(float(np.exp(inst.pred_mass[top].item())), 3)
                row["abs_pct_err"] = abs(round(row["pred_kg"] - row["gt_kg"], 3)) / row["gt_kg"] * 100
            rows.append(row)
        sys.stderr.write(f"\r  fold {fold}: {i + 1}/{len(records)}")
        sys.stderr.flush()
    sys.stderr.write("\n")
    return rows


# ====================================================================================
# REPORT
# ====================================================================================

def valid(row):
    return row["pred_kg"] != "" and float(row["iou_gt"]) >= IOU_VALID


def mape(rows):
    return float(np.mean([float(r["abs_pct_err"]) for r in rows])) if rows else float("nan")


def cluster_mean(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["mass_g"], r["session"])].append(float(r["abs_pct_err"]))
    return float(np.mean([np.mean(v) for v in groups.values()])) if groups else float("nan")


def difference_interval(rows, mode):
    """Per-cluster MAPE(mode) - MAPE(rgb_only), with a 95% bootstrap over clusters."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[(r["mass_g"], r["session"])][r["mode"]].append(float(r["abs_pct_err"]))
    diffs = np.array([np.mean(g[mode]) - np.mean(g["rgb_only"]) for g in groups.values()])
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    samples = [rng.choice(diffs, len(diffs)).mean() for _ in range(BOOTSTRAP_SAMPLES)]
    low, high = np.percentile(samples, [2.5, 97.5])
    return diffs.mean(), low, high


def bias_g(rows):
    return float(np.mean([(float(r["pred_kg"]) - float(r["gt_kg"])) * 1000 for r in rows])) if rows else float("nan")


def write_report(out, k):
    rows = []
    for fold in range(k):
        with open(os.path.join(out, f"fold_{fold}.csv")) as fh:
            for row in csv.DictReader(fh):
                row["mass_g"] = int(row["mass_g"])
                rows.append(row)

    by_file = defaultdict(dict)
    for r in rows:
        by_file[r["file"]][r["mode"]] = r
    common = {f for f, modes in by_file.items() if all(valid(modes[m]) for m in MODES)}
    same = [r for r in rows if r["file"] in common]

    n_images = len(by_file)
    lines = [f"{n_images} images. Same frames = {len(common)} images with a valid detection (IoU >= {IOU_VALID}) in all modes.",
             "", "Detection per mode: no detection / top detection off the cluster"]
    for mode in MODES:
        mode_rows = [r for r in rows if r["mode"] == mode]
        none = sum(r["pred_kg"] == "" for r in mode_rows)
        off = sum(r["pred_kg"] != "" and float(r["iou_gt"]) < IOU_VALID for r in mode_rows)
        lines.append(f"  {mode:<14} {none:>4} / {off:>4}")

    def table(title, subset):
        lines.extend(["", title, f"{'':<18}{'images':>7}" + "".join(f"{m:>22}" for m in MODES)])
        for label, group in subset:
            cells = ""
            for mode in MODES:
                mode_rows = [r for r in group if r["mode"] == mode]
                cells += f"{mape(mode_rows):>9.1f}{cluster_mean(mode_rows):>7.1f}{bias_g(mode_rows):>+6.0f}"
            lines.append(f"{label:<18}{len(group) // len(MODES):>7}{cells}")

    lines.append("\nSame frames. Each mode: MAPE per image, MAPE per cluster, bias in g")
    table("Overall", [("all", same)])
    table("Per session", [(s, [r for r in same if r["session"] == s])
                          for s in sorted({r["session"] for r in same}, reverse=True)])
    table("Per mass range", [(f"{low}-{high} g" if high < 10 ** 6 else f">= {low} g",
                              [r for r in same if low <= r["mass_g"] < high]) for low, high in MASS_RANGES])

    lines.extend(["", "Change vs unchanged RGB, per cluster (95% over clusters)"])
    for mode in MODES[1:]:
        for label, group in [("all", same)] + [(s, [r for r in same if r["session"] == s])
                                               for s in sorted({r["session"] for r in same}, reverse=True)]:
            mean, low, high = difference_interval(group, mode)
            lines.append(f"  {mode:<14} {label:<12} {mean:+6.1f} points ({low:+.1f} to {high:+.1f})")

    report = "\n".join(lines) + "\n"
    with open(os.path.join(out, "report.txt"), "w") as fh:
        fh.write(report)
    print(report)


def save_examples(examples, path):
    sessions = sorted(examples[MODES[0]])
    rows = []
    for session in sessions:
        tiles = []
        for mode in MODES:
            tile = cv2.resize(examples[mode][session], (480, 270))
            cv2.putText(tile, f"{session} {mode}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            tiles.append(tile)
        rows.append(np.hstack(tiles))
    cv2.imwrite(path, np.vstack(rows))


def main():
    parser = argparse.ArgumentParser(description="Test how much RGB-based mass estimates depend on the background")
    parser.add_argument("--name", required=True, help="Output folder: results/eval/<name>_background/")
    parser.add_argument("--run_template", required=True, help="Run name with {} for the fold")
    parser.add_argument("--dataset_template", required=True, help="Fold dataset path with {} for the fold")
    parser.add_argument("--folds", type=int, nargs="+", required=True, help="Folds to run in this process")
    parser.add_argument("--k", type=int, default=5, help="Total folds: the report is written once all exist")
    parser.add_argument("--conf_thr", type=float, default=0.7)
    parser.add_argument("--mask_margin", type=int, default=5, help="Dilate the mask by this many px")
    parser.add_argument("--blur_sigma", type=float, default=25.0, help="Gaussian blur sigma in px")
    args = parser.parse_args()

    out = os.path.join("results", "eval", f"{args.name}_background")
    os.makedirs(out, exist_ok=True)
    for fold in args.folds:
        path = os.path.join(out, f"fold_{fold}.csv")
        if os.path.exists(path):
            raise SystemExit(f"Results already exist: {path}")

    examples = {}
    for fold in args.folds:
        rows = run_fold(fold, args, examples)
        with open(os.path.join(out, f"fold_{fold}.csv"), "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    save_examples(examples, os.path.join(out, "examples.jpg"))

    if all(os.path.exists(os.path.join(out, f"fold_{fold}.csv")) for fold in range(args.k)):
        write_report(out, args.k)
    else:
        print(f"Report not written yet: not all {args.k} folds are in {out}/")


if __name__ == "__main__":
    main()
