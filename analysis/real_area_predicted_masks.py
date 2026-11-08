"""
Does real area still predict mass on an unseen session when the mask comes from the model?

Real area of a mask = sum of Z^2 over its valid depth pixels (Z in m). Multiplied by 1 / (fx * fy) this is the
area in m^2; the focal length is a constant that the fit absorbs, so it is not needed here. This assumes
the same camera intrinsics in all sessions.

For each session: fit log(mass) = c + s * log(real area) per image on the annotated masks of the other two
sessions, then predict the held-out session from the masks of the model that never saw it (RGB-D input,
top detection, same as evaluate.py). The annotated-mask result on the same frames shows how much the
predicted masks cost. Variants: one fit for everything, and a separate fit per tomato type
(hand-labelled in analysis/cluster_inventory.csv), so "known type" assumes something else supplies the type.

Writes results/eval/real_area_predicted_masks/{per_image.csv, report.txt}.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=analysis ./run.sh analysis/real_area_predicted_masks.py
"""
import csv
import json
import os
from argparse import Namespace

import cv2
import numpy as np
import torch

import utils_detection
from evaluate import build_eval_model, build_input, input_normalization
from dataset_audit import spearman
from tomato_count_analysis import cluster_key

SESSIONS = [("home", "home", "2025-07-13"), ("lab1", "2023", "2023-12-01"), ("lab2", "lab", "2025-03-28")]
OUT = "results/eval/real_area_predicted_masks"
CONF_THR = 0.7


def real_area(depth, mask):
    """Sum of Z^2 (m^2) over valid depth pixels in the mask; 0 if there are none."""
    z = depth[mask & (depth > 0)].astype(np.float64) / 1000.0
    return float((z * z).sum())


def fill_mask(annotations, height, width):
    mask = np.zeros((height, width), np.uint8)
    for annotation in annotations:
        for polygon in annotation["segmentation"]:
            cv2.fillPoly(mask, [np.array(polygon, np.float32).reshape(-1, 2).round().astype(np.int32)], 1)
    return mask.astype(bool)


def collect(inventory):
    rows = []
    for session, folder, _ in SESSIONS:
        run_dir = os.path.join("output", f"loso_{folder}_20k")
        dataset = f"./data_min120/loso/hold_{folder}/"
        mean, std, _ = input_normalization(Namespace(pixel_std=None), run_dir)
        model = build_eval_model(os.path.join(run_dir, "model_final.pth"), mean, std, CONF_THR, dataset)
        records = utils_detection.get_tomato_dataset_dicts(dataset, "val")
        for i, record in enumerate(records):
            bgr = cv2.imread(record["file_name"])
            depth = np.load(record["depth_file"]).astype(np.float32)
            if depth.shape != bgr.shape[:2]:
                depth = cv2.resize(depth, bgr.shape[1::-1], interpolation=cv2.INTER_NEAREST)
            image = build_input(bgr, depth, "rgbd")
            with torch.no_grad():
                inst = model([{"image": torch.from_numpy(image.transpose(2, 0, 1)).float(),
                               "height": image.shape[0], "width": image.shape[1]}])[0]["instances"].to("cpu")
            gt_mask = fill_mask(record["annotations"], bgr.shape[0], bgr.shape[1])
            name = os.path.basename(record["file_name"])
            key = cluster_key(name)
            row = {"file": name, "session": session, "key": key, "type": inventory[key]["type"],
                   "mass_g": inventory[key]["mass"], "area_gt": real_area(depth, gt_mask), "area_pred": "",
                   "mask_iou": "", "cnn_pred_g": ""}
            if len(inst) > 0:
                top = int(inst.scores.argmax())
                pred_mask = inst.pred_masks[top].numpy().astype(bool)
                row["area_pred"] = real_area(depth, pred_mask)
                union = (pred_mask | gt_mask).sum()
                row["mask_iou"] = round(float((pred_mask & gt_mask).sum() / union), 4) if union else 0.0
                row["cnn_pred_g"] = float(np.exp(inst.pred_mass[top].item())) * 1000
            rows.append(row)
            print(f"\r  {session}: {i + 1}/{len(records)}", end="", flush=True)
        print()
        del model
        torch.cuda.empty_cache()
    return rows


def scores(gt, pred, keys):
    """MAPE, MAE, R2, and how predictions follow mass over clusters (rank, slope)."""
    err = np.abs(pred - gt)
    by_cluster = {}
    for k, g, p in zip(keys, gt, pred):
        by_cluster.setdefault(k, ([], []))
        by_cluster[k][0].append(g)
        by_cluster[k][1].append(np.log(p))
    masses = [np.mean(v[0]) for v in by_cluster.values()]
    means = [np.exp(np.mean(v[1])) for v in by_cluster.values()]
    per_cluster = [np.mean(np.abs(np.exp(v[1]) - np.array(v[0])) / np.array(v[0])) * 100 for v in by_cluster.values()]
    r2 = 1 - ((pred - gt) ** 2).sum() / ((gt - gt.mean()) ** 2).sum()
    slope = np.polyfit(np.log(masses), np.log(means), 1)[0]
    return (f"MAPE {np.mean(err / gt) * 100:6.1f}%  per cluster {np.mean(per_cluster):6.1f}%  MAE {err.mean():4.0f} g  "
            f"R2 {r2:+6.2f}  rank {spearman(masses, means):+.2f}  slope {slope:+.2f}")


def fit(rows):
    x = np.log([r["area_gt"] for r in rows])
    y = np.log([r["mass_g"] for r in rows])
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept


def predict(model, area):
    return np.exp(model[0] * np.log(area) + model[1])


def main():
    with open("analysis/cluster_inventory.csv") as fh:
        inventory = {r["key"]: dict(type=r["type"], mass=int(r["mass_g"])) for r in csv.DictReader(fh)}
    os.makedirs(OUT, exist_ok=True)
    rows = collect(inventory)
    with open(os.path.join(OUT, "per_image.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = []
    features = {}
    with open("analysis/features.csv") as fh:
        for r in csv.DictReader(fh):
            features[r["file_name"]] = float(r["area_m2"])
    ratios = np.array([features[r["file"]] / r["area_gt"] for r in rows if r["area_gt"] > 0])
    lines.append(f"Check against features.csv area_m2: ratio to sum(Z^2) spread (max/min) {ratios.max() / ratios.min():.4f} "
                 f"over {len(ratios)} images\n")

    for session, _, _ in SESSIONS:
        test = [r for r in rows if r["session"] == session]
        train = [r for r in rows if r["session"] != session]
        hit = [r for r in test if r["area_pred"] != "" and r["area_pred"] > 0]
        iou = np.array([r["mask_iou"] for r in hit])
        lines.append(f"=== {session}: {len(test)} images, {len(test) - len(hit)} with no detection. Mask IoU with the annotation: "
                     f"median {np.median(iou):.2f}, under 0.5: {(iou < 0.5).sum()}")
        all_fit = fit(train)
        lines.append(f"  fit on the other sessions, all types: slope {all_fit[0]:.2f}")

        def report(label, subset, model_of):
            if not subset:
                return
            gt = np.array([r["mass_g"] for r in subset], float)
            keys = [r["key"] for r in subset]
            pred_mask = np.array([predict(model_of(r), r["area_pred"]) for r in subset])
            gt_mask = np.array([predict(model_of(r), r["area_gt"]) for r in subset])
            cnn = np.array([r["cnn_pred_g"] for r in subset])
            lines.append(f"  {label}")
            lines.append(f"    predicted masks   {scores(gt, pred_mask, keys)}")
            lines.append(f"    annotated masks   {scores(gt, gt_mask, keys)}")
            lines.append(f"    CNN mass head     {scores(gt, cnn, keys)}")

        report("One fit, type unknown, all clusters", hit, lambda r: all_fit)
        by_type = {t: fit([r for r in train if r["type"] == t]) for t in ("cherry", "normal")}
        report("Separate fit per type, all clusters", hit, lambda r: by_type[r["type"]])
        for kind in ("normal", "cherry"):
            report(f"Separate fit, {kind} only", [r for r in hit if r["type"] == kind], lambda r: by_type[r["type"]])
        lines.append("")

    report = "\n".join(lines)
    with open(os.path.join(OUT, "report.txt"), "w") as fh:
        fh.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
