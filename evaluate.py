"""
Evaluate a trained model on one or more splits, with input-modality ablations.

For every image the model is run once per selected input mode (--input_modes,
default: all three; e.g. --input_modes rgbd for a plain evaluation):
    rgbd        real BGR + real depth
    rgb_only    depth replaced by its mean (-> 0 after normalization)
    depth_only  BGR replaced by its mean
Invalid depth (0 mm) is filled with the depth mean, as in training.

Depth ablations (real BGR, only the depth channel changed; opt-in via --input_modes):
    rgbd_zero      raw 0 mm everywhere, NOT filled (-> -2.2 sigma; out of distribution)
    rgbd_flat      the cluster's own distance everywhere (median valid depth in the GT box):
                   keeps "how far", removes the shape
    rgbd_shuffled  the real depth map of a frame from a DIFFERENT cluster (fixed seed 0)
When rgbd is evaluated too, the report adds "change vs rgbd": mean |pred - pred_rgbd| / pred_rgbd
over frames where both predict = how much that input change moves the prediction.

Per image, the highest-scoring detection (score >= --conf_thr) is compared with the
GT mass. Images with no detection are counted as misses, not as errors. Distance =
mean valid depth inside that detection's mask.

Writes results/eval/<name>/<split>/ (refuses to overwrite unless --overwrite):
    per_image.csv   one row per image x input mode
    summary.json    settings + overall and per-cluster MAE / MAPE / bias / misses per mode
    report.txt      readable report: summary table + per-cluster frame listing per mode
With --save_viz, 3-panel images go to output/_viz/<name>/<split>/<mode>/.

Examples:
    ./run.sh evaluate.py --run s0713_zoom_drop05 --splits val test_0328
    ./run.sh evaluate.py --run s0713_zoom_drop05 --splits val --input_modes rgbd --save_viz 20
    ./run.sh evaluate.py --run s0713_zoom_drop065 --splits val test_0328 --tag depth_ablation \
        --input_modes rgbd rgb_only rgbd_zero rgbd_flat rgbd_shuffled
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
import torch
import yaml

from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

import utils_detection
from train import BGR_MEAN, BGR_STD, DEPTH_MEAN_MM, DEPTH_STD_MM

INPUT_MODES = ["rgbd", "rgb_only", "depth_only"]
DEPTH_ABLATIONS = ["rgbd_zero", "rgbd_flat", "rgbd_shuffled"]


# ====================================================================================
# MODEL
# ====================================================================================

def resolve_model(args):
    """-> (name used for result folders, checkpoint path, run folder)."""
    if args.model_path:
        ckpt = args.model_path
        run_dir = os.path.dirname(ckpt)
        name = os.path.basename(run_dir.rstrip("/"))
    else:
        run_dir = os.path.join("output", args.run)
        ckpt = os.path.join(run_dir, args.checkpoint)
        name = args.run
    stem = os.path.splitext(os.path.basename(ckpt))[0]
    if stem != "model_final":
        name = f"{name}__{stem}"
    if args.tag:
        name = f"{name}__{args.tag}"
    if not os.path.exists(ckpt):
        raise SystemExit(f"Checkpoint not found: {ckpt}")
    return name, ckpt, run_dir


def input_normalization(args, run_dir):
    """PIXEL_MEAN/STD: --pixel_std > the run's config.yaml > current defaults."""
    mean = [*BGR_MEAN, DEPTH_MEAN_MM]
    std = [*BGR_STD, DEPTH_STD_MM]
    source = "defaults (train.py)"
    cfg_path = os.path.join(run_dir, "config.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            model_cfg = yaml.safe_load(fh)["MODEL"]
        mean, std, source = model_cfg["PIXEL_MEAN"], model_cfg["PIXEL_STD"], "run config.yaml"
    if args.pixel_std:
        std, source = args.pixel_std, "--pixel_std"
    return mean, std, source


def build_eval_model(ckpt, pixel_mean, pixel_std, conf_thr, dataset_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.META_ARCHITECTURE = "RGBDGeneralizedRCNN"
    cfg.MODEL.PIXEL_MEAN = list(pixel_mean)
    cfg.MODEL.PIXEL_STD = list(pixel_std)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_thr
    cfg.MODEL.WEIGHTS = ckpt
    # Required by the vendored heads to build; they do not affect inference.
    cfg.MODEL.ROI_MASK_HEAD.LOSS_WEIGHT = 1.0
    cfg.MODEL.ROI_MASS_HEAD.MASS_LOSS_WEIGHT = 1.0
    cfg.DATASET_PATH = dataset_path
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(ckpt)
    return model


# ====================================================================================
# INPUT
# ====================================================================================

def build_input(image_bgr, depth_mm_raw, mode, flat_mm=None, other_depth_raw=None):
    """4-channel (H, W) float32 input [B, G, R, depth_mm] for one input mode.
    flat_mm: the value for rgbd_flat; other_depth_raw: the depth map for rgbd_shuffled."""
    H, W = image_bgr.shape[:2]
    if mode == "rgbd_shuffled":
        depth_mm_raw = other_depth_raw
    depth_mm = cv2.resize(depth_mm_raw.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    depth_mm[depth_mm == 0] = DEPTH_MEAN_MM   # invalid depth -> neutral, as in training

    image = np.zeros((H, W, 4), dtype=np.float32)
    image[:, :, :3] = image_bgr
    image[:, :, 3] = depth_mm
    if mode == "rgb_only":
        image[:, :, 3] = DEPTH_MEAN_MM
    elif mode == "depth_only":
        image[:, :, :3] = BGR_MEAN
    elif mode == "rgbd_zero":
        image[:, :, 3] = 0.0
    elif mode == "rgbd_flat":
        image[:, :, 3] = flat_mm
    return image


def cluster_distance_mm(depth_valid, annotations):
    """Median valid depth inside the union of the GT boxes (the cluster's distance)."""
    x0 = int(min(a["bbox"][0] for a in annotations))
    y0 = int(min(a["bbox"][1] for a in annotations))
    x1 = int(max(a["bbox"][0] + a["bbox"][2] for a in annotations))
    y1 = int(max(a["bbox"][1] + a["bbox"][3] for a in annotations))
    box = depth_valid[y0:y1, x0:x1]
    box = box[box > 0]
    return float(np.median(box)) if box.size else DEPTH_MEAN_MM


def shuffled_partners(dataset_dicts, seed=0):
    """For each frame, the index of a random frame from a different cluster (GT mass)."""
    rng = np.random.RandomState(seed)
    masses = np.array([d["cluster_mass"] for d in dataset_dicts])
    return [int(rng.choice(np.where(masses != m)[0])) for m in masses]


# ====================================================================================
# METRICS
# ====================================================================================

def summarize(rows):
    """Aggregate per-image rows (of one input mode, optionally one cluster)."""
    hit = [r for r in rows if r["pred_kg"] is not None]
    out = {"n_images": len(rows), "n_with_prediction": len(hit), "n_no_detection": len(rows) - len(hit)}
    if not hit:
        return out
    gt = np.array([r["gt_kg"] for r in hit])
    err = np.array([r["err_kg"] for r in hit])
    dist = [r["distance_mm"] for r in hit if r["distance_mm"] is not None]
    out.update({
        "mae_kg": float(np.abs(err).mean()),
        "mape_pct": float((np.abs(err) / gt).mean() * 100),
        "bias_kg": float(err.mean()),
        "worst_abs_err_kg": float(np.abs(err).max()),
        "r2": float(1 - (err ** 2).sum() / ((gt - gt.mean()) ** 2).sum()) if len(set(gt)) > 1 else None,
        "distance_mm_range": [float(min(dist)), float(max(dist))] if dist else None,
    })
    return out


def summarize_common_frames(rows_by_mode):
    """Each mode again, but only on the frames where EVERY mode has a prediction.
    A mode that misses the hard frames otherwise looks better than it is."""
    files = [{r["file"] for r in rows if r["pred_kg"] is not None} for rows in rows_by_mode.values()]
    common = set.intersection(*files)
    return {"n_frames": len(common),
            "modes": {m: summarize([r for r in rows if r["file"] in common])
                      for m, rows in rows_by_mode.items()}}


def change_vs_rgbd(rows_by_mode):
    """Per mode: how much the prediction moves relative to real RGB-D (mean and median
    |pred - pred_rgbd| / pred_rgbd in %, over frames where both predict)."""
    ref = {r["file"]: r["pred_kg"] for r in rows_by_mode["rgbd"] if r["pred_kg"] is not None}
    out = {}
    for m, rows in rows_by_mode.items():
        if m == "rgbd":
            continue
        rel = [abs(r["pred_kg"] - ref[r["file"]]) / ref[r["file"]] * 100
               for r in rows if r["pred_kg"] is not None and r["file"] in ref]
        if rel:
            out[m] = {"n_frames": len(rel), "mean_pct": float(np.mean(rel)), "median_pct": float(np.median(rel))}
    return out


# ====================================================================================
# VISUALIZATION
# ====================================================================================

def draw_panels(image_bgr, depth_raw, row, instances, top):
    """
    3 panels: RGB + GT | top prediction on RGB (error %, score, distance) raw depth, colour-mapped per image.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    left = image_bgr.copy()
    cv2.putText(left, f"GT {row['gt_kg']:.3f} kg   input: {row['input_mode']}",
                (10, 30), font, 0.8, (0, 255, 0), 2)

    mid = image_bgr.copy()
    if top is None:
        cv2.putText(mid, "no detection", (10, 30), font, 0.8, (0, 0, 255), 2)
    else:
        mask = instances.pred_masks[top].numpy()
        mid[mask] = (0.5 * mid[mask] + (0, 0, 127)).astype(np.uint8)
        contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[-2]
        cv2.drawContours(mid, contours, -1, (0, 255, 255), 2)
        pct = 100 * row["err_kg"] / row["gt_kg"]
        dist = f"{row['distance_mm']:.0f} mm" if row["distance_mm"] is not None else "n/a"
        cv2.putText(mid, f"pred {row['pred_kg']:.3f} kg ({pct:+.0f}%)  score {row['score']:.2f}",
                    (10, 30), font, 0.8, (0, 255, 0), 2)
        cv2.putText(mid, f"detections {row['n_detections']}  distance {dist}",
                    (10, 60), font, 0.7, (0, 255, 0), 2)

    valid = depth_raw > 0
    if valid.any():
        d_min, d_max = depth_raw[valid].min(), depth_raw[valid].max()
        depth_norm = np.clip((depth_raw - d_min) / (d_max - d_min + 1e-6) * 255, 0, 255)
    else:
        depth_norm = np.zeros_like(depth_raw)
    depth_color = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.putText(depth_color, "Depth (color-mapped)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return np.hstack([left, mid, depth_color])


# ====================================================================================
# REPORT
# ====================================================================================

def write_report(path, header, rows_by_mode, summary):
    lines = [header, "", f"{'input mode':<12} {'MAPE %':>7} {'MAE kg':>7} {'bias kg':>8} {'R2':>6} {'no det':>8}"]
    for mode, s in summary["modes"].items():
        miss = f"{s['n_no_detection']}/{s['n_images']}"
        if "mape_pct" in s:
            r2 = f"{s['r2']:.3f}" if s["r2"] is not None else "-"
            lines.append(f"{mode:<12} {s['mape_pct']:7.1f} {s['mae_kg']:7.3f} {s['bias_kg']:+8.3f} {r2:>6} {miss:>8}")
        else:
            lines.append(f"{mode:<12} {'no predictions':>30} {miss:>8}")

    if "common_frames" in summary:
        c = summary["common_frames"]
        lines += ["", f"Same frames only ({c['n_frames']} frames where every mode has a prediction):"]
        for mode, s in c["modes"].items():
            if "mape_pct" in s:
                lines.append(f"{mode:<12} {s['mape_pct']:7.1f} {s['mae_kg']:7.3f} {s['bias_kg']:+8.3f}")

    if "change_vs_rgbd" in summary:
        lines += ["", "Change vs rgbd (how much the prediction moves; mean / median |pred - pred_rgbd| / pred_rgbd):"]
        for mode, c in summary["change_vs_rgbd"].items():
            lines.append(f"{mode:<14} {c['mean_pct']:6.1f}% / {c['median_pct']:5.1f}%   ({c['n_frames']} frames)")

    for mode, rows in rows_by_mode.items():
        lines += ["", "=" * 78, f"{mode}: results by cluster (GT mass), frames sorted by distance", "=" * 78]
        groups = defaultdict(list)
        for r in rows:
            groups[f"{r['gt_kg']:.3f}"].append(r)
        for key in sorted(groups, key=float):
            g = sorted(groups[key], key=lambda r: (r["distance_mm"] is None, r["distance_mm"] or 0.0))
            lines.append(f"\n----- GT mass {key} kg  ({len(g)} frames) -----")
            for r in g:
                d = f"{r['distance_mm']:7.1f} mm" if r["distance_mm"] is not None else "    n/a   "
                if r["pred_kg"] is None:
                    lines.append(f"  {r['file']:42} {d}  GT={r['gt_kg']:.3f}  pred=(no detection)")
                else:
                    lines.append(f"  {r['file']:42} {d}  GT={r['gt_kg']:.3f}  pred={r['pred_kg']:.3f}  "
                                 f"err={r['err_kg']:+.3f} ({r['abs_pct_err']:4.1f}%)")
            s = summary["clusters"][mode][key]
            if "mape_pct" in s:
                dr = s["distance_mm_range"]
                dr = f"{dr[0]:.0f}-{dr[1]:.0f} mm" if dr else "n/a"
                lines.append(f"    -> n={s['n_with_prediction']}  MAE={s['mae_kg']:.3f} kg  "
                             f"MAPE={s['mape_pct']:.1f}%  bias={s['bias_kg']:+.3f} kg  dist={dr}")
        s = summary["modes"][mode]
        if "mape_pct" in s:
            lines += ["", f"  MAE = {s['mae_kg']:.3f} kg   MAPE = {s['mape_pct']:.1f}%   "
                          f"bias = {s['bias_kg']:+.3f} kg   worst = {s['worst_abs_err_kg']:.3f} kg"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ====================================================================================
# MAIN
# ====================================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a model with input-modality ablations")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="Run name under output/ (uses --checkpoint inside it)")
    src.add_argument("--model_path", help="Explicit checkpoint path")
    p.add_argument("--checkpoint", default="model_final.pth", help="Checkpoint file inside the run folder")
    p.add_argument("--dataset_path", default="./data_s0713/")
    p.add_argument("--splits", nargs="+", default=["val"], help="e.g. val test_0328 test_1201")
    p.add_argument("--input_modes", nargs="+", default=INPUT_MODES, choices=INPUT_MODES + DEPTH_ABLATIONS)
    p.add_argument("--tag", default=None,
                   help="Suffix for the result folder (results/eval/<run>__<tag>/), e.g. depth_ablation")
    p.add_argument("--conf_thr", type=float, default=0.7, help="Minimum detection score")
    p.add_argument("--pixel_std", type=float, nargs=4, default=None,
                   help="Override PIXEL_STD (runs trained before the norm fix: 1 1 1 411.683)")
    p.add_argument("--save_viz", type=int, nargs="?", const=0, default=None,
                   help="Save 3-panel images. No value = all frames; N = the N worst frames per "
                        "mode (misses count as worst)")
    p.add_argument("--overwrite", action="store_true", help="Replace existing results for this run/split")
    return p.parse_args()


def evaluate_split(model, args, split):
    """-> rows_by_mode, viz items. Runs every input mode on every image of the split."""
    dataset_dicts = utils_detection.get_tomato_dataset_dicts(args.dataset_path, split)
    rows_by_mode = {m: [] for m in args.input_modes}
    viz = {m: [] for m in args.input_modes}
    partners = shuffled_partners(dataset_dicts) if "rgbd_shuffled" in args.input_modes else None

    for i, d in enumerate(dataset_dicts):
        image_bgr = cv2.imread(d["file_name"])
        depth_raw = np.load(d["depth_file"])
        depth_valid = cv2.resize(depth_raw.astype(np.float32), image_bgr.shape[1::-1],
                                 interpolation=cv2.INTER_AREA)
        flat_mm = cluster_distance_mm(depth_valid, d["annotations"]) if "rgbd_flat" in args.input_modes else None
        other_depth = np.load(dataset_dicts[partners[i]]["depth_file"]) if partners else None
        for mode in args.input_modes:
            image = build_input(image_bgr, depth_raw, mode, flat_mm, other_depth)
            with torch.no_grad():
                inst = model([{"image": torch.from_numpy(image.transpose(2, 0, 1)).float(),
                               "height": image.shape[0], "width": image.shape[1]}])[0]["instances"].to("cpu")

            row = {"file": os.path.basename(d["file_name"]), "input_mode": mode, "gt_kg": d["cluster_mass"],
                   "n_detections": len(inst), "score": None, "pred_kg": None, "err_kg": None,
                   "abs_pct_err": None, "distance_mm": None}
            top = None
            if len(inst) > 0:
                top = int(inst.scores.argmax())
                row["score"] = round(float(inst.scores[top]), 4)
                row["pred_kg"] = round(float(np.exp(inst.pred_mass[top].item())), 3)
                row["err_kg"] = round(row["pred_kg"] - row["gt_kg"], 3)
                row["abs_pct_err"] = abs(row["err_kg"]) / row["gt_kg"] * 100
                in_mask = inst.pred_masks[top].numpy() & (depth_valid > 0)
                if in_mask.any():
                    row["distance_mm"] = round(float(depth_valid[in_mask].mean()), 1)
            rows_by_mode[mode].append(row)

            if args.save_viz is not None:
                rank = float("inf") if row["pred_kg"] is None else row["abs_pct_err"]
                viz[mode].append((rank, i, row, draw_panels(image_bgr, depth_valid, row, inst, top)))
                if args.save_viz > 0:   # keep only the N worst in memory
                    viz[mode] = sorted(viz[mode], key=lambda v: -v[0])[:args.save_viz]
        sys.stderr.write(f"\r  {split}: {i + 1}/{len(dataset_dicts)}")
        sys.stderr.flush()
    sys.stderr.write("\n")
    return rows_by_mode, viz


def main():
    args = parse_args()
    name, ckpt, run_dir = resolve_model(args)
    pixel_mean, pixel_std, norm_source = input_normalization(args, run_dir)

    out_dirs = {s: os.path.join("results", "eval", name, s) for s in args.splits}
    for d in out_dirs.values():
        if os.path.exists(d) and not args.overwrite:
            raise SystemExit(f"Results already exist: {d}  (use --overwrite to replace)")

    print(f"Model     : {ckpt}")
    print(f"Splits    : {args.splits}   modes: {args.input_modes}   conf_thr: {args.conf_thr}")
    print(f"PIXEL_STD : {pixel_std}   ({norm_source})")
    model = build_eval_model(ckpt, pixel_mean, pixel_std, args.conf_thr, args.dataset_path)

    for split in args.splits:
        rows_by_mode, viz = evaluate_split(model, args, split)

        summary = {
            "model": ckpt, "name": name, "split": split, "dataset_path": args.dataset_path,
            "conf_thr": args.conf_thr, "pixel_mean": list(pixel_mean), "pixel_std": list(pixel_std),
            "pixel_norm_source": norm_source, "evaluated_at": datetime.now().isoformat(timespec="seconds"),
            "command": "python " + " ".join(sys.argv),
            "modes": {m: summarize(rows_by_mode[m]) for m in args.input_modes},
            "clusters": {},
        }
        for m in args.input_modes:
            by_cluster = defaultdict(list)
            for r in rows_by_mode[m]:
                by_cluster[f"{r['gt_kg']:.3f}"].append(r)
            summary["clusters"][m] = {k: summarize(v) for k, v in sorted(by_cluster.items())}
        if len(args.input_modes) > 1:
            summary["common_frames"] = summarize_common_frames(rows_by_mode)
        if "rgbd" in args.input_modes and len(args.input_modes) > 1:
            summary["change_vs_rgbd"] = change_vs_rgbd(rows_by_mode)

        out = out_dirs[split]
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "per_image.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_by_mode[args.input_modes[0]][0]))
            w.writeheader()
            for m in args.input_modes:
                w.writerows(rows_by_mode[m])
        with open(os.path.join(out, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        write_report(os.path.join(out, "report.txt"),
                     f"{name}  |  split {split}  |  {ckpt}  |  conf_thr {args.conf_thr}",
                     rows_by_mode, summary)

        if args.save_viz is not None:
            for m, items in viz.items():
                vdir = os.path.join("output", "_viz", name, split, m)
                os.makedirs(vdir, exist_ok=True)
                for _, i, row, panels in items:
                    cv2.imwrite(os.path.join(vdir, f"{i:03d}_{row['file']}"), panels)
            print(f"  images -> output/_viz/{name}/{split}/")

        print(f"\n{split}:")
        for m, s in summary["modes"].items():
            if "mape_pct" in s:
                print(f"  {m:<11} MAPE {s['mape_pct']:5.1f}%  MAE {s['mae_kg']:.3f} kg  "
                      f"bias {s['bias_kg']:+.3f}  no detection {s['n_no_detection']}/{s['n_images']}")
            else:
                print(f"  {m:<11} no predictions ({s['n_images']} images)")
        if "common_frames" in summary:
            c = summary["common_frames"]
            print(f"  same {c['n_frames']} frames: " + "  ".join(
                f"{m} {s['mape_pct']:.1f}%" for m, s in c["modes"].items() if "mape_pct" in s))
        if "change_vs_rgbd" in summary:
            print("  change vs rgbd: " + "  ".join(
                f"{m} {c['mean_pct']:.1f}%" for m, c in summary["change_vs_rgbd"].items()))
        print(f"  -> {out}/")


if __name__ == "__main__":
    main()
