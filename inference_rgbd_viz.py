"""
Visualize RGB-D mass predictions side by side with ground truth.

Usage:
    python inference_rgbd_viz.py \\
        --model_path output/clean_baseline_v1_LF/model_0009499.pth \\
        --dataset_path ./data/ \\
        --split val \\
        --n 20 \\
        --out_dir ./qual_val
"""
import os, sys, argparse, copy
import cv2
import numpy as np
import torch

from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

import utils_detection


DEPTH_MEAN_MM = 915.043
DEPTH_STD_MM  = 411.683


def build_rgbd_input(image_bgr, depth_mm_raw):
    """Build 4-channel input matching train_new.py mapper exactly."""
    H, W = image_bgr.shape[:2]
    depth_mm = cv2.resize(depth_mm_raw.astype(np.float32), (W, H),
                          interpolation=cv2.INTER_AREA)
    # MUST match training (load_rgbd_image): invalid (0mm) pixels -> depth mean,
    # which becomes 0 after normalization (neutral). Without this, 0mm -> -2.22 sigma.
    depth_mm[depth_mm == 0] = DEPTH_MEAN_MM
    image_4ch = np.zeros((H, W, 4), dtype=np.float32)
    image_4ch[:, :, :3] = image_bgr.astype(np.float32)
    #image_4ch[:, :, 0] = 103.530  # B
    #image_4ch[:, :, 1] = 116.280  # G
    #image_4ch[:, :, 2] = 123.675  # R

    image_4ch[:, :, 3]  = depth_mm #np.zeros((H, W), dtype=np.float32)
    return image_4ch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',   required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--split',        default='val', help='train or val')
    parser.add_argument('--n',            type=int, default=20)
    parser.add_argument('--conf_thr',     type=float, default=0.7)
    parser.add_argument('--out_dir',      default='./qual_inference')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Config matching train_new.py ---
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.META_ARCHITECTURE       = "RGBDGeneralizedRCNN"
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, DEPTH_MEAN_MM]
    #cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, DEPTH_STD_MM]
    cfg.MODEL.PIXEL_STD = [57.375, 57.120, 58.395, 411.683]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES   = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.conf_thr
    cfg.MODEL.ROI_MASK_HEAD.LOSS_WEIGHT      = 1.0
    cfg.MODEL.ROI_MASS_HEAD.MASS_LOSS_WEIGHT = 1.0
    cfg.MODEL.WEIGHTS                 = args.model_path
    cfg.DATASET_PATH                  = args.dataset_path

    # --- Build model and load weights ---
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    print(f"Loaded model from {args.model_path}")

    # --- Register dataset ---
    dataset_name = f"TomatoDataset_{args.split}"
    if dataset_name not in DatasetCatalog.list():
        DatasetCatalog.register(
            dataset_name,
            lambda: utils_detection.get_tomato_dataset_dicts(args.dataset_path, args.split))
        MetadataCatalog.get(dataset_name).set(thing_classes=["tomato_bunch"])
    metadata = MetadataCatalog.get(dataset_name)
    dataset_dicts = DatasetCatalog.get(dataset_name)

    n_images = len(dataset_dicts) if args.n == 0 else min(args.n, len(dataset_dicts))
    print(f"Running on {n_images} images from {args.split} split")

    # Per-image error rows for a small summary
    summary_rows = []

    for i in range(n_images):
        d = dataset_dicts[i]
        fname = os.path.basename(d["file_name"])

        image_bgr = cv2.imread(d["file_name"])
        if image_bgr is None:
            print(f"Could not load: {d['file_name']}")
            continue
        depth_mm_raw = np.load(d["depth_file"])

        image_4ch = build_rgbd_input(image_bgr, depth_mm_raw)
        H, W = image_4ch.shape[:2]

        inputs = {
            "image":  torch.from_numpy(image_4ch.transpose(2, 0, 1)).float().to(model.device),
            "height": H,
            "width":  W,
        }

        with torch.no_grad():
            outputs = model([inputs])[0]

        instances = outputs["instances"].to("cpu")
        n_det = len(instances)

        gt_mass = d.get("cluster_mass", None)

        # Pred mass (model predicts log-mass)
        pred_kgs = []
        if instances.has("pred_mass") and n_det > 0:
            pred_log = instances.pred_mass.numpy().ravel()
            pred_kgs = np.exp(pred_log).round(3).tolist()

        # If multiple detections, take the highest-confidence one for the comparison
        top_pred_kg = None
        depth_mean_mm = None
        if n_det > 0 and len(pred_kgs) > 0:
            top_idx = int(instances.scores.argmax().item())
            top_pred_kg = pred_kgs[top_idx]

            # Mean camera->cluster distance INSIDE the predicted mask (top detection).
            # depth_mm_raw is already aligned/same-size as the image and mask.
            if instances.has("pred_masks"):
                mask = instances.pred_masks[top_idx].numpy().astype(bool)
                valid_px = mask & (depth_mm_raw > 0)
                if valid_px.any():
                    depth_mean_mm = float(depth_mm_raw[valid_px].mean())

        err = (top_pred_kg - gt_mass) if (top_pred_kg is not None and gt_mass is not None) else None
        summary_rows.append((fname, gt_mass, top_pred_kg, err, depth_mean_mm))

        # --- Visualize ---
        # Panel 1: RGB with GT mass
        vis_gt = image_bgr.copy()
        if gt_mass is not None:
            cv2.putText(vis_gt, f"GT mass: {gt_mass:.3f} kg",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Panel 2: RGB with predictions overlaid
        v = Visualizer(image_bgr[:, :, ::-1], metadata=metadata, scale=1.0,
                       instance_mode=ColorMode.IMAGE)
        vis_out = v.draw_instance_predictions(instances)
        panel_pred = np.ascontiguousarray(vis_out.get_image()[:, :, ::-1])

        # Annotate predicted masses
        if pred_kgs:
            y = 30
            cv2.putText(panel_pred, f"Detections: {n_det}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 30
            for k, m in enumerate(pred_kgs):
                star = " *" if (top_pred_kg is not None and k == top_idx) else ""
                cv2.putText(panel_pred, f"pred {k}: {m:.3f} kg{star}",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y += 25
        else:
            cv2.putText(panel_pred, "no detections",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Panel 3: depth visualization
        depth_vis = depth_mm_raw.astype(np.float32)
        depth_vis = cv2.resize(depth_vis, (W, H), interpolation=cv2.INTER_AREA)
        valid = depth_vis > 0
        if valid.any():
            d_min, d_max = depth_vis[valid].min(), depth_vis[valid].max()
            depth_norm = np.clip((depth_vis - d_min) / (d_max - d_min + 1e-6) * 255, 0, 255)
        else:
            depth_norm = np.zeros_like(depth_vis)
        depth_color = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.putText(depth_color, "Depth (color-mapped)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        combined = np.hstack([vis_gt, panel_pred, depth_color])
        out_path = os.path.join(args.out_dir, f"{i:03d}_{fname}")
        cv2.imwrite(out_path, combined)

        # Lightweight progress to stderr so the (redirected) stdout stays clean
        # for the grouped-by-cluster listing below.
        sys.stderr.write(f"\r  processed {i+1}/{n_images}")
        sys.stderr.flush()

    sys.stderr.write("\n")

    # --- Results organized by cluster (GT mass), sorted by distance within each ---
    from collections import defaultdict
    groups = defaultdict(list)
    for row in summary_rows:                       # (fname, gt, pred, err, depth_mm)
        gt = row[1]
        groups[round(gt, 3) if gt is not None else None].append(row)

    print("\n" + "=" * 78)
    print("RESULTS BY CLUSTER (GT mass) — frames sorted by mean in-mask distance")
    print("=" * 78)
    for gt_kg in sorted(k for k in groups if k is not None):
        rows = groups[gt_kg]
        # sort by distance ascending; frames with no distance go last
        rows.sort(key=lambda r: (r[4] is None, r[4] if r[4] is not None else 0.0))
        print(f"\n----- GT mass {gt_kg:.3f} kg  ({len(rows)} frames) -----")
        for fname, gt, pred, err, dmm in rows:
            dstr = f"{dmm:7.1f} mm" if dmm is not None else "    n/a   "
            if pred is not None and gt is not None:
                rel = abs(err) / gt * 100 if gt > 0 else float('nan')
                print(f"  {fname:42} {dstr}  GT={gt:.3f}  pred={pred:.3f}  "
                      f"err={err:+.3f} ({rel:4.1f}%)")
            else:
                print(f"  {fname:42} {dstr}  GT={gt}  pred=(no detection)")
        # per-cluster stats (over frames with a prediction)
        v = [(e, dmm) for _, _, _, e, dmm in rows if e is not None]
        if v:
            errs_g = np.array([abs(e) for e, _ in v])
            signed_g = np.array([e for e, _ in v])
            mape_g = (errs_g / gt_kg * 100).mean() if gt_kg > 0 else float('nan')
            dd = [dm for _, dm in v if dm is not None]
            drange = f"{min(dd):.0f}-{max(dd):.0f} mm" if dd else "n/a"
            print(f"    -> n={len(v)}  MAE={errs_g.mean():.3f} kg  MAPE={mape_g:.1f}%  "
                  f"bias={signed_g.mean():+.3f} kg  dist={drange}")

    # --- Overall summary ---
    valid = [(f, g, p, e, dm) for f, g, p, e, dm in summary_rows if e is not None]
    if valid:
        errs = np.array([abs(e) for _, _, _, e, _ in valid])
        signed = np.array([e for _, _, _, e, _ in valid])
        rel_errs = np.array([abs(e) / g * 100 for _, g, _, e, _ in valid if g])
        print("\n" + "=" * 78)
        print(f"Overall ({len(valid)}/{len(summary_rows)} with predictions):")
        print(f"  MAE = {errs.mean():.3f} kg   MAPE = {rel_errs.mean():.1f}%   "
              f"bias = {signed.mean():+.3f} kg   worst = {errs.max():.3f} kg")

    print(f"\nDone. Visualizations saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
