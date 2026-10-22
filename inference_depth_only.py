"""
Inference script for the depth-only segmentation model.
Applies the same depth->BGR transform as train.py, with corrected normalization (mm scale).

Saves visualizations showing: depth input | predicted masks overlaid on depth | predicted masks overlaid on RGB

Usage:
    python inference_depth_only.py \
        --model_path ./output/depth_only_segm/model_final.pth \
        --dataset_path /path/to/data/ \
        --split val \
        --n 20
"""
import matplotlib.pyplot as plt
import torch
import numpy as np
import os, cv2, argparse

import detectron2
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


def build_depth_only_4ch(image_bgr, depth_mm_raw):
    """
    Build a 4-channel depth-only image, exactly matching train.py _build_4ch_image_and_annos().

    Returns:
        image_4ch: (H, W, 4) float32 — depth replicated into BGR + raw depth in ch3
        depth_cache: (H, W) float32 — raw depth in mm with zeros for invalid (for mass head)
    """
    H, W = image_bgr.shape[:2]

    # Resize depth to match image
    depth_mm = cv2.resize(depth_mm_raw.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)

    # Cache: raw depth with zeros for invalid (same as train.py)
    depth_cache = depth_mm.copy()

    # Fill invalid pixels with mean (same as train.py)
    valid = depth_mm > 0
    depth_mm[~valid] = 915.043  # depth mean in mm



    # Build 4-channel image
    image_4ch = np.zeros((H, W, 4), dtype=np.float32)

    # Depth-only: replicate depth into BGR channels (same as train.py depth-only block).
    # GLOBAL normalization [200, 2000] mm — MUST match the constants in train.py
    # _build_4ch_image_and_annos(). If you change them there, change them here too.
    DEPTH_GLOBAL_MIN_MM = 200.0
    DEPTH_GLOBAL_MAX_MM = 2000.0
    depth_visual = np.clip(
        (depth_mm - DEPTH_GLOBAL_MIN_MM)
        / (DEPTH_GLOBAL_MAX_MM - DEPTH_GLOBAL_MIN_MM) * 255.0,
        0, 255,
    )
    image_4ch[:, :, 0] = depth_visual  # B
    image_4ch[:, :, 1] = depth_visual  # G
    image_4ch[:, :, 2] = depth_visual  # R

    #depth_cache[:] = np.zeros_like(depth_cache) 
    depth_mm[:] = DEPTH_MEAN_MM#np.zeros_like(depth_mm)

    # Channel 3: raw depth in mm (for backbone + mass head)
    image_4ch[:, :, 3] = depth_mm

    return image_4ch, depth_cache, depth_visual.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description='Depth-only model inference + visualization')
    parser.add_argument('--model_path', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--dataset_path', required=True, help='Root data folder')
    parser.add_argument('--split', default='val', help='train or val')
    parser.add_argument('--n', type=int, default=20, help='Number of images to process (0=all)')
    parser.add_argument('--conf_thr', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--out_dir', default='./depth_only_inference', help='Output folder')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Config (matching train.py) ---
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.META_ARCHITECTURE = "RGBDGeneralizedRCNN"
    # CORRECT normalization: mm scale (same as train.py)
    #cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 915.043]
    #cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 411.683]
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 915.043]
    cfg.MODEL.PIXEL_STD = [57.375, 57.120, 58.395, 411.683]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.conf_thr
    cfg.MODEL.ROI_MASK_HEAD.LOSS_WEIGHT = 1.0
    cfg.MODEL.ROI_MASS_HEAD.MASS_LOSS_WEIGHT = 1.0
    cfg.MODEL.WEIGHTS = args.model_path
    cfg.DATASET_PATH = args.dataset_path

    # --- Build model ---
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    print(f"Loaded model from {args.model_path}")

    # --- Register dataset ---
    dataset_name = f"TomatoDataset_{args.split}"
    DatasetCatalog.register(
        dataset_name,
        lambda: utils_detection.get_tomato_dataset_dicts(args.dataset_path, args.split)
    )
    MetadataCatalog.get(dataset_name).set(thing_classes=["tomato_bunch"])
    metadata = MetadataCatalog.get(dataset_name)

    dataset_dicts = DatasetCatalog.get(dataset_name)
    n_images = len(dataset_dicts) if args.n == 0 else min(args.n, len(dataset_dicts))
    print(f"Running inference on {n_images} images from {args.split} split")

    per_image_results = []  # list of (fname, gt_kg, pred_kg)
    no_pred_count = 0

    for i in range(n_images):
        d = dataset_dicts[i]
        fname = os.path.basename(d["file_name"])

        # Load RGB and depth
        image_bgr = cv2.imread(d["file_name"])
        if image_bgr is None:
            print(f"Could not load: {d['file_name']}")
            continue
        depth_mm_raw = np.load(d["depth_file"])

        # Build depth-only 4-channel image (same as train.py)
        image_4ch, depth_cache, depth_vis_gray = build_depth_only_4ch(image_bgr, depth_mm_raw)
        H, W = image_4ch.shape[:2]

        # Run inference
        inputs = {
            "image": torch.from_numpy(image_4ch.transpose(2, 0, 1)).float().to(model.device),
            "depth": torch.from_numpy(depth_cache).unsqueeze(0).float().to(model.device),
            "height": H,
            "width": W,
        }

        with torch.no_grad():
            outputs = model([inputs])[0]

        instances = outputs["instances"].to("cpu")
        n_det = len(instances)

        gt_mass = d.get("cluster_mass", None)

        # Pick the highest-score detection as the per-image prediction.
        if n_det == 0 or not instances.has("pred_mass") or gt_mass is None:
            print(f"[{i+1:>4}/{n_images}] {fname}  GT={gt_mass}  no detections")
            no_pred_count += 1
            pred_kg = None
        else:
            scores = instances.scores.numpy()
            top_idx = int(scores.argmax())
            pred_log = float(instances.pred_mass[top_idx].numpy())
            pred_kg = float(np.exp(pred_log))
            err_kg = pred_kg - gt_mass
            err_pct = abs(err_kg) / gt_mass * 100.0
            sign = "+" if err_kg >= 0 else "-"
            print(f"[{i+1:>4}/{n_images}] {fname}  "
                  f"GT={gt_mass:.3f}  pred={pred_kg:.3f}  "
                  f"err={sign}{abs(err_kg):.3f} ({err_pct:.1f}%)")
            per_image_results.append((fname, gt_mass, pred_kg))

        # --- Visualize ---
        # Panel 1: Depth input (what model sees)
        depth_vis_bgr = cv2.merge([depth_vis_gray, depth_vis_gray, depth_vis_gray])

        # Panel 2: Predictions overlaid on depth
        vis_depth = Visualizer(depth_vis_bgr[:, :, ::-1], metadata=metadata, scale=1.0,
                               instance_mode=ColorMode.IMAGE)
        vis_depth_out = vis_depth.draw_instance_predictions(instances)
        panel_depth = np.ascontiguousarray(vis_depth_out.get_image()[:, :, ::-1])

        # Panel 3: Predictions overlaid on original RGB
        vis_rgb = Visualizer(image_bgr[:, :, ::-1], metadata=metadata, scale=1.0,
                             instance_mode=ColorMode.IMAGE)
        vis_rgb_out = vis_rgb.draw_instance_predictions(instances)
        panel_rgb = np.ascontiguousarray(vis_rgb_out.get_image()[:, :, ::-1])

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        gt_str = f" GT={gt_mass:.3f}kg" if gt_mass is not None else ""
        pred_lbl = f" pred={pred_kg:.3f}kg" if pred_kg is not None else " no pred"
        cv2.putText(depth_vis_bgr, "Depth input (model sees this)", (10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(panel_depth, f"Predictions on depth ({n_det} det){pred_lbl}", (10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(panel_rgb, f"Predictions on RGB (reference){gt_str}", (10, 30), font, 0.7, (0, 255, 0), 2)

        combined = np.hstack([depth_vis_bgr, panel_depth, panel_rgb])
        out_path = os.path.join(args.out_dir, f"{i:03d}_{fname}")
        cv2.imwrite(out_path, combined)

    # ================ SUMMARY ================
    print()
    n = len(per_image_results)
    if n == 0:
        print(f"No predictions produced ({no_pred_count} images had no detections / no GT).")
    else:
        gt_arr   = np.array([r[1] for r in per_image_results], dtype=np.float64)
        pred_arr = np.array([r[2] for r in per_image_results], dtype=np.float64)
        err_arr  = pred_arr - gt_arr
        mae   = float(np.mean(np.abs(err_arr)))
        mape  = float(np.mean(np.abs(err_arr / gt_arr)) * 100.0)
        bias  = float(np.mean(err_arr))
        worst = float(np.max(np.abs(err_arr)))
        print(f"Overall summary ({n}/{n_images} with predictions):")
        print(f"  MAE = {mae:.3f} kg")
        print(f"  MAPE = {mape:.1f}%")
        print(f"  Mean signed error (bias) = {bias:+.3f} kg")
        print(f"  Worst error: {worst:.3f} kg")
        if no_pred_count > 0:
            print(f"  ({no_pred_count} images had no detections; excluded from summary)")
        print()

        # Per-mass-group breakdown
        unique_gt = sorted(set(gt_arr.tolist()))
        print(f"Per-mass-group breakdown ({len(unique_gt)} unique GT values):")
        print(f"{'GT (kg)':>8} {'n':>5} {'mean_pred':>11} {'MAE':>9} {'MAPE':>8} {'bias':>9}")
        print("-" * 60)
        for g in unique_gt:
            mask = (gt_arr == g)
            sub_pred = pred_arr[mask]
            sub_err  = sub_pred - g
            n_g    = int(mask.sum())
            mean_p = float(np.mean(sub_pred))
            mae_g  = float(np.mean(np.abs(sub_err)))
            mape_g = float(np.mean(np.abs(sub_err)) / g * 100.0)
            bias_g = float(np.mean(sub_err))
            print(f"{g:>8.3f} {n_g:>5d} {mean_p:>11.3f} {mae_g:>9.3f} {mape_g:>7.1f}% {bias_g:>+9.3f}")

    print()
    print(f"Done! Visualizations saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
