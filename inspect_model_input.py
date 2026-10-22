"""
Inspect exactly what the model receives as input, end-to-end.

Pipeline reminder:
    dataset_dict
        |
        v
    test_mapper (in train.py)   ─── outputs raw 4-channel image + "depth" tensor (raw mm)
        |
        v
    model.preprocess_image()     ─── normalizes EVERY channel:
        |                            (x - PIXEL_MEAN[c]) / PIXEL_STD[c]
        v
    backbone.conv1               ─── receives the normalized 4-ch tensor

In parallel, the "depth" tensor flows through:
    RGBDGeneralizedRCNN._cache_depth()
        valid = depth > 200
        depth_norm = (depth - 915.043) / 411.683
        cached_depth = depth_norm * valid
        |
        v
    roi_heads._concat_cached_depth_to_features()  ── attaches cached_depth as ch256
        |
        v
    mass_head.mass_fcn1          ─── receives 257-channel ROI features

This script reports per-channel stats at every stage above, for a few sample images,
so you can verify what the backbone and the mass head actually see — before pressing
"train" on any modification.

Usage:
    python inspect_model_input.py --dataset_path ./data/ --split train --n 3
    python inspect_model_input.py --dataset_path ./data/ --split train --n 3 --simulate_ch3_zero

The --simulate_ch3_zero flag does NOT modify train.py. It only overrides the
mapper's output AFTER the fact, setting channel 3 = 915.043 (mean depth) and the
"depth" tensor = 0, so you can preview the values the model would receive if you
made that change. Compare the printed stats with and without the flag.
"""
import argparse
import os
import sys
import numpy as np
import torch

# Make sibling modules importable
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from detectron2.data import DatasetCatalog, MetadataCatalog
import cv2
import utils_detection
# train.py exposes `test_mapper` (no augmentation → reproducible inspection)
import train as train_module


# These MUST match cfg.MODEL.PIXEL_MEAN / cfg.MODEL.PIXEL_STD in train.py main()
PIXEL_MEAN = np.array([103.530, 116.280, 123.675, 915.043], dtype=np.float64)
PIXEL_STD  = np.array([57.375, 57.120, 58.395, 411.683],            dtype=np.float64)

# These MUST match _cache_depth in rcnn.py
LATE_FUSION_VALID_THRESH_MM = 200.0


# ---------- helpers ----------

def stats(arr):
    a = np.asarray(arr, dtype=np.float64)
    return dict(min=a.min(), max=a.max(), mean=a.mean(), std=a.std(),
                shape=tuple(a.shape))


def print_stats_row(label, arr):
    s = stats(arr)
    print(f"    {label:<40}  min={s['min']:+10.4f}  max={s['max']:+10.4f}  "
          f"mean={s['mean']:+10.4f}  std={s['std']:.4f}")


def save_grayscale(arr, path, vmin=None, vmax=None):
    """Save a 2D float array as a 0-255 grayscale PNG using the given range."""
    a = np.asarray(arr, dtype=np.float64)
    if vmin is None:
        vmin = float(a.min())
    if vmax is None:
        vmax = float(a.max())
    if vmax - vmin < 1e-12:
        a8 = np.full(a.shape, 128, dtype=np.uint8)  # constant → mid-gray
    else:
        a8 = np.clip((a - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(path, a8)


def normalize_channels(img_chw):
    """Apply (x - PIXEL_MEAN) / PIXEL_STD per channel. img_chw shape (4, H, W)."""
    out = img_chw.copy().astype(np.float64)
    for c in range(4):
        out[c] = (img_chw[c] - PIXEL_MEAN[c]) / PIXEL_STD[c]
    return out


def cache_depth_replay(depth_hw):
    """
    Replay what RGBDGeneralizedRCNN._cache_depth does:
        valid = depth > 200
        depth_norm = (depth - 915.043) / 411.683
        cached_depth = depth_norm * valid
    Returns (cached_depth, valid_mask).
    """
    depth_hw = np.asarray(depth_hw, dtype=np.float64)
    valid = (depth_hw > LATE_FUSION_VALID_THRESH_MM).astype(np.float64)
    depth_norm = (depth_hw - PIXEL_MEAN[3]) / PIXEL_STD[3]
    return depth_norm * valid, valid


# ---------- main per-sample inspection ----------

def inspect_one(dataset_dict, simulate_ch3_zero, out_dir, idx):
    fname = os.path.basename(dataset_dict["file_name"])
    gt_kg = dataset_dict.get("cluster_mass", None)

    print()
    print("=" * 100)
    print(f"SAMPLE [{idx}]  {fname}    cluster_mass={gt_kg} kg")
    print("=" * 100)

    # Run the mapper exactly as the test loader would — no augmentation
    mapped = train_module.test_mapper(dataset_dict)
    image_chw = mapped["image"].numpy().astype(np.float64)   # (4, H, W) raw
    depth_chw = mapped["depth"].numpy().astype(np.float64)   # (1, H, W) raw mm
    depth_hw  = depth_chw[0]

    if simulate_ch3_zero:
        print("  [simulation] overriding ch3 -> 915.043 mm, depth tensor -> 0")
        image_chw[3] = PIXEL_MEAN[3]      # raw value that normalizes to 0
        depth_hw[:]  = 0.0                # zero late-fusion path

    # --- 1) Mapper output (RAW, before normalization) ---
    print(f"\n  STAGE 1 — Mapper output (RAW, before the model normalizes):")
    print(f"  image tensor shape: {image_chw.shape}, depth tensor shape: {depth_hw.shape}")
    for c, name in enumerate(("ch0 B (BGR-replicated depth)",
                              "ch1 G (BGR-replicated depth)",
                              "ch2 R (BGR-replicated depth)",
                              "ch3 (raw depth, mm)")):
        print_stats_row(name, image_chw[c])
    print_stats_row("'depth' tensor (raw mm, for late fusion)", depth_hw)

    # --- 2) After model's preprocess_image normalization ---
    img_norm = normalize_channels(image_chw)
    print(f"\n  STAGE 2 — After preprocess_image normalization "
          f"(this is what backbone.conv1 receives):")
    print(f"  formula: (raw - PIXEL_MEAN[c]) / PIXEL_STD[c]")
    print(f"  PIXEL_MEAN = {PIXEL_MEAN.tolist()}")
    print(f"  PIXEL_STD  = {PIXEL_STD.tolist()}")
    for c, name in enumerate(("ch0 normalized",
                              "ch1 normalized",
                              "ch2 normalized",
                              "ch3 normalized")):
        print_stats_row(name, img_norm[c])

    # --- 3) Late-fusion path (what mass head ch256 receives) ---
    cached_depth, valid_mask = cache_depth_replay(depth_hw)
    valid_pct = 100.0 * valid_mask.sum() / valid_mask.size
    print(f"\n  STAGE 3 — Late-fusion depth (what mass head ch256 receives):")
    print(f"  formula: ((depth - 915.043) / 411.683) * (depth > 200)")
    print(f"  valid pixels (depth > 200 mm): {valid_pct:.2f}%")
    print_stats_row("cached_depth (mass head ch256)", cached_depth)

    # --- 4) Save grayscale visualizations ---
    if out_dir is not None:
        prefix = os.path.join(out_dir, f"sample{idx:02d}")
        ch_names = ("B", "G", "R", "ch3depth")
        for c in range(4):
            save_grayscale(image_chw[c], f"{prefix}_raw_{ch_names[c]}.png")
            save_grayscale(img_norm[c],   f"{prefix}_norm_{ch_names[c]}.png")
        save_grayscale(depth_hw,     f"{prefix}_depth_tensor.png")
        save_grayscale(cached_depth, f"{prefix}_late_fusion_ch256.png")
        # also save valid_mask
        save_grayscale(valid_mask, f"{prefix}_valid_mask.png", vmin=0, vmax=1)
        print(f"\n  Saved 11 PNGs to {out_dir}/ with prefix sample{idx:02d}_")


# ---------- entry point ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", default="./data/",
                    help="Root data folder (containing images/<split>/ and depth files).")
    ap.add_argument("--split", default="train",
                    choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=3,
                    help="Number of sample images to inspect.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for sample selection (reproducible).")
    ap.add_argument("--out_dir", default="./inspect_input_viz",
                    help="Folder to save grayscale PNG visualizations "
                         "(set to '' to disable visualizations).")
    ap.add_argument("--simulate_ch3_zero", action="store_true",
                    help="Simulate the proposed mapper change: ch3 = 915.043 mm, "
                         "depth tensor = 0. Use to preview model input "
                         "BEFORE modifying train.py.")
    args = ap.parse_args()

    # Register dataset
    dset_name = f"TomatoDataset_{args.split}"
    DatasetCatalog.register(
        dset_name,
        lambda: utils_detection.get_tomato_dataset_dicts(args.dataset_path, args.split),
    )
    MetadataCatalog.get(dset_name).set(thing_classes=["tomato_bunch"])

    dicts = DatasetCatalog.get(dset_name)
    print(f"Loaded {len(dicts)} samples from split='{args.split}'.")

    rng = np.random.RandomState(args.seed)
    n = min(args.n, len(dicts))
    chosen = rng.choice(len(dicts), size=n, replace=False)

    out_dir = args.out_dir.strip()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for f in os.listdir(out_dir):
            if f.endswith(".png"):
                os.remove(os.path.join(out_dir, f))
    else:
        out_dir = None

    print(f"Inspecting {n} sample(s)  "
          f"(simulate_ch3_zero={args.simulate_ch3_zero}, out_dir={out_dir})")

    for i, di in enumerate(chosen):
        inspect_one(dicts[int(di)], args.simulate_ch3_zero, out_dir, i)

    # Final checklist
    print()
    print("=" * 100)
    print("CHECKLIST — what to verify in the numbers above")
    print("=" * 100)
    print("""
  CURRENT depth-only mapper (no --simulate_ch3_zero flag):
    - ch0/ch1/ch2 RAW:           ~0..255, std > 0 (BGR-replicated depth, spatially varying)
    - ch0/ch1/ch2 NORMALIZED:    roughly (-1, +1.5), std > 0 (signal alive)
    - ch3 RAW:                   roughly 200..2000 mm (with invalid filled to 915.043)
    - ch3 NORMALIZED:            roughly (-1.7, +2.6), mean near 0, std > 0 (signal alive)
    - depth tensor RAW:          raw mm with zeros for invalid pixels
    - late-fusion ch256:         non-zero where depth was valid, zero where invalid

  PROPOSED change (--simulate_ch3_zero):
    - ch0/ch1/ch2 unchanged (depth signal preserved on BGR path)
    - ch3 RAW:                   exactly 915.043 everywhere
    - ch3 NORMALIZED:            exactly 0.0 everywhere, std = 0
    - depth tensor RAW:          exactly 0 everywhere
    - late-fusion ch256:         exactly 0 everywhere (both pathways dead)

  PNG visualizations saved to the out_dir:
    sample00_raw_*.png        — raw values per channel, per-image-normalized to 0-255 for display
    sample00_norm_*.png       — normalized values per channel, per-image-normalized to 0-255 for display
    sample00_depth_tensor.png — raw 'depth' tensor (mm)
    sample00_late_fusion_ch256.png — what the mass head sees as ch256
    sample00_valid_mask.png   — late-fusion valid mask (white = valid)

  ch3 and late_fusion images should look like flat gray (constant) under --simulate_ch3_zero.
""")


if __name__ == "__main__":
    main()
