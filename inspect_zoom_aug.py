"""
Visual sanity-check for the DEPTH-CONSISTENT ZOOM augmentation, BEFORE training.

For a few random training samples it renders, at several zoom factors s:
  [ RGB + GT mask ]   [ color-mapped depth ]
stacked one row per s, and prints the numeric checks that prove the aug is
physically correct:
  - in-mask depth ratio (after/before) should be ~ 1/s   (depth scaled by 1/s)
  - mask area ratio (after/before)     should be ~ s^2   (apparent size scaled by s)
If those hold and the mask still lines up on BOTH the RGB and the depth panel,
the augmentation is sound (apparent_size * depth stays invariant = mass unchanged).

Reads the COCO json directly (no detectron2 import). Run with the project env:
  LD_PRELOAD=$HOME/miniconda3/envs/tomato_mass/lib/libstdc++.so.6 \
    ~/miniconda3/envs/tomato_mass/bin/python3 inspect_zoom_aug.py \
      --dataset_path ./data_s0713/ --split train --n 8 --out_dir ./aug_preview
"""
import os, json, argparse, random
import cv2
import numpy as np

DEPTH_MEAN_MM = 915.043
BGR_MEAN = (103.530, 116.280, 123.675)

# Must match train_new_clean.py's _COLOR_PRESETS exactly so the preview is faithful.
_COLOR_PRESETS = {
    "light":  dict(wb=0.10, hue=5,  sat=(0.85, 1.15), bri=(0.8, 1.2), con=(0.8, 1.2)),
    "strong": dict(wb=0.20, hue=10, sat=(0.6, 1.4),   bri=(0.6, 1.4), con=(0.6, 1.4)),
}


def color_jitter_bgr(bgr_float, strength):
    """BGR-only photometric jitter (returns a jittered copy). Mirrors training."""
    p = _COLOR_PRESETS.get(strength)
    if p is None:
        return bgr_float.copy()
    bgr = bgr_float.astype(np.float32).copy()
    for c in range(3):                               # white balance
        bgr[:, :, c] *= random.uniform(1 - p["wb"], 1 + p["wb"])
    np.clip(bgr, 0, 255, out=bgr)
    hsv = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-p["hue"], p["hue"])) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(*p["sat"]), 0, 255)
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    bgr *= random.uniform(*p["bri"])                 # brightness
    m = bgr.mean()
    bgr = m + (bgr - m) * random.uniform(*p["con"])  # contrast
    return np.clip(bgr, 0, 255, out=bgr)


# --------------------------------------------------------------------------------------
# The augmentation under test (mirror this exactly in the train mapper later)
# --------------------------------------------------------------------------------------
def depth_consistent_zoom(image_bgr, depth_mm, mask, s):
    """
    Simulate the camera moving closer/farther by factor s, keeping mass invariant.
      - resize pixels by s        (apparent size  x s)
      - scale valid depth by 1/s  (object distance x 1/s)
      - recanvas to original HxW, centered on the cluster
      - holes (sensor 0) and padding -> DEPTH_MEAN_MM (neutral after norm)
    Returns: rgb_out(H,W,3 float32), depth_filled(H,W float32), mask_out(bool),
             valid_out(bool)   [valid_out = real depth, not hole/pad]
    """
    H, W = depth_mm.shape
    invalid = depth_mm == 0

    newW, newH = max(1, round(W * s)), max(1, round(H * s))
    rgb_i  = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
    rgb_r   = cv2.resize(image_bgr, (newW, newH), interpolation=rgb_i).astype(np.float32)
    depth_r = cv2.resize(depth_mm,  (newW, newH), interpolation=cv2.INTER_NEAREST)
    inval_r = cv2.resize(invalid.astype(np.uint8), (newW, newH), interpolation=cv2.INTER_NEAREST).astype(bool)
    mask_r  = cv2.resize(mask.astype(np.uint8),    (newW, newH), interpolation=cv2.INTER_NEAREST).astype(bool)

    # scale ONLY valid depths -> physical distance change
    valid_r = ~inval_r
    depth_r = depth_r.astype(np.float32)
    depth_r[valid_r] = depth_r[valid_r] / s

    # recanvas to (H, W) centered on the cluster centroid (crop if s>1, pad if s<1)
    if mask_r.any():
        ys, xs = np.where(mask_r)
        cy, cx = int(ys.mean()), int(xs.mean())
    else:
        cy, cx = newH // 2, newW // 2
    top, left = cy - H // 2, cx - W // 2  # window top-left in resized frame

    rgb_out   = np.zeros((H, W, 3), np.float32); rgb_out[:] = BGR_MEAN
    depth_out = np.zeros((H, W), np.float32)     # 0 = hole/pad for now
    mask_out  = np.zeros((H, W), bool)
    valid_out = np.zeros((H, W), bool)

    # intersection of window [top:top+H, left:left+W] with resized image bounds
    sy0, sx0 = max(0, top), max(0, left)
    sy1, sx1 = min(newH, top + H), min(newW, left + W)
    dy0, dx0 = sy0 - top, sx0 - left
    dy1, dx1 = dy0 + (sy1 - sy0), dx0 + (sx1 - sx0)
    if sy1 > sy0 and sx1 > sx0:
        rgb_out[dy0:dy1, dx0:dx1]   = rgb_r[sy0:sy1, sx0:sx1]
        depth_out[dy0:dy1, dx0:dx1] = depth_r[sy0:sy1, sx0:sx1]
        mask_out[dy0:dy1, dx0:dx1]  = mask_r[sy0:sy1, sx0:sx1]
        valid_out[dy0:dy1, dx0:dx1] = valid_r[sy0:sy1, sx0:sx1]

    depth_filled = depth_out.copy()
    depth_filled[depth_out == 0] = DEPTH_MEAN_MM   # holes + padding -> neutral
    return rgb_out, depth_filled, mask_out, valid_out


# --------------------------------------------------------------------------------------
# rendering helpers
# --------------------------------------------------------------------------------------
def colorize_depth(depth, valid, vmin, vmax):
    d = np.clip((depth - vmin) / (vmax - vmin + 1e-6) * 255, 0, 255).astype(np.uint8)
    col = cv2.applyColorMap(d, cv2.COLORMAP_JET)
    col[~valid] = (0, 0, 0)  # holes/pad shown black
    return col

def draw_mask(bgr, mask, color=(0, 255, 0)):
    out = bgr.astype(np.uint8).copy()
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cv2.drawContours(out, cnts, -1, color, 2)
    return out

def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------------------
def load_sample(coco_dir, img_dir, depth_dir, img_entry, ann_by_img):
    fn = img_entry["file_name"]
    bgr = cv2.imread(os.path.join(img_dir, fn))
    depth = np.load(os.path.join(depth_dir, os.path.splitext(fn)[0] + ".npy")).astype(np.float32)
    H, W = bgr.shape[:2]
    if depth.shape != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
    mask = np.zeros((H, W), np.uint8)
    for a in ann_by_img.get(img_entry["id"], []):
        seg = a.get("segmentation", [])
        if isinstance(seg, list):
            for poly in seg:
                pts = np.array(poly, np.float32).reshape(-1, 2).round().astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
    mass_g = float(os.path.basename(fn).split("_")[0])
    return bgr, depth, mask.astype(bool), mass_g, fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", default="./data_s0713/")
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--scales", default="0.7,1.0,1.3", help="comma list of zoom factors to preview")
    p.add_argument("--out_dir", default="./aug_preview")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--preview_w", type=int, default=440, help="panel width in the saved image")
    p.add_argument("--color_aug", choices=["off", "light", "strong"], default="off",
                   help="Also preview photometric (BGR) jitter at this strength.")
    p.add_argument("--color_variants", type=int, default=4,
                   help="How many random color-jittered variants to show per sample.")
    args = p.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    scales = [float(x) for x in args.scales.split(",")]

    root = args.dataset_path.rstrip("/") + "/"
    coco = json.load(open(f"{root}gt_json/{args.split}/{args.split}.json"))
    img_dir   = f"{root}images/{args.split}"
    depth_dir = f"{root}iter_4_depth_matched"
    ann_by_img = {}
    for a in coco["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)

    sample = random.sample(coco["images"], min(args.n, len(coco["images"])))
    print(f"{'file':42} {'s':>4} {'mass_g':>6} {'d_in_before':>11} {'d_in_after':>10} "
          f"{'ratio(≈1/s)':>11} {'area_ratio(≈s^2)':>16}")

    for img_entry in sample:
        bgr, depth, mask, mass_g, fn = load_sample(root, img_dir, depth_dir, img_entry, ann_by_img)
        valid0 = depth > 0
        d_in_before = float(depth[mask & valid0].mean()) if (mask & valid0).any() else float("nan")
        area_before = int(mask.sum())
        # shared depth colormap range from the original valid depth (2..98 pct)
        vlo, vhi = np.percentile(depth[valid0], [2, 98]) if valid0.any() else (200, 2000)

        rows = []
        for s in scales:
            rgb_a, depth_a, mask_a, valid_a = depth_consistent_zoom(bgr, depth, mask, s)
            d_in_after = float(depth_a[mask_a & valid_a].mean()) if (mask_a & valid_a).any() else float("nan")
            area_after = int(mask_a.sum())
            ratio = d_in_after / d_in_before if d_in_before else float("nan")
            arr   = area_after / area_before if area_before else float("nan")
            print(f"{fn:42} {s:>4.2f} {mass_g:>6.0f} {d_in_before:>11.1f} {d_in_after:>10.1f} "
                  f"{ratio:>11.2f} {arr:>16.2f}")

            rgb_panel   = label(draw_mask(rgb_a, mask_a),
                                f"s={s:.2f}  RGB+mask  (area x{arr:.2f}, want {s*s:.2f})")
            depth_panel = label(colorize_depth(depth_a, valid_a, vlo, vhi),
                                f"depth  in-mask {d_in_after:.0f}mm (x{ratio:.2f}, want {1/s:.2f})")
            row = np.hstack([rgb_panel, depth_panel])
            rows.append(row)

        composite = np.vstack(rows)
        scale = args.preview_w * 2 / composite.shape[1]
        composite = cv2.resize(composite, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Color-jitter preview strip (BGR only; depth unaffected) at full resolution, s=1
        if args.color_aug != "off":
            panels = [label(draw_mask(bgr, mask), "original")]
            for k in range(args.color_variants):
                jit = color_jitter_bgr(bgr.astype(np.float32), args.color_aug)
                panels.append(label(draw_mask(jit, mask), f"{args.color_aug} #{k+1}"))
            strip = np.hstack(panels)
            w = composite.shape[1]
            strip = cv2.resize(strip, (w, int(strip.shape[0] * w / strip.shape[1])),
                               interpolation=cv2.INTER_AREA)
            composite = np.vstack([composite, strip])

        out = os.path.join(args.out_dir, f"{os.path.splitext(fn)[0]}_zoom.jpg")
        cv2.imwrite(out, composite)

    print(f"\nSaved {len(sample)} previews to {args.out_dir}/")
    print("CHECK:  depth ratio ≈ 1/s,  area ratio ≈ s^2,  mask aligned on BOTH panels,  "
          "holes/padding black in depth & gray in RGB.")
    if args.color_aug != "off":
        print(f"COLOR:  bottom strip = '{args.color_aug}' jitter — check hue stays plausible "
              "(tomatoes still red-ish) and lighting/WB variation looks realistic.")


if __name__ == "__main__":
    main()
