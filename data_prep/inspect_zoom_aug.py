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

Uses the training code itself (train.py: load_rgbd_image, depth_consistent_zoom,
color_jitter_bgr), so the preview is exactly what the model sees.

  ./run.sh data_prep/inspect_zoom_aug.py --dataset_path ./data_s0713/ --split train --n 8 --out_dir ./aug_preview
"""
import os, argparse, random
import cv2
import numpy as np

import utils_detection
from train import load_rgbd_image, depth_consistent_zoom, color_jitter_bgr


def annotations_to_mask(annotations, H, W):
    """Rasterize the (polygon) segmentations of all instances into one bool mask."""
    mask = np.zeros((H, W), np.uint8)
    for a in annotations:
        seg = a.get("segmentation", [])
        if isinstance(seg, list):
            for poly in seg:
                pts = np.array(poly, np.float32).reshape(-1, 2).round().astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


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

    dataset_dicts = utils_detection.get_tomato_dataset_dicts(args.dataset_path, args.split)
    sample = random.sample(dataset_dicts, min(args.n, len(dataset_dicts)))
    print(f"{'file':42} {'s':>4} {'mass_g':>6} {'d_in_before':>11} {'d_in_after':>10} "
          f"{'ratio(≈1/s)':>11} {'area_ratio(≈s^2)':>16}")

    for d in sample:
        _, image, invalid = load_rgbd_image(d)
        H, W = invalid.shape
        bgr, depth = image[:, :, :3], image[:, :, 3]
        mask = annotations_to_mask(d["annotations"], H, W)
        mass_g = d["cluster_mass"] * 1000
        fn = os.path.basename(d["file_name"])
        valid0 = ~invalid
        d_in_before = float(depth[mask & valid0].mean()) if (mask & valid0).any() else float("nan")
        area_before = int(mask.sum())
        # shared depth colormap range from the original valid depth (2..98 pct)
        vlo, vhi = np.percentile(depth[valid0], [2, 98]) if valid0.any() else (200, 2000)

        rows = []
        for s in scales:
            image_a, invalid_a, annos_a = depth_consistent_zoom(image, invalid, d["annotations"], s)
            rgb_a, depth_a, valid_a = image_a[:, :, :3], image_a[:, :, 3], ~invalid_a
            mask_a = annotations_to_mask(annos_a, H, W)
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
                jit = image.copy()
                color_jitter_bgr(jit, args.color_aug)   # in place, BGR channels only
                panels.append(label(draw_mask(jit[:, :, :3], mask), f"{args.color_aug} #{k+1}"))
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
