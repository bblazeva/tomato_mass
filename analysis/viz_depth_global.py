"""
Preview depth maps with the global normalization that train.py will use.

For each session (2025-07-13 .jpg, 2025-03-28 .png, 2023-12-01 .png), pick a
handful of sample images and render the depth map two ways:
  - Per-image min/max normalization (the OLD behavior in train.py)
  - Global [DEPTH_GLOBAL_MIN_MM, DEPTH_GLOBAL_MAX_MM] normalization (the NEW behavior)

Saves a side-by-side comparison PNG per sample under analysis/depth_viz/.
Run this and inspect the images before approving the train.py change.
"""
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# These MUST match the constants in train.py
DEPTH_GLOBAL_MIN_MM = 200.0
DEPTH_GLOBAL_MAX_MM = 2000.0

DATA_DIR     = Path("/home/blagicab/master_tomato/tomato_multitask_RGB-D/data")
IMAGES_DIR   = DATA_DIR / "images"
DEPTH_DIR    = DATA_DIR / "iter_4_depth_matched"
FEATURES_CSV = Path("/home/blagicab/master_tomato/analysis/features.csv")
OUT_DIR      = Path("/home/blagicab/master_tomato/analysis/depth_viz")
N_PER_SESSION = 4   # how many examples per capture session


def parse_session(file_name):
    fn = file_name.lower()
    ext  = "jpg" if fn.endswith(".jpg") else ("png" if fn.endswith(".png") else "?")
    m    = re.search(r"(20\d{2}-\d{2}-\d{2})", file_name)
    date = m.group(1) if m else "?"
    return ext, date


def load_depth_mm(stem):
    """Load depth file by filename stem. Returns (H,W) float array in mm."""
    path = DEPTH_DIR / f"{stem}.npy"
    return np.load(path).astype(np.float32)


def normalize_per_image(depth_mm):
    """OLD behavior: per-image min/max → 0-255."""
    valid = depth_mm > 0
    if not valid.any():
        return np.zeros_like(depth_mm, dtype=np.uint8)
    d_min = depth_mm[valid].min()
    d_max = depth_mm[valid].max()
    out = np.clip((depth_mm - d_min) / (d_max - d_min + 1e-6) * 255.0, 0, 255)
    # invalid → fill with the mid (so they don't dominate the visual)
    out[~valid] = 0
    return out.astype(np.uint8)


def normalize_global(depth_mm):
    """NEW behavior: global [200, 2000] mm → 0-255 (clipped)."""
    out = np.clip(
        (depth_mm - DEPTH_GLOBAL_MIN_MM)
        / (DEPTH_GLOBAL_MAX_MM - DEPTH_GLOBAL_MIN_MM) * 255.0,
        0, 255,
    )
    return out.astype(np.uint8)


def pick_samples_per_session(df, n_per_session):
    """For each (ext, date) session present in TRAIN, pick n samples with varied masses."""
    train = df[df.split == "train"].copy()
    train["session"] = train.file_name.apply(parse_session)
    sessions = sorted(train.session.unique(), key=lambda s: (s[0], s[1]))
    chosen = []
    for s in sessions:
        sub = train[train.session == s].sort_values("mass_g")
        if len(sub) <= n_per_session:
            chosen.extend(sub.to_dict("records"))
        else:
            # evenly spaced by mass quantile
            idx = np.linspace(0, len(sub) - 1, n_per_session).round().astype(int)
            chosen.extend(sub.iloc[idx].to_dict("records"))
    return chosen


def make_preview(row, out_path):
    file_name = row["file_name"]
    mass_g    = row["mass_g"]
    ext, date = parse_session(file_name)

    stem      = Path(file_name).stem
    img_path  = IMAGES_DIR / "train" / file_name

    try:
        rgb_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    except Exception:
        rgb_bgr = None

    try:
        depth_mm = load_depth_mm(stem)
    except Exception as e:
        print(f"  skip {file_name}: {e}")
        return

    # If sizes differ (depth at different resolution), resize depth to match RGB.
    if rgb_bgr is not None and depth_mm.shape != rgb_bgr.shape[:2]:
        depth_mm = cv2.resize(depth_mm, (rgb_bgr.shape[1], rgb_bgr.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

    per_img  = normalize_per_image(depth_mm)
    global_  = normalize_global(depth_mm)

    valid = depth_mm > 0
    stats_lines = [
        f"{file_name}  GT={mass_g} g   session={ext} {date}",
        (f"valid depth range: [{depth_mm[valid].min():.0f}, {depth_mm[valid].max():.0f}] mm   "
         f"mean={depth_mm[valid].mean():.0f}   std={depth_mm[valid].std():.0f}   "
         f"valid_pct={100*valid.sum()/valid.size:.1f}%"),
        f"global norm range: [{DEPTH_GLOBAL_MIN_MM:.0f}, {DEPTH_GLOBAL_MAX_MM:.0f}] mm  (fixed across all images)",
    ]

    n_panels = 3 if rgb_bgr is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    col = 0
    if rgb_bgr is not None:
        rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        axes[col].imshow(rgb_rgb)
        axes[col].set_title("RGB (reference only)")
        axes[col].axis("off")
        col += 1
    axes[col].imshow(per_img, cmap="gray", vmin=0, vmax=255)
    axes[col].set_title("Per-image normalize (OLD)")
    axes[col].axis("off")
    col += 1
    axes[col].imshow(global_, cmap="gray", vmin=0, vmax=255)
    axes[col].set_title(f"Global normalize (NEW)\n[{int(DEPTH_GLOBAL_MIN_MM)}-{int(DEPTH_GLOBAL_MAX_MM)}] mm")
    axes[col].axis("off")

    fig.suptitle("\n".join(stats_lines), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # clean previous previews so the user sees only this run's output
    for old in OUT_DIR.glob("*.png"):
        old.unlink()

    df = pd.read_csv(FEATURES_CSV)

    # global stats across the dataset
    valid_min = df.min_depth_m.min() * 1000
    valid_max = df.max_depth_m.max() * 1000
    print("=" * 70)
    print("GLOBAL DEPTH NORMALIZATION PREVIEW")
    print("=" * 70)
    print(f"Dataset depth extremes:  [{valid_min:.0f}, {valid_max:.0f}] mm")
    print(f"Global norm range used:  [{DEPTH_GLOBAL_MIN_MM:.0f}, {DEPTH_GLOBAL_MAX_MM:.0f}] mm")
    print(f"Margin below dataset min: {valid_min - DEPTH_GLOBAL_MIN_MM:.0f} mm")
    print(f"Margin above dataset max: {DEPTH_GLOBAL_MAX_MM - valid_max:.0f} mm")
    print("(any pixel outside the global range will be clipped to 0 or 255)")
    print()

    chosen = pick_samples_per_session(df, N_PER_SESSION)
    print(f"Rendering {len(chosen)} sample previews to {OUT_DIR}/")

    for i, row in enumerate(chosen):
        ext, date = parse_session(row["file_name"])
        stem = Path(row["file_name"]).stem
        out_name = f"{date}_{ext}_{row['mass_g']:04d}g_{stem}.png"
        out_path = OUT_DIR / out_name
        print(f"  [{i+1}/{len(chosen)}] {out_name}")
        make_preview(row, out_path)

    print()
    print(f"Done. Open {OUT_DIR}/ and review the side-by-side comparisons.")
    print("If the GLOBAL panels look reasonable across sessions, train.py is good to go.")
    print("If a session's depth looks too dark/bright, adjust DEPTH_GLOBAL_MIN_MM / MAX_MM")
    print("in BOTH this script and train.py and re-run this preview.")


if __name__ == "__main__":
    main()
