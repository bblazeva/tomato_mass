"""
extract_features.py
-------------------
Per-image RGBD feature extraction for the tomato-cluster mass dataset.

For every annotated image in data/json/{train,val,test}.json:
  - rasterizes the polygon segmentation into a binary mask
  - loads the matching depth map from data/depth/<stem>.npy (uint16 mm -> meters)
  - computes geometric + depth-based features inside the mask
  - derives a few volume proxies
  - reads the GT cluster mass from the filename prefix (grams)

Outputs (under analysis/):
  features.csv               one row per annotation, all features
  correlations.csv           Pearson / Spearman / log-log Pearson, per feature × split
  feature_correlations.png   bar chart of correlations with mass (overall)
  scatter_grid.png           per-feature scatter vs mass (linear)
  scatter_grid_loglog.png    same, log-log scale

Run with the Python env that has numpy + opencv + matplotlib.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "analysis"
SPLITS = ("train", "val", "test")
FNAME_RE = re.compile(r"^(\d+)_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.(png|jpg)$")

# Order in which features show up in tables and plots. Proxies last.
FEATURES_FOR_REPORT = [
    "area_px",
    "area_m2",
    "mean_depth_m",
    "median_depth_m",
    "std_depth_m",
    "min_depth_m",
    "max_depth_m",
    "depth_extent_m",
    "bbox_w_px",
    "bbox_h_px",
    "bbox_aspect",
    "solidity",
    "perimeter_px",
    "n_polygons",
    "proxy_area",
    "proxy_vol_extent",
    "proxy_vol_std",
    "proxy_pseudo_vol",
]


# ---------- IO helpers ----------

def load_intrinsics(path: Path) -> dict:
    with path.open() as f:
        intr = json.load(f)
    print(
        f"Intrinsics: fx={intr['fx']:.2f}, fy={intr['fy']:.2f}, "
        f"cx={intr['cx']:.2f}, cy={intr['cy']:.2f} "
        f"({intr['width']}x{intr['height']})"
    )
    return intr


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def parse_mass_from_filename(file_name: str) -> int | None:
    m = FNAME_RE.match(file_name)
    return int(m.group(1)) if m else None


# ---------- feature extraction ----------

def polygons_to_mask(segmentation: list[list[float]], h: int, w: int) -> np.ndarray:
    polys = []
    for poly in segmentation:
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        polys.append(pts.round().astype(np.int32))
    mask = np.zeros((h, w), dtype=np.uint8)
    if polys:
        cv2.fillPoly(mask, polys, 1)
    return mask


def mask_shape_features(mask: np.ndarray) -> dict:
    """Pixel-space shape features: solidity, perimeter."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {"solidity": float("nan"), "perimeter_px": 0.0}
    pts = np.vstack(contours)
    hull = cv2.convexHull(pts)
    hull_area = float(cv2.contourArea(hull))
    perim = float(sum(cv2.arcLength(c, True) for c in contours))
    area_px = float(mask.sum())
    solidity = area_px / hull_area if hull_area > 0 else float("nan")
    return {"solidity": solidity, "perimeter_px": perim}


def features_for_image(
    file_name: str,
    anns: list[dict],
    depth_path: Path,
    h: int,
    w: int,
    fx: float,
    fy: float,
) -> dict | None:
    """Returns a feature row, or None if the image should be skipped."""
    depth_mm = np.load(depth_path)
    depth_m = depth_mm.astype(np.float32) / 1000.0  # uint16 mm -> meters
    if depth_m.shape != (h, w):
        depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)

    mask = np.zeros((h, w), dtype=np.uint8)
    n_polys = 0
    for ann in anns:
        mask |= polygons_to_mask(ann["segmentation"], h, w)
        n_polys += len(ann["segmentation"])

    valid = mask.astype(bool) & np.isfinite(depth_m) & (depth_m > 0)
    if not valid.any():
        return None

    Z = depth_m[valid]
    area_px = int(mask.sum())
    # metric area: each pixel at depth Z covers (Z/fx)*(Z/fy) m^2.
    # Sum over masked pixels -> approximately distance-invariant frontal area.
    area_m2 = float((Z * Z).sum() / (fx * fy))
    mean_depth = float(Z.mean())
    median_depth = float(np.median(Z))
    std_depth = float(Z.std())
    min_depth = float(Z.min())
    max_depth = float(Z.max())
    extent = max_depth - min_depth

    ys, xs = np.where(mask > 0)
    bbox_w = int(xs.max() - xs.min() + 1)
    bbox_h = int(ys.max() - ys.min() + 1)
    bbox_aspect = bbox_w / bbox_h if bbox_h > 0 else float("nan")

    shape = mask_shape_features(mask)

    return {
        "file_name": file_name,
        "area_px": area_px,
        "area_m2": area_m2,
        "mean_depth_m": mean_depth,
        "median_depth_m": median_depth,
        "std_depth_m": std_depth,
        "min_depth_m": min_depth,
        "max_depth_m": max_depth,
        "depth_extent_m": extent,
        "bbox_w_px": bbox_w,
        "bbox_h_px": bbox_h,
        "bbox_aspect": bbox_aspect,
        "solidity": shape["solidity"],
        "perimeter_px": shape["perimeter_px"],
        "n_polygons": n_polys,
        # Volume proxies — same metric area scaled by different "thickness" guesses.
        # proxy_pseudo_vol is the one from the old script (uses distance-from-camera);
        # included only so you can see why it's worse than the others.
        "proxy_area": area_m2,
        "proxy_vol_extent": area_m2 * extent,
        "proxy_vol_std": area_m2 * std_depth,
        "proxy_pseudo_vol": area_m2 * mean_depth,
    }


def extract_for_split(split: str, intr: dict, depth_dir: Path) -> list[dict]:
    fx, fy = intr["fx"], intr["fy"]
    json_path = DATA / "json" / f"{split}.json"
    with json_path.open() as f:
        coco = json.load(f)
    images_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image[a["image_id"]].append(a)

    rows: list[dict] = []
    skipped = 0
    for image_id, anns in anns_by_image.items():
        meta = images_by_id.get(image_id)
        if meta is None:
            skipped += 1
            continue
        file_name = meta["file_name"]
        mass_g = parse_mass_from_filename(file_name)
        if mass_g is None:
            print(f"  [{split}] cannot parse mass from {file_name}")
            skipped += 1
            continue
        depth_path = depth_dir / (Path(file_name).stem + ".npy")
        if not depth_path.exists():
            print(f"  [{split}] missing depth: {depth_path.name}")
            skipped += 1
            continue

        feats = features_for_image(
            file_name=file_name,
            anns=anns,
            depth_path=depth_path,
            h=meta["height"],
            w=meta["width"],
            fx=fx,
            fy=fy,
        )
        if feats is None:
            print(f"  [{split}] no valid depth inside mask: {file_name}")
            skipped += 1
            continue
        feats = {"split": split, "mass_g": mass_g, **feats}
        rows.append(feats)

    print(f"[{split}] processed {len(rows)}, skipped {skipped}")
    return rows


# ---------- correlations ----------

def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman = Pearson of ranks. Avoids the scipy dependency."""
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return pearson(rx, ry)


def compute_correlations(rows: list[dict]) -> list[dict]:
    by_split: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)
    by_split["overall"] = rows

    results: list[dict] = []
    for split, split_rows in by_split.items():
        mass = np.array([r["mass_g"] for r in split_rows], dtype=float)
        for feat in FEATURES_FOR_REPORT:
            vals = np.array([r[feat] for r in split_rows], dtype=float)
            ok = np.isfinite(vals) & np.isfinite(mass)
            v, m = vals[ok], mass[ok]
            p = pearson(v, m)
            s = spearman(v, m)
            pos = (v > 0) & (m > 0)
            p_log = (
                pearson(np.log(v[pos]), np.log(m[pos]))
                if pos.sum() > 1
                else float("nan")
            )
            results.append(
                {
                    "feature": feat,
                    "split": split,
                    "n": int(ok.sum()),
                    "pearson": p,
                    "spearman": s,
                    "pearson_loglog": p_log,
                }
            )
    return results


def print_correlations(corr_rows: list[dict]) -> None:
    overall = [r for r in corr_rows if r["split"] == "overall"]
    overall.sort(
        key=lambda r: abs(r["pearson"]) if np.isfinite(r["pearson"]) else 0,
        reverse=True,
    )
    print()
    print(
        f"{'feature':<22} {'n':>5}  {'Pearson':>8}  {'Spearman':>9}  {'Pearson(log)':>13}"
    )
    print("-" * 64)
    for r in overall:
        print(
            f"{r['feature']:<22} {r['n']:>5}  "
            f"{r['pearson']:>8.3f}  {r['spearman']:>9.3f}  {r['pearson_loglog']:>13.3f}"
        )


# ---------- plots ----------

def plot_correlation_bar(corr_rows: list[dict], path: Path) -> None:
    overall = [r for r in corr_rows if r["split"] == "overall"]

    def best(r: dict) -> float:
        a = abs(r["pearson"]) if np.isfinite(r["pearson"]) else 0
        b = abs(r["pearson_loglog"]) if np.isfinite(r["pearson_loglog"]) else 0
        return max(a, b)

    overall = sorted(overall, key=best, reverse=True)
    feats = [r["feature"] for r in overall]
    pear = [r["pearson"] for r in overall]
    spr = [r["spearman"] for r in overall]

    x = np.arange(len(feats))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 0.18, pear, width=0.36, label="Pearson")
    ax.bar(x + 0.18, spr, width=0.36, label="Spearman")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(feats, rotation=45, ha="right")
    ax.set_ylabel("correlation with mass_g (overall)")
    ax.set_title("Feature ↔ mass correlation")
    ax.set_ylim(-1, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter_grid(rows: list[dict], path: Path, log: bool = False) -> None:
    feats = FEATURES_FOR_REPORT
    n = len(feats)
    cols = 4
    n_rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 4, n_rows * 3))
    axes = axes.flatten()
    split_colors = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}

    for ax, feat in zip(axes, feats):
        for split, color in split_colors.items():
            split_rows = [r for r in rows if r["split"] == split]
            x = np.array([r[feat] for r in split_rows], dtype=float)
            y = np.array([r["mass_g"] for r in split_rows], dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if log:
                ok &= (x > 0) & (y > 0)
                xp, yp = np.log(x[ok]), np.log(y[ok])
            else:
                xp, yp = x[ok], y[ok]
            ax.scatter(xp, yp, s=6, alpha=0.5, color=color, label=split)
        ax.set_title(feat, fontsize=9)
        ax.tick_params(labelsize=7)
        if log:
            ax.set_xlabel(f"log({feat})", fontsize=7)
            ax.set_ylabel("log(mass_g)", fontsize=7)
        else:
            ax.set_xlabel(feat, fontsize=7)
            ax.set_ylabel("mass_g", fontsize=7)
        ax.grid(alpha=0.3)

    for ax in axes[n:]:
        ax.set_visible(False)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, label=s)
        for s, c in split_colors.items()
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------- main ----------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    intr = load_intrinsics(DATA / "camera_intrinsics.json")
    depth_dir = DATA / "depth"

    all_rows: list[dict] = []
    for split in SPLITS:
        all_rows.extend(extract_for_split(split, intr, depth_dir))

    if not all_rows:
        raise SystemExit("no rows extracted — check data/ paths")

    write_csv(all_rows, OUT / "features.csv")
    print(f"wrote {OUT / 'features.csv'}  ({len(all_rows)} rows)")

    corr = compute_correlations(all_rows)
    write_csv(corr, OUT / "correlations.csv")
    print(f"wrote {OUT / 'correlations.csv'}")
    print_correlations(corr)

    plot_correlation_bar(corr, OUT / "feature_correlations.png")
    plot_scatter_grid(all_rows, OUT / "scatter_grid.png", log=False)
    plot_scatter_grid(all_rows, OUT / "scatter_grid_loglog.png", log=True)
    print(f"wrote {OUT / 'feature_correlations.png'}")
    print(f"wrote {OUT / 'scatter_grid.png'}")
    print(f"wrote {OUT / 'scatter_grid_loglog.png'}")


if __name__ == "__main__":
    main()
