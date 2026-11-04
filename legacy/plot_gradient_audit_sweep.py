"""
Plot the gradient-audit sweep produced by gradient_audit_sweep2.py.

One FIGURE PER INPUT MODE (real_rgbd, rgb_only, depth_only). Each figure has three
stacked panels sharing the iteration x-axis:

  TOP    — gradient-flow RATIOS (mass relative to detection):
             mass/det @ backbone_res, mass/det @ fpn, and (except depth_only)
             MASS stem depth/rgb. A dashed y=1 line marks the crossover where the
             mass objective pushes a part as hard as detection does.

  MIDDLE — ABSOLUTE shared-trunk gradients (log y), DET vs MASS, so a change in a
             top-panel ratio can be read correctly: did mass climb, or did
             detection decay? Lines:
             det @backbone, mass @backbone, det @fpn, mass @fpn.

  BOTTOM — MASS stem ABSOLUTE per-channel grad (log y): rgb_abs and depth_abs —
             who drives the input stem. For depth_only, rgb_abs is 0 by
             construction, so only depth_abs is shown and the depth/rgb ratio is
             dropped from the top panel.

No pandas dependency — stdlib csv + matplotlib only, so it runs without the
detectron2 / CUDA environment.

Run:
  ~/miniconda3/envs/tomato_mass/bin/python3 plot_gradient_audit_sweep.py \
      --csv output/s0713_zoom_drop065/gradient_audit_sweep.csv \
      --outdir output/s0713_zoom_drop065/audit_plots
"""
import os, csv, math, argparse
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODES = ["real_rgbd", "rgb_only", "depth_only"]

# Column that holds the numeric iteration. gradient_audit_sweep2.py writes
# 'checkpoint_iteration_numeric'; we fall back to the older 'checkpoint_iteration'.
ITER_COLS = ["checkpoint_iteration_numeric", "checkpoint_iteration"]


def _fnum(s):
    try:
        v = float(s)
        return v if not math.isnan(v) else None
    except (ValueError, TypeError):
        return None


def load(csv_path):
    """rows[mode] -> dict of column -> list, sorted by iteration."""
    by_mode = defaultdict(list)
    iter_col = None
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        iter_col = next((c for c in ITER_COLS if c in reader.fieldnames), None)
        if iter_col is None:
            raise SystemExit(
                f"No iteration column found in {csv_path}. Expected one of {ITER_COLS}.")
        for r in reader:
            by_mode[r["input_mode"]].append(r)

    out = {}
    for mode, rows in by_mode.items():
        # numeric iterations first, push blank/'final' to the end
        def itkey(r):
            v = _fnum(r[iter_col])
            return (v is None, v if v is not None else float("inf"))
        rows = sorted(rows, key=itkey)
        cols = defaultdict(list)
        for r in rows:
            cols["iter"].append(_fnum(r[iter_col]))
            for k, v in r.items():
                cols[k].append(_fnum(v))
        out[mode] = cols
    return out


def _plot(ax, x, y, label, marker="o"):
    # drop points where y is None so a missing/NaN value breaks the line cleanly
    pts = [(xi, yi) for xi, yi in zip(x, y) if yi is not None]
    if pts:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker=marker, ms=4, label=label)


def plot_mode(mode, cols, outdir):
    x = cols["iter"]
    fig, (ax_r, ax_abs, ax_stem) = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    # --- TOP: ratios (mass relative to detection) ---
    _plot(ax_r, x, cols["mass_det_backbone_res"], "mass/det @ backbone_res")
    _plot(ax_r, x, cols["mass_det_fpn"],          "mass/det @ fpn")
    if mode != "depth_only":
        _plot(ax_r, x, cols["mass_stem_depth_over_rgb"], "MASS stem depth/rgb")
    ax_r.axhline(1.0, color="grey", ls="--", lw=1, alpha=0.7)
    ax_r.set_ylabel("ratio")
    ax_r.set_title(f"Gradient audit over training — input_mode = {mode}")
    ax_r.legend(fontsize=8, loc="best")
    ax_r.grid(True, alpha=0.3)

    # --- MIDDLE: absolute shared-trunk gradients, DET vs MASS (log y) ---
    _plot(ax_abs, x, cols["det_backbone_res_abs"],  "DET  @ backbone_res")
    _plot(ax_abs, x, cols["mass_backbone_res_abs"], "MASS @ backbone_res")
    _plot(ax_abs, x, cols["det_fpn_abs"],           "DET  @ fpn")
    _plot(ax_abs, x, cols["mass_fpn_abs"],          "MASS @ fpn")
    ax_abs.set_yscale("log")
    ax_abs.set_ylabel("RMS grad (abs, log)")
    ax_abs.legend(fontsize=8, loc="best", ncol=2)
    ax_abs.grid(True, which="both", alpha=0.3)

    # --- BOTTOM: MASS stem absolute per-channel grad (log y) ---
    if mode != "depth_only":
        _plot(ax_stem, x, cols["mass_stem_rgb_abs"], "MASS stem rgb (abs)")
    _plot(ax_stem, x, cols["mass_stem_depth_abs"], "MASS stem depth (abs)")
    ax_stem.set_yscale("log")
    ax_stem.set_ylabel("stem RMS grad (abs, log)")
    ax_stem.set_xlabel("training iteration")
    ax_stem.legend(fontsize=8, loc="best")
    ax_stem.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(outdir, f"audit_{mode}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.csv)), "audit_plots")
    os.makedirs(outdir or ".", exist_ok=True)

    data = load(args.csv)
    for mode in MODES:
        if mode in data:
            plot_mode(mode, data[mode], outdir)
        else:
            print(f"(skipping {mode}: not in CSV)")


if __name__ == "__main__":
    main()
