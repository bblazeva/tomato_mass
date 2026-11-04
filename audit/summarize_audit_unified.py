"""
Summarize the unified gradient audit (gradient_audit_sweep2.py, same 60 fixed val
images, seed 0, mass weight 1) into one table + two figures for reporting.

Per model it reduces the audit to three questions:
  1. depth/RGB push   - MASS-loss gradient on conv1 depth slice / RGB slices
                        (real RGB-D input). Is depth being trained?
  2. mass/det backbone - MASS vs DETECTION gradient in the shared ResNet.
                        Does the mass loss reach the shared features?
  3. val mass loss with RGB-D / RGB-only / depth-only input.
                        Which input is the easier path for mass?

"final" = the last audited checkpoint of each run. The range column is min-max
over the last (up to) 4 audited checkpoints, as a noise indicator.

Outputs (in results/audit_2026-10/):
  summary.md, summary.csv
  fig_audit_models.png    - the three questions, one panel each, one row per model
  fig_audit_training.png  - drop_05_00 (model-5 replicate) over training

Run (from the repo root; paths below are relative to it):
  ./run.sh audit/summarize_audit_unified.py
"""
import csv
import json
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

OUT_DIR = "results/audit_2026-10"

# (short id, label, audit dir, metrics.json for val MAPE)
RUNS = [
    ("4.1", "4.1  no norm + RGB-drop 0.5", "output/audit_unified/4.1_notNorm_drop05",
     "/mnt/md0/blagicab/models/clean_baseline_v1_5conv_aug_fix_noLF_stip_drop05/metrics.json"),
    ("4.2", "4.2  norm + RGB-drop 0.5", "output/audit_unified/4.2_norm_drop05",
     "output/clean_baseline_v1_5conv_aug_fix_noLF_norm_stip_drop05/metrics.json"),
    ("5", "5    + zoom  (best)", "output/audit_unified/5_zoom_drop05",
     "output/s0713_zoom_drop05/metrics.json"),
    ("7c", "7c   RGB-drop 0.5, 10k it (= 5 setup)", "output/audit_unified/5r_drop_05_00",
     "output/s0713_zoom_drop_05_00/metrics.json"),
    ("5c", "5c   + colour aug", "output/audit_unified/5c_color_aug",
     "output/s0713_zoom_drop05_color_aug/metrics.json"),
    ("6b", "6b   mass weight 2", "output/audit_unified/6b_weight2",
     "output/s0713_zoom_drop05_mass_loss_weight2/metrics.json"),
    ("6d", "6d   mass schedule 0.5>1>2", "output/audit_unified/6d_schedule",
     "output/s0713_zoom_drop05_mass_loss_schedule/metrics.json"),
    ("7a", "7a   RGB-drop 0.65", "output/s0713_zoom_drop065/grad_audit",
     "output/s0713_zoom_drop065/metrics.json"),
    ("7b", "7b   RGB-drop 0.4 + depth-drop 0.1", "output/s0713_zoom_drop_04_01/grad_audit",
     "output/s0713_zoom_drop_04_01/metrics.json"),
]
TRAINING_RUN = "7c"
# The training figure uses the full-val (190 image) audit with per-image output, for error bands.
TRAINING_AUDIT_FULL = "results/audit_2026-10/val190/7c_drop_05_00"
MODES = ["real_rgbd", "rgb_only", "depth_only"]
MODE_LABEL = {"real_rgbd": "RGB-D", "rgb_only": "RGB only", "depth_only": "depth only"}

# Reference palette (light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = {"real_rgbd": "#2a78d6", "rgb_only": "#eb6834", "depth_only": "#1baf7a"}


def fnum(s):
    try:
        v = float(s)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def load_sweep(audit_dir):
    """rows[mode] -> list of row dicts sorted by iteration."""
    path = os.path.join(audit_dir, "gradient_audit_sweep.csv")
    by_mode = defaultdict(list)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            it = fnum(r.get("checkpoint_iteration_numeric") or r.get("checkpoint_iteration"))
            r["_it"] = int(it) if it is not None else -1
            by_mode[r["input_mode"]].append(r)
    for m in by_mode:
        by_mode[m].sort(key=lambda r: r["_it"])
    return by_mode


def subset_files(audit_dir):
    path = os.path.join(audit_dir, "gradient_audit_subset.csv")
    with open(path, newline="") as fh:
        return [r["file_name"] for r in csv.DictReader(fh)]


def val_mape(metrics_path):
    """(final val MAPE %, list of (iter, MAPE %)) from training-time evals."""
    if not os.path.exists(metrics_path):
        return None, []
    curve = []
    with open(metrics_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            if "mass_mre" in d:
                curve.append((d["iteration"], 100 * d["mass_mre"]))
    return (curve[-1][1] if curve else None), curve


def summarize():
    ref_subset = None
    rows = []
    for rid, label, audit_dir, metrics in RUNS:
        if not os.path.exists(os.path.join(audit_dir, "gradient_audit_sweep.csv")):
            print("MISSING audit for", rid, audit_dir)
            continue
        files = subset_files(audit_dir)
        if ref_subset is None:
            ref_subset = files
        elif files != ref_subset:
            print("WARNING: %s audited a different image subset than the first run" % rid)

        sweep = load_sweep(audit_dir)
        real = sweep["real_rgbd"]
        last4 = real[-4:]
        ratios = [fnum(r["mass_stem_depth_over_rgb"]) for r in last4]
        ratios = [v for v in ratios if v is not None]
        final = real[-1]
        loss = {m: fnum(sweep[m][-1]["loss_mass"]) for m in MODES if sweep.get(m)}
        mape, _ = val_mape(metrics)
        rows.append({
            "id": rid,
            "label": label,
            "checkpoint": final["_it"],
            "depth_over_rgb": fnum(final["mass_stem_depth_over_rgb"]),
            "depth_over_rgb_min": min(ratios) if ratios else None,
            "depth_over_rgb_max": max(ratios) if ratios else None,
            "n_ckpt_range": len(ratios),
            "mass_over_det_backbone": fnum(final["mass_det_backbone_res"]),
            "loss_mass_rgbd": loss.get("real_rgbd"),
            "loss_mass_rgb": loss.get("rgb_only"),
            "loss_mass_depth": loss.get("depth_only"),
            "val_mape_final": mape,
        })
    return rows


def fmt(v, nd=3):
    return "-" if v is None else ("%.*f" % (nd, v))


def write_tables(rows):
    cols = ["id", "label", "checkpoint", "depth_over_rgb", "depth_over_rgb_min",
            "depth_over_rgb_max", "n_ckpt_range", "mass_over_det_backbone",
            "loss_mass_rgbd", "loss_mass_rgb", "loss_mass_depth", "val_mape_final"]
    with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    lines = [
        "# Unified gradient audit — 60 fixed val images, seed 0, mass weight 1",
        "",
        "| Model | ckpt | depth/RGB push (mass, RGB-D input) | range last ckpts | "
        "mass/det @ backbone | val mass loss RGB-D / RGB / depth | depth ÷ RGB-D loss | val MAPE |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        rng = "%s–%s (%d)" % (fmt(r["depth_over_rgb_min"], 2), fmt(r["depth_over_rgb_max"], 2),
                              r["n_ckpt_range"])
        lr = r["loss_mass_depth"] / r["loss_mass_rgbd"] if r["loss_mass_rgbd"] and r["loss_mass_depth"] else None
        lines.append("| %s | %dk | **%s** | %s | %s | %s / %s / %s | %s× | %s%% |" % (
            r["label"].strip(), round(r["checkpoint"] / 1000), fmt(r["depth_over_rgb"], 2), rng,
            fmt(r["mass_over_det_backbone"], 2),
            fmt(r["loss_mass_rgbd"]), fmt(r["loss_mass_rgb"]), fmt(r["loss_mass_depth"]),
            fmt(lr, 1), fmt(r["val_mape_final"], 1)))
    with open(os.path.join(OUT_DIR, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_models(rows):
    rows = rows[::-1]  # first model on top
    y = list(range(len(rows)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.48 * len(rows) + 1.8), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1, 1.35]})
    fig.patch.set_facecolor(SURFACE)

    # 1. depth/RGB push
    ax = axes[0]
    style_axis(ax)
    vals = [r["depth_over_rgb"] or 0 for r in rows]
    ax.barh(y, vals, height=0.55, color=SERIES["real_rgbd"])
    for yi, r in zip(y, rows):
        label_x = r["depth_over_rgb"] or 0
        if r["n_ckpt_range"] > 1:
            ax.plot([r["depth_over_rgb_min"], r["depth_over_rgb_max"]], [yi, yi],
                    color=INK, linewidth=1.2)
            label_x = max(label_x, r["depth_over_rgb_max"])
        ax.text(label_x + 0.02, yi, fmt(r["depth_over_rgb"], 2),
                va="center", fontsize=8.5, color=INK2)
    ax.axvline(1.0, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlim(0, 1.1)
    ax.set_title("1. Is depth trained?\nmass-loss push on depth ÷ on RGB (conv1)",
                 fontsize=10, color=INK, loc="left")
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9, color=INK)

    # 2. mass/det at backbone
    ax = axes[1]
    style_axis(ax)
    vals = [r["mass_over_det_backbone"] or 0 for r in rows]
    ax.barh(y, vals, height=0.55, color=SERIES["real_rgbd"])
    for yi, v in zip(y, vals):
        ax.text(v + 0.03, yi, fmt(v, 2), va="center", fontsize=8.5, color=INK2)
    ax.axvline(1.0, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlim(0, max(2.0, max(vals) * 1.15))
    ax.set_title("2. Does mass reach the backbone?\nmass ÷ detection gradient (ResNet)",
                 fontsize=10, color=INK, loc="left")

    # 3. val mass loss per input
    ax = axes[2]
    style_axis(ax)
    h = 0.24
    keys = [("real_rgbd", "loss_mass_rgbd", h), ("rgb_only", "loss_mass_rgb", 0.0),
            ("depth_only", "loss_mass_depth", -h)]
    for mode, key, off in keys:
        vals = [r[key] or 0 for r in rows]
        ax.barh([yi + off for yi in y], vals, height=h * 0.9, color=SERIES[mode],
                label=MODE_LABEL[mode])
    ax.set_title("3. Which input is easier?\nval mass loss, per input given (lower = better)",
                 fontsize=10, color=INK, loc="left")
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)

    fig.suptitle("Gradient audit, final checkpoints — same 60 val images for every model  "
                 "(dashed line = equal push; black whisker = range over last checkpoints)",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(OUT_DIR, "fig_audit_models.png")
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    print("wrote", path)


def per_image_curves(per_image_csv, n_boot=1000, seed=0):
    """From a per-image audit CSV: curves[(quantity, mode)] -> (iters, value, lo95, hi95).
    Values are computed like the sweep (ratio of means); bands are image bootstraps."""
    import numpy as np
    raw = defaultdict(lambda: defaultdict(list))
    with open(per_image_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            key = (int(float(r["checkpoint_iteration_numeric"])), r["input_mode"])
            for c in ("mass_stem_depth_abs", "mass_stem_rgb_abs", "loss_mass"):
                raw[key][c].append(float(r[c] or 0.0))
    rng = np.random.default_rng(seed)
    fns = {
        "depth_rgb": lambda d, i: d["mass_stem_depth_abs"][i].mean() / d["mass_stem_rgb_abs"][i].mean(),
        "loss_mass": lambda d, i: d["loss_mass"][i].mean(),
    }
    curves = defaultdict(lambda: ([], [], [], []))
    for (it, mode) in sorted(raw):
        d = {c: np.array(v) for c, v in raw[(it, mode)].items()}
        n = len(d["loss_mass"])
        for q, fn in fns.items():
            if q == "depth_rgb" and mode != "real_rgbd":
                continue
            boots = [fn(d, rng.integers(0, n, n)) for _ in range(n_boot)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            c = curves[(q, mode)]
            c[0].append(it / 1000)
            c[1].append(fn(d, np.arange(n)))
            c[2].append(lo)
            c[3].append(hi)
    return curves, n


def fig_training():
    rid, label, audit_dir, metrics = next(r for r in RUNS if r[0] == TRAINING_RUN)
    per_image = os.path.join(TRAINING_AUDIT_FULL, "gradient_audit_per_image.csv")
    if not os.path.exists(per_image):
        print("no full-val per-image audit for", rid, "at", per_image)
        return
    curves, n_img = per_image_curves(per_image)
    _, mape_curve = val_mape(metrics)

    fig, axes = plt.subplots(3, 1, figsize=(8, 8.5), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        style_axis(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.grid(False, axis="x")

    ax = axes[0]
    xs, v, lo, hi = curves[("depth_rgb", "real_rgbd")]
    ax.fill_between(xs, lo, hi, color=SERIES["real_rgbd"], alpha=0.18, linewidth=0)
    ax.plot(xs, v, color=SERIES["real_rgbd"], linewidth=2, marker="o", markersize=5)
    ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1)
    ax.set_ylim(0, 1.2)
    ax.set_title("Is depth trained? mass-loss push on depth ÷ RGB (RGB-D input)",
                 fontsize=10, color=INK, loc="left")

    ax = axes[1]
    for mode in MODES:
        xs, v, lo, hi = curves[("loss_mass", mode)]
        ax.fill_between(xs, lo, hi, color=SERIES[mode], alpha=0.15, linewidth=0)
        ax.plot(xs, v, color=SERIES[mode], linewidth=2, marker="o", markersize=5,
                label=MODE_LABEL[mode])
    ax.set_yscale("log")
    ax.set_yticks([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.tick_params(which="minor", length=0)
    ax.set_title("Which input is easier? val mass loss per input (log scale, lower = better)",
                 fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8.5, frameon=False)

    ax = axes[2]
    if mape_curve:
        ax.plot([i / 1000 for i, _ in mape_curve], [m for _, m in mape_curve], color=INK2,
                linewidth=2, marker="o", markersize=4)
    ax.set_title("Val MAPE during training (%)", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("training iteration (thousands)", fontsize=9, color=INK2)

    fig.suptitle("%s — gradient audit over training, all %d val images (bands = 95%% bootstrap)"
                 % (label.strip(), n_img), fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(OUT_DIR, "fig_audit_training.png")
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    print("wrote", path)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = summarize()
    write_tables(rows)
    fig_models(rows)
    fig_training()
