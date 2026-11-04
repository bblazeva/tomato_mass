"""
Figures for the 23.10 progress update. Reads only saved results (no GPU):
  results/eval/<run>/{val,test_0328}/per_image.csv
  results/eval/<run>__depth_ablation/val/summary.json
  results/audit_2026-10/summary.csv
  output/<run>/metrics.json   (grad/depth, grad/rgb logged during training)

  ./run.sh docs/progress/2026-10-23/make_figures.py
"""
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"

MODELS = [  # id, run, label
    ("5", "s0713_zoom_drop05", "5\nRGB-drop 0.5\n(15k)"),
    ("7c", "s0713_zoom_drop_05_00", "7c\nRGB-drop 0.5\n(10k, = 5)"),
    ("6d", "s0713_zoom_drop05_mass_loss_schedule", "6d\nmass-weight\nschedule"),
    ("7b", "s0713_zoom_drop_04_01", "7b\nRGB-drop 0.4\n+ depth-drop 0.1"),
    ("7a", "s0713_zoom_drop065", "7a\nRGB-drop 0.65"),
]
MODES = [("rgbd", "RGB-D", BLUE), ("rgb_only", "RGB only", ORANGE), ("depth_only", "depth only", AQUA)]

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200,
})


def load(run, split):
    rows = {}
    for r in csv.DictReader(open(f"results/eval/{run}/{split}/per_image.csv")):
        rows[(r["input_mode"], r["file"])] = r
    return rows


def style(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def bar_labels(ax, bars, fmt="{:.0f}"):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + ax.get_ylim()[1] * 0.01,
                fmt.format(b.get_height()), ha="center", va="bottom", fontsize=8, color=MUTED)


# ------------------------------------------------------------------------------------------
# Fig 1: in-domain vs cross-session MAPE per input, on frames every model+mode predicts
# ------------------------------------------------------------------------------------------
def fig_input_modes():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, split, title in [(axes[0], "val", "In-domain (val, home session)"),
                             (axes[1], "test_0328", "Cross-session (lab session)")]:
        data = {rid: load(run, split) for rid, run, _ in MODELS}
        files = {k[1] for k in data["5"]}
        common = [f for f in files if all(data[rid][(m, f)]["pred_kg"] for rid, _, _ in MODELS for m, _, _ in MODES)]
        x = np.arange(len(MODELS))
        w = 0.26
        for j, (m, lab, col) in enumerate(MODES):
            vals = [np.mean([float(data[rid][(m, f)]["abs_pct_err"]) for f in common]) for rid, _, _ in MODELS]
            bars = ax.bar(x + (j - 1) * w, vals, w - 0.03, color=col, label=lab)
            bar_labels(ax, bars)
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, _, lab in MODELS], fontsize=8)
        ax.set_ylabel("MAPE (%)")
        ax.set_title(f"{title}\n{len(common)} frames that every model and input detects", fontsize=10)
        style(ax)
    axes[0].set_ylim(0, 25)
    axes[1].set_ylim(0, 60)
    axes[1].legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_input_modes.png")


# ------------------------------------------------------------------------------------------
# Fig 2: gradient audit — (a) is depth trained at all, (b) RGB dropout dose during training
# ------------------------------------------------------------------------------------------
def fig_audit():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.2]})

    ax = axes[0]
    rows = {r["id"]: r for r in csv.DictReader(open("results/audit_2026-10/summary.csv"))}
    order = [("4.1", "4.1 no norm. fix"), ("4.2", "4.2 norm. fix"), ("5", "5 + zoom"),
             ("7c", "7c (= 5, 10k)"), ("6d", "6d schedule"), ("7b", "7b drop 0.4+0.1"), ("7a", "7a drop 0.65")]
    vals = [float(rows[i]["depth_over_rgb"]) for i, _ in order]
    y = np.arange(len(order))[::-1]
    bars = ax.barh(y, vals, 0.6, color=[MUTED] + [BLUE] * (len(order) - 1))
    for b, v in zip(bars, vals):
        ax.text(v + 0.015, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in order], fontsize=9)
    ax.axvline(1.0, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("mass-loss gradient on depth filters ÷ on RGB filters (conv1)")
    ax.set_title("(a) Is depth trained? Final checkpoints, same 60 val images\n"
                 "(1.0 = depth pushed as hard as RGB)", fontsize=10)
    style(ax, "x")

    ax = axes[1]
    curves = [("clean_baseline_v1_5conv_aug_fix_noLF_norm_stip", "no RGB dropout (run 3)", MUTED),
              ("s0713_zoom_drop_04_01", "RGB-drop 0.4 + depth-drop 0.1 (7b)", YELLOW),
              ("s0713_zoom_drop05", "RGB-drop 0.5 (5)", BLUE),
              ("s0713_zoom_drop065", "RGB-drop 0.65 (7a)", ORANGE)]
    for run, lab, col in curves:
        it, ratio = [], []
        for line in open(f"output/{run}/metrics.json"):
            r = json.loads(line)
            if r.get("grad/rgb"):
                it.append(r["iteration"])
                ratio.append(r["grad/depth"] / r["grad/rgb"])
        it, ratio = np.array(it), np.array(ratio)
        k = 25   # running median over 25 logged points (dropout makes single values jump)
        smooth = np.array([np.median(ratio[max(0, i - k):i + 1]) for i in range(len(ratio))])
        ax.plot(it / 1000, smooth, color=col, linewidth=2, label=lab)
    ax.set_yscale("log")
    ax.set_yticks([0.3, 1, 3, 10])
    ax.set_yticklabels(["0.3", "1", "3", "10"])
    ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlabel("training iteration (thousands)")
    ax.set_ylabel("depth ÷ RGB gradient (conv1)")
    ax.set_title("(b) During training: more RGB dropout → depth filters learn more\n"
                 "(total training loss, running median)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.17))
    style(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_gradient_audit.png")


# ------------------------------------------------------------------------------------------
# Fig 3: depth ablation — how much the prediction moves when the depth channel is changed
# ------------------------------------------------------------------------------------------
def fig_depth_ablation():
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    versions = [("rgb_only", "no depth\n(mean value)"), ("rgbd_zero", "0 mm\neverywhere"),
                ("rgbd_flat", "flat = cluster's\nown distance"), ("rgbd_shuffled", "shuffled\n(other cluster's depth)")]
    x = np.arange(len(versions))
    w = 0.36
    for j, (run, lab, col) in enumerate([("s0713_zoom_drop05", "model 5 (RGB-drop 0.5)", BLUE),
                                         ("s0713_zoom_drop065", "model 7a (RGB-drop 0.65)", ORANGE)]):
        s = json.load(open(f"results/eval/{run}__depth_ablation/val/summary.json"))["change_vs_rgbd"]
        bars = ax.bar(x + (j - 0.5) * w, [s[v]["mean_pct"] for v, _ in versions], w - 0.04, color=col, label=lab)
        bar_labels(ax, bars, "{:.1f}")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in versions], fontsize=9)
    ax.set_ylabel("mean change of the predicted mass (%)")
    ax.set_ylim(0, 45)
    ax.set_title("Real RGB + modified depth vs real RGB-D (in-domain val, 190 frames)", fontsize=10)
    ax.legend(frameon=False, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_depth_ablation.png")


# ------------------------------------------------------------------------------------------
# Fig 4: averaging RGB-D and depth-only predictions (fixed 50/50, nothing tuned)
# ------------------------------------------------------------------------------------------
def fig_averaging():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, split, title, ymax in [(axes[0], "val", "In-domain (val)", 25),
                                   (axes[1], "test_0328", "Cross-session (lab)", 90)]:
        rgbd, avg = [], []
        for rid, run, _ in MODELS:
            rows = load(run, split)
            e1, e2 = [], []
            for (m, f), a in rows.items():
                if m != "rgbd" or not a["pred_kg"]:
                    continue
                d = rows[("depth_only", f)]
                gt, pa = float(a["gt_kg"]), float(a["pred_kg"])
                pm = np.sqrt(pa * float(d["pred_kg"])) if d["pred_kg"] else pa   # depth-only missed -> RGB-D
                e1.append(abs(pa - gt) / gt * 100)
                e2.append(abs(pm - gt) / gt * 100)
            rgbd.append(np.mean(e1))
            avg.append(np.mean(e2))
        x = np.arange(len(MODELS))
        w = 0.36
        bar_labels(ax, ax.bar(x - w / 2, rgbd, w - 0.04, color=BLUE, label="RGB-D model"))
        bar_labels(ax, ax.bar(x + w / 2, avg, w - 0.04, color=AQUA, label="average of RGB-D and depth-only prediction"))
        ax.set_ylim(0, ymax)
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, _, lab in MODELS], fontsize=8)
        ax.set_ylabel("MAPE (%)")
        ax.set_title(f"{title}, all frames", fontsize=10)
        style(ax)
    axes[1].legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_averaging.png")


if __name__ == "__main__":
    fig_input_modes()
    fig_audit()
    fig_depth_ablation()
    fig_averaging()
    print("figures ->", OUT)
