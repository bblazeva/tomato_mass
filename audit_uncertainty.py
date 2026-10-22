"""
How much can we trust the gradient-audit numbers? Uses the per-image CSV written
by gradient_audit_sweep2.py (gradient_audit_per_image.csv).

For each checkpoint x input mode it reports, for the key quantities:
  value      - computed exactly like the sweep (ratio of means over all images)
  boot95     - 95% bootstrap interval, resampling images (1000 draws, seed 0)
  per-cluster value (208 / 490 / 670 / 940 g)
  LOCO range - min-max when each cluster in turn is left out

Images within a cluster are near-duplicate frames, so the image bootstrap is
optimistic; the per-cluster values and LOCO range are the honest check.

Quantities:
  depth_rgb   mass-loss stem gradient, depth / RGB          (real_rgbd only)
  mass_det    mass / detection gradient at backbone_res
  loss_mass   val mass loss (mean)

Run:
  ~/miniconda3/envs/tomato_mass/bin/python3 audit_uncertainty.py \
      output/audit_val190/5_zoom_drop05/gradient_audit_per_image.csv
"""
import csv
import sys
from collections import defaultdict

import numpy as np

N_BOOT = 1000


def ratio(num, den):
    d = np.mean(den)
    return float(np.mean(num) / d) if d > 0 else float("nan")


QUANTITIES = {
    "depth_rgb": (lambda r: ratio(r["mass_stem_depth_abs"], r["mass_stem_rgb_abs"]), ["real_rgbd"]),
    "mass_det": (lambda r: ratio(r["mass_backbone_res_abs"], r["det_backbone_res_abs"]), None),
    "loss_mass": (lambda r: float(np.mean(r["loss_mass"])), None),
}
NUM_COLS = ["mass_stem_depth_abs", "mass_stem_rgb_abs", "mass_backbone_res_abs",
            "det_backbone_res_abs", "loss_mass"]


def load(path):
    """groups[(iteration, mode)] -> dict of column -> np.array, plus 'cluster'."""
    raw = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            key = (int(float(r["checkpoint_iteration_numeric"])), r["input_mode"])
            for c in NUM_COLS:
                raw[key][c].append(float(r[c] or 0.0))
            raw[key]["cluster"].append(int(round(float(r["cluster_mass"]) * 1000)))
    return {k: {c: np.array(v) for c, v in d.items()} for k, d in raw.items()}


def take(d, mask):
    return {c: v[mask] for c, v in d.items()}


def analyse(d, fn, rng):
    n = len(d["cluster"])
    value = fn(d)
    boots = [fn(take(d, rng.integers(0, n, n))) for _ in range(N_BOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    clusters = sorted(set(d["cluster"].tolist()))
    per_cluster = {c: fn(take(d, d["cluster"] == c)) for c in clusters}
    loco = [fn(take(d, d["cluster"] != c)) for c in clusters]
    return value, (lo, hi), per_cluster, (min(loco), max(loco)), n


def main(path):
    data = load(path)
    rng = np.random.default_rng(0)
    out_rows = []
    print("file:", path)
    for qname, (fn, modes) in QUANTITIES.items():
        print("\n== %s" % qname)
        print("%6s %-10s %4s %8s %17s %17s   per cluster (208/490/670/940)" % (
            "iter", "mode", "n", "value", "boot95", "leave-1-cluster"))
        for (it, mode) in sorted(data):
            if modes and mode not in modes:
                continue
            v, (lo, hi), pc, (l1, l2), n = analyse(data[(it, mode)], fn, rng)
            pcs = " / ".join("%.3f" % pc[c] for c in sorted(pc))
            print("%6d %-10s %4d %8.3f   [%6.3f, %6.3f]   [%6.3f, %6.3f]   %s" % (
                it, mode, n, v, lo, hi, l1, l2, pcs))
            row = {"quantity": qname, "iteration": it, "input_mode": mode, "n_images": n,
                   "value": v, "boot95_lo": lo, "boot95_hi": hi,
                   "loco_min": l1, "loco_max": l2}
            row.update({"cluster_%dg" % c: pc[c] for c in pc})
            out_rows.append(row)

    out = path.replace("gradient_audit_per_image.csv", "audit_uncertainty.csv")
    fields = sorted({k for r in out_rows for k in r}, key=lambda k: (
        ["quantity", "iteration", "input_mode", "n_images", "value", "boot95_lo", "boot95_hi",
         "loco_min", "loco_max"] + [k]).index(k))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print("\nwrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
