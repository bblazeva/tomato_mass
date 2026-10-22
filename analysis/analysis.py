"""
Analysis for the tomato RGBD mass dataset.

Parses filenames in data/{train,val,test}/, extracts mass (grams) from the
filename prefix, and writes:
  - analysis/samples.csv       one row per image
  - analysis/summary.txt       per-split stats (counts, range, binned histogram)
  - analysis/hist_mass.png     overlaid mass histogram per split
  - analysis/bar_per_mass.png  images-per-unique-mass bar chart (per split)

Filename format: <mass_g>_<YYYY-MM-DD-HH-MM-SS>.<png|jpg>
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "analysis"
SPLITS = ("train", "val", "test")
FNAME_RE = re.compile(r"^(\d+)_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.(png|jpg)$")
# 100 g buckets up to 1100 g — adjusted automatically if masses exceed this.
BIN_STEP = 100


def collect() -> list[dict]:
    rows: list[dict] = []
    for split in SPLITS:
        split_dir = DATA_DIR / split
        if not split_dir.is_dir():
            print(f"warning: missing split dir {split_dir}")
            continue
        for f in sorted(split_dir.iterdir()):
            m = FNAME_RE.match(f.name)
            if not m:
                print(f"skipping unparseable filename: {f.name}")
                continue
            rows.append(
                {
                    "filename": f.name,
                    "split": split,
                    "mass_g": int(m.group(1)),
                    "timestamp": m.group(2),
                    "ext": m.group(3),
                }
            )
    return rows


def bin_edges(max_mass: int) -> list[int]:
    top = ((max_mass // BIN_STEP) + 1) * BIN_STEP
    return list(range(0, top + 1, BIN_STEP))


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)

        for split in SPLITS:
            masses = [r["mass_g"] for r in rows if r["split"] == split]
            counter = Counter(masses)
            total_samples = sum(counter.values())

            writer.writerow([f"{split}: {total_samples} samples"])
            writer.writerow(["value[g]", "number of samples"])

            for mass_g, count in sorted(counter.items()):
                writer.writerow([mass_g, count])

            writer.writerow([])  # blank line between train/test/val sections


def split_stats(rows: list[dict]) -> dict[str, dict]:
    by_split: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r["mass_g"])
    by_split["overall"] = [r["mass_g"] for r in rows]

    out: dict[str, dict] = {}
    for split, masses in by_split.items():
        arr = np.array(masses)
        out[split] = {
            "masses": masses,
            "n_images": int(arr.size),
            "n_unique_masses": int(np.unique(arr).size),
            "min": int(arr.min()),
            "max": int(arr.max()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
        }
    return out


def write_summary(stats: dict[str, dict], path: Path) -> None:
    global_max = stats["overall"]["max"]
    edges = bin_edges(global_max)

    lines: list[str] = []
    lines.append("Tomato RGBD dataset — mass analysis\n")
    lines.append("=" * 60 + "\n\n")

    for split in (*SPLITS, "overall"):
        s = stats[split]
        lines.append(f"[{split}]\n")
        lines.append(f"  images           : {s['n_images']}\n")
        lines.append(f"  unique mass vals : {s['n_unique_masses']}\n")
        lines.append(
            f"  mass range       : {s['min']}–{s['max']} g  "
            f"(mean {s['mean']:.1f}, median {s['median']:.1f}, std {s['std']:.1f})\n"
        )
        counts, _ = np.histogram(s["masses"], bins=edges)
        lines.append("  binned counts:\n")
        for i, c in enumerate(counts):
            label = f"{edges[i]}-{edges[i+1]}g"
            lines.append(f"    {label:>12} : {int(c)}\n")
        lines.append("\n")

    text = "".join(lines)
    path.write_text(text)
    print(text)


def plot_histogram(stats: dict[str, dict], path: Path) -> None:
    edges = bin_edges(stats["overall"]["max"])
    fig, ax = plt.subplots(figsize=(10, 5))
    for split in SPLITS:
        masses = stats[split]["masses"]
        ax.hist(
            masses,
            bins=edges,
            alpha=0.55,
            label=f"{split} (n={len(masses)})",
        )
    ax.set_xlabel("mass (g)")
    ax.set_ylabel("image count")
    ax.set_title("Mass distribution by split")
    ax.set_xticks(edges)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_mass_bar(stats: dict[str, dict], rows: list[dict], path: Path) -> None:
    all_masses = sorted({r["mass_g"] for r in rows})
    x = np.arange(len(all_masses))
    fig, axes = plt.subplots(len(SPLITS), 1, figsize=(12, 9), sharex=True)
    for ax, split in zip(axes, SPLITS):
        counter = Counter(r["mass_g"] for r in rows if r["split"] == split)
        counts = [counter.get(m, 0) for m in all_masses]
        ax.bar(x, counts, width=1.0)
        ax.set_ylabel("count")
        ax.set_title(
            f"{split}: images per unique mass value "
            f"({len(counter)} unique, {sum(counts)} images)"
        )
        ax.grid(axis="y", alpha=0.3)
    axes[-1].set_xlabel("unique mass values [g]")
    step = max(1, len(all_masses) // 20)
    ticks = list(range(0, len(all_masses), step))
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([all_masses[i] for i in ticks], rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect()
    if not rows:
        raise SystemExit("no parseable files found under data/{train,val,test}/")
    write_csv(rows, OUT_DIR / "samples.csv")
    stats = split_stats(rows)
    write_summary(stats, OUT_DIR / "summary.txt")
    plot_histogram(stats, OUT_DIR / "hist_mass.png")
    plot_per_mass_bar(stats, rows, OUT_DIR / "bar_per_mass.png")
    print(f"\nwrote outputs to {OUT_DIR}/")


if __name__ == "__main__":
    main()
