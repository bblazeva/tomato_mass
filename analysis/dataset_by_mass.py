"""
Overview of the whole annotated dataset: images per cluster, ranked by mass, coloured by capture session.

Pools the train / val / test annotations of the mixed-session dataset (--dataset_path) and writes:
  analysis/dataset_by_mass.png   bar chart, one bar per cluster
  analysis/dataset_by_mass.csv   mass_g, session, n_images

A cluster = one mass value within one session. The mass and session are parsed from the file name,
e.g. 208_2025-07-13-18-11-23.jpg.

  ./run.sh analysis/dataset_by_mass.py
"""
import argparse
import csv
import json
import os
import re
import statistics
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SPLITS = ["train", "val", "test"]
SESSION_LABELS = {"2025-07-13": "2025-07-13 (home)", "2025-03-28": "2025-03-28 (lab)", "2023-12-01": "2023-12-01"}
SESSION_COLORS = {"2025-07-13": "#2a78d6", "2025-03-28": "#eb6834", "2023-12-01": "#1baf7a"}
FILE_NAME = re.compile(r"^(\d+)_(\d{4}-\d{2}-\d{2})-")
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"


def collect(dataset_path):
    """-> Counter {(mass_g, session): number of images}, over all splits."""
    clusters = Counter()
    for split in SPLITS:
        with open(os.path.join(dataset_path, "gt_json", split, split + ".json")) as fh:
            images = json.load(fh)["images"]
        for image in images:
            name = os.path.basename(image["file_name"])
            match = FILE_NAME.match(name)
            if match is None:
                raise SystemExit(f"Cannot parse mass and session from: {name}")
            clusters[(int(match.group(1)), match.group(2))] += 1
    return clusters


def print_summary(clusters):
    sessions = sorted({session for _, session in clusters})
    masses = [mass for (mass, _), n in clusters.items() for _ in range(n)]
    print(f"All: {sum(clusters.values())} images, {len(clusters)} clusters, mass {min(masses)}-{max(masses)} g, "
          f"mean {statistics.mean(masses):.1f} g, median {statistics.median(masses):.1f} g")
    for session in sessions:
        counts = {mass: n for (mass, s), n in clusters.items() if s == session}
        print(f"  {SESSION_LABELS.get(session, session):<18} {sum(counts.values()):>4} images, "
              f"{len(counts):>2} clusters, mass {min(counts)}-{max(counts)} g, "
              f"{min(counts.values())}-{max(counts.values())} images per cluster")


def plot(clusters, path):
    ranked = sorted(clusters.items())
    x = range(len(ranked))
    colors = [SESSION_COLORS.get(session, MUTED) for (_, session), _ in ranked]

    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.bar(x, [n for _, n in ranked], width=0.8, color=colors)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(mass) for (mass, _), _ in ranked], rotation=90, fontsize=8, color=MUTED)
    ax.set_xlim(-0.8, len(ranked) - 0.2)
    ax.set_xlabel("cluster mass [g], ranked (not to scale)", color=INK)
    ax.set_ylabel("images", color=INK)
    ax.tick_params(axis="y", colors=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"Images per cluster, ranked by mass ({len(ranked)} clusters, {sum(clusters.values())} images)",
                 color=INK, fontsize=11)

    counts = Counter()
    for (_, session), n in ranked:
        counts[session] += n
    handles = [Patch(color=SESSION_COLORS.get(s, MUTED),
                     label=f"{SESSION_LABELS.get(s, s)}: {counts[s]} images, "
                           f"{sum(1 for _, x_s in clusters if x_s == s)} clusters")
               for s in sorted(counts, reverse=True)]
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_path", default="./data/")
    parser.add_argument("--out_dir", default="./analysis/")
    args = parser.parse_args()

    clusters = collect(args.dataset_path)
    print_summary(clusters)

    csv_path = os.path.join(args.out_dir, "dataset_by_mass.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mass_g", "session", "n_images"])
        for (mass, session), n in sorted(clusters.items()):
            writer.writerow([mass, session, n])

    png_path = os.path.join(args.out_dir, "dataset_by_mass.png")
    plot(clusters, png_path)
    print(f"-> {png_path}\n-> {csv_path}")


if __name__ == "__main__":
    main()
