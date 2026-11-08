"""
Prepare hand-counting of the tomatoes per cluster: image pages plus a CSV to fill in.

For each cluster (ranked by mass) two crops are shown: the frame where the cluster looks largest, and the
largest frame from the most distant moment in the capture, for a second view of hidden tomatoes.

Writes (never overwrites the CSV, so filled-in counts are safe):
  analysis/tomato_counts.csv              id, mass_g, session, n_images, n_tomatoes (empty), notes
  analysis/tomato_count_sheet/page_*.jpg  labelled with the same ids

    ./run.sh analysis/make_tomato_count_sheet.py
"""
import argparse
import csv
import json
import os
import re
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np

FILE_NAME = re.compile(r"^(\d+)_(\d{4}-\d{2}-\d{2})-(\d{2}-\d{2}-\d{2})")
TILE = 480
LABEL_HEIGHT = 44
PADDING = 25
CLUSTERS_PER_ROW = 2
ROWS_PER_PAGE = 4
LATER_FRAME_MIN_GAP_S = 5


def load_clusters(dataset):
    """-> {(mass_g, session): [(time, box_area, file_name, bbox), ...]}."""
    with open(os.path.join(dataset, "gt_json", "all", "all.json")) as fh:
        coco = json.load(fh)
    bbox = {a["image_id"]: a["bbox"] for a in coco["annotations"]}
    clusters = defaultdict(list)
    for image in coco["images"]:
        match = FILE_NAME.match(image["file_name"])
        time = datetime.strptime(match.group(2) + match.group(3), "%Y-%m-%d%H-%M-%S")
        box = bbox[image["id"]]
        clusters[(int(match.group(1)), match.group(2))].append((time, box[2] * box[3], image["file_name"], box))
    return clusters


def pick_two(frames):
    """The largest view, and the largest view from a distant moment (if there is one)."""
    first = max(frames, key=lambda f: f[1])
    gap = lambda f: abs((f[0] - first[0]).total_seconds())
    candidates = [f for f in frames if gap(f) >= LATER_FRAME_MIN_GAP_S]
    if not candidates:
        return [first]
    farthest = max(gap(f) for f in candidates)
    return [first, max((f for f in candidates if gap(f) >= farthest / 2), key=lambda f: f[1])]


def crop_tile(dataset, frame):
    """Square crop around the box, scaled up to TILE x TILE, padded with black."""
    image = cv2.imread(os.path.join(dataset, "images", "all", frame[2]))
    x, y, w, h = frame[3]
    height, width = image.shape[:2]
    crop = image[max(0, int(y) - PADDING):min(height, int(y + h) + PADDING),
                 max(0, int(x) - PADDING):min(width, int(x + w) + PADDING)]
    scale = TILE / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    tile = np.zeros((TILE, TILE, 3), np.uint8)
    top, left = (TILE - crop.shape[0]) // 2, (TILE - crop.shape[1]) // 2
    tile[top:top + crop.shape[0], left:left + crop.shape[1]] = crop
    return tile


def cluster_panel(dataset, cluster_id, mass, session, frames):
    tiles = [crop_tile(dataset, f) for f in pick_two(frames)]
    while len(tiles) < 2:
        tiles.append(np.zeros((TILE, TILE, 3), np.uint8))
    label = np.full((LABEL_HEIGHT, TILE * 2, 3), 255, np.uint8)
    cv2.putText(label, f"#{cluster_id}   {mass} g   {session}", (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2,
                cv2.LINE_AA)
    return np.vstack([label, np.hstack(tiles)])


def main():
    parser = argparse.ArgumentParser(description="Prepare hand-counting of tomatoes per cluster")
    parser.add_argument("--dataset", default="./data_min120")
    parser.add_argument("--csv", default="analysis/tomato_counts.csv")
    parser.add_argument("--out_dir", default="analysis/tomato_count_sheet")
    args = parser.parse_args()

    clusters = load_clusters(args.dataset)
    ranked = sorted(clusters)

    if os.path.exists(args.csv):
        print(f"Keeping the existing {args.csv} (not overwritten)")
    else:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id", "mass_g", "session", "n_images", "n_tomatoes", "notes"])
            for i, (mass, session) in enumerate(ranked, start=1):
                writer.writerow([i, mass, session, len(clusters[(mass, session)]), "", ""])

    os.makedirs(args.out_dir, exist_ok=True)
    panels = [cluster_panel(args.dataset, i, mass, session, clusters[(mass, session)])
              for i, (mass, session) in enumerate(ranked, start=1)]
    blank = np.full_like(panels[0], 255)
    per_page = CLUSTERS_PER_ROW * ROWS_PER_PAGE
    n_pages = 0
    for start in range(0, len(panels), per_page):
        page = panels[start:start + per_page]
        page += [blank] * (per_page - len(page))
        rows = [np.hstack(page[r:r + CLUSTERS_PER_ROW]) for r in range(0, per_page, CLUSTERS_PER_ROW)]
        n_pages += 1
        cv2.imwrite(os.path.join(args.out_dir, f"page_{n_pages}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"{len(ranked)} clusters -> {args.csv}, {n_pages} pages in {args.out_dir}/")


if __name__ == "__main__":
    main()
