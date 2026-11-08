"""
One image sheet per cluster, for looking at the data by eye: a crop of every frame, in capture order.

Crops work as in make_tomato_count_sheet.py, and the #ids match analysis/tomato_counts.csv (the excluded
clusters are not in that file, so they have no id). A cluster here is one mass in one session.

Covers the pool of make_min_mass_dataset.py: data/gt_json/{train,val,test}. Only reads the source files.

Output (refuses to run if the folder already exists):
    <out>/<session>/<mass>g.jpg     mass zero-padded, so the files sort by mass
    <out>/excluded/<mass>g.jpg      below --min_mass_g, not used by the data_min120 models

    ./run.sh analysis/make_cluster_sheets.py
"""
import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict

import cv2
import numpy as np

FILE_NAME = re.compile(r"^(\d+)_(\d{4}-\d{2}-\d{2})-(\d{2}-\d{2}-\d{2})")
SRC_SPLITS = ["train", "val", "test"]
TILE = 320
COLUMNS = 6
TITLE_HEIGHT = 50
CAPTION_HEIGHT = 26
PADDING = 25


def load_clusters():
    """-> {(mass_g, session): [(time, image_path, [x0, y0, x1, y1] or None), ...]}, sorted by time."""
    clusters = defaultdict(list)
    for split in SRC_SPLITS:
        with open(os.path.join("data", "gt_json", split, split + ".json")) as fh:
            coco = json.load(fh)
        boxes = defaultdict(list)
        for annotation in coco["annotations"]:
            x, y, w, h = annotation["bbox"]
            boxes[annotation["image_id"]].append((x, y, x + w, y + h))
        for image in coco["images"]:
            name = os.path.basename(image["file_name"])
            match = FILE_NAME.match(name)
            found = boxes.get(image["id"])
            box = [min(b[0] for b in found), min(b[1] for b in found),
                   max(b[2] for b in found), max(b[3] for b in found)] if found else None
            clusters[(int(match.group(1)), match.group(2))].append(
                (match.group(3), os.path.join("data", "images", split, name), box))
    return {key: sorted(frames, key=lambda f: (f[0], f[1])) for key, frames in clusters.items()}


def crop_tile(frame):
    """Crop around the box, scaled to fit TILE x TILE on black, with the capture time below it."""
    time, path, box = frame
    image = cv2.imread(path)
    if image is None:
        raise SystemExit(f"Cannot read {path}")
    height, width = image.shape[:2]
    if box is None:
        crop, time = image, time + " NO BOX"
    else:
        crop = image[max(0, int(box[1]) - PADDING):min(height, int(box[3]) + PADDING),
                     max(0, int(box[0]) - PADDING):min(width, int(box[2]) + PADDING)]
    scale = TILE / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    tile = np.zeros((TILE + CAPTION_HEIGHT, TILE, 3), np.uint8)
    top, left = (TILE - crop.shape[0]) // 2, (TILE - crop.shape[1]) // 2
    tile[top:top + crop.shape[0], left:left + crop.shape[1]] = crop
    tile[TILE:] = 255
    cv2.putText(tile, time, (8, TILE + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return tile


def make_sheet(title, frames):
    tiles = [crop_tile(f) for f in frames]
    n_rows = math.ceil(len(tiles) / COLUMNS)
    tiles += [np.full_like(tiles[0], 255)] * (n_rows * COLUMNS - len(tiles))
    grid = np.vstack([np.hstack(tiles[r * COLUMNS:(r + 1) * COLUMNS]) for r in range(n_rows)])
    header = np.full((TITLE_HEIGHT, grid.shape[1], 3), 255, np.uint8)
    cv2.putText(header, title, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([header, grid])


def main():
    parser = argparse.ArgumentParser(description="One image sheet per cluster")
    parser.add_argument("--min_mass_g", type=int, default=120)
    parser.add_argument("--counts", default="analysis/tomato_counts.csv")
    parser.add_argument("--out", default="/mnt/md0/blagicab/data_inspect/cluster_sheets")
    args = parser.parse_args()

    if os.path.exists(args.out):
        raise SystemExit(f"Output already exists: {args.out}")

    with open(args.counts, newline="") as fh:
        ids = {(int(r["mass_g"]), r["session"]): r["id"] for r in csv.DictReader(fh)}

    clusters = load_clusters()
    for (mass, session), frames in sorted(clusters.items()):
        excluded = mass < args.min_mass_g
        cluster_id = "" if excluded else f"#{ids[(mass, session)]}   "
        note = "   EXCLUDED" if excluded else ""
        title = f"{cluster_id}{mass} g   {session}   {len(frames)} frames{note}"
        folder = os.path.join(args.out, "excluded" if excluded else session)
        os.makedirs(folder, exist_ok=True)
        cv2.imwrite(os.path.join(folder, f"{mass:04d}g.jpg"), make_sheet(title, frames),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    no_box = sum(f[2] is None for frames in clusters.values() for f in frames)
    print(f"{len(clusters)} sheets, {sum(len(f) for f in clusters.values())} frames, {no_box} without a box -> {args.out}")


if __name__ == "__main__":
    main()
