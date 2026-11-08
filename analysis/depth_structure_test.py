"""
Does the model read the cluster's structure from the depth map? Two tests, editing depth only inside known regions.

A. Silhouette test (depth-only input, cross-validation models):
     real          the real depth map
     cluster only  real depth on the cluster, background neutral
     silhouette    flat depth on the cluster (its own median distance), background neutral: only the outline is left
   Comparing silhouette with cluster only shows whether surface detail matters. Comparing cluster only with real
   shows the effect of removing the depth background.

B. Pasted-depth test (RGB-D and depth-only). Only the cluster's depth is replaced, in its own place:
     paste other   depth of a DIFFERENT cluster, resized to this cluster's box, scaled to this cluster's median distance
     paste same    depth of ANOTHER IMAGE OF THE SAME cluster, handled the same way (the control)
   Everything else stays real, so position and distance do not change. If the model reads the cluster from depth, the
   prediction should move towards the other cluster's mass. Pull = slope of log(pasted / real prediction) on
   log(other mass / own mass): 1 = fully, 0 = not at all. Frames where the top detection leaves the cluster (box IoU
   with the annotation < 0.5) are counted and left out, as they say nothing about mass.

Annotated masks define the cluster. The partner comes from the same split, so the model has not seen it either,
and from the same session when possible. Runs on up to MAX_PER_CLUSTER evenly spaced images per cluster.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=analysis ./run.sh analysis/depth_structure_test.py --folds 0 2 4 --run s0713_zoom_drop05
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=analysis ./run.sh analysis/depth_structure_test.py --folds 1 3 --run s0713_zoom_drop065
  PYTHONPATH=analysis ./run.sh analysis/depth_structure_test.py --report

Writes results/eval/depth_structure_test/{<tag>.csv, report.txt, examples.jpg}
"""
import argparse
import csv
import glob
import os
from argparse import Namespace
from collections import defaultdict

import cv2
import numpy as np
import torch

import utils_detection
from depth_geometry_test import apply_input, base_image
from evaluate import build_eval_model, input_normalization
from tomato_count_analysis import cluster_key
from train import DEPTH_MEAN_MM

OUT = "results/eval/depth_structure_test"
MAX_PER_CLUSTER = 12
IOU_VALID = 0.5
CONDITIONS = {"depth_only": ["real", "cluster only", "silhouette", "paste other", "paste same"],
              "rgbd": ["real", "paste other", "paste same"]}
BOOTSTRAP_SEED = 0
BOOTSTRAP_SAMPLES = 3000


def fill_mask(annotations, height, width):
    mask = np.zeros((height, width), np.uint8)
    for annotation in annotations:
        for polygon in annotation["segmentation"]:
            cv2.fillPoly(mask, [np.array(polygon, np.float32).reshape(-1, 2).round().astype(np.int32)], 1)
    return mask.astype(bool)


def load_frame(record):
    """-> 4-channel image (invalid depth = 0), cluster mask, integer box (x, y, w, h)."""
    bgr = cv2.imread(record["file_name"])
    image, _ = base_image(bgr, np.load(record["depth_file"]))
    mask = fill_mask(record["annotations"], bgr.shape[0], bgr.shape[1])
    x, y, w, h = record["annotations"][0]["bbox"]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(bgr.shape[1], int(np.ceil(x + w))), min(bgr.shape[0], int(np.ceil(y + h)))
    return image, mask, (x0, y0, x1 - x0, y1 - y0)


def median_depth(depth, mask):
    values = depth[mask & (depth > 0)]
    return float(np.median(values)) if values.size else DEPTH_MEAN_MM


def paste_depth(image, mask, box, other_image, other_mask, other_box):
    """Replace the cluster's depth by another cluster's, resized to this box, at this cluster's median distance."""
    out = image.copy()
    depth = out[:, :, 3]
    x, y, w, h = box
    px, py, pw, ph = other_box
    own = median_depth(depth, mask)

    # remove this cluster: fill with the surrounding depth, from a margin around the box
    margin = np.zeros_like(mask)
    margin[max(0, y - 20):y + h + 20, max(0, x - 20):x + w + 20] = True
    margin &= ~cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    background = median_depth(depth, margin)
    depth[mask] = background

    crop_depth = cv2.resize(other_image[:, :, 3][py:py + ph, px:px + pw], (w, h), interpolation=cv2.INTER_NEAREST)
    crop_mask = cv2.resize(other_mask[py:py + ph, px:px + pw].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    scale = own / median_depth(crop_depth, crop_mask)
    window = depth[y:y + h, x:x + w]
    window[crop_mask] = crop_depth[crop_mask] * scale   # invalid pixels stay 0
    return out, out[:, :, 3] == 0


def apply_condition(condition, image, mask, box, other, same):
    out = image.copy()
    depth = out[:, :, 3]
    if condition == "real":
        return out, depth == 0
    if condition == "cluster only":
        depth[~mask] = DEPTH_MEAN_MM
        return out, depth == 0
    if condition == "silhouette":
        own = median_depth(depth, mask)
        depth[:] = DEPTH_MEAN_MM
        depth[mask] = own
        return out, depth == 0
    source = other if condition == "paste other" else same
    return paste_depth(image, mask, box, source[0], source[1], source[2])


def box_iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / union if union > 0 else 0.0


def run_model(tag, run_dir, dataset, split, examples=None):
    mean, std, _ = input_normalization(Namespace(pixel_std=None), run_dir)
    model = build_eval_model(os.path.join(run_dir, "model_final.pth"), mean, std, 0.7, dataset)
    groups = defaultdict(list)
    for record in utils_detection.get_tomato_dataset_dicts(dataset, split):
        groups[cluster_key(os.path.basename(record["file_name"]))].append(record)
    for records in groups.values():
        records.sort(key=lambda r: r["file_name"])

    rng = np.random.RandomState(BOOTSTRAP_SEED)
    rows = []
    for key in sorted(groups):
        records = groups[key]
        session = os.path.basename(records[0]["file_name"]).split("_")[1][:10]
        picks = np.unique(np.linspace(0, len(records) - 1, MAX_PER_CLUSTER).round().astype(int))
        for index in picks:
            record = records[int(index)]
            name = os.path.basename(record["file_name"])
            mass = record["cluster_mass"]
            candidates = [k for k in groups if groups[k][0]["cluster_mass"] != mass]
            same_session = [k for k in candidates if session in groups[k][0]["file_name"]]
            other_key = (same_session or candidates)[rng.randint(len(same_session or candidates))]
            other_record = groups[other_key][rng.randint(len(groups[other_key]))]
            same_index = (int(index) + len(records) // 2) % len(records)
            same_record = records[same_index] if len(records) > 1 else record

            image, mask, box = load_frame(record)
            other, same = load_frame(other_record), load_frame(same_record)
            x, y, w, h = record["annotations"][0]["bbox"]
            gt_box = [x, y, x + w, y + h]
            if examples is not None and session not in examples:
                examples[session] = {c: apply_condition(c, image, mask, box, other, same)[0][:, :, 3].copy()
                                     for c in CONDITIONS["depth_only"]}

            for mode, conditions in CONDITIONS.items():
                for condition in conditions:
                    changed, invalid = apply_condition(condition, image, mask, box, other, same)
                    x_input = apply_input(changed, invalid, mode)
                    with torch.no_grad():
                        inst = model([{"image": torch.from_numpy(x_input.transpose(2, 0, 1)).float(),
                                       "height": x_input.shape[0], "width": x_input.shape[1]}])[0]["instances"].to("cpu")
                    row = {"tag": tag, "file": name, "key": key, "session": session, "mass_kg": mass, "other_mass_kg": other_record["cluster_mass"],
                           "input": mode, "condition": condition, "pred_kg": "", "box_iou": ""}
                    if len(inst):
                        top = int(inst.scores.argmax())
                        row["pred_kg"] = float(np.exp(inst.pred_mass[top].item()))
                        row["box_iou"] = round(box_iou(inst.pred_boxes.tensor[top].tolist(), gt_box), 3)
                    rows.append(row)
        print(f"{tag}: {key}", flush=True)

    with open(os.path.join(OUT, f"{tag}.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_examples(examples, path):
    rows = []
    for session, panels in sorted(examples.items()):
        real = panels["real"]
        low, high = np.percentile(real[real > 0], [2, 98])
        tiles = []
        for condition, depth in panels.items():
            scaled = np.clip((depth - low) / (high - low) * 255, 0, 255).astype(np.uint8)
            tile = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
            tile[depth == 0] = 0
            tile = cv2.resize(tile, (384, 216), interpolation=cv2.INTER_NEAREST)
            cv2.putText(tile, f"{session} {condition}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            tiles.append(tile)
        rows.append(np.hstack(tiles))
    cv2.imwrite(path, np.vstack(rows))


def valid(row):
    return row["pred_kg"] != "" and float(row["box_iou"]) >= IOU_VALID


def bootstrap_mean(values):
    values = np.array(values)
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    samples = [rng.choice(values, len(values)).mean() for _ in range(BOOTSTRAP_SAMPLES)]
    return values.mean(), np.percentile(samples, 2.5), np.percentile(samples, 97.5)


def section_silhouette(rows, lines):
    lines.append("A. Silhouette test: depth-only input, cross-validation models")
    data = defaultdict(dict)
    for r in rows:
        if r["input"] == "depth_only" and r["tag"].startswith("cv5"):
            data[r["file"]][r["condition"]] = r
    conditions = ["real", "cluster only", "silhouette"]
    for condition in conditions:
        chosen = [m[condition] for m in data.values()]
        none = sum(r["pred_kg"] == "" for r in chosen)
        off = sum(r["pred_kg"] != "" and float(r["box_iou"]) < IOU_VALID for r in chosen)
        lines.append(f"  {condition:<14} no detection {none:>3}, top detection off the cluster {off:>3}, of {len(chosen)} images")
    common = {f: m for f, m in data.items() if all(valid(m[c]) for c in conditions)}
    lines.append(f"  MAPE on the {len(common)} images valid in all three:")

    def table(label, items):
        cells = ""
        for condition in conditions:
            err = [abs(float(m[condition]["pred_kg"]) - float(m[condition]["mass_kg"])) / float(m[condition]["mass_kg"]) * 100 for m in items]
            per = defaultdict(list)
            for m, e in zip(items, err):
                per[m[condition]["key"]].append(e)
            cells += f"{np.mean(err):>9.1f}{np.mean([np.mean(v) for v in per.values()]):>9.1f}"
        lines.append(f"    {label:<14}{len(items):>6}{cells}")

    lines.append(f"    {'':<14}{'images':>6}" + "".join(f"{c:>18}" for c in conditions) + "   (per image, per cluster)")
    table("all", list(common.values()))
    for session in ("2025-07-13", "2023-12-01", "2025-03-28"):
        table(session, [m for m in common.values() if m["real"]["session"] == session])

    per = defaultdict(lambda: defaultdict(list))
    for m in common.values():
        for condition in conditions:
            r = m[condition]
            per[r["key"]][condition].append(abs(float(r["pred_kg"]) - float(r["mass_kg"])) / float(r["mass_kg"]) * 100)
    for a, b in (("cluster only", "real"), ("silhouette", "cluster only"), ("silhouette", "real")):
        mean, low, high = bootstrap_mean([np.mean(v[a]) - np.mean(v[b]) for v in per.values()])
        lines.append(f"  {a} - {b}, per cluster: {mean:+.1f} points ({low:+.1f} to {high:+.1f})")


def section_paste(rows, lines):
    lines.extend(["", "B. Pasted-depth test. Pull: slope of log(pasted / real prediction) on log(other mass / own mass), 95% over clusters.",
                  "   Change: median |pasted / real - 1| per image. Box moved = top detection off the cluster after pasting."])
    groups = [("cross-validation", lambda t: t.startswith("cv5")), ("model 5, Home val", lambda t: t == "s0713_zoom_drop05"),
              ("7a, Home val", lambda t: t == "s0713_zoom_drop065")]
    for label, keep in groups:
        for mode in ("rgbd", "depth_only"):
            data = defaultdict(dict)
            for r in rows:
                if keep(r["tag"]) and r["input"] == mode:
                    data[r["file"]][r["condition"]] = r
            usable = [m for m in data.values() if valid(m["real"]) and valid(m["paste other"]) and valid(m["paste same"])]
            moved = sum(not valid(m["paste other"]) for m in data.values() if valid(m["real"]))
            moved_same = sum(not valid(m["paste same"]) for m in data.values() if valid(m["real"]))
            if len(usable) < 5:
                lines.append(f"  {label:<18}{mode:<11} too few valid images ({len(usable)})")
                continue
            ratio = np.array([np.log(float(m["paste other"]["pred_kg"]) / float(m["real"]["pred_kg"])) for m in usable])
            control = np.array([np.log(float(m["paste same"]["pred_kg"]) / float(m["real"]["pred_kg"])) for m in usable])
            target = np.array([np.log(float(m["real"]["other_mass_kg"]) / float(m["real"]["mass_kg"])) for m in usable])
            keys = np.array([m["real"]["key"] for m in usable])
            unique = sorted(set(keys))
            rng = np.random.RandomState(BOOTSTRAP_SEED)
            samples = []
            for _ in range(BOOTSTRAP_SAMPLES):
                chosen = rng.choice(unique, len(unique))
                idx = np.concatenate([np.where(keys == k)[0] for k in chosen])
                if np.ptp(target[idx]) > 0:
                    samples.append(np.polyfit(target[idx], ratio[idx], 1)[0])
            pull = np.polyfit(target, ratio, 1)[0]
            low, high = np.percentile(samples, [2.5, 97.5])
            big = np.abs(np.exp(ratio) - 1) > 0.25
            direction = np.mean(np.sign(ratio[big]) == np.sign(target[big])) * 100 if big.any() else float("nan")
            lines.append(f"  {label:<18}{mode:<11} n={len(usable):>3}  pull {pull:+.2f} ({low:+.2f} to {high:+.2f})   "
                         f"change: other {np.median(np.abs(np.exp(ratio) - 1)) * 100:5.1f}%, same (control) {np.median(np.abs(np.exp(control) - 1)) * 100:5.1f}%   "
                         f"over 25%: {big.sum()}, towards other's mass {direction:.0f}%   box moved: other {moved}, same {moved_same}")


def report():
    rows = []
    for path in sorted(glob.glob(os.path.join(OUT, "*.csv"))):
        with open(path) as fh:
            rows += list(csv.DictReader(fh))
    lines = [f"{len({r['file'] for r in rows})} images, {len({r['key'] for r in rows})} clusters", ""]
    section_silhouette(rows, lines)
    section_paste(rows, lines)
    text = "\n".join(lines)
    with open(os.path.join(OUT, "report.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)


def main():
    parser = argparse.ArgumentParser(description="Test whether the model reads cluster structure from depth")
    parser.add_argument("--folds", type=int, nargs="*", default=[])
    parser.add_argument("--run", nargs="*", default=[], help="Home-only runs, tested on data_s0713 val")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    examples = {}
    for fold in args.folds:
        run_model(f"cv5_fold{fold}", f"output/cv5_fold{fold}_20k", f"./data_min120/folds/fold_{fold}/", "val", examples)
    for run in args.run:
        run_model(run, os.path.join("output", run), "./data_s0713/", "val")
    if examples:
        save_examples(examples, os.path.join(OUT, f"examples_{'_'.join(map(str, args.folds))}.jpg"))
    if args.report:
        report()


if __name__ == "__main__":
    main()
