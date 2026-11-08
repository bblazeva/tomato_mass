"""
Look inside the trained networks: what do the depth filters do, and what does the mass head know?

A. First-layer depth filters (from the checkpoints). DC share = |sum of weights| / sum of |weights|. Near 0: the filter
   detects local patterns (edges) and ignores a constant shift in depth. Near 1: it responds to absolute values.
B. Probes: simple regressors on the mass head's features, from each leave-one-session-out model.
     head input = the ROIAlign features (14x14x256, averaged over space); head final = the last 128-d layer before the mass.
   "known" = cross-validated by cluster within the sessions the model trained on; "unseen" = fitted on those
   sessions, tested on the held-out one. A probe shows the information is present, not that the head uses it.
C. Do the head's features change when the cluster looks larger (image zoom), or when depth values change? Compared
   with natural differences: another image of the same cluster, and another cluster of the same session.
   If a size change alters the features less than natural variation, the head can hardly react to size.
D. Does adding real area to the head's final features help mass on an unseen session? (annotated masks; a preview only)

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=analysis ./run.sh analysis/look_inside.py --extract lab 2023
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=analysis ./run.sh analysis/look_inside.py --extract home
  PYTHONPATH=analysis ./run.sh analysis/look_inside.py --report

Writes results/eval/look_inside/{<model>.npz, report.txt}
"""
import argparse
import csv
import glob
import os
from argparse import Namespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import utils_detection
from dataset_audit import spearman
from depth_geometry_test import apply_condition, apply_input, base_image
from evaluate import build_eval_model, input_normalization
from tomato_count_analysis import cluster_key

OUT = "results/eval/look_inside"
MODELS = {"lab": "2025-03-28", "home": "2025-07-13", "2023": "2023-12-01"}
MODES = ["rgbd", "depth_only"]
MAX_PER_CLUSTER = 10
SENSITIVITY_PER_CLUSTER = 4
CONDITIONS = ["image zoom", "depth x1.25", "consistent zoom"]
ALPHAS = [1, 10, 100, 1000, 10000]
TARGETS = [("mass", "log_mass"), ("real area", "log_area"), ("distance", "log_distance"), ("pixel area", "log_px"),
           ("tomato count", "log_count"), ("tomato size", "log_fruit")]


def load_labels():
    inventory = {}
    with open("analysis/cluster_inventory.csv") as fh:
        for r in csv.DictReader(fh):
            inventory[r["key"]] = dict(type=r["type"], count=int(r["n_tomatoes"]), mass=int(r["mass_g"]))
    features = {}
    with open("analysis/features.csv") as fh:
        for r in csv.DictReader(fh):
            features[r["file_name"]] = (float(r["area_m2"]), float(r["area_px"]), float(r["mean_depth_m"]))
    return inventory, features


def evenly(items, n):
    items = sorted(items, key=lambda r: r["file_name"])
    return [items[i] for i in np.unique(np.linspace(0, len(items) - 1, n).round().astype(int))]


class Capture:
    """Keeps the head's input and last hidden layer for every detection of the last forward pass."""

    def __init__(self, model):
        head = model.roi_heads.mass_head
        head.register_forward_pre_hook(self._input)
        head.fc1.register_forward_hook(self._final)
        self.roi = self.final = None

    def _input(self, _, inputs):
        self.roi = inputs[0].detach().cpu()

    def _final(self, _, __, output):
        self.final = F.relu(output).detach().cpu()


def forward(model, capture, x):
    capture.roi = capture.final = None
    with torch.no_grad():
        inst = model([{"image": torch.from_numpy(x.transpose(2, 0, 1)).float(),
                       "height": x.shape[0], "width": x.shape[1]}])[0]["instances"].to("cpu")
    if len(inst) == 0 or capture.final is None:
        return None
    top = int(inst.scores.argmax())
    return capture.roi[top].numpy(), capture.final[top].numpy(), float(inst.pred_mass[top].item())


def extract(name):
    session = MODELS[name]
    inventory, features = load_labels()
    run_dir = f"output/loso_{name}_20k"
    mean, std, _ = input_normalization(Namespace(pixel_std=None), run_dir)
    model = build_eval_model(os.path.join(run_dir, "model_final.pth"), mean, std, 0.7, "./data_min120/")
    capture = Capture(model)

    groups = {}
    for record in utils_detection.get_tomato_dataset_dicts("./data_min120/", "all"):
        groups.setdefault(cluster_key(os.path.basename(record["file_name"])), []).append(record)

    rows = {m: [] for m in MODES}
    for key, records in sorted(groups.items()):
        for record in evenly(records, MAX_PER_CLUSTER):
            image, invalid = base_image(cv2.imread(record["file_name"]), np.load(record["depth_file"]))
            file_name = os.path.basename(record["file_name"])
            area, px, distance = features[file_name]
            label = inventory[key]
            for mode in MODES:
                out = forward(model, capture, apply_input(image, invalid, mode))
                if out is not None:
                    rows[mode].append(dict(
                        key=key, unseen=session in file_name, session=file_name.split("_")[1][:10], type=label["type"],
                        log_mass=np.log(label["mass"]), log_area=np.log(area), log_distance=np.log(distance),
                        log_px=np.log(px), log_count=np.log(label["count"]), log_fruit=np.log(label["mass"] / label["count"]),
                        roi=out[0].mean(axis=(1, 2)), final=out[1], pred=out[2]))
        print(f"{name}: {key}", flush=True)

    sensitivity = {}
    for mode in MODES:
        records = [r for k, v in sorted(groups.items()) for r in evenly(v, SENSITIVITY_PER_CLUSTER)
                   if session in r["file_name"]]
        base, changed = [], {c: [] for c in CONDITIONS}
        keys = []
        for record in records:
            image, invalid = base_image(cv2.imread(record["file_name"]), np.load(record["depth_file"]))
            ref = forward(model, capture, apply_input(image, invalid, mode))
            variants = {}
            for condition in CONDITIONS:
                x, inv = apply_condition(image, invalid, record["annotations"], condition)
                variants[condition] = forward(model, capture, apply_input(x, inv, mode))
            if ref is None or any(v is None for v in variants.values()):
                continue
            base.append(ref)
            keys.append(cluster_key(os.path.basename(record["file_name"])))
            for condition in CONDITIONS:
                changed[condition].append(variants[condition])
        sensitivity[mode] = dict(keys=keys, base=base, changed=changed)

    np.savez_compressed(os.path.join(OUT, f"{name}.npz"), rows=np.array([rows], dtype=object),
                        sensitivity=np.array([sensitivity], dtype=object))


def section_filters(lines):
    def conv1(path):
        checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint.get("model", checkpoint)
        for key in ("backbone.bottom_up.stem.conv1.weight", "backbone.bottom_up.res1.conv1.weight"):
            if key in state:
                value = state[key]
                return value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)
        return None

    def dc_share(w):
        return float(np.mean(np.abs(w.sum(axis=(1, 2))) / (np.abs(w).sum(axis=(1, 2)) + 1e-9)))

    lines.append("A. First-layer filters (64 filters; channels B, G, R, depth)")
    lines.append(f"  {'model':<26}{'DC share RGB':>13}{'DC share depth':>16}{'depth/RGB norm':>16}{'depth vs start (cosine)':>25}")
    start = conv1("weights/mask_rcnn_R50_FPN_3x_RGBD.pth")
    runs = [("start weights", "weights/mask_rcnn_R50_FPN_3x_RGBD.pth")]
    runs += [(f"loso_{n}", f"output/loso_{n}_20k/model_final.pth") for n in MODELS]
    runs += [("CV fold 0", "output/cv5_fold0_20k/model_final.pth"), ("model 5 (zoom)", "output/s0713_zoom_drop05/model_final.pth"),
             ("model 4.2 (no zoom)", "output/clean_baseline_v1_5conv_aug_fix_noLF_norm_stip_drop05/model_final.pth")]
    for label, path in runs:
        w = conv1(path)
        if w is None:
            lines.append(f"  {label:<26} not found")
            continue
        rgb, depth = w[:, :3], w[:, 3]
        cosine = float((depth * start[:, 3]).sum() / (np.linalg.norm(depth) * np.linalg.norm(start[:, 3]) + 1e-9)) if start is not None else float("nan")
        lines.append(f"  {label:<26}{dc_share(rgb.reshape(-1, 7, 7)):>13.2f}{dc_share(depth):>16.2f}"
                     f"{np.linalg.norm(depth) / (np.linalg.norm(rgb) / np.sqrt(3)):>16.2f}{cosine:>25.2f}")


def best_alpha(X, y, groups):
    best, best_score = ALPHAS[0], -np.inf
    n_splits = min(5, len(set(groups)))
    for alpha in ALPHAS:
        predicted = np.zeros(len(y))
        for train, test in GroupKFold(n_splits).split(X, y, groups):
            predicted[test] = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X[train], y[train]).predict(X[test])
        score = 1 - ((y - predicted) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        if score > best_score:
            best, best_score = alpha, score
    return best, best_score


def rank_over_clusters(keys, y, predicted):
    truth, pred = {}, {}
    for k, a, b in zip(keys, y, predicted):
        truth.setdefault(k, []).append(a)
        pred.setdefault(k, []).append(b)
    names = list(truth)
    return spearman([np.mean(truth[k]) for k in names], [np.mean(pred[k]) for k in names])


def section_probes(models, lines):
    lines.extend(["", "B. Probes on the head's features. known = R2 by cluster within trained sessions; unseen = R2 and "
                      "rank over clusters on the held-out session"])
    for name, data in models.items():
        rows = data["rows"]
        for mode in MODES:
            for layer, title in (("final", "head final"), ("roi", "head input")):
                r = rows[mode]
                train = [x for x in r if not x["unseen"]]
                test = [x for x in r if x["unseen"]]
                Xa, Xb = np.array([x[layer] for x in train]), np.array([x[layer] for x in test])
                groups = np.array([x["key"] for x in train])
                lines.append(f"  held out {name}, {mode}, {title}: {len(train)} train / {len(test)} test images")
                for label, target in TARGETS:
                    ya, yb = np.array([x[target] for x in train]), np.array([x[target] for x in test])
                    alpha, known = best_alpha(Xa, ya, groups)
                    predicted = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(Xa, ya).predict(Xb)
                    unseen = 1 - ((yb - predicted) ** 2).sum() / ((yb - yb.mean()) ** 2).sum()
                    lines.append(f"    {label:<14} known R2 {known:+6.2f}   unseen R2 {unseen:+7.2f}   "
                                 f"unseen rank {rank_over_clusters([x['key'] for x in test], yb, predicted):+.2f}")
                types = np.array([x["type"] == "cherry" for x in train])
                if len(set(types)) > 1:
                    correct = np.zeros(len(types))
                    for tr, te in GroupKFold(min(5, len(set(groups)))).split(Xa, types, groups):
                        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000)).fit(Xa[tr], types[tr])
                        correct[te] = clf.predict(Xa[te]) == types[te]
                    test_types = np.array([x["type"] == "cherry" for x in test])
                    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000)).fit(Xa, types)
                    lines.append(f"    cherry/normal  known accuracy {correct.mean():.2f}   unseen accuracy {np.mean(clf.predict(Xb) == test_types):.2f}")
                sessions = np.array([x["session"] for x in train])
                correct = np.zeros(len(sessions))
                for tr, te in GroupKFold(min(5, len(set(groups)))).split(Xa, sessions, groups):
                    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000)).fit(Xa[tr], sessions[tr])
                    correct[te] = clf.predict(Xa[te]) == sessions[te]
                lines.append(f"    which session  known accuracy {correct.mean():.2f} (2 sessions; majority "
                             f"{max(np.mean(sessions == s) for s in set(sessions)):.2f})")


def section_sensitivity(models, lines):
    lines.extend(["", "C. Feature change caused by a change in the input, relative to natural differences (held-out session images).",
                  "   Ratio < 1: the change alters the features less than another view of the same cluster does.",
                  "   Numbers: median |change| / |same cluster, other image|, then / |other cluster|; last: median prediction factor"])
    for name, data in models.items():
        for mode in MODES:
            s = data["sensitivity"][mode]
            if len(s["base"]) < 3:
                continue
            keys = s["keys"]
            for layer, index, title in (("head input", 0, "input"), ("head final", 1, "final")):
                base = np.array([b[index].ravel() for b in s["base"]])
                same, other = [], []
                for i, key in enumerate(keys):
                    d = np.linalg.norm(base - base[i], axis=1)
                    same_mask = np.array([k == key for k in keys])
                    same_mask[i] = False
                    other_mask = np.array([k != key for k in keys])
                    same.append(np.median(d[same_mask]) if same_mask.any() else np.nan)
                    other.append(np.median(d[other_mask]) if other_mask.any() else np.nan)
                same, other = np.array(same), np.array(other)
                for condition in CONDITIONS:
                    changed = np.array([c[index].ravel() for c in s["changed"][condition]])
                    delta = np.linalg.norm(changed - base, axis=1)
                    factor = np.exp(np.median([c[2] - b[2] for c, b in zip(s["changed"][condition], s["base"])]))
                    lines.append(f"  held out {name:<5}{mode:<11}{title:<6}{condition:<16} "
                                 f"same-cluster x{np.nanmedian(delta / same):.2f}   other-cluster x{np.nanmedian(delta / other):.2f}   pred x{factor:.2f}")


def section_add_area(models, lines):
    lines.extend(["", "D. Mass on the unseen session from: head final features; real area alone; both (ridge; MAPE %, rank over clusters)"])
    for name, data in models.items():
        for mode in MODES:
            r = data["rows"][mode]
            train = [x for x in r if not x["unseen"]]
            test = [x for x in r if x["unseen"]]
            groups = np.array([x["key"] for x in train])
            ya, yb = np.array([x["log_mass"] for x in train]), np.array([x["log_mass"] for x in test])
            parts = {
                "features": (np.array([x["final"] for x in train]), np.array([x["final"] for x in test])),
                "real area": (np.array([[x["log_area"]] for x in train]), np.array([[x["log_area"]] for x in test])),
            }
            parts["both"] = tuple(np.hstack([parts["features"][i], parts["real area"][i]]) for i in (0, 1))
            cells = ""
            for label, (Xa, Xb) in parts.items():
                alpha, _ = best_alpha(Xa, ya, groups)
                predicted = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(Xa, ya).predict(Xb)
                mape = np.mean(np.abs(np.exp(predicted) - np.exp(yb)) / np.exp(yb)) * 100
                cells += f"   {label} {mape:6.1f}% ({rank_over_clusters([x['key'] for x in test], yb, predicted):+.2f})"
            lines.append(f"  held out {name:<5}{mode:<11}{cells}")


def report():
    lines = []
    section_filters(lines)
    models = {}
    for path in sorted(glob.glob(os.path.join(OUT, "*.npz"))):
        data = np.load(path, allow_pickle=True)
        models[os.path.basename(path)[:-4]] = dict(rows=data["rows"][0], sensitivity=data["sensitivity"][0])
    section_probes(models, lines)
    section_sensitivity(models, lines)
    section_add_area(models, lines)
    text = "\n".join(lines)
    with open(os.path.join(OUT, "report.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)


def main():
    parser = argparse.ArgumentParser(description="Look inside the trained networks")
    parser.add_argument("--extract", nargs="+", choices=list(MODELS), default=[])
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for name in args.extract:
        extract(name)
    if args.report:
        report()


if __name__ == "__main__":
    main()
