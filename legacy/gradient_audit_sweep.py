"""
GRADIENT AUDIT SWEEP — run the gradient audit across every saved checkpoint of a
run, so the modality/loss balance can be tracked *over training*, not just at the
final weights.

It reuses the exact internals of gradient_audit.py (same grouping, same RMS, same
two-pass det/mass backward, same input-mode ablation), so a single row here is
numerically identical to running gradient_audit.py on that checkpoint.

WHY THE BATCHES ARE FIXED ACROSS CHECKPOINTS (paired / repeated-measures design)
-------------------------------------------------------------------------------
The model is the only thing that should change between checkpoints. So we:
  * select a fixed set of images ONCE, STRATIFIED by mass bin exactly like the
    trainer (cluster_mass < 0.25 / < 0.50 / else), with a seeded RNG so the set
    is reproducible and representative (not just the alphabetically-first images);
  * keep only images with GT instances, so detectron2's PROPOSAL_APPEND_GT
    guarantees foreground ROIs and the fg-skip almost never fires, even for a
    bad early detector;
  * map them through val_mapper ONCE (no augmentation, no rgb_dropout) and cache
    the input tensors; the SAME tensors are reused for every checkpoint and mode.

WHY WE ALSO SEED EACH FORWARD
-----------------------------
Fixed inputs are not enough: the ROI head randomly SUBSAMPLES proposals every
forward (subsample_labels uses torch RNG), so each checkpoint would otherwise see
different fg/bg ROIs -> noise in the mass gradient. We seed deterministically per
batch index, so batch i gets identical proposal sampling at every checkpoint.

NORMALIZATION MUST MATCH TRAINING
---------------------------------
No config.yaml is dumped by train_new_clean.py, so PIXEL_MEAN/PIXEL_STD/FREEZE_AT
are CLI args. Pass the values the run actually trained with, or cross-checkpoint
numbers are contaminated. Defaults match the active values in gradient_audit.py.

Output: a tidy CSV with one row per (checkpoint_iteration, input_mode):
  checkpoint_iteration, input_mode,
  mass_det_backbone_res, mass_det_fpn,
  mass_stem_depth_over_rgb, mass_stem_rgb_abs, mass_stem_depth_abs,
  loss_det_total, loss_cls, loss_box_reg, loss_mask, loss_rpn_cls,
  loss_rpn_loc, loss_mass, n_used
(mass_stem_depth_over_rgb is NaN for depth_only, where RGB is 0 by construction.)

Run:
  LD_PRELOAD=$HOME/miniconda3/envs/tomato_mass/lib/libstdc++.so.6 \
    ~/miniconda3/envs/tomato_mass/bin/python3 gradient_audit_sweep.py \
      --run_dir output/s0713_zoom_drop065 --dataset_path ./data_s0713/ \
      --split val --n_batches 60 --mass_loss_weight 1.0 \
      --pixel_std 57.375 57.120 58.395 411.683 \
      --out output/s0713_zoom_drop065/gradient_audit_sweep.csv
"""
import os, re, csv, copy, glob, random, argparse
from collections import defaultdict

import numpy as np
import torch

from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.events import EventStorage

import utils_detection
from train_new_clean import val_mapper

# Reuse the EXACT audit internals so rows match gradient_audit.py one-to-one.
from gradient_audit import (
    DET_KEYS, MASS_KEYS, group_of, snapshot_grads, backward_one, apply_input_mode,
)

MODES = ["real_rgbd", "rgb_only", "depth_only"]

# Mirror the trainer's mass-bin stratification (train_new_clean.build_train_loader).
def mass_bin(m):
    if m < 0.25: return 0
    if m < 0.50: return 1
    return 2


def set_seed(s):
    """Make the ROI-head proposal subsampling reproducible for a given batch."""
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def build_cfg(model_path, dataset_path, split, pixel_mean, pixel_std, freeze_at):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.META_ARCHITECTURE = "RGBDGeneralizedRCNN"
    cfg.MODEL.PIXEL_MEAN = list(pixel_mean)
    cfg.MODEL.PIXEL_STD  = list(pixel_std)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_MASK_HEAD.LOSS_WEIGHT = 1.0
    cfg.MODEL.ROI_MASS_HEAD.MASS_LOSS_WEIGHT = 1.0
    cfg.MODEL.BACKBONE.FREEZE_AT = freeze_at
    cfg.MODEL.WEIGHTS = model_path
    cfg.DATASET_PATH = dataset_path

    name = f"TomatoDataset_{split}"
    if name not in DatasetCatalog.list():
        DatasetCatalog.register(
            name, lambda: utils_detection.get_tomato_dataset_dicts(dataset_path, split))
        MetadataCatalog.get(name).set(thing_classes=["tomato_bunch"])
    cfg.DATASETS.TRAIN = (name,)
    cfg.DATASETS.TEST  = (name,)
    return cfg


def iter_of(path):
    """model_0000999.pth -> 1000 (detectron2 saves the 0-indexed iter; +1 = steps done)."""
    m = re.search(r"model_(\d+)\.pth$", os.path.basename(path))
    return int(m.group(1)) + 1 if m else None


def find_checkpoints(run_dir, include_final):
    paths = glob.glob(os.path.join(run_dir, "model_[0-9]*.pth"))
    paths = sorted([p for p in paths if iter_of(p) is not None], key=iter_of)
    if include_final:
        fin = os.path.join(run_dir, "model_final.pth")
        if os.path.exists(fin):
            # place final one step after the last numbered checkpoint
            paths.append(fin)
    return paths


def select_stratified(dataset_name, n, seed):
    """Deterministic, mass-stratified set of GT-bearing images (representative,
    not alphabetically-first). Falls back to filling from leftovers if a bin is short."""
    dicts = [d for d in DatasetCatalog.get(dataset_name)
             if len(d.get("annotations", [])) >= 1 and d.get("cluster_mass") is not None]
    rng = np.random.default_rng(seed)

    bins = {0: [], 1: [], 2: []}
    for d in dicts:
        bins[mass_bin(d["cluster_mass"])].append(d)

    per = n // 3
    targets = [per + (1 if i < (n - per * 3) else 0) for i in range(3)]

    chosen, leftovers = [], []
    for b in range(3):
        pool = sorted(bins[b], key=lambda d: d["file_name"])
        perm = rng.permutation(len(pool))
        take = min(targets[b], len(pool))
        chosen += [pool[i] for i in perm[:take]]
        leftovers += [pool[i] for i in perm[take:]]

    if len(chosen) < n and leftovers:  # a bin was short -> top up from the rest
        rng.shuffle(leftovers)
        chosen += leftovers[: n - len(chosen)]

    counts = {b: sum(1 for d in chosen if mass_bin(d["cluster_mass"]) == b) for b in range(3)}
    print(f"Stratified fixed set: {len(chosen)} images | per-bin counts (small/med/large) = "
          f"{counts[0]}/{counts[1]}/{counts[2]}")
    return chosen


def build_fixed_inputs(dataset_name, n, seed, device):
    """Map the stratified image set ONCE and cache on-device tensors."""
    dicts = select_stratified(dataset_name, n, seed)
    batches = []
    for d in dicts:
        mapped = val_mapper(copy.deepcopy(d))   # val_mapper pops 'annotations'
        mapped["image"] = mapped["image"].to(device)
        if "instances" in mapped:
            mapped["instances"] = mapped["instances"].to(device)
        batches.append([mapped])
    return batches


def audit_checkpoint(model, fixed_batches, mode, mass_weight, pixel_mean, seed_base):
    """Two-pass audit over the fixed batch set for one input mode.
    Returns grad accumulators + averaged scalar losses."""
    det_groups, mass_groups = defaultdict(list), defaultdict(list)
    det_stem  = {"rgb": [], "depth": []}
    mass_stem = {"rgb": [], "depth": []}
    loss_vals = defaultdict(list)
    used = 0

    for i, batch in enumerate(fixed_batches):
        set_seed(seed_base + i)   # identical proposal subsampling at every checkpoint
        with EventStorage() as storage:
            inputs = apply_input_mode(batch, mode, pixel_mean)
            losses = model(inputs)
            if "loss_mass" not in losses:
                continue
            fg = storage.latest().get("roi_head/num_fg_samples", (0, 0))
            fg = fg[0] if isinstance(fg, tuple) else fg
            if fg is not None and fg < 1:
                continue
            for k, v in losses.items():
                loss_vals[k].append(float(v.detach().item()))
            _, (dg, ds) = backward_one(model, losses, DET_KEYS, 1.0, retain=True)
            _, (mg, ms) = backward_one(model, losses, MASS_KEYS, mass_weight, retain=False)
        for k, v in dg.items():  det_groups[k].append(v)
        for k, v in mg.items():  mass_groups[k].append(v)
        det_stem["rgb"].append(ds["rgb"]);   det_stem["depth"].append(ds["depth"])
        mass_stem["rgb"].append(ms["rgb"]);  mass_stem["depth"].append(ms["depth"])
        used += 1
    return used, det_groups, mass_groups, det_stem, mass_stem, loss_vals


def reduce_row(used, det_groups, mass_groups, det_stem, mass_stem, loss_vals, mode):
    def gmean(d, k):
        return float(np.mean(d[k])) if d.get(k) else 0.0

    def lmean(k):
        return float(np.mean(loss_vals[k])) if loss_vals.get(k) else 0.0

    m_rgb   = float(np.mean(mass_stem["rgb"]))   if mass_stem["rgb"]   else 0.0
    m_depth = float(np.mean(mass_stem["depth"])) if mass_stem["depth"] else 0.0

    row = {
        "mass_det_backbone_res":    gmean(mass_groups, "backbone_res") / (gmean(det_groups, "backbone_res") + 1e-12),
        "mass_det_fpn":             gmean(mass_groups, "fpn")          / (gmean(det_groups, "fpn") + 1e-12),
        # depth/rgb is meaningless for depth_only (RGB == 0 by construction)
        "mass_stem_depth_over_rgb": float("nan") if mode == "depth_only" else m_depth / (m_rgb + 1e-12),
        "mass_stem_rgb_abs":        m_rgb,
        "mass_stem_depth_abs":      m_depth,
        "loss_det_total":           sum(lmean(k) for k in DET_KEYS),
        "loss_cls":                 lmean("loss_cls"),
        "loss_box_reg":             lmean("loss_box_reg"),
        "loss_mask":                lmean("loss_mask"),
        "loss_rpn_cls":             lmean("loss_rpn_cls"),
        "loss_rpn_loc":             lmean("loss_rpn_loc"),
        "loss_mass":                lmean("loss_mass"),
        "n_used":                   used,
    }
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="Directory with model_*.pth checkpoints.")
    p.add_argument("--dataset_path", default="./data_s0713/")
    p.add_argument("--split", default="val")
    p.add_argument("--n_batches", type=int, default=60,
                   help="Number of fixed (stratified) images audited at EVERY checkpoint.")
    p.add_argument("--mass_loss_weight", type=float, default=1.0,
                   help="1.0 = intrinsic (does the task want depth?). Set to the run's "
                        "training weight for the 'effective' audit.")
    p.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the stratified image selection AND per-batch forward seeding.")
    # Normalization / freezing MUST match training (no config.yaml is dumped).
    p.add_argument("--pixel_mean", type=float, nargs=4,
                   default=[103.530, 116.280, 123.675, 915.043])
    p.add_argument("--pixel_std", type=float, nargs=4,
                   default=[57.375, 57.120, 58.395, 411.683])
    p.add_argument("--freeze_at", type=int, default=0)
    p.add_argument("--include_final", action="store_true",
                   help="Also audit model_final.pth (duplicates the last numbered checkpoint).")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out_csv = args.out or os.path.join(args.run_dir, "gradient_audit_sweep.csv")

    checkpoints = find_checkpoints(args.run_dir, args.include_final)
    if not checkpoints:
        raise SystemExit(f"No model_*.pth checkpoints found in {args.run_dir}")
    print("=" * 78)
    print(f"Run dir       : {args.run_dir}")
    print(f"Checkpoints   : {len(checkpoints)}  (iter {iter_of(checkpoints[0])} .. "
          f"{iter_of(checkpoints[-1]) if iter_of(checkpoints[-1]) else 'final'})")
    print(f"PIXEL_MEAN    : {args.pixel_mean}")
    print(f"PIXEL_STD     : {args.pixel_std}   <-- MUST match this run's training")
    print(f"FREEZE_AT     : {args.freeze_at}")
    print(f"mass_weight   : {args.mass_loss_weight}   n_batches: {args.n_batches}   seed: {args.seed}")
    print("=" * 78)

    # Build the model + cfg ONCE; only reload weights per checkpoint.
    cfg = build_cfg(checkpoints[0], args.dataset_path, args.split,
                    args.pixel_mean, args.pixel_std, args.freeze_at)
    device = cfg.MODEL.DEVICE
    model = build_model(cfg)
    model.train()  # train mode so forward() returns losses

    # Sanity: no parameter should fall outside the known groups (guards group_of/FPN).
    ungrouped = [n for n, _ in model.named_parameters() if group_of(n) == "other"]
    if ungrouped:
        print(f"WARNING: {len(ungrouped)} params grouped as 'other' (e.g. {ungrouped[:3]})")

    fixed_batches = build_fixed_inputs(cfg.DATASETS.TRAIN[0], args.n_batches, args.seed, device)

    fieldnames = ["checkpoint_iteration", "input_mode",
                  "mass_det_backbone_res", "mass_det_fpn",
                  "mass_stem_depth_over_rgb", "mass_stem_rgb_abs", "mass_stem_depth_abs",
                  "loss_det_total", "loss_cls", "loss_box_reg", "loss_mask",
                  "loss_rpn_cls", "loss_rpn_loc", "loss_mass", "n_used"]

    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for ckpt in checkpoints:
            it = iter_of(ckpt)
            label = it if it is not None else "final"
            DetectionCheckpointer(model).load(ckpt)
            model.train()
            for mode in args.modes:
                out = audit_checkpoint(model, fixed_batches, mode,
                                       args.mass_loss_weight, cfg.MODEL.PIXEL_MEAN, args.seed)
                row = reduce_row(*out, mode=mode)
                row.update({"checkpoint_iteration": label, "input_mode": mode})
                writer.writerow(row)
                fh.flush()
                print(f"[iter {str(label):>6} | {mode:10}] "
                      f"mass/det back={row['mass_det_backbone_res']:.3f} "
                      f"fpn={row['mass_det_fpn']:.3f} "
                      f"depth/rgb={row['mass_stem_depth_over_rgb']:.3f} "
                      f"loss_mass={row['loss_mass']:.4f} (n={row['n_used']})")

    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
