"""
GRADIENT AUDIT SWEEP — run the gradient audit across every saved checkpoint of a
run, so the modality/loss balance can be tracked *over training*, not just at the
final weights.

Purpose
-------
This script answers:

  1. Does loss_mass reach the shared backbone/FPN over training?
  2. Does loss_mass push the RGB or depth input stem more strongly?
  3. Do these quantities change across checkpoints?
  4. How do these patterns differ under real_rgbd / rgb_only / depth_only input?

Important interpretation
------------------------
This is a checkpoint-landscape diagnostic. It measures what gradients each loss
would produce at each frozen checkpoint on a fixed diagnostic subset.

It is NOT exactly the optimizer update that happened during training, because the
real update also included optimizer momentum, LR schedule, stochastic augmentation,
and the live batch sampled at that iteration.

Design choices
--------------
- Fixed diagnostic images across all checkpoints and input modes.
- Stratified by mass bin, mirroring the trainer:
    cluster_mass < 0.25 / < 0.50 / else
- Fixed RNG seed per batch index to reduce noise from Detectron2 ROI proposal
  subsampling.
- Uses val_mapper once and caches tensors, so there is no random augmentation,
  no RGB dropout, and no random zoom during the audit.
- Logs both ratios and absolute gradient magnitudes, so changes in ratios can be
  interpreted correctly.

Output
------
Main CSV:
  one row per checkpoint x input_mode

Subset CSV:
  selected fixed diagnostic samples

Metadata JSON:
  audit config and checkpoint list

Run example
-----------
LD_PRELOAD=$HOME/miniconda3/envs/tomato_mass/lib/libstdc++.so.6 \
  ~/miniconda3/envs/tomato_mass/bin/python3 gradient_audit_sweep.py \
    --run_dir output/s0713_zoom_drop05 \
    --dataset_path ./data_s0713/ \
    --split val \
    --n_batches 60 \
    --mass_loss_weight 1.0 \
    --pixel_std 57.375 57.120 58.395 411.683 \
    --out output/s0713_zoom_drop05/gradient_audit_sweep.csv
"""

import os
import re
import csv
import copy
import glob
import json
import random
import argparse
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

# Reuse your existing single-checkpoint audit internals.
# Keep gradient_audit.py in the same directory.
from gradient_audit import (
    DET_KEYS,
    MASS_KEYS,
    group_of,
    snapshot_grads,
    backward_one,
    apply_input_mode,
)

MODES = ["real_rgbd", "rgb_only", "depth_only"]


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def mass_bin(m):
    """Mirror trainer mass-bin stratification."""
    m = float(m)
    if m < 0.25:
        return 0
    if m < 0.50:
        return 1
    return 2


def set_seed(seed):
    """
    Make ROI-head proposal subsampling reproducible for a given batch index.

    Important:
    - Use the same seed for the same batch across checkpoints and input modes.
    - Do not include checkpoint iteration or input_mode in the seed if the goal is
      paired comparison.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def iter_of(path):
    """
    model_0000999.pth -> 1000
    Detectron2 checkpoint filenames are usually 0-indexed iteration numbers.
    """
    m = re.search(r"model_(\d+)\.pth$", os.path.basename(path))
    return int(m.group(1)) + 1 if m else None


def find_checkpoints(run_dir, include_final=False, final_iteration=None):
    """
    Returns a list of checkpoint records:
      {
        "path": ...,
        "label": ...,
        "iteration_numeric": ...
      }

    model_final.pth is optional because it often duplicates the last numbered checkpoint.
    """
    paths = glob.glob(os.path.join(run_dir, "model_[0-9]*.pth"))
    paths = sorted([p for p in paths if iter_of(p) is not None], key=iter_of)

    records = []
    for p in paths:
        it = iter_of(p)
        records.append({
            "path": p,
            "label": str(it),
            "iteration_numeric": it,
        })

    if include_final:
        final_path = os.path.join(run_dir, "model_final.pth")
        if os.path.exists(final_path):
            records.append({
                "path": final_path,
                "label": "final",
                "iteration_numeric": final_iteration if final_iteration is not None else "",
            })

    return records


def build_cfg(model_path, dataset_path, split, pixel_mean, pixel_std, freeze_at):
    """
    Build the model config.

    WARNING:
    PIXEL_MEAN, PIXEL_STD, and FREEZE_AT must match the run being audited.
    If these do not match training, the gradient audit is contaminated.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))

    cfg.MODEL.META_ARCHITECTURE = "RGBDGeneralizedRCNN"

    cfg.MODEL.PIXEL_MEAN = list(pixel_mean)
    cfg.MODEL.PIXEL_STD = list(pixel_std)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT = True

    cfg.MODEL.ROI_MASK_HEAD.LOSS_WEIGHT = 1.0

    # Set internal mass-head weight to 1.0 so loss_mass is raw.
    # The audit applies --mass_loss_weight explicitly in backward_one().
    cfg.MODEL.ROI_MASS_HEAD.MASS_LOSS_WEIGHT = 1.0

    cfg.MODEL.BACKBONE.FREEZE_AT = freeze_at
    cfg.MODEL.WEIGHTS = model_path

    cfg.DATASET_PATH = dataset_path

    name = f"TomatoDataset_{split}"
    if name not in DatasetCatalog.list():
        DatasetCatalog.register(
            name,
            lambda: utils_detection.get_tomato_dataset_dicts(dataset_path, split)
        )
        MetadataCatalog.get(name).set(thing_classes=["tomato_bunch"])

    cfg.DATASETS.TRAIN = (name,)
    cfg.DATASETS.TEST = (name,)

    return cfg


def summarize_param_groups(model):
    """Print number of tensors and parameters assigned to each audit group."""
    counts = defaultdict(int)
    numels = defaultdict(int)
    examples = defaultdict(list)

    for name, p in model.named_parameters():
        g = group_of(name)
        counts[g] += 1
        numels[g] += p.numel()
        if len(examples[g]) < 5:
            examples[g].append(name)

    print("\nParameter group sanity check:")
    for g in sorted(counts.keys()):
        print(f"  {g:14} tensors={counts[g]:4d}  params={numels[g]:10d}")
        if g == "other":
            print(f"    examples: {examples[g]}")

    if counts.get("other", 0) > 0:
        print("\nWARNING: Some parameters are grouped as 'other'.")
        print("If these are important backbone/FPN/head parameters, fix group_of() in gradient_audit.py.\n")


# ---------------------------------------------------------------------
# Fixed diagnostic subset
# ---------------------------------------------------------------------

def select_stratified(dataset_name, n, seed):
    """
    Deterministic, mass-stratified set of GT-bearing images.

    This mirrors the trainer's mass bins:
      small  < 0.25 kg
      medium < 0.50 kg
      large  >= 0.50 kg

    If a bin is short, fill from leftovers.
    """
    dicts = [
        d for d in DatasetCatalog.get(dataset_name)
        if len(d.get("annotations", [])) >= 1 and d.get("cluster_mass") is not None
    ]

    if not dicts:
        raise RuntimeError(
            f"No GT-bearing samples with cluster_mass found in dataset {dataset_name}."
        )

    rng = np.random.default_rng(seed)

    bins = {0: [], 1: [], 2: []}
    for d in dicts:
        bins[mass_bin(d["cluster_mass"])].append(d)

    per = n // 3
    targets = [per + (1 if i < (n - per * 3) else 0) for i in range(3)]

    chosen = []
    leftovers = []

    for b in range(3):
        pool = sorted(bins[b], key=lambda d: d["file_name"])
        if len(pool) == 0:
            continue

        perm = rng.permutation(len(pool))
        take = min(targets[b], len(pool))

        chosen += [pool[i] for i in perm[:take]]
        leftovers += [pool[i] for i in perm[take:]]

    if len(chosen) < n and leftovers:
        rng.shuffle(leftovers)
        chosen += leftovers[: n - len(chosen)]

    # Final deterministic order for stable batch index.
    chosen = sorted(chosen, key=lambda d: d["file_name"])

    counts = {
        b: sum(1 for d in chosen if mass_bin(d["cluster_mass"]) == b)
        for b in range(3)
    }

    print(
        f"Stratified fixed set: {len(chosen)} images | "
        f"per-bin counts small/med/large = {counts[0]}/{counts[1]}/{counts[2]}"
    )

    if len(chosen) < n:
        print(
            f"WARNING: Requested {n} images but only selected {len(chosen)} "
            f"GT-bearing images with cluster_mass."
        )

    return chosen


def write_subset_csv(selected_dicts, subset_csv):
    """Save selected diagnostic subset for reproducibility."""
    os.makedirs(os.path.dirname(subset_csv) or ".", exist_ok=True)

    with open(subset_csv, "w", newline="") as fh:
        fieldnames = [
            "audit_index",
            "file_name",
            "image_id",
            "cluster_mass",
            "mass_bin",
            "height",
            "width",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for i, d in enumerate(selected_dicts):
            writer.writerow({
                "audit_index": i,
                "file_name": d.get("file_name", ""),
                "image_id": d.get("image_id", ""),
                "cluster_mass": d.get("cluster_mass", ""),
                "mass_bin": mass_bin(d["cluster_mass"]) if d.get("cluster_mass") is not None else "",
                "height": d.get("height", ""),
                "width": d.get("width", ""),
            })

    print(f"Subset CSV written: {subset_csv}")


def build_fixed_inputs(dataset_name, n, seed, device, subset_csv=None):
    """
    Select, save, map once, and cache fixed diagnostic inputs.

    Uses val_mapper:
      - no augmentation
      - no RGB dropout
      - no random zoom
    """
    selected = select_stratified(dataset_name, n, seed)

    if subset_csv:
        write_subset_csv(selected, subset_csv)

    batches = []
    for d in selected:
        mapped = val_mapper(copy.deepcopy(d))  # val_mapper may pop annotations
        mapped["image"] = mapped["image"].to(device)

        if "instances" in mapped:
            mapped["instances"] = mapped["instances"].to(device)

        batches.append([mapped])

    return selected, batches


# ---------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------

def audit_checkpoint(model, fixed_batches, mode, mass_weight, pixel_mean, seed_base,
                     per_image=None):
    """
    Run the two-pass audit over the fixed batch set for one input mode.

    For each fixed batch:
      1. apply input mode
      2. forward once
      3. backward detection losses only, snapshot gradients
      4. zero gradients
      5. backward mass loss only, snapshot gradients
      6. zero gradients

    If per_image is a list, one dict per used batch (keyed by audit_index) is
    appended to it, so error bars / leave-one-cluster-out can be computed later.

    Returns:
      used, det_groups, mass_groups, det_stem, mass_stem, loss_vals
    """
    det_groups = defaultdict(list)
    mass_groups = defaultdict(list)

    det_stem = {"rgb": [], "depth": []}
    mass_stem = {"rgb": [], "depth": []}

    loss_vals = defaultdict(list)
    used = 0

    for i, batch in enumerate(fixed_batches):
        # Same seed for same batch index across checkpoints and input modes.
        set_seed(seed_base + i)

        with EventStorage() as storage:
            inputs = apply_input_mode(batch, mode, pixel_mean)
            losses = model(inputs)

            if "loss_mass" not in losses:
                continue

            # Skip unusable batches. With PROPOSAL_APPEND_GT and GT-bearing images,
            # this should almost never happen.
            fg = storage.latest().get("roi_head/num_fg_samples", (0, 0))
            fg = fg[0] if isinstance(fg, tuple) else fg

            if fg is not None and fg < 1:
                continue

            # Record scalar losses before backward.
            for k, v in losses.items():
                try:
                    loss_vals[k].append(float(v.detach().item()))
                except Exception:
                    pass

            # Pass A: detection/segmentation losses.
            _, (dg, ds) = backward_one(
                model,
                losses,
                DET_KEYS,
                weight=1.0,
                retain=True,
            )

            # Pass B: mass loss only.
            _, (mg, ms) = backward_one(
                model,
                losses,
                MASS_KEYS,
                weight=mass_weight,
                retain=False,
            )

        for k, v in dg.items():
            det_groups[k].append(v)

        for k, v in mg.items():
            mass_groups[k].append(v)

        det_stem["rgb"].append(ds["rgb"])
        det_stem["depth"].append(ds["depth"])

        mass_stem["rgb"].append(ms["rgb"])
        mass_stem["depth"].append(ms["depth"])

        if per_image is not None:
            per_image.append({
                "audit_index": i,
                "loss_mass": float(losses["loss_mass"].detach().item()),
                "loss_det_total": float(sum(
                    losses[k].detach().item() for k in DET_KEYS if k in losses)),
                "mass_stem_rgb_abs": ms["rgb"],
                "mass_stem_depth_abs": ms["depth"],
                "det_stem_rgb_abs": ds["rgb"],
                "det_stem_depth_abs": ds["depth"],
                "mass_backbone_res_abs": mg.get("backbone_res", 0.0),
                "det_backbone_res_abs": dg.get("backbone_res", 0.0),
                "mass_fpn_abs": mg.get("fpn", 0.0),
                "det_fpn_abs": dg.get("fpn", 0.0),
            })

        used += 1

    return used, det_groups, mass_groups, det_stem, mass_stem, loss_vals


def reduce_row(
    used,
    det_groups,
    mass_groups,
    det_stem,
    mass_stem,
    loss_vals,
    mode,
    mass_weight,
    n_requested,
    n_fixed,
):
    """Collapse accumulators into one CSV row."""

    def gmean(d, k):
        return float(np.mean(d[k])) if d.get(k) else 0.0

    def smean(d, k):
        return float(np.mean(d[k])) if d.get(k) else 0.0

    def lmean(k):
        return float(np.mean(loss_vals[k])) if loss_vals.get(k) else 0.0

    det_backbone = gmean(det_groups, "backbone_res")
    mass_backbone = gmean(mass_groups, "backbone_res")

    det_fpn = gmean(det_groups, "fpn")
    mass_fpn = gmean(mass_groups, "fpn")

    det_stem_rgb = smean(det_stem, "rgb")
    det_stem_depth = smean(det_stem, "depth")

    mass_stem_rgb = smean(mass_stem, "rgb")
    mass_stem_depth = smean(mass_stem, "depth")

    loss_det_total = sum(lmean(k) for k in DET_KEYS)
    loss_mass_raw = lmean("loss_mass")
    loss_mass_effective = mass_weight * loss_mass_raw
    loss_total_effective = loss_det_total + loss_mass_effective

    row = {
        # Absolute gradient magnitudes.
        "det_backbone_res_abs": det_backbone,
        "mass_backbone_res_abs": mass_backbone,
        "det_fpn_abs": det_fpn,
        "mass_fpn_abs": mass_fpn,

        "det_stem_rgb_abs": det_stem_rgb,
        "det_stem_depth_abs": det_stem_depth,
        "mass_stem_rgb_abs": mass_stem_rgb,
        "mass_stem_depth_abs": mass_stem_depth,

        # Ratios.
        "mass_det_backbone_res": mass_backbone / (det_backbone + 1e-12),
        "mass_det_fpn": mass_fpn / (det_fpn + 1e-12),
        "det_stem_depth_over_rgb": det_stem_depth / (det_stem_rgb + 1e-12),

        # For depth_only, RGB is zero by construction, so this ratio is not meaningful.
        "mass_stem_depth_over_rgb": (
            float("nan") if mode == "depth_only"
            else mass_stem_depth / (mass_stem_rgb + 1e-12)
        ),

        # Scalar losses.
        "loss_det_total": loss_det_total,
        "loss_total_effective": loss_total_effective,
        "loss_cls": lmean("loss_cls"),
        "loss_box_reg": lmean("loss_box_reg"),
        "loss_mask": lmean("loss_mask"),
        "loss_rpn_cls": lmean("loss_rpn_cls"),
        "loss_rpn_loc": lmean("loss_rpn_loc"),
        "loss_mass": loss_mass_raw,
        "loss_mass_effective": loss_mass_effective,

        # Bookkeeping.
        "n_requested": n_requested,
        "n_fixed": n_fixed,
        "n_used": used,
    }

    return row


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------

def write_metadata_json(meta_path, args, checkpoints, subset_csv, out_csv):
    """Write audit metadata for reproducibility."""
    os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)

    meta = {
        "run_dir": args.run_dir,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "n_batches_requested": args.n_batches,
        "modes": args.modes,
        "seed": args.seed,
        "mass_loss_weight": args.mass_loss_weight,
        "pixel_mean": args.pixel_mean,
        "pixel_std": args.pixel_std,
        "freeze_at": args.freeze_at,
        "include_final": args.include_final,
        "final_iteration": args.final_iteration,
        "subset_csv": subset_csv,
        "output_csv": out_csv,
        "checkpoints": checkpoints,
        "important_note": (
            "This audit measures per-loss gradient structure at frozen checkpoints "
            "on a fixed diagnostic subset. It is not the exact optimizer update "
            "history during training."
        ),
    }

    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Metadata JSON written: {meta_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_dir",
        required=True,
        help="Directory with model_*.pth checkpoints."
    )
    parser.add_argument(
        "--dataset_path",
        default="./data_s0713/"
    )
    parser.add_argument(
        "--split",
        default="val"
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        default=60,
        help="Number of fixed stratified images audited at every checkpoint."
    )
    parser.add_argument(
        "--mass_loss_weight",
        type=float,
        default=1.0,
        help=(
            "1.0 = intrinsic audit. Set to the run's training mass weight for "
            "the effective audit."
        )
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=MODES,
        choices=MODES
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for fixed subset selection and deterministic per-batch forward seeding."
    )

    # These must match training.
    parser.add_argument(
        "--pixel_mean",
        type=float,
        nargs=4,
        default=[103.530, 116.280, 123.675, 915.043],
        help="Must match the run's training PIXEL_MEAN."
    )
    parser.add_argument(
        "--pixel_std",
        type=float,
        nargs=4,
        default=[57.375, 57.120, 58.395, 411.683],
        help="Must match the run's training PIXEL_STD."
    )
    parser.add_argument(
        "--freeze_at",
        type=int,
        default=0,
        help="Must match the run's training BACKBONE.FREEZE_AT."
    )

    parser.add_argument(
        "--include_final",
        action="store_true",
        help="Also audit model_final.pth. Usually duplicates the last numbered checkpoint."
    )
    parser.add_argument(
        "--final_iteration",
        type=int,
        default=None,
        help="Optional numeric iteration to assign to model_final.pth."
    )

    parser.add_argument(
        "--out",
        default=None,
        help="Main CSV path. Default: <run_dir>/gradient_audit_sweep.csv"
    )
    parser.add_argument(
        "--subset_out",
        default=None,
        help="Subset CSV path. Default: <run_dir>/gradient_audit_subset.csv"
    )
    parser.add_argument(
        "--meta_out",
        default=None,
        help="Metadata JSON path. Default: <run_dir>/gradient_audit_sweep_meta.json"
    )
    parser.add_argument(
        "--per_image_out",
        default=None,
        help="Per-image CSV path (one row per checkpoint x mode x image). "
             "Default: gradient_audit_per_image.csv next to the main CSV."
    )

    args = parser.parse_args()

    out_csv = args.out or os.path.join(args.run_dir, "gradient_audit_sweep.csv")
    subset_csv = args.subset_out or os.path.join(args.run_dir, "gradient_audit_subset.csv")
    meta_json = args.meta_out or os.path.join(args.run_dir, "gradient_audit_sweep_meta.json")
    per_image_csv = args.per_image_out or os.path.join(
        os.path.dirname(out_csv) or ".", "gradient_audit_per_image.csv")

    checkpoints = find_checkpoints(
        args.run_dir,
        include_final=args.include_final,
        final_iteration=args.final_iteration,
    )

    if not checkpoints:
        raise SystemExit(f"No model_*.pth checkpoints found in {args.run_dir}")

    first_iter = checkpoints[0]["iteration_numeric"]
    last_iter = checkpoints[-1]["iteration_numeric"] or checkpoints[-1]["label"]

    print("=" * 88)
    print("GRADIENT AUDIT SWEEP")
    print("=" * 88)
    print(f"Run dir        : {args.run_dir}")
    print(f"Dataset path   : {args.dataset_path}")
    print(f"Split          : {args.split}")
    print(f"Checkpoints    : {len(checkpoints)}  ({first_iter} .. {last_iter})")
    print(f"Modes          : {args.modes}")
    print(f"n_batches      : {args.n_batches}")
    print(f"Seed           : {args.seed}")
    print(f"PIXEL_MEAN     : {args.pixel_mean}")
    print(f"PIXEL_STD      : {args.pixel_std}   <-- MUST match training")
    print(f"FREEZE_AT      : {args.freeze_at}   <-- MUST match training")
    print(f"mass_weight    : {args.mass_loss_weight}")
    print(f"Output CSV     : {out_csv}")
    print(f"Subset CSV     : {subset_csv}")
    print(f"Metadata JSON  : {meta_json}")
    print("=" * 88)

    # Build the model and config once.
    cfg = build_cfg(
        model_path=checkpoints[0]["path"],
        dataset_path=args.dataset_path,
        split=args.split,
        pixel_mean=args.pixel_mean,
        pixel_std=args.pixel_std,
        freeze_at=args.freeze_at,
    )

    device = torch.device(cfg.MODEL.DEVICE)

    model = build_model(cfg)
    model.to(device)
    model.train()  # train mode is required so Detectron2 returns losses

    summarize_param_groups(model)

    # Build fixed diagnostic set.
    selected_dicts, fixed_batches = build_fixed_inputs(
        dataset_name=cfg.DATASETS.TRAIN[0],
        n=args.n_batches,
        seed=args.seed,
        device=device,
        subset_csv=subset_csv,
    )

    n_fixed = len(fixed_batches)

    write_metadata_json(
        meta_path=meta_json,
        args=args,
        checkpoints=checkpoints,
        subset_csv=subset_csv,
        out_csv=out_csv,
    )

    fieldnames = [
        "checkpoint_iteration_numeric",
        "checkpoint_label",
        "checkpoint_path",
        "input_mode",

        # Absolute shared-trunk gradients.
        "det_backbone_res_abs",
        "mass_backbone_res_abs",
        "det_fpn_abs",
        "mass_fpn_abs",

        # Absolute stem gradients.
        "det_stem_rgb_abs",
        "det_stem_depth_abs",
        "mass_stem_rgb_abs",
        "mass_stem_depth_abs",

        # Ratios.
        "mass_det_backbone_res",
        "mass_det_fpn",
        "det_stem_depth_over_rgb",
        "mass_stem_depth_over_rgb",

        # Scalar losses.
        "loss_det_total",
        "loss_total_effective",
        "loss_cls",
        "loss_box_reg",
        "loss_mask",
        "loss_rpn_cls",
        "loss_rpn_loc",
        "loss_mass",
        "loss_mass_effective",

        # Bookkeeping.
        "n_requested",
        "n_fixed",
        "n_used",
    ]

    per_image_fields = [
        "checkpoint_iteration_numeric", "checkpoint_label", "input_mode",
        "audit_index", "file_name", "cluster_mass",
        "loss_mass", "loss_det_total",
        "mass_stem_rgb_abs", "mass_stem_depth_abs",
        "det_stem_rgb_abs", "det_stem_depth_abs",
        "mass_backbone_res_abs", "det_backbone_res_abs",
        "mass_fpn_abs", "det_fpn_abs",
    ]

    checkpointer = DetectionCheckpointer(model)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    with open(out_csv, "w", newline="") as fh, open(per_image_csv, "w", newline="") as fh_img:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        img_writer = csv.DictWriter(fh_img, fieldnames=per_image_fields)
        img_writer.writeheader()

        for ckpt in checkpoints:
            ckpt_path = ckpt["path"]
            ckpt_label = ckpt["label"]
            ckpt_iter = ckpt["iteration_numeric"]

            checkpointer.load(ckpt_path)
            model.train()

            for mode in args.modes:
                per_image = []
                out = audit_checkpoint(
                    model=model,
                    fixed_batches=fixed_batches,
                    mode=mode,
                    mass_weight=args.mass_loss_weight,
                    pixel_mean=cfg.MODEL.PIXEL_MEAN,
                    seed_base=args.seed,
                    per_image=per_image,
                )

                for rec in per_image:
                    d = selected_dicts[rec["audit_index"]]
                    rec.update({
                        "checkpoint_iteration_numeric": ckpt_iter,
                        "checkpoint_label": ckpt_label,
                        "input_mode": mode,
                        "file_name": d["file_name"],
                        "cluster_mass": d["cluster_mass"],
                    })
                    img_writer.writerow(rec)
                fh_img.flush()

                row = reduce_row(
                    *out,
                    mode=mode,
                    mass_weight=args.mass_loss_weight,
                    n_requested=args.n_batches,
                    n_fixed=n_fixed,
                )

                row.update({
                    "checkpoint_iteration_numeric": ckpt_iter,
                    "checkpoint_label": ckpt_label,
                    "checkpoint_path": ckpt_path,
                    "input_mode": mode,
                })

                writer.writerow(row)
                fh.flush()

                if row["n_used"] != n_fixed:
                    print(
                        f"WARNING: n_used={row['n_used']} but n_fixed={n_fixed} "
                        f"for checkpoint={ckpt_label}, mode={mode}"
                    )

                print(
                    f"[ckpt {str(ckpt_label):>8} | {mode:10}] "
                    f"mass/det_back={row['mass_det_backbone_res']:.3f} "
                    f"mass/det_fpn={row['mass_det_fpn']:.3f} "
                    f"mass_depth/rgb={row['mass_stem_depth_over_rgb']:.3f} "
                    f"mass_depth_abs={row['mass_stem_depth_abs']:.3e} "
                    f"loss_mass={row['loss_mass']:.4f} "
                    f"(n={row['n_used']}/{n_fixed})"
                )

    print("\nDone.")
    print(f"Main CSV      : {out_csv}")
    print(f"Subset CSV    : {subset_csv}")
    print(f"Per-image CSV : {per_image_csv}")
    print(f"Metadata JSON : {meta_json}")


if __name__ == "__main__":
    main()