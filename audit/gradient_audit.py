"""
GRADIENT AUDIT — which loss actually updates which part of the model.

The optimizer trains via gradients, not loss scalars. This script isolates the
gradient each task produces, so we can answer:
  - Does the MASS loss reach the shared backbone, or only its own head?
  - Does the MASS loss push the DEPTH input channel (conv1[:,3]) vs the RGB channels?
  - Does DETECTION dominate the shared trunk?
  - Do norm/dropout/zoom change which modality the mass objective learns from?

Method, per batch:
  forward() once -> dict of separable losses
  loss_det.backward(retain_graph=True); record grads; zero
  (mass_loss_weight * loss_mass).backward(); record grads; zero
Averaged over many batches (skipping batches with no foreground ROI samples).

Run:
  ./run.sh audit/gradient_audit.py \
      --model_path output/s0713_zoom_drop05/model_final.pth \
      --dataset_path ./data_s0713/ --split train --n_batches 40
"""
import os, sys, argparse
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
from train import val_mapper   # reuse the exact training input pipeline


DET_KEYS  = ["loss_cls", "loss_box_reg", "loss_mask", "loss_rpn_cls", "loss_rpn_loc"]
MASS_KEYS = ["loss_mass"]


def rms(t):
    """RMS of a gradient tensor = ||g|| / sqrt(numel). Removes parameter-count bias,
    so values are comparable across layers of different sizes."""
    if t is None:
        return 0.0
    return t.pow(2).mean().sqrt().item()


def group_of(name):
    """Map a parameter name to a coarse group, for 'where does the gradient land'."""
    if name == "backbone.bottom_up.stem.conv1.weight":
        return "stem_conv1"
    if name.startswith("backbone.bottom_up.res"):
        return "backbone_res"
    if name.startswith("backbone.fpn"):
        return "fpn"
    if name.startswith("proposal_generator"):
        return "rpn_head"
    if name.startswith("roi_heads.box") :
        return "box_head"
    if name.startswith("roi_heads.mask_head"):
        return "mask_head"
    if name.startswith("roi_heads.mass_head"):
        return "mass_head"
    if name.startswith("backbone.bottom_up.stem"):
        return "backbone_res"
    return "other"


def snapshot_grads(model):
    """After a .backward(), summarize grads into per-group mean RMS, plus the
    per-channel stem-conv1 RMS (B,G,R,Depth split)."""
    group_vals = defaultdict(list)
    stem = {"rgb": 0.0, "depth": 0.0}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if name == "backbone.bottom_up.stem.conv1.weight":
            g = p.grad  # (64, 4, 7, 7)
            # per-input-channel RMS, then average the three RGB channels
            b = rms(g[:, 0:1]); gg = rms(g[:, 1:2]); r = rms(g[:, 2:3]); d = rms(g[:, 3:4])
            stem["rgb"] = float(np.mean([b, gg, r]))
            stem["depth"] = d
        group_vals[group_of(name)].append(rms(p.grad))
    group_mean = {k: float(np.mean(v)) for k, v in group_vals.items()}
    return group_mean, stem


def backward_one(model, losses, keys, weight=1.0, retain=False):
    """Sum the chosen loss keys, backprop, and return the grad snapshot.
    Zeroes grads first so passes don't contaminate each other."""
    model.zero_grad(set_to_none=True)
    total = sum(losses[k] for k in keys if k in losses) * weight
    total.backward(retain_graph=retain)
    snap = snapshot_grads(model)
    model.zero_grad(set_to_none=True)
    return float(total.detach().item()), snap

def clone_inputs(inputs):
    """Clone image tensors so input-mode ablations do not modify the dataloader batch in-place."""
    cloned = []
    for x in inputs:
        y = dict(x)
        y["image"] = x["image"].clone()
        cloned.append(y)
    return cloned


def apply_input_mode(inputs, mode, pixel_mean):
    """
    Modify raw BGRD input before model normalization.

    Assumption:
      inputs[i]["image"] is raw-ish [4,H,W], and the model normalizes internally:
        (image - PIXEL_MEAN) / PIXEL_STD

    Therefore setting a channel to PIXEL_MEAN makes that normalized channel ~0.
    """
    inputs = clone_inputs(inputs)

    for x in inputs:
        img = x["image"]  # [4, H, W]

        if mode == "real_rgbd":
            pass

        elif mode == "rgb_only":
            # keep BGR, neutralize depth
            img[3, :, :] = float(pixel_mean[3])

        elif mode == "depth_only":
            # neutralize BGR, keep depth
            img[0, :, :] = float(pixel_mean[0])
            img[1, :, :] = float(pixel_mean[1])
            img[2, :, :] = float(pixel_mean[2])

        else:
            raise ValueError(f"Unknown input_mode: {mode}")

    return inputs

def audit(model, data_loader, n_batches, mass_weight, input_mode, pixel_mean):
    det_groups, det_stem = defaultdict(list), defaultdict(list)
    mass_groups, mass_stem = defaultdict(list), defaultdict(list)
    used = 0
    printed_sanity = False

    for inputs in data_loader:
        if used >= n_batches:
            break

        with EventStorage() as storage:
            inputs = apply_input_mode(inputs, input_mode, pixel_mean)

            if not printed_sanity:
                img = inputs[0]["image"]
                print(f"\nInput sanity check | mode={input_mode}")
                print("shape:", tuple(img.shape))
                print("channel means:", [round(img[c].mean().item(), 3) for c in range(img.shape[0])])
                print("channel mins: ", [round(img[c].min().item(), 3) for c in range(img.shape[0])])
                print("channel maxs: ", [round(img[c].max().item(), 3) for c in range(img.shape[0])])
                printed_sanity = True

            losses = model(inputs)
            if "loss_mass" not in losses:
                continue
            # skip batches with no foreground ROI samples (mass grad would be noise)
            fg = storage.latest().get("roi_head/num_fg_samples", (0, 0))
            fg = fg[0] if isinstance(fg, tuple) else fg
            if fg is not None and fg < 1:
                continue
            # Pass A: detection/segmentation (retain graph for the second backward)
            _, (dg, ds) = backward_one(model, losses, DET_KEYS, 1.0, retain=True)
            # Pass B: mass only (weighted as in training)
            _, (mg, ms) = backward_one(model, losses, MASS_KEYS, mass_weight, retain=False)
        for k, v in dg.items():  det_groups[k].append(v)
        for k, v in mg.items():  mass_groups[k].append(v)
        det_stem["rgb"].append(ds["rgb"]);   det_stem["depth"].append(ds["depth"])
        mass_stem["rgb"].append(ms["rgb"]);  mass_stem["depth"].append(ms["depth"])
        used += 1
    return used, det_groups, mass_groups, det_stem, mass_stem


def ms_str(vals):
    a = np.array(vals) if vals else np.array([0.0])
    return f"{a.mean():.3e} ± {a.std():.1e}"

def report(used, det_groups, mass_groups, det_stem, mass_stem, input_mode):
    groups = ["stem_conv1", "backbone_res", "fpn", "rpn_head", "box_head", "mask_head", "mass_head"]
    print(f"\n=== Gradient audit over {used} batches | input_mode={input_mode} | mean RMS ± std ===")
    print(f"{'group':14} {'DET grad':>22} {'MASS grad':>22} {'mass/det':>9}")
    for g in groups:
        dv = det_groups.get(g, [0.0]); mv = mass_groups.get(g, [0.0])
        ratio = np.mean(mv) / (np.mean(dv) + 1e-12)
        print(f"{g:14} {ms_str(dv):>22} {ms_str(mv):>22} {ratio:>9.3f}")

    print("\n--- STEM conv1 per-channel (the headline: who drives the DEPTH input) ---")
    d_rgb, d_dep = np.mean(det_stem['rgb']), np.mean(det_stem['depth'])
    m_rgb, m_dep = np.mean(mass_stem['rgb']), np.mean(mass_stem['depth'])
    print(f"{'':10} {'rgb (mean B,G,R)':>18} {'depth (ch3)':>14} {'depth/rgb':>10}")
    print(f"{'DET':10} {d_rgb:>18.3e} {d_dep:>14.3e} {d_dep/(d_rgb+1e-12):>10.3f}")
    print(f"{'MASS':10} {m_rgb:>18.3e} {m_dep:>14.3e} {m_dep/(m_rgb+1e-12):>10.3f}")
    print("\nKEY READINGS:")
    print(f"  • MASS depth/rgb ratio = {m_dep/(m_rgb+1e-12):.3f}  "
          "(>1 ⇒ mass objective pushes depth harder than RGB at the input)")
    print(f"  • mass/det in backbone_res = "
          f"{np.mean(mass_groups.get('backbone_res',[0]))/(np.mean(det_groups.get('backbone_res',[1e-12]))+1e-12):.3f}  "
          "(does mass reach the shared trunk, or only its head?)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--dataset_path", default="./data_s0713/")
    p.add_argument("--split", default="train")
    p.add_argument("--n_batches", type=int, default=40)
    p.add_argument("--mass_loss_weight", type=float, default=1.0,
                   help="Weight on mass loss in Pass B. 1.0 = raw (does the task want depth?); "
                        "set to your training weight for the 'effective' audit.")
    p.add_argument(
        "--input_mode",
        default="real_rgbd",
        choices=["real_rgbd", "rgb_only", "depth_only"],
        help="Input ablation mode for gradient audit."
    )
    args = p.parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.META_ARCHITECTURE = "RGBDGeneralizedRCNN"
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 915.043]
    #cfg.MODEL.PIXEL_STD  = [1.0, 1.0, 1.0, 411.683]

    cfg.MODEL.PIXEL_STD  = [57.375, 57.120, 58.395, 411.683]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_MASK_HEAD.LOSS_WEIGHT = 1.0
    cfg.MODEL.ROI_MASS_HEAD.MASS_LOSS_WEIGHT = 1.0
    cfg.MODEL.BACKBONE.FREEZE_AT = 0   # MUST match training, else stem (incl. depth ch) is frozen → 0 grad
    cfg.MODEL.WEIGHTS = args.model_path
    cfg.DATASET_PATH = args.dataset_path

    split = args.split
    name = f"TomatoDataset_{split}"
    if name not in DatasetCatalog.list():
        DatasetCatalog.register(name,
            lambda: utils_detection.get_tomato_dataset_dicts(args.dataset_path, split))
        MetadataCatalog.get(name).set(thing_classes=["tomato_bunch"])
    cfg.DATASETS.TRAIN = (name,)

    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.train()   # train mode so forward() returns losses
    print(f"Loaded {args.model_path}  | split={split} | mass_weight={args.mass_loss_weight}")

    from detectron2.data import build_detection_train_loader
    loader = build_detection_train_loader(cfg, mapper=val_mapper, total_batch_size=1)

    out = audit(
        model,
        loader,
        args.n_batches,
        args.mass_loss_weight,
        input_mode=args.input_mode,
        pixel_mean=cfg.MODEL.PIXEL_MEAN,
    )
    report(*out, input_mode=args.input_mode)

if __name__ == "__main__":
    main()
