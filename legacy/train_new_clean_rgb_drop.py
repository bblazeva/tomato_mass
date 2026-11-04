import os
import copy
import random
import argparse
from collections import Counter

import cv2
import numpy as np
import torch
import pickle5 as pickle

from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import HookBase, PeriodicWriter
from detectron2.utils.events import TensorboardXWriter
from detectron2.utils.logger import log_every_n_seconds
from detectron2.utils import comm
from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data import detection_utils as utils_detectron
import detectron2.data.transforms as T
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from torch.utils.data import WeightedRandomSampler, Sampler

import utils_detection
from mass_evaluator import MassEvaluator


DEPTH_MEAN_MM = 915.043
DEPTH_STD_MM  = 411.683
BGR_MEAN = (103.530, 116.280, 123.675)


class _InfiniteSampler(Sampler):
    """Wrap any finite sampler to loop forever."""
    def __init__(self, base_sampler):
        self.base = base_sampler

    def __iter__(self):
        while True:
            yield from self.base

    def __len__(self):
        return 2 ** 31


# ====================================================================================
# DATA LOADING
# ====================================================================================

def load_rgbd_image(dataset_dict):
    """
    Load 4-channel [B, G, R, Depth_mm] image.

    - BGR channels: actual RGB pixel values (0-255)
    - Depth channel: RAW millimeter values; invalid (0mm) pixels are NOT filled here —
      the caller fills them with DEPTH_MEAN_MM *after* augmentation (so the zoom can
      scale only real depths and keep holes neutral). `invalid` marks those pixels.

    Returns: (dataset_dict_copy, image_hwc_float32, invalid_hw_bool)
    """
    dataset_dict = copy.deepcopy(dataset_dict)

    image_bgr = utils_detectron.read_image(dataset_dict["file_name"], format="BGR")
    H, W = image_bgr.shape[:2]

    depth_mm = np.load(dataset_dict["depth_file"]).astype(np.float32)
    if depth_mm.shape != (H, W):
        depth_mm = cv2.resize(depth_mm, (W, H), interpolation=cv2.INTER_NEAREST)

    invalid = depth_mm == 0 

    image = np.zeros((H, W, 4), dtype=np.float32)
    image[:, :, :3] = image_bgr.astype(np.float32)
    image[:, :, 3]  = depth_mm

    return dataset_dict, image, invalid


def depth_consistent_zoom(image, invalid, annos, s):
    """
    Simulate the camera moving closer/farther by factor s (mass-invariant):
      - resize pixels by s            (apparent size  × s)
      - scale VALID depth by 1/s      (object distance × 1/s)   → apparent×depth stays const
      - recanvas to original H×W centered on the cluster (crop if s>1, pad if s<1)
      - holes + padding stay `invalid` (caller fills them with DEPTH_MEAN_MM)
    Depth uses INTER_NEAREST (never blend across depth edges / holes); RGB uses
    AREA (down) / LINEAR (up). Annotations get the same s + crop/pad offset.

    Returns: (image_out_hwc4, invalid_out_hw_bool, annos_out)
    """
    H, W = invalid.shape
    bgr, depth = image[:, :, :3], image[:, :, 3]
    newW, newH = max(1, round(W * s)), max(1, round(H * s))

    bgr_interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
    bgr_r   = cv2.resize(bgr,   (newW, newH), interpolation=bgr_interp).astype(np.float32)
    depth_r = cv2.resize(depth, (newW, newH), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    inval_r = cv2.resize(invalid.astype(np.uint8), (newW, newH),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
    depth_r[~inval_r] /= s   # scale only real depths → physical distance change

    # crop window centered on the cluster (union-bbox center) in the resized frame
    if annos:
        x0 = min(a["bbox"][0] for a in annos);                 y0 = min(a["bbox"][1] for a in annos)
        x1 = max(a["bbox"][0] + a["bbox"][2] for a in annos);  y1 = max(a["bbox"][1] + a["bbox"][3] for a in annos)
        cx, cy = int((x0 + x1) / 2 * s), int((y0 + y1) / 2 * s)
    else:
        cx, cy = newW // 2, newH // 2
    top, left = cy - H // 2, cx - W // 2

    img_out = np.zeros((H, W, 4), np.float32)
    img_out[:, :, :3] = BGR_MEAN
    inval_out = np.ones((H, W), bool)   # padding is invalid until overwritten by real data

    sy0, sx0 = max(0, top), max(0, left)
    sy1, sx1 = min(newH, top + H), min(newW, left + W)
    dy0, dx0 = sy0 - top, sx0 - left
    dy1, dx1 = dy0 + (sy1 - sy0), dx0 + (sx1 - sx0)
    if sy1 > sy0 and sx1 > sx0:
        img_out[dy0:dy1, dx0:dx1, :3] = bgr_r[sy0:sy1, sx0:sx1]
        img_out[dy0:dy1, dx0:dx1, 3]  = depth_r[sy0:sy1, sx0:sx1]
        inval_out[dy0:dy1, dx0:dx1]   = inval_r[sy0:sy1, sx0:sx1]

    # transform annotations: (x, y) -> (x*s - left, y*s - top)
    annos_out = []
    for a in annos:
        a = copy.deepcopy(a)
        x, y, w, h = a["bbox"]
        a["bbox"] = [x * s - left, y * s - top, w * s, h * s]
        seg = a.get("segmentation")
        if isinstance(seg, list):
            new_seg = []
            for poly in seg:
                pts = np.asarray(poly, np.float32).reshape(-1, 2)
                pts[:, 0] = pts[:, 0] * s - left
                pts[:, 1] = pts[:, 1] * s - top
                new_seg.append(pts.reshape(-1).tolist())
            a["segmentation"] = new_seg
        annos_out.append(a)

    return img_out, inval_out, annos_out


# Photometric jitter presets (BGR only). `wb` = per-channel white-balance gain range,
# `hue` in OpenCV units (0-179 = 0-360deg), `sat`/`bri`/`con` are multiplicative ranges.
_COLOR_PRESETS = {
    "light":  dict(wb=0.10, hue=5,  sat=(0.85, 1.15), bri=(0.8, 1.2), con=(0.8, 1.2)),
    "strong": dict(wb=0.20, hue=10, sat=(0.6, 1.4),   bri=(0.6, 1.4), con=(0.6, 1.4)),
}


def color_jitter_bgr(image, strength):
    """
    Photometric augmentation on the BGR channels ONLY (channel 3 / depth untouched).
    Simulates the day/lighting/appearance shift that breaks RGB across sessions:
    white-balance (per-channel gain) + hue + saturation + brightness + contrast.
    Operates in-place on image[:, :, :3] (float32, 0-255).
    """
    p = _COLOR_PRESETS.get(strength)
    if p is None:
        return
    bgr = image[:, :, :3]

    # White balance: independent per-channel gain (color-temperature shift)
    for c in range(3):
        bgr[:, :, c] *= random.uniform(1 - p["wb"], 1 + p["wb"])
    np.clip(bgr, 0, 255, out=bgr)

    # Hue + saturation (uint8 HSV round-trip)
    hsv = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-p["hue"], p["hue"])) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(*p["sat"]), 0, 255)
    bgr[:] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # Brightness, then contrast
    bgr *= random.uniform(*p["bri"])
    m = bgr.mean()
    bgr[:] = m + (bgr - m) * random.uniform(*p["con"])
    np.clip(bgr, 0, 255, out=bgr)


def train_mapper(dataset_dict, rgb_dropout=0.0, zoom_prob=0.0, zoom_min=0.7, zoom_max=1.3,
                 color_aug="light"):
    """
    Training data mapper.

    - Depth-consistent zoom: with prob zoom_prob, simulate a distance change
      (apparent size ×s, depth ×1/s) so the model must read scale, not memorize area
    - Geometric augmentation (flip) applied to all 4 channels
    - RGB dropout: with prob rgb_dropout, neutralize BGR (forces reliance on depth)
    - Color jitter (white-balance/hue/saturation/brightness/contrast), BGR only,
      strength = color_aug ("off"/"light"/"strong"), when not dropped out
    """
    dataset_dict, image, invalid = load_rgbd_image(dataset_dict)
    annos_in = dataset_dict.pop("annotations")

    # Depth-consistent zoom (geometric + physical distance change) — before flip
    if zoom_prob > 0.0 and random.random() < zoom_prob:
        s = random.uniform(zoom_min, zoom_max)
        image, invalid, annos_in = depth_consistent_zoom(image, invalid, annos_in, s)

    # Fill depth holes + padding with the neutral mean (→ 0 after normalization)
    image[:, :, 3][invalid] = DEPTH_MEAN_MM

    # Geometric augmentation — all 4 channels transformed together
    augs = T.AugmentationList([
        T.RandomFlip(horizontal=True, vertical=False),
    ])
    aug_input = T.AugInput(image)
    transform = augs(aug_input)
    image = aug_input.image

    # RGB dropout or color jitter (mutually exclusive)
    if rgb_dropout > 0.0 and random.random() < rgb_dropout:
        image[:, :, 0] = 103.530  # B
        image[:, :, 1] = 116.280  # G
        image[:, :, 2] = 123.675  # R
        #image[:, :, :3] = 0.0  # zero BGR, depth channel untouched
    elif color_aug != "off" and random.random() < 0.5:
        color_jitter_bgr(image, color_aug)

    annos = [
        utils_detectron.transform_instance_annotations(ann, [transform], image.shape[:2])
        for ann in annos_in
    ]

    image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()

    return {
        "filename": dataset_dict["file_name"],
        "image_id": dataset_dict["image_id"],
        "height":   dataset_dict["height"],
        "width":    dataset_dict["width"],
        "image":    image_tensor,
        "instances": utils_detectron.annotations_to_instances(annos, image_tensor.shape[1:]),
    }


def val_mapper(dataset_dict):
    """Validation mapper — no augmentation."""
    dataset_dict, image, invalid = load_rgbd_image(dataset_dict)
    image[:, :, 3][invalid] = DEPTH_MEAN_MM   # fill holes neutral (matches train path)

    annos = [
        utils_detectron.transform_instance_annotations(ann, [], image.shape[:2])
        for ann in dataset_dict.pop("annotations")
    ]

    image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()

    return {
        "filename": dataset_dict["file_name"],
        "image_id": dataset_dict["image_id"],
        "height":   dataset_dict["height"],
        "width":    dataset_dict["width"],
        "image":    image_tensor,
        "instances": utils_detectron.annotations_to_instances(annos, image_tensor.shape[1:]),
    }


# ====================================================================================
# HOOKS
# ====================================================================================

class ValidationLossHook(HookBase):
    """
    Computes and logs loss on the validation set every eval_period iterations.
    Logs val/loss_total, val/loss_mass (unweighted), val/loss_det to TensorBoard.
    """

    def __init__(self, cfg, model, data_loader):
        self._model      = model
        self._period     = cfg.TEST.EVAL_PERIOD
        self._data_loader = data_loader

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final  = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()

    def _do_loss_eval(self):
        losses      = []
        mass_losses = []

        with torch.no_grad():
            for inputs in self._data_loader:
                loss_dict = self._model(inputs)
                loss_dict = {
                    k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
                    for k, v in loss_dict.items()
                }
                losses.append(sum(loss_dict.values()))
                mass_losses.append(loss_dict.get("loss_mass", 0.0))

        mean_total = float(np.mean(losses))
        mean_mass  = float(np.mean(mass_losses))
        mean_det   = mean_total - mean_mass

        # Unweight mass loss for interpretability (weight may have changed via scheduler)
        model = self._model
        if hasattr(model, "module"):
            model = model.module
        current_weight = getattr(model.roi_heads.mass_head, "loss_weight", 1.0)
        mean_mass_unweighted = mean_mass / max(current_weight, 1e-6)

        self.trainer.storage.put_scalar("val/loss_total", mean_total)
        self.trainer.storage.put_scalar("val/loss_mass",  mean_mass_unweighted)
        self.trainer.storage.put_scalar("val/loss_det",   mean_det)
        comm.synchronize()


class MassLossWeightSchedulerHook(HookBase):
    """
    Updates mass_head.loss_weight during training according to a schedule.

    schedule: list of (start_iter, weight) pairs, e.g.:
        [(0, 0.0), (3000, 0.5), (5000, 1.0)]

    Logs to stdout when the weight changes.
    """

    def __init__(self, schedule):
        self._schedule       = sorted(schedule, key=lambda x: x[0])
        self._current_weight = None

    def after_step(self):
        it     = self.trainer.iter
        weight = self._schedule[0][1]
        for start_iter, w in self._schedule:
            if it >= start_iter:
                weight = w

        if weight != self._current_weight:
            self._current_weight = weight
            model = self.trainer.model
            if hasattr(model, "module"):
                model = model.module
            model.roi_heads.mass_head.loss_weight = weight
            print(f"[iter {it}] mass_loss_weight → {weight}")


class DepthGradientHook(HookBase):
    """
    Every 20 iterations, logs the gradient magnitude for the depth channel
    vs RGB channels in the first conv layer of the backbone.

    If grad/depth << grad/rgb consistently, the model is ignoring depth.
    """

    def after_step(self):
        if self.trainer.iter % 20 != 0:
            return

        for name, param in self.trainer.model.named_parameters():
            if name == "backbone.bottom_up.stem.conv1.weight":
                if param.grad is None:
                    print(f"[iter {self.trainer.iter}] grad/depth: conv1 grad is None (backbone frozen?)")
                    return
                g = param.grad
                rgb_g = g[:, :3].abs().mean().item()
                depth_g = g[:,  3].abs().mean().item()
                self.trainer.storage.put_scalar("grad/rgb",   rgb_g)
                self.trainer.storage.put_scalar("grad/depth", depth_g)
                print(f"[iter {self.trainer.iter}] grad/rgb={rgb_g:.2e}  grad/depth={depth_g:.2e}  ratio={depth_g/max(rgb_g, 1e-12):.3f}")
                return


# ====================================================================================
# TRAINER
# ====================================================================================

class TomatoTrainer(DefaultTrainer):

    _mass_loss_schedule = None

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return DatasetEvaluators([
            COCOEvaluator(dataset_name, cfg, True, output_folder),
            MassEvaluator(),
        ])

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, ValidationLossHook(
            self.cfg,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                mapper=val_mapper,
            ),
        ))
        hooks.append(DepthGradientHook())
        if self._mass_loss_schedule is not None:
            hooks.append(MassLossWeightSchedulerHook(self._mass_loss_schedule))
        return hooks

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """Use 4-channel val_mapper at evaluation time (no augmentation)."""
        return build_detection_test_loader(cfg, dataset_name, mapper=val_mapper)

    @classmethod
    def build_train_loader(cls, cfg):
        """Stratified sampling by mass bin (small / medium / large)."""
        dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])

        def mass_bin(m):
            if m < 0.25: return 0
            if m < 0.50: return 1
            return 2

        for d in dataset_dicts:
            d["mass_bin"] = mass_bin(d["cluster_mass"])

        bin_counts  = Counter(d["mass_bin"] for d in dataset_dicts)
        bin_weights = {b: 1.0 / c for b, c in bin_counts.items()}
        sample_wts  = [bin_weights[d["mass_bin"]] for d in dataset_dicts]

        sampler = _InfiniteSampler(
            WeightedRandomSampler(sample_wts, len(sample_wts), replacement=True)
        )

        rgb_dropout = getattr(cfg, "RGB_DROPOUT", 0.0)
        zoom_prob   = getattr(cfg, "ZOOM_PROB", 0.0)
        zoom_min    = getattr(cfg, "ZOOM_MIN", 0.7)
        zoom_max    = getattr(cfg, "ZOOM_MAX", 1.3)
        color_aug   = getattr(cfg, "COLOR_AUG", "light")
        return build_detection_train_loader(
            cfg,
            mapper=lambda d: train_mapper(d, rgb_dropout=rgb_dropout,
                                          zoom_prob=zoom_prob,
                                          zoom_min=zoom_min, zoom_max=zoom_max,
                                          color_aug=color_aug),
            sampler=sampler,
        )


# ====================================================================================
# ARGUMENT PARSING
# ====================================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train RGB-D tomato multitask model")

    # Data
    p.add_argument("--dataset_path",    required=True,
                   help="Root data folder")
    p.add_argument("--weights",         default="./mask_rcnn_R50_FPN_3x_RGBD.pth",
                   help="Starting checkpoint (.pth)")
    p.add_argument("--experiment_name", default="exp",
                   help="Output folder name under ./output/")

    # Training
    p.add_argument("--num_iterations",  type=int,   default=10000)
    p.add_argument("--batch_size",      type=int,   default=2)
    p.add_argument("--learning_rate",   type=float, default=0.00025)
    p.add_argument("--mass_loss_weight",type=float, default=1.0,
                   help="Weight of mass loss relative to detection/segmentation loss")
    p.add_argument("--mass_loss_schedule", type=str, default=None,
                   help="Override --mass_loss_weight with a schedule: 'iter:weight,...' "
                        "e.g. '0:0,3000:0.5,5000:1.0'")

    # Evaluation / checkpointing
    p.add_argument("--eval_period",       type=int, default=500)
    p.add_argument("--checkpoint_period", type=int, default=1000)

    # Freezing
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze the ResNet backbone (train heads only)")
    p.add_argument("--freeze_det",      action="store_true",
                   help="Freeze detection/segmentation branch (train mass head only)")
    p.add_argument("--freeze_mass",     action="store_true",
                   help="Freeze mass head (train detection/segmentation only)")

    # Depth
    p.add_argument("--init_depth_weights", action="store_true",
                   help="Re-initialize depth channel weights in conv1 from RGB average. "
                        "Use when starting from a fresh 4-channel checkpoint.")
    p.add_argument("--rgb_dropout", type=float, default=0.0,
                   help="Probability of zeroing out RGB channels per sample during training. "
                        "Forces the backbone to learn from depth.")

    # Depth-consistent zoom augmentation
    p.add_argument("--zoom_prob", type=float, default=0.0,
                   help="Probability of applying depth-consistent zoom per sample "
                        "(apparent size ×s, depth ×1/s; mass-invariant). 0 disables it.")
    p.add_argument("--zoom_min", type=float, default=0.7,
                   help="Min zoom factor s (s<1 = farther/smaller).")
    p.add_argument("--zoom_max", type=float, default=1.3,
                   help="Max zoom factor s (s>1 = closer/bigger).")
    p.add_argument("--color_aug", choices=["off", "light", "strong"], default="light",
                   help="Photometric jitter strength on BGR (white-balance/hue/sat/"
                        "brightness/contrast). 'off' = no color jitter.")

    return p.parse_args()


# ====================================================================================
# MAIN
# ====================================================================================

if __name__ == "__main__":
    args = parse_args()

    print(f"Experiment  : {args.experiment_name}")
    print(f"LR          : {args.learning_rate}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Iterations  : {args.num_iterations}")
    print(f"RGB dropout : {args.rgb_dropout}")
    print(f"Zoom aug    : prob={args.zoom_prob}  s∈[{args.zoom_min}, {args.zoom_max}]")
    print(f"Color aug   : {args.color_aug}")

    # --- Register datasets ---
    for split in ["train", "val"]:
        DatasetCatalog.register(
            f"TomatoDataset_{split}",
            lambda s=split: utils_detection.get_tomato_dataset_dicts(args.dataset_path, s),
        )
        MetadataCatalog.get(f"TomatoDataset_{split}").set(thing_classes=["tomato_bunch"])

    # --- Config ---
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))

    cfg.MODEL.META_ARCHITECTURE = "RGBDGeneralizedRCNN"
    if args.weights:
        print(f"Loading weights from: {args.weights}")
        cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 915.043]
    cfg.MODEL.PIXEL_STD = [57.375, 57.120, 58.395, 411.683]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_MASK_HEAD.LOSS_WEIGHT = 1.0
    cfg.MODEL.ROI_MASS_HEAD.MASS_LOSS_WEIGHT = args.mass_loss_weight
    cfg.MODEL.BACKBONE.FREEZE_AT = 0

    cfg.DATASETS.TRAIN = ("TomatoDataset_train",)
    cfg.DATASETS.TEST = ("TomatoDataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.SOLVER.MAX_ITER = args.num_iterations
    cfg.TEST.EVAL_PERIOD = args.eval_period
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period

    # LR schedule: linear warmup, then ×0.1 decay at 80% of training
    cfg.SOLVER.WARMUP_ITERS  = min(500, args.num_iterations // 10)
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.STEPS = (int(args.num_iterations * 0.8),)
    cfg.SOLVER.GAMMA = 0.1
    print(f"LR decay scheduled at iter: {cfg.SOLVER.STEPS[0]}")

    cfg.OUTPUT_DIR   = os.path.join("./output", args.experiment_name)
    cfg.DATASET_PATH = args.dataset_path
    cfg.RGB_DROPOUT  = args.rgb_dropout
    cfg.ZOOM_PROB    = args.zoom_prob
    cfg.ZOOM_MIN     = args.zoom_min
    cfg.ZOOM_MAX     = args.zoom_max
    cfg.COLOR_AUG    = args.color_aug

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # --- Mass loss weight schedule ---
    if args.mass_loss_schedule:
        TomatoTrainer._mass_loss_schedule = [
            (int(part.split(":")[0]), float(part.split(":")[1]))
            for part in args.mass_loss_schedule.split(",")
        ]
        # Set initial weight from schedule (iter 0 value)
        cfg.MODEL.ROI_MASS_HEAD.MASS_LOSS_WEIGHT = TomatoTrainer._mass_loss_schedule[0][1]
        print(f"Mass loss schedule: {TomatoTrainer._mass_loss_schedule}")
    else:
        TomatoTrainer._mass_loss_schedule = None

    # --- Build trainer ---
    trainer = TomatoTrainer(cfg)
    trainer.resume_or_load(resume=False)

    # --- Re-initialize depth channel weights (optional) ---
    # mask_rcnn_R50_FPN_3x_RGBD.pth already has depth channel initialized from RGB average.
    # Use --init_depth_weights only if you want to reset them.
    if args.init_depth_weights:
        conv1 = trainer.model.backbone.bottom_up.stem.conv1  # shape: (64, 4, 7, 7)
        with torch.no_grad():
            avg = conv1.weight[:, :3].mean(dim=1, keepdim=True)
            conv1.weight[:, 3:4].copy_(avg)
        print("Re-initialized depth channel (conv1 ch3) weights from RGB average.")

    # --- Freeze layers (optional) ---
    if args.freeze_backbone:
        for name, param in trainer.model.named_parameters():
            if name.startswith("backbone."):
                param.requires_grad = False
        print("Backbone frozen.")

    if args.freeze_det:
        for name, param in trainer.model.named_parameters():
            if "mass" not in name:
                param.requires_grad = False
        print("Detection/segmentation branch frozen. Training mass head only.")

    if args.freeze_mass:
        for name, param in trainer.model.named_parameters():
            if "mass" in name:
                param.requires_grad = False
        print("Mass head frozen. Training detection/segmentation only.")

    if args.freeze_backbone or args.freeze_det or args.freeze_mass:
        trainer.optimizer = trainer.build_optimizer(cfg, trainer.model)

    # --- Train ---
    trainer.train()
    print("Training complete.")
