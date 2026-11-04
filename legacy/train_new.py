"""
RGB-D Tomato Multitask Training Script
=======================================
Trains a Mask R-CNN model with an additional mass regression head.
Input: 4-channel RGB-D images [B, G, R, Depth_mm].

Depth enters the model through early fusion: depth as the 4th input channel,
processed jointly with RGB through the backbone and FPN.

Usage:
    python train_new.py \\
        --experiment_name my_run \\
        --dataset_path ./data/ \\
        --num_iterations 20000 \\
        --learning_rate 0.00025 \\
        --mass_loss_schedule "0:0,3000:0.5,5000:1.0" \\
        --rgb_dropout 0.3
"""

import os
import copy
import time
import random
import logging
import argparse
import datetime
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


# ====================================================================================
# CONSTANTS
# ====================================================================================

DEPTH_MEAN_MM = 915.043   # mean depth of training set (mm)
DEPTH_STD_MM  = 411.683   # std depth of training set (mm)


# ====================================================================================
# STRATIFIED SAMPLER
# ====================================================================================

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
    - Depth channel: raw millimeter values; 0 (invalid) filled with depth mean (915.043mm → 0 after norm)

    Returns: (dataset_dict_copy, image_hwc_float32)
    """
    dataset_dict = copy.deepcopy(dataset_dict)

    image_bgr = utils_detectron.read_image(dataset_dict["file_name"], format="BGR")
    H, W = image_bgr.shape[:2]

    depth_mm = np.load(dataset_dict["depth_file"]).astype(np.float32)
    if depth_mm.shape != (H, W):
        depth_mm = cv2.resize(depth_mm, (W, H), interpolation=cv2.INTER_AREA)

    # Replace invalid depth pixels (0mm = no sensor return) with the depth mean.
    # After normalization: (915.043 - 915.043) / 411.683 = 0 → neutral signal.
    # Without this, 0mm → (0 - 915.043) / 411.683 = -2.22, a large spurious value.
    DEPTH_MEAN_MM = 915.043
    depth_mm[depth_mm == 0] = DEPTH_MEAN_MM

    image = np.zeros((H, W, 4), dtype=np.float32)
    image[:, :, :3] = image_bgr.astype(np.float32)
    image[:, :, 3]  = depth_mm

    return dataset_dict, image


def train_mapper(dataset_dict, rgb_dropout=0.0):
    """
    Training data mapper.

    - Geometric augmentation (flip + multi-scale resize) applied to all 4 channels
    - RGB dropout: with probability rgb_dropout, zero BGR channels entirely
      (forces backbone to rely on depth when RGB is absent)
    - Color jitter on BGR channels only (when not dropped out)
    """
    dataset_dict, image = load_rgbd_image(dataset_dict)

    # Geometric augmentation — all 4 channels transformed together
    augs = T.AugmentationList([
        T.RandomFlip(horizontal=True, vertical=False),
    ])
    aug_input = T.AugInput(image)
    transform = augs(aug_input)
    image = aug_input.image  # (H, W, 4), float32

    # RGB dropout or color jitter (mutually exclusive per sample)
    if rgb_dropout > 0.0 and random.random() < rgb_dropout:
        image[:, :, 0] = 103.530  # B
        image[:, :, 1] = 116.280  # G
        image[:, :, 2] = 123.675  # R
        #image[:, :, :3] = 0.0  # zero BGR, depth channel untouched
    elif random.random() < 0.5:
        # Brightness
        bri = random.uniform(0.6, 1.4)
        image[:, :, :3] = np.clip(image[:, :, :3] * bri, 0, 255)
        # Contrast
        gray_mean = image[:, :, :3].mean()
        con = random.uniform(0.6, 1.4)
        image[:, :, :3] = np.clip(
            gray_mean + (image[:, :, :3] - gray_mean) * con, 0, 255
        )

    annos = [
        utils_detectron.transform_instance_annotations(ann, [transform], image.shape[:2])
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


def val_mapper(dataset_dict):
    """Validation mapper — no augmentation."""
    dataset_dict, image = load_rgbd_image(dataset_dict)

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
                g       = param.grad
                rgb_g   = g[:, :3].abs().mean().item()
                depth_g = g[:,  3].abs().mean().item()
                self.trainer.storage.put_scalar("grad/rgb",   rgb_g)
                self.trainer.storage.put_scalar("grad/depth", depth_g)
                print(f"[iter {self.trainer.iter}] grad/rgb={rgb_g:.2e}  grad/depth={depth_g:.2e}  ratio={depth_g/max(rgb_g, 1e-12):.3f}")
                return


# ====================================================================================
# TRAINER
# ====================================================================================

class TomatoTrainer(DefaultTrainer):

    _mass_loss_schedule = None  # set before instantiation if using a schedule

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
        hooks.append(PeriodicWriter([TensorboardXWriter(self.cfg.OUTPUT_DIR)], period=20))
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
        return build_detection_train_loader(
            cfg,
            mapper=lambda d: train_mapper(d, rgb_dropout=rgb_dropout),
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

    cfg.MODEL.META_ARCHITECTURE       = "RGBDGeneralizedRCNN"
    if args.weights:
        print(f"Loading weights from: {args.weights}")
        cfg.MODEL.WEIGHTS             = args.weights
    # cfg.MODEL.PIXEL_MEAN              = [103.530, 116.280, 123.675, DEPTH_MEAN_MM]
    cfg.MODEL.PIXEL_STD               = [1.0,     1.0,     1.0,     DEPTH_STD_MM]
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 915.043]
    #cfg.MODEL.PIXEL_STD = [57.375, 57.120, 58.395, 411.683]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES   = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST     = 0.7
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE  = 512
    cfg.MODEL.ROI_MASK_HEAD.LOSS_WEIGHT       = 1.0
    cfg.MODEL.ROI_MASS_HEAD.MASS_LOSS_WEIGHT  = args.mass_loss_weight
    cfg.MODEL.BACKBONE.FREEZE_AT              = 0

    cfg.DATASETS.TRAIN           = ("TomatoDataset_train",)
    cfg.DATASETS.TEST            = ("TomatoDataset_val",)
    cfg.DATALOADER.NUM_WORKERS   = 4

    cfg.SOLVER.IMS_PER_BATCH     = args.batch_size
    cfg.SOLVER.BASE_LR           = args.learning_rate
    cfg.SOLVER.MAX_ITER          = args.num_iterations
    cfg.TEST.EVAL_PERIOD         = args.eval_period
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period

    # LR schedule: linear warmup, then ×0.1 decay at 80% of training
    cfg.SOLVER.WARMUP_ITERS  = min(500, args.num_iterations // 10)
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.STEPS         = (int(args.num_iterations * 0.8),)
    cfg.SOLVER.GAMMA         = 0.1
    print(f"LR decay scheduled at iter: {cfg.SOLVER.STEPS[0]}")

    cfg.OUTPUT_DIR   = os.path.join("./output", args.experiment_name)
    cfg.DATASET_PATH = args.dataset_path
    cfg.RGB_DROPOUT  = args.rgb_dropout

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
