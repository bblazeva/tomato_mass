"""
====================================================================================
MULTITASK RGB-D FRUIT DETECTION AND SIZING - TRAINING SCRIPT
====================================================================================

This script trains a Mask R-CNN model with an additional mass regression head.
The model uses RGB-D (4-channel) input: RGB + Depth.

ARCHITECTURE OVERVIEW:
----------------------
1. Input: 4-channel image [B, G, R, Depth]
2. Backbone: ResNet-50 + FPN (modified first conv: 3->4 channels)
3. Heads:
   - Box head: bounding box detection
   - Mask head: instance segmentation
   - Mass head: mass regression (custom addition)

DEPTH FLOW:
------------------------------------
There are TWO paths for depth information:

PATH 1 - Backbone (4th channel):
   train_mapper()
   -> image[:,:,3] = depth / 1000.0  (normalize to meters)
   -> preprocess_image() in rcnn.py normalizes with PIXEL_MEAN[3], PIXEL_STD[3]
   -> backbone processes all 4 channels together
   -> FPN features contain depth information implicitly

PATH 2 - Mass head (explicit depth concat):
   train_mapper()
   -> returns separate "depth" tensor
   -> RGBDGeneralizedRCNN._cache_depth() stores normalized depth
   -> roi_heads._concat_cached_depth_to_features() concatenates to FPN features
   -> mass_head receives features with +1 channel (depth)

POTENTIAL DEPTH LEARNING ISSUES:
--------------------------------
1. If depth has low variance after normalization, network may ignore it
2. If backbone already learned to ignore 4th channel, mass head depth concat won't help
3. Check: Are gradients flowing to depth channel? (DepthGradHook monitors this)
4. Check: Is depth signal strong enough? Print depth stats in _cache_depth()

====================================================================================
"""

import os
import json
import cv2
import copy
import time
import datetime
import logging
import random
import argparse
import inspect
from collections import Counter

import numpy as np
import torch
import torchvision
import pickle5 as pickle

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import wandb

from torch.utils.data import DataLoader, WeightedRandomSampler, Sampler

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
    DatasetMapper,
    build_batch_data_loader,
)
from detectron2.data import detection_utils as utils_detectron
import detectron2.data.transforms as T

from detectron2.evaluation import COCOEvaluator, DatasetEvaluators

import utils_detection
from mass_evaluator import MassEvaluator

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling import META_ARCH_REGISTRY

import uuid


# ====================================================================================
# STRATIFIED SAMPLER - Ensures balanced sampling across mass bins
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
# DATA MAPPERS - Build 4-channel RGB-D images
# ====================================================================================
#
# DEPTH DEBUG POINT #1: This is where depth enters the pipeline
# The depth is loaded from .npy files and combined with RGB
#
# ====================================================================================

def _build_4ch_image_and_annos(dataset_dict):
    """
    Build 4-channel image. 
   
    DEPTH PROCESSING:
    -----------------
    1. Load depth from .npy file
    2. Resize to match RGB image dimensions
    3. Normalize: depth_meters = depth_mm / 1000.0
    4. Combine into 4-channel: [B, G, R, depth_meters]

    DEBUG: Add prints here to check depth statistics:
        print(f"Depth stats: min={depth_map.min()}, max={depth_map.max()}, mean={depth_map.mean()}")
    """
    dataset_dict = copy.deepcopy(dataset_dict)

    image_bgr = utils_detectron.read_image(dataset_dict["file_name"], format="BGR")

    if not os.path.exists(dataset_dict["depth_file"]):
        print("DEPTH FILE NOT FOUND:", dataset_dict["depth_file"])

    # =========================================================================
    # DEPTH DEBUG POINT #2: Raw depth loading
    # =========================================================================
    depth_map = np.load(dataset_dict["depth_file"])

    # DEBUG: Uncomment to check raw depth values
    # print(f"[DEBUG] Raw depth: min={depth_map.min():.3f}, max={depth_map.max():.3f}, "
    #       f"mean={depth_map.mean():.3f}, std={depth_map.std():.3f}")

    try:
        depth_map = cv2.resize(
            depth_map,
            (image_bgr.shape[1], image_bgr.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    except Exception as e:
        raise Exception("COULD NOT RESIZE " + dataset_dict["depth_file"]) from e

    # =========================================================================
    # DEPTH DEBUG POINT #3: Depth normalization (mm -> meters)
    # =========================================================================
    # Build 4-channel image: BGR + depth (in meters)
    image = np.zeros((image_bgr.shape[0], image_bgr.shape[1], 4), dtype=np.float32)
    image[:, :, 0:3] = image_bgr.astype(np.float32)
    depth_mm = depth_map.astype(np.float32)
    depth_cache = depth_mm.copy()
    valid = depth_mm > 0
    depth_mm[~valid] = 915.043  # depth mean in mm
    image[:, :, 3] = 915.043
    depth_cache[:] = 0.0

    # =========================================================================
    # DEPTH-ONLY SANITY CHECK: replicate depth into BGR channels
    # Standard approach for feeding single-channel data into pretrained backbones
    # Comment out these lines to go back to normal RGB-D
    #
    # GLOBAL depth normalization: same [200, 2000] mm range across every image,
    # so absolute depth scale is preserved (was per-image min/max before, which
    # destroyed cross-image scale comparability).
    # Range covers the dataset: actual extremes are ~274-1901 mm, with margin.
    # =========================================================================
    DEPTH_GLOBAL_MIN_MM = 200.0
    DEPTH_GLOBAL_MAX_MM = 2000.0
    depth_visual = depth_mm.copy()
    depth_visual = np.clip(
        (depth_visual - DEPTH_GLOBAL_MIN_MM)
        / (DEPTH_GLOBAL_MAX_MM - DEPTH_GLOBAL_MIN_MM) * 255.0,
        0, 255,
    )
    image[:, :, 0] = depth_visual  # B
    image[:, :, 1] = depth_visual  # G
    image[:, :, 2] = depth_visual  # R
    #image[:, :, 3] = depth_map.astype(np.float32) #/ 1000.0  # Convert mm to 

    # DEBUG: Uncomment to check normalized depth
    # print(f"[DEBUG] Normalized depth (meters): min={image[:,:,3].min():.3f}, "
    #       f"max={image[:,:,3].max():.3f}, mean={image[:,:,3].mean():.3f}")

    return dataset_dict, image, depth_cache


# Debug flags for saving augmented images
DEBUG_SAVE_AUG = False
DEBUG_SAVE_PROB = 0.02  # 2% of samples will be saved


def train_mapper(dataset_dict):
    """
    Mapper with augmentation (train only).
    - Geometric aug on all 4 channels (flip + resize)
    - Light Gaussian noise on depth channel

    DEPTH FLOW:
    -----------
    1. _build_4ch_image_and_annos() creates 4-channel image
    2. Augmentations applied to all 4 channels together
    3. Depth noise added (~1cm standard deviation)   - removed
    4. Returns BOTH:
       - "image": 4-channel tensor for backbone (depth is 4th channel)
       - "depth": separate depth tensor for mass head (via _cache_depth)

    WHY TWO DEPTH PATHS?
    The backbone sees depth as 4th channel (learns features jointly).
    The mass head receives depth concatenated to FPN features (explicit depth).
    This redundancy may actually cause issues - the network might rely on
    one path and ignore the other.
    """
    dataset_dict, image, depth_map = _build_4ch_image_and_annos(dataset_dict)

    # =========================================================================
    # GEOMETRIC AUGMENTATION (applied to RGB+Depth together)
    # Only horizontal flip — no multi-scale resize (was producing train/val
    # scale mismatch), no color jitter (was corrupting depth signal when BGR
    # channels carry replicated depth).
    # =========================================================================
    augs = T.AugmentationList([
        T.RandomFlip(horizontal=True, vertical=False)
    ])

    aug_input = T.AugInput(image)
    transform = augs(aug_input)
    image = aug_input.image  # transformed 4-channel image (float32, HWC)

    depth_raw_hwc = depth_map[:, :, None]
    depth_raw_aug = transform.apply_image(depth_raw_hwc)[:, :, 0]
    depth_tensor_cache = torch.from_numpy(depth_raw_aug.copy()).unsqueeze(0).float()

    # =========================================================================
    # ANNOTATIONS (use same geometric transform)
    # =========================================================================
    annos = [
        utils_detectron.transform_instance_annotations(
            annotation, [transform], image.shape[0:2]
        )
        for annotation in dataset_dict.pop("annotations")
    ]

    # =========================================================================
    # CONVERT TO TENSORS
    # =========================================================================
    # image_tensor: (4, H, W) - goes to backbone
    # depth_tensor: (1, H, W) - goes to _cache_depth for mass head
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()

    # =========================================================================
    # (OPTIONAL) DEBUG: Save augmented images
    # =========================================================================
    if DEBUG_SAVE_AUG and random.random() < DEBUG_SAVE_PROB:
        save_folder = "./augmented_debug"
        os.makedirs(save_folder, exist_ok=True)
        img_4ch = image_tensor.numpy().transpose(1, 2, 0)
        aug_bgr = np.clip(img_4ch[:, :, :3], 0, 255).astype("uint8")
        depth_channel = img_4ch[:, :, 3]
        depth_norm = cv2.normalize(depth_channel, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype("uint8")
        uid = f"{dataset_dict['image_id']}_{uuid.uuid4().hex[:6]}"
        cv2.imwrite(os.path.join(save_folder, f"{uid}_rgb.jpg"), aug_bgr)
        cv2.imwrite(os.path.join(save_folder, f"{uid}_depth.jpg"), depth_norm)
        metadata = MetadataCatalog.get("TomatoDataset_train")
        annotated_dict = {
            "file_name": dataset_dict["file_name"],
            "image_id": dataset_dict["image_id"],
            "height": image_tensor.shape[1],
            "width": image_tensor.shape[2],
            "annotations": annos,
        }
        visualizer = Visualizer(aug_bgr, metadata=metadata, scale=1.0)
        vis_output = visualizer.draw_dataset_dict(annotated_dict)
        cv2.imwrite(os.path.join(save_folder, f"{uid}_ann.jpg"), vis_output.get_image())

    # =========================================================================
    # RETURN DICT
    # =========================================================================
    return {
        "filename": dataset_dict["file_name"],
        "image_id": dataset_dict["image_id"],
        "height": dataset_dict["height"],
        "width": dataset_dict["width"],
        "image": image_tensor,      # 4-channel for backbone
        "depth": depth_tensor_cache,      # separate depth for mass head
        "instances": utils_detectron.annotations_to_instances(
            annos, image_tensor.shape[1:]
        ),
    }


def test_mapper(dataset_dict):
    """
    Mapper WITHOUT augmentation (for val/test).
    Same depth handling as train_mapper, just no augmentation.
    """
    dataset_dict, image, depth_map = _build_4ch_image_and_annos(dataset_dict)

    # No augmentation
    annos = [
        utils_detectron.transform_instance_annotations(
            annotation, [], image.shape[0:2]
        )
        for annotation in dataset_dict.pop("annotations")
    ]

    image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
    depth_tensor = torch.from_numpy((image[:, :, 3]).copy()).unsqueeze(0).float()
    depth_tensor_cache = torch.from_numpy(depth_map.copy()).unsqueeze(0).float()

    return {
        "filename": dataset_dict["file_name"],
        "image_id": dataset_dict["image_id"],
        "height": dataset_dict["height"],
        "width": dataset_dict["width"],
        "image": image_tensor,
        "depth": depth_tensor_cache,
        "instances": utils_detectron.annotations_to_instances(
            annos, image_tensor.shape[1:]
        ),
    }


# ====================================================================================
# VALIDATION LOSS HOOK - Computes loss on validation set during training
# ====================================================================================

class LossEvalHook(HookBase):
    def __init__(self, cfg, model, data_loader):
        self._model = model
        self._period = cfg.TEST.EVAL_PERIOD
        self._data_loader = data_loader
        self._output_dir = cfg.OUTPUT_DIR
        self._mass_loss_weight = cfg.MODEL.ROI_MASS_HEAD.MASS_LOSS_WEIGHT

    def _do_loss_eval(self):
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        mass_losses = []

        for idx, inputs in enumerate(self._data_loader):
            with open(os.path.join(self._output_dir, 'current_images.txt'), 'w') as f:
                f.write("%s\n" % inputs[0]['filename'])

            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start

            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(
                    seconds=int(total_seconds_per_img * (total - idx - 1))
                )
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )

            loss_batch, mass_loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
            mass_losses.append(mass_loss_batch)

        mean_loss = np.mean(losses)
        mean_mass_loss = np.mean(mass_losses)
        mean_det_loss = mean_loss - mean_mass_loss
        

        # The corrected loss - actual loss scaled with the loss weight 
        if self._mass_loss_weight > 0:
            mean_mass_loss_corrected = mean_mass_loss / self._mass_loss_weight
            mean_loss_corrected = mean_det_loss + mean_mass_loss_corrected
        else:
            mean_mass_loss_corrected = 0.0
            mean_loss_corrected = mean_det_loss

        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        self.trainer.storage.put_scalar('validation_mass_loss', mean_mass_loss)
        self.trainer.storage.put_scalar('validation_det_loss', mean_det_loss)
        self.trainer.storage.put_scalar('validation_mass_loss_corr', mean_mass_loss_corrected)
        self.trainer.storage.put_scalar('validation_loss_corr', mean_loss_corrected)
        comm.synchronize()

        return losses

    def _get_loss(self, data):
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        val_losses_file = os.path.join(self._output_dir, 'validation_losses.pkl')
        if os.path.exists(val_losses_file):
            with open(val_losses_file, 'rb') as handle:
                saved_losses = pickle.load(handle)
            saved_losses.append(metrics_dict)
            with open(val_losses_file, 'wb') as handle:
                pickle.dump(saved_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            saved_losses = [metrics_dict]
            with open(val_losses_file, 'wb') as handle:
                pickle.dump(saved_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)

        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        mass_losses_reduced = metrics_dict['loss_mass']
        return total_losses_reduced, mass_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)


# ====================================================================================
# DEPTH GRADIENT HOOK - Monitors if gradients are flowing to depth channel
# ====================================================================================
#
# THIS IS CRITICAL FOR DEBUGGING YOUR DEPTH LEARNING ISSUE!
#
# If grad/depth is always 0 or much smaller than grad/rgb, the network
# is not learning to use depth information.
#
# ====================================================================================

class DepthGradHook(HookBase):
    """
    Monitors gradient magnitude for RGB vs Depth channels in the first conv layer.

    WHAT TO LOOK FOR:
    -----------------
    - grad/rgb: average gradient magnitude for RGB channels
    - grad/depth: average gradient magnitude for depth channel

    If grad/depth << grad/rgb consistently, the network is ignoring depth.
    If grad/depth ~= 0, depth gradients are not flowing at all.

    DEBUGGING TIPS:
    ---------------
    1. If grad/depth is 0: Check if depth values have variance
    2. If grad/depth << grad/rgb: Try increasing depth signal (different normalization)
    3. If gradients are similar but depth still ignored: Architecture issue
    """
    def after_step(self):
        """Called after each training step (after optimizer.step())"""
        # Only check every 20 iterations to reduce spam
        if self.trainer.iter % 20 != 0:
            return

        found = False
        for name, param in self.trainer.model.named_parameters():
            if name == "backbone.bottom_up.stem.conv1.weight":
                found = True
                if param.grad is None:
                    print("[GRAD DEBUG] *** BACKBONE IS FROZEN *** - No gradients for conv1.weight!")
                    print(f"[GRAD DEBUG] requires_grad = {param.requires_grad}")
                    return
                g = param.grad
                # RGB channels: indices 0,1,2
                rgb_g = g[:, :3].abs().mean().item()
                # Depth channel: index 3
                d_g = g[:, 3].abs().mean().item()

                self.trainer.storage.put_scalar("grad/rgb", rgb_g)
                self.trainer.storage.put_scalar("grad/depth", d_g)

                # DEBUG: Print gradient ratio
                ratio = d_g / (rgb_g + 1e-8)
                print(f"[GRAD DEBUG] RGB: {rgb_g:.6f}, Depth: {d_g:.6f}, Ratio: {ratio:.4f}")
                return
        if not found:
            print("[GRAD DEBUG] WARNING: Layer 'backbone.bottom_up.stem.conv1.weight' not found!")


# ====================================================================================
# CUSTOM TRAINER
# ====================================================================================

class MyTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Create evaluator(s) for a given dataset."""
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = []
        evaluators.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        evaluators.append(MassEvaluator())
        return DatasetEvaluators(evaluators)

    def build_hooks(self):
        print('BUILD_DETECTION_TEST_MODEL')
        hooks = super().build_hooks()

        # Validation loss hook
        hooks.insert(
            -1,
            LossEvalHook(
                self.cfg,
                self.model,
                build_detection_test_loader(
                    self.cfg,
                    self.cfg.DATASETS.TEST[0],
                    test_mapper
                )
            )
        )

        # Depth gradient monitoring hook
        hooks.append(DepthGradHook())

        # TensorBoard writer
        writer = TensorboardXWriter(self.cfg.OUTPUT_DIR)
        hooks.append(PeriodicWriter([writer], period=1))
        return hooks

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """Use 4-channel mapper at test time (no augmentation)."""
        return build_detection_test_loader(
            cfg,
            dataset_name,
            mapper=test_mapper,
        )

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Build train loader with stratified sampling by mass bins.

        Mass bins:
        - Small: < 250g
        - Medium: 250-500g
        - Large: > 500g

        This ensures the model sees equal representation of all mass ranges.
        """
        dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])

        def mass_bin(m):
            if m < 0.25:
                return 0  # Small
            elif m < 0.5:
                return 1  # Medium
            else:
                return 2  # Large

        for d in dataset_dicts:
            d["mass_bin"] = mass_bin(d["cluster_mass"])

        bin_counts = Counter(d["mass_bin"] for d in dataset_dicts)
        bin_weights = {b: 1.0 / c for b, c in bin_counts.items()}
        sample_wts = [bin_weights[d["mass_bin"]] for d in dataset_dicts]

        weighted_sampler = WeightedRandomSampler(
            weights=sample_wts,
            num_samples=len(sample_wts),
            replacement=True,
        )

        sampler = _InfiniteSampler(weighted_sampler)

        return build_detection_train_loader(
            cfg,
            mapper=train_mapper,
            sampler=sampler,
            output_dir=cfg.OUTPUT_DIR,
        )


# ====================================================================================
# ARGUMENT PARSING
# ====================================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train multitask RGB-D tomato model')
    parser.add_argument('--num_iterations', dest='num_iterations', default=10000,
                        help='maximum number of iterations (not epochs)')
    parser.add_argument('--checkpoint_period', dest='checkpoint_period', default=500,
                        help='save checkpoint every X iterations')
    parser.add_argument('--eval_period', dest='eval_period', default=500,
                        help='evaluate every X iterations')
    parser.add_argument('--batch_size', dest='batch_size', default=2)
    parser.add_argument('--learing_rate', dest='learing_rate', default=0.00025)
    parser.add_argument('--mass_loss_weight', dest='mass_loss_weight', default=1)
    parser.add_argument('--experiment_name', dest='experiment_name', default='trial_PLS')
    parser.add_argument('--dataset_path', dest='dataset_path',
                        default='/home/student/Downloads/blagica/multitask_RGB-D_FruitDetectionAndSizing/data/')
    parser.add_argument('--batch_size_per_image', dest='bs_per_image', default=512)
    parser.add_argument('--weights', dest='weights', default="./mask_rcnn_R50_FPN_3x_RGBD.pth")
    parser.add_argument('--freeze_det', dest='freeze_det', default=0,
                        help='1 to freeze detection branch, only train mass head')
    args = parser.parse_args()
    return args


# ====================================================================================
# MAIN TRAINING LOOP
# ====================================================================================

if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    lr = float(args.learing_rate)
    mass_loss_weight = float(args.mass_loss_weight)
    bs = int(args.batch_size)
    max_iter = int(args.num_iterations)
    checkpoint_period = int(args.checkpoint_period)
    eval_period = int(args.eval_period)
    experiment_name = args.experiment_name
    dataset_path = args.dataset_path
    bs_per_image = int(args.bs_per_image)
    weights_file = args.weights
    freeze_det_arg = int(args.freeze_det)

    print('***EXPERIMENT:*** ', experiment_name)
    print('***BATCH_SIZE: ', bs)
    print('***LR: ', lr)

    # =========================================================================
    # REGISTER DATASETS
    # =========================================================================
    print('PREPARING DATA...')
    for d in ["train", "val"]:
        DatasetCatalog.register(
            "TomatoDataset_" + d,
            lambda d=d: utils_detection.get_tomato_dataset_dicts(dataset_path, d)
        )
        MetadataCatalog.get("TomatoDataset_" + d).set(thing_classes=["tomato_bunch"])
    TomatoDataset_metadata = MetadataCatalog.get("TomatoDataset_train")

    # =========================================================================
    # MODEL CONFIGURATION
    # =========================================================================
    cfg = get_cfg()

    # =========================================================================
    # DEPTH DEBUG POINT #4: Pixel normalization for 4-channel input
    # =========================================================================
    # These values normalize the input:
    # - RGB channels: standard ImageNet-ish means (BGR order)
    # - Depth channel: mean=0.915043m, std=0.411683m
    #
    # CRITICAL: If your depth values are outside this range, normalization
    # will produce extreme values that the network can't handle!
    #
    # DEBUG: Check if your depth values match these statistics
    # =========================================================================

    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    #cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 915.043]
    #cfg.MODEL.PIXEL_STD = [57.375, 57.120, 58.395, 411.683]
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 915.043]
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 411.683]
    
    # Use our custom RGB-D architecture
    cfg.MODEL.META_ARCHITECTURE = "RGBDGeneralizedRCNN"

    cfg.DATASETS.TRAIN = ("TomatoDataset_train",)
    cfg.DATASETS.TEST = ("TomatoDataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.MODEL.WEIGHTS = weights_file

    cfg.SOLVER.IMS_PER_BATCH = bs
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = bs_per_image
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # tomato_bunch
    cfg.TEST.EVAL_PERIOD = eval_period
    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period
    cfg.OUTPUT_DIR = "./output/" + str(experiment_name)
    cfg.DATASET_PATH = dataset_path

    cfg.MODEL.ROI_MASK_HEAD.LOSS_WEIGHT = 1.0
    cfg.MODEL.ROI_MASS_HEAD.MASS_LOSS_WEIGHT = mass_loss_weight
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.BACKBONE.FREEZE_AT = 0

    # Freeze flags
    freeze_mass = 0
    freeze_det = freeze_det_arg

    # depth channel weights
    stage1 = True

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # BUILD AND INITIALIZE MODEL
    # =========================================================================
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)

    # =========================================================================
    # DEPTH DEBUG POINT #5: Initialize depth channel weights
    # =========================================================================
    # The backbone's first conv layer was expanded from 3 to 4 channels.
    # We initialize the 4th channel (depth) weights by averaging the RGB weights.
    #
    # POTENTIAL ISSUE: This initialization assumes depth should be processed
    # similarly to grayscale. If depth has very different characteristics,
    # this initialization might not be ideal.
    # =========================================================================
    # Q: This should only be dont the first time no? when loading the inital model for train, not in the other stages
    if stage1:
        conv1 = trainer.model.backbone.bottom_up.stem.conv1  # (64, 4, 7, 7)
        with torch.no_grad():
            avg = conv1.weight[:, :3].mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
            conv1.weight[:, 3:4].copy_(avg)

        print("[INFO] Initialized depth channel weights from RGB average")

    if torch.cuda.is_available():
        print(
            'Number of cuda devices = ' + str(torch.cuda.device_count()) +
            ' ; Current cuda device = ' +
            torch.cuda.get_device_name(torch.cuda.current_device())
        )
    else:
        print('Not using cuda devices')

    # =========================================================================
    # OPTIONAL: Freeze backbone (for fine-tuning mass head only)
    # Note: cfg.MODEL.BACKBONE.FREEZE_AT = 0 is set above to enable depth learning
    # =========================================================================
    freeze_backbone = 0  # Set to 1 for stage 2 training

    if freeze_backbone == 1:
        for name, param in trainer.model.named_parameters():
            if name.startswith("backbone."):
                param.requires_grad = False
                print(f"{name} frozen (backbone)")

    # =========================================================================
    # OPTIONAL: Freeze detection branch (only train mass head)
    # =========================================================================
    if freeze_det == 1:
        for name, param in trainer.model.named_parameters():
            if name.find('mass') == -1:
                param.requires_grad = False
                print(name + ' frozen')
            else:
                param.requires_grad = True
                print(name + ' will be trained')

    # =========================================================================
    # OPTIONAL: Freeze mass head (only train detection)
    # =========================================================================
    if freeze_mass == 1:
        for name, param in trainer.model.named_parameters():
            if 'mass' in name:
                param.requires_grad = False
                print(f"{name} frozen (mass head)")
            else:
                param.requires_grad = True
                print(f"{name} will be trained (segmentation)")

    # Rebuild optimizer with updated requires_grad
    trainer.optimizer = trainer.build_optimizer(cfg, trainer.model)

    print("PIXEL_MEAN:", cfg.MODEL.PIXEL_MEAN)
    print("PIXEL_STD :", cfg.MODEL.PIXEL_STD)

    # =========================================================================
    # START TRAINING
    # =========================================================================
    trainer.train()
    print('FINISHED_TRAINING')
