"""
====================================================================================
MASS PREDICTION HEAD - Custom Head for Fruit Mass Regression
====================================================================================

This module implements the mass prediction head for estimating fruit mass from
RGB-D features.

ARCHITECTURE:
=============

    ROI-pooled FPN features (N, 256, 14, 14)
            |
            v
    5 x Conv3x3 + ReLU            (operates at 14x14; 5th replaces the old deconv.
                                   norm is config-controlled via ROI_MASK_HEAD.NORM)
            |
            v
    AdaptiveAvgPool2d (to 4x4)
            |
            v
    Flatten -> FC(4096, 128) -> ReLU -> FC(128, 1)
            |
            v
    log(mass_kg) prediction


LOSS FUNCTION:
==============
MSE loss in log-space: MSE(log(pred), log(gt))

Why log-space?
- Mass varies over wide range (e.g., 0.1kg to 1.0kg)
- Log-space weights relative errors equally
- A 50g error matters more for a 100g fruit than a 500g fruit


DEPTH INPUT:
============
There is NO explicit depth channel in the mass head input. An earlier late-fusion
variant concatenated the raw depth as a 257th channel (in
roi_heads._concat_cached_depth_to_features), but it was removed after ablations
showed no improvement. The head now consumes the 256 FPN channels only; depth
reaches it implicitly via the 4th input channel through the shared backbone.

====================================================================================
"""

# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
import pdb
from torch.nn import functional as F
import cv2
import os
import numpy as np

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, cat, get_norm
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from sklearn.metrics import mean_squared_error


__all__ = [
    "BaseMassRCNNHead",
    "MassRCNNConvUpsampleHead",
    "build_mass_head",
    "ROI_MASS_HEAD_REGISTRY",
]


ROI_MASS_HEAD_REGISTRY = Registry("ROI_MASS_HEAD")
ROI_MASS_HEAD_REGISTRY.__doc__ = """
Registry for mass prediction heads.
The registered object will be called with `obj(cfg, input_shape)`.
"""


# ====================================================================================
# LOSS FUNCTION
# ====================================================================================

@torch.jit.unused
def mass_regr_loss(pred_mass_logits: torch.Tensor, instances: List[Instances], vis_period: int = 0):
    """
    Compute the mass regression loss in LOG-SPACE.

    =====================================================================
    LOSS COMPUTATION
    =====================================================================

    The loss is: MSE(log(pred_mass), log(gt_mass))

    This means:
    - pred_mass_logits is already in log-space (model outputs log(mass))
    - gt_mass is in linear space (kg), so we convert: log(gt_mass)
    - MSE in log-space penalizes relative errors equally

    Args:
        pred_mass_logits: Predicted log(mass) values, shape (N, 1)
        instances: List of Instances containing gt_mass field
        vis_period: Not used

    Returns:
        Scalar loss tensor
    =====================================================================
    """
    gt_mass = []

    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        gt_mass.append(instances_per_image.gt_mass)

    if len(gt_mass) == 0:
        return pred_mass_logits.sum() * 0

    gt_mass = cat(gt_mass, dim=0)

    # Squeeze prediction to 1D
    pred_mass_logits = pred_mass_logits[:, 0]

    # =====================================================================
    # DEBUG: Print ground truth masses
    # =====================================================================
    print('gt_mass (kg):', gt_mass)

    if gt_mass.dim() > 1:
        gt_mass = gt_mass[:, 0]

    if not gt_mass.is_cuda:
        gt_mass = gt_mass.cuda()

    # =====================================================================
    # CONVERT GT TO LOG-SCALE
    # =====================================================================
    # gt_mass is in kg (linear scale)
    # Model predicts log(mass), so we need log(gt_mass) for comparison
    gt_log_mass = torch.log(gt_mass)

    # pred_mass_logits is already log(mass) from the model
    pred_log_mass = pred_mass_logits

    # =====================================================================
    # DEBUG: Print predictions in kg (for interpretability)
    # =====================================================================
    pred_mass_kg = torch.exp(pred_log_mass)
    print('pred mass (kg):', pred_mass_kg)

    # =====================================================================
    # MSE LOSS IN LOG-SPACE
    # =====================================================================
    mass_loss = F.mse_loss(pred_log_mass, gt_log_mass)

    return mass_loss


# ====================================================================================
# INFERENCE FUNCTION
# ====================================================================================

def mass_inference(pred_mass_logits: torch.Tensor, pred_instances: List[Instances]):
    """
    Attach predicted mass to instances for inference.

    Note: pred_mass_logits is in LOG-SCALE.
    The evaluator (MassEvaluator) will convert back using exp().
    """
    for instances in pred_instances:
        instances.pred_mass = pred_mass_logits


# ====================================================================================
# BASE MASS HEAD
# ====================================================================================

class BaseMassRCNNHead(nn.Module):
    """
    Base class for mass prediction heads.
    """

    @configurable
    def __init__(self, *, loss_weight: float = 1, vis_period: int = 0,
                 output_dir: str = "/output", dataset_path: str = "/data"):
        super().__init__()
        self.vis_period = vis_period
        self.loss_weight = loss_weight
        self.output_dir = output_dir
        self.dataset_path = dataset_path
        print("Mass head initialized:")
        print(f"  - loss_weight = {self.loss_weight}")
        print(f"  - output_dir = {self.output_dir}")

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "vis_period": cfg.VIS_PERIOD,
            "loss_weight": cfg.MODEL.ROI_MASS_HEAD.MASS_LOSS_WEIGHT,
            "output_dir": cfg.OUTPUT_DIR,
            "dataset_path": cfg.DATASET_PATH
        }

    def forward(self, x, instances: List[Instances]):
        """
        Forward pass for mass prediction.

        Args:
            x: ROI-pooled FPN features, shape (N, 256, 14, 14)
            instances: Instances with gt_mass (training) or pred_boxes (inference)

        Returns:
            Training: dict with "loss_mass"
            Inference: updated instances with "pred_mass"
        """
        x = self.layers(x)

        if self.training:
            return {"loss_mass": mass_regr_loss(x, instances, self.vis_period) * self.loss_weight}
        else:
            mass_inference(x, instances)
            return instances

    def layers(self, x):
        """Neural network layers - to be implemented by subclasses."""
        raise NotImplementedError


# ====================================================================================
# MASS HEAD IMPLEMENTATION
# ====================================================================================

@ROI_MASS_HEAD_REGISTRY.register()
class MassRCNNConvUpsampleHead(BaseMassRCNNHead, nn.Sequential):
    """
    Mass prediction head with conv layers followed by MLP.

    Architecture:
    - 5 x (Conv3x3 + ReLU)   (5th replaces the old ConvTranspose2d; no upsample.
                              norm optional via ROI_MASK_HEAD.NORM, default none)
    - AdaptiveAvgPool2d (to 4x4)
    - FC(in_dim, 128) + ReLU
    - FC(128, 1)   (bias initialized to -0.8064, the mean log(mass) of the train set)

    Input: ROI-pooled FPN features, shape (N, 256, 14, 14). There is no explicit
    depth channel (late fusion removed); the first conv (mass_fcn1) has weights of
    shape (conv_dim, 256, 3, 3).
    """

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, num_classes, conv_dims, conv_norm="", **kwargs):
        super().__init__(**kwargs)
        assert len(conv_dims) >= 1, "conv_dims have to be non-empty!"

        self.conv_norm_relus = []

        # FPN features only (no late-fusion depth concat); typically 256 channels.
        cur_channels = input_shape.channels
        print(f"[MASS HEAD] Input channels: {cur_channels}")

        # NOTE: the ConvTranspose2d upsample layer was removed. It upsampled
        # 14x14 -> 28x28 only for AdaptiveAvgPool2d((4,4)) to immediately
        # collapse it back down — the spatial upsampling was wasted. However,
        # an ablation showed the deconv's ~262k learned params were contributing
        # capacity (R^2 dropped 0.757 -> 0.712 when it was simply removed).
        # So instead of dropping it, we replace it with an extra plain Conv3x3:
        # the loop now iterates over ALL of conv_dims (5 conv layers) instead of
        # conv_dims[:-1] (4). This keeps the learned capacity, drops the
        # pointless upsample, and operates entirely at 14x14.
        for k, conv_dim in enumerate(conv_dims):
            print(f'[MASS HEAD] Conv layer {k+1}: {cur_channels} -> {conv_dim}')
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            self.add_module("mass_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            cur_channels = conv_dim

        # Pooling and MLP
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(cur_channels * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 1)

        # Initialize weights
        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)

        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)
        # FC2 bias initialized to the mean log(mass_kg) of the training set, so
        # the head starts out predicting the dataset-average mass.

        #nn.init.constant_(self.fc2.bias, -0.8064)
        nn.init.constant_(self.fc2.bias, 0.0)


    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        ret.update({
            "input_shape": input_shape,
            "num_classes": 1 if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
                           else cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "conv_dims": [conv_dim] * (num_conv + 1),
            "conv_norm": cfg.MODEL.ROI_MASK_HEAD.NORM,
        })
        return ret

    def layers(self, x):
        """
        Forward through conv layers and MLP.

        Input x: ROI-pooled FPN features, shape (N, 256, 14, 14).
        """
        if x.shape[0] == 0:
            return x.new_empty((0, 1))

        # Conv layers (5 x Conv3x3+BN+ReLU; the 5th replaces the old deconv)
        for conv in self.conv_norm_relus:
            x = conv(x)

        # Pool 14x14 -> 4x4
        P = self.pool(x)

        # MLP
        flat = P.flatten(start_dim=1)
        h = F.relu(self.fc1(flat))
        return self.fc2(h)  # Output: log(mass)


# ====================================================================================
# BUILD FUNCTION
# ====================================================================================

def build_mass_head(cfg, input_shape):
    """Build the mass prediction head."""
    cfg.MODEL.ROI_MASS_HEAD.NAME = "MassRCNNConvUpsampleHead"
    name = cfg.MODEL.ROI_MASS_HEAD.NAME
    return ROI_MASS_HEAD_REGISTRY.get(name)(cfg, input_shape)
