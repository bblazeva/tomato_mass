"""
====================================================================================
GENERALIZED R-CNN AND RGB-D VARIANT
====================================================================================

This file contains the main model architectures:
- GeneralizedRCNN: Standard Mask R-CNN
- RGBDGeneralizedRCNN: Custom variant for RGB-D (4-channel) input

DEPTH FLOW:
===========

                    train_mapper()
                         |
                  "image" (4-ch BGRD)
                         |
                         v
                  preprocess_image()        normalizes all 4 channels;
                         |                  PIXEL_MEAN[3]/PIXEL_STD[3] for depth
                         v
                  backbone(images.tensor)
                         |
                         v
                  FPN features (depth implicit in the features)
                         |
                         v
                  all heads, incl. mass head

Depth is consumed only as the 4th input channel (no explicit late-fusion path;
see RGBDGeneralizedRCNN). preprocess_image() normalizes it with PIXEL_MEAN[3] and
PIXEL_STD[3] — if depth values are out of the expected range, normalization fails.

====================================================================================
"""

# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import os

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY

__all__ = ["GeneralizedRCNN", "ProposalNetwork", "RGBDGeneralizedRCNN"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        output_dir: str,
        vis_period: int = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.output_dir = output_dir

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "output_dir": cfg.OUTPUT_DIR
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Forward pass for training.

        FLOW:
        1. preprocess_image() - normalizes 4-channel input
        2. backbone() - extracts FPN features
        3. proposal_generator() - generates region proposals
        4. roi_heads() - predicts boxes, masks, and mass
        """
        if not self.training:
            return self.inference(batched_inputs)

        # =====================================================================
        # DEPTH DEBUG POINT: preprocess_image normalizes depth (4th channel)
        # =====================================================================
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        # =====================================================================
        # DEPTH DEBUG POINT: Backbone processes 4-channel input
        # The first conv layer has been expanded from 3->4 channels
        # Depth information flows through backbone implicitly
        # =====================================================================
        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """Run inference on the given inputs."""
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        # Log current inference images
        os.makedirs(self.output_dir, exist_ok=True)
        txt_path = os.path.join(self.output_dir, "current_images_inference.txt")
        with open(txt_path, "w") as f:
            for inp in batched_inputs:
                if "filename" in inp:
                    f.write(inp["filename"] + "\n")

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.

        =====================================================================
        DEPTH DEBUG POINT: This normalizes ALL channels including depth
        =====================================================================

        For 4-channel input:
        - Channels 0,1,2 (BGR): normalized with ImageNet-like stats
        - Channel 3 (Depth): normalized with PIXEL_MEAN[3], PIXEL_STD[3]

        If your depth values are outside the expected range:
        - Expected: mean ~0.915m, std ~0.41m (based on your dataset)
        - Check: Print depth before/after normalization to verify
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """Rescale the output instances to the target size."""
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


# ====================================================================================
# RGB-D GENERALIZED R-CNN - Custom variant for depth-aware mass prediction
# ====================================================================================

@META_ARCH_REGISTRY.register()
class RGBDGeneralizedRCNN(GeneralizedRCNN):
    """
    RGB-D variant of GeneralizedRCNN for depth-aware mass prediction.

    Depth reaches the model through a single path: the 4th channel of the input
    image is normalized in preprocess_image() (PIXEL_MEAN/PIXEL_STD are 4-element)
    and consumed by the backbone stem, so depth is implicit in the FPN features
    that every head — including the mass head — reads.

    NOTE (late fusion removed): an earlier variant additionally cached the raw
    depth (_cache_depth) and concatenated it as an extra channel to the FPN
    features feeding the mass head (late fusion). That path was removed after
    ablations showed it gave no improvement, so this class no longer overrides
    forward()/inference() or caches depth. It is kept only to preserve the
    registered meta-architecture name used by configs
    (cfg.MODEL.META_ARCHITECTURE = "RGBDGeneralizedRCNN").
    """
    pass


# ====================================================================================
# PROPOSAL NETWORK (Standard Detectron2 - not modified)
# ====================================================================================

@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    """A meta architecture that only predicts object proposals."""

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        output_dir: str,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.output_dir = output_dir
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)

        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
