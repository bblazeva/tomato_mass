"""
====================================================================================
DATASET UTILITIES FOR TOMATO DETECTION
====================================================================================

This module handles loading and preparing the tomato dataset in COCO format.

KEY FUNCTIONS:
- get_tomato_dataset_dicts(): Main function to load dataset

DATA FORMAT EXPECTED:
---------------------
root_dir/
├── gt_json/
│   ├── train/train.json   (COCO format annotations)
│   └── val/val.json
├── images/
│   ├── train/*.jpg
│   └── val/*.jpg
└── iter_4_depth_matched/
    └── *.npy              (depth maps, one per image)

IMAGE NAMING CONVENTION:
------------------------
Images should be named: MASS_*.jpg
Where MASS is the ground truth mass in grams.
Example: 250_image_001.jpg -> mass = 250g = 0.25kg

====================================================================================
"""

from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os
import json
import cv2
import random
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer, GenericMask
import matplotlib.pyplot as plt
import pdb
# Note: preprocess_input was imported but never used - removed


def get_tomato_dataset_dicts(root_dir, split):
    """
    Load dataset dictionaries for Detectron2.

    Args:
        root_dir: Path to data directory
        split: "train" or "val"

    Returns:
        List of dataset dicts, each containing:
        - file_name: path to RGB image
        - depth_file: path to depth .npy file
        - image_id, height, width
        - cluster_mass: mass in kg (extracted from filename)
        - annotations: list of instance annotations

    DEPTH FILE LOCATION:
    --------------------
    Depth files are expected at: root_dir/iter_4_depth_matched/<base_name>.npy
    Where <base_name> is the image filename without extension.
    """
    depth_path = root_dir + 'iter_4_depth_matched'
    json_file_path = os.path.join(root_dir + 'gt_json/' + split, str(split) + ".json")

    with open(json_file_path, "r") as f:
        coco_dict = json.load(f)

    # Create mapping from image_id to annotations
    annos_per_image = {}
    for ann in coco_dict["annotations"]:
        img_id = ann["image_id"]
        annos_per_image.setdefault(img_id, []).append(ann)

    dataset_dicts = []
    for image in coco_dict["images"]:
        record = {}

        file_name = image["file_name"]
        full_file_name = os.path.join(root_dir, "images", split, file_name)
        record["file_name"] = full_file_name
        record["image_id"] = image["id"]
        record["height"] = image["height"]
        record["width"] = image["width"]

        # Build depth file path
        base_name = os.path.splitext(file_name)[0]
        depth_file = os.path.join(depth_path, base_name + ".npy")
        record["depth_file"] = depth_file

        # =====================================================================
        # MASS EXTRACTION FROM FILENAME
        # =====================================================================
        # Filename format: MASS_*.jpg (e.g., "250_image_001.jpg")
        # Mass is extracted from first token before underscore
        # Converted from grams to kg
        # =====================================================================
        base_name = os.path.basename(file_name)
        try:
            mass_val = float(base_name.split('_')[0]) / 1000.0  # g to kg
        except Exception as e:
            print("Error extracting mass from", base_name, ":", e)
            mass_val = None

        record["cluster_mass"] = mass_val

        # Build annotations
        objs = []
        for ann in annos_per_image.get(image["id"], []):
            obj = {}
            obj["bbox"] = ann["bbox"]  # COCO format: [x, y, width, height]
            obj["bbox_mode"] = BoxMode.XYWH_ABS
            obj["segmentation"] = ann["segmentation"]
            obj["category_id"] = ann["category_id"]
            obj["mass"] = mass_val  # Add mass to each instance
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


# ====================================================================================
# UTILITY FUNCTIONS FOR EVALUATION
# ====================================================================================

def plot_masks(mask_plot, gt_mask, im):
    """Overlay masks on image for visualization."""
    for i in range(len(gt_mask)):
        if gt_mask[i] is not None:
            mask_plot[:, :, 2] = gt_mask[i] * 255 + mask_plot[:, :, 2]
    return mask_plot


def voc_ap(rec, prec, use_07_metric=False):
    """
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the VOC 07 11-point method.
    """
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_overlaps_masks(masks1, masks2):
    """Compute IoU overlaps between two sets of masks."""
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))

    intersection = np.logical_and(masks1, masks2)
    union = np.logical_or(masks1, masks2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def iou_bbox(boxA, boxB):
    """Compute IoU between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def extract_polys(mask):
    """Extract polygon contours from binary mask."""
    mm, contours, h = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    poly = {}
    if len(contours) > 1:
        base = np.load('base_cont.npy', allow_pickle=True)
        for j, cont in enumerate(contours):
            if j > 0:
                og = np.concatenate((og, cont), axis=0)
            else:
                og = cont
        base[0] = og
        contours = base
    for i, cont in enumerate(contours):
        poly[i] = {}
        poly[i]['all_points_x'] = cont[:, 0, 0]
        poly[i]['all_points_y'] = cont[:, 0, 1]
    return poly
