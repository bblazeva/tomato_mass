"""
Dataset loading for the tomato RGB-D data (COCO-format annotations).

Expected layout under root_dir (e.g. ./data_s0713/):
    gt_json/<split>/<split>.json     COCO annotations (one cluster per image)
    images/<split>/<name>.jpg|png    RGB images
    depth/<name>.npy                 depth maps in mm, aligned to the RGB image

The ground-truth mass is encoded in the file name: <mass_g>_<timestamp>.<ext>,
e.g. 208_2025-07-13-18-11-23.jpg -> 0.208 kg.
"""

import os
import json

from detectron2.structures import BoxMode


def get_tomato_dataset_dicts(root_dir, split):
    """
    Build the Detectron2 dataset list for one split (train / val / test_0328 / ...).

    Each record contains:
        file_name, depth_file, image_id, height, width
        cluster_mass:   mass in kg, parsed from the file name (None if unparsable)
        annotations:    list of instances: bbox (XYWH), segmentation, category_id, mass
    The per-instance "mass" is turned into `instances.gt_mass` by the vendored
    detectron2 (data/detection_utils.py: annotations_to_instances); the mass head
    and MassEvaluator read it from there.
    """
    depth_dir = os.path.join(root_dir, "depth")
    json_file = os.path.join(root_dir, "gt_json", split, split + ".json")

    with open(json_file, "r") as f:
        coco_dict = json.load(f)

    annos_per_image = {}
    for ann in coco_dict["annotations"]:
        annos_per_image.setdefault(ann["image_id"], []).append(ann)

    dataset_dicts = []
    for image in coco_dict["images"]:
        file_name = image["file_name"]
        base_name = os.path.splitext(file_name)[0]

        # Mass from the file name prefix, grams -> kg
        try:
            mass_kg = float(os.path.basename(file_name).split("_")[0]) / 1000.0
        except ValueError as e:
            print("Error extracting mass from", file_name, ":", e)
            mass_kg = None

        annotations = [
            {
                "bbox": ann["bbox"],  # COCO format: [x, y, width, height]
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": ann["segmentation"],
                "category_id": ann["category_id"],
                "mass": mass_kg,
            }
            for ann in annos_per_image.get(image["id"], [])
        ]

        dataset_dicts.append({
            "file_name": os.path.join(root_dir, "images", split, file_name),
            "image_id": image["id"],
            "height": image["height"],
            "width": image["width"],
            "depth_file": os.path.join(depth_dir, base_name + ".npy"),
            "cluster_mass": mass_kg,
            "annotations": annotations,
        })

    return dataset_dicts
