"""
Detectron2 evaluator for the mass head (used during training evaluations).

The mass head predicts log(mass_kg); predictions are converted back with exp().

Which predictions count:
    Each predicted instance is matched to the ground-truth box with the highest IoU.
    Only matches with IoU >= 0.5 are scored (each GT at most once). Missed clusters and
    false positives are skipped, so these metrics never penalize a missed detection —
    read them together with bbox AP.

Metrics returned (keys as in metrics.json; masses in kg):
    mass_mse, mass_mae, mass_rmse
    mass_mre   mean relative error |pred - gt| / gt   (this x 100 is the MAPE we report)
    mape       = 100 * mass_mre
    r2         coefficient of determination
"""

import numpy as np
import torch

from detectron2.evaluation import DatasetEvaluator
from detectron2.structures import pairwise_iou

IOU_MATCH_THRESHOLD = 0.5


class MassEvaluator(DatasetEvaluator):

    def __init__(self):
        super().__init__()
        self._predictions = []
        self._ground_truths = []

    def reset(self):
        self._predictions.clear()
        self._ground_truths.clear()

    def process(self, inputs, outputs):
        """Collect (predicted kg, GT kg) for every prediction matched to a GT box."""
        for input_dict, output in zip(inputs, outputs):
            pred_inst = output["instances"]
            gt_inst = input_dict["instances"]
            if len(pred_inst) == 0 or len(gt_inst) == 0:
                continue

            gt_mass = gt_inst.gt_mass
            if isinstance(gt_mass, torch.Tensor):
                gt_kg = gt_mass.detach().cpu().numpy().ravel()
            else:
                gt_kg = np.array(gt_mass, dtype=float).ravel()

            pred_kg = np.exp(pred_inst.pred_mass.detach().cpu().numpy()).ravel().round(5)

            # Greedy matching: each prediction to its best-IoU GT box
            pred_boxes = pred_inst.pred_boxes
            gt_boxes = gt_inst.gt_boxes.to(pred_boxes.device)
            iou = pairwise_iou(gt_boxes, pred_boxes).cpu().numpy()

            matched_gt = set()
            for p in range(len(pred_inst)):
                best_gt = iou[:, p].argmax()
                if iou[best_gt, p] >= IOU_MATCH_THRESHOLD and best_gt not in matched_gt:
                    matched_gt.add(best_gt)
                    self._predictions.append(float(pred_kg[p]))
                    self._ground_truths.append(float(gt_kg[best_gt]))

    def evaluate(self):
        if not self._predictions:
            return {}

        preds = np.array(self._predictions, dtype=float)
        gts = np.array(self._ground_truths, dtype=float)
        err = preds - gts

        mse = (err ** 2).mean()
        mre = np.abs(err / gts).mean()
        r2 = 1 - np.sum(err ** 2) / np.sum((gts - gts.mean()) ** 2)

        return {
            "mass_mse": mse,
            "mass_mae": np.abs(err).mean(),
            "mass_mre": mre,
            "mape": 100 * mre,
            "mass_rmse": np.sqrt(mse),
            "r2": r2,
        }
