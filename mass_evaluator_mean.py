"""
====================================================================================
MASS EVALUATOR - Custom Detectron2 Evaluator for Mass Prediction
====================================================================================

This evaluator computes metrics for the mass regression task.

METRICS COMPUTED:
-----------------
- MSE: Mean Squared Error
- MAE: Mean Absolute Error
- MRE: Mean Relative Error (|pred - gt| / gt)
- MAPE: Mean Absolute Percentage Error (MRE * 100)
- RMSE: Root Mean Squared Error
- R²: Coefficient of determination

IMPORTANT NOTE ON LOG-SCALE:
----------------------------
The mass head predicts in LOG-SCALE (log(mass_kg)).
This evaluator converts predictions back to kg using exp().

WHY LOG-SCALE?
--------------
1. Mass values vary over a wide range (e.g., 0.1kg to 1kg)
2. Log-scale makes the loss function weight relative errors equally
3. Prevents large masses from dominating the loss

====================================================================================
"""

from detectron2.evaluation import DatasetEvaluator
import torch
import numpy as np


class MassEvaluator(DatasetEvaluator):
    """
    Evaluator for mass prediction task.

    Collects predictions and ground truths during evaluation,
    then computes aggregate metrics.
    """

    def __init__(self):
        super().__init__()
        self._predictions = []
        self._ground_truths = []

    def reset(self):
        """Clear stored predictions and ground truths."""
        self._predictions.clear()
        self._ground_truths.clear()

    def process(self, inputs, outputs):
        """
        Process a batch of inputs and outputs.

        Matches predicted instances to GT instances by IoU, then
        compares mass only for matched pairs. False positives and
        missed detections are skipped.

        The model outputs pred_mass in LOG-SCALE.
        Ground truth (gt_mass) is in kg (linear scale).
        """
        from detectron2.structures import pairwise_iou

        for input_dict, output in zip(inputs, outputs):
            pred_inst = output["instances"]
            if len(pred_inst) == 0:
                continue

            gt_inst = input_dict["instances"]
            if len(gt_inst) == 0:
                continue

            # Get ground truth mass
            gt_mass_field = gt_inst.gt_mass
            if isinstance(gt_mass_field, torch.Tensor):
                gt_np = gt_mass_field.detach().cpu().numpy().ravel()
            else:
                gt_np = np.array(gt_mass_field, dtype=float).ravel()

            # Get predicted mass (in log-scale) and convert to kg
            pred_log = pred_inst.pred_mass.detach().cpu().numpy()
            pred_kg = np.exp(pred_log).ravel().round(5)

            # Match predictions to GT by IoU (greedy, threshold=0.5)
            pred_boxes = pred_inst.pred_boxes
            gt_boxes = gt_inst.gt_boxes.to(pred_boxes.device)
            iou_matrix = pairwise_iou(gt_boxes, pred_boxes).cpu().numpy()

            matched_gt = set()
            for pred_idx in range(len(pred_inst)):
                if iou_matrix.shape[0] == 0:
                    break
                best_gt = iou_matrix[:, pred_idx].argmax()
                if iou_matrix[best_gt, pred_idx] >= 0.5 and best_gt not in matched_gt:
                    matched_gt.add(best_gt)
                    self._predictions.append(float(pred_kg[pred_idx]))
                    self._ground_truths.append(float(gt_np[best_gt]))

    def evaluate(self):
        """
        Compute and return evaluation metrics.

        Returns:
            Dict of metric_name -> metric_value
        """
        if not self._predictions:
            return {}

        preds = np.array(self._predictions, dtype=float)
        gts = np.array(self._ground_truths, dtype=float)

        # Mean Squared Error
        mse = ((preds - gts) ** 2).mean()

        # Mean Absolute Error
        mae = np.abs(preds - gts).mean()

        # Mean Relative Error
        mre = np.abs((preds - gts) / gts).mean()

        # Mean Absolute Percentage Error
        mape = 100 * mre

        # Root Mean Squared Error
        rmse = np.sqrt(mse)

        # R² Score (coefficient of determination)
        ss_res = np.sum((preds - gts) ** 2)
        ss_tot = np.sum((gts - gts.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot

        return {
            "mass_mse": mse,
            "mass_mae": mae,
            "mass_mre": mre,
            "mape": mape,
            "mass_rmse": rmse,
            "r2": r2,
        }
