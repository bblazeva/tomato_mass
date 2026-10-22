#from typing import final
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
import json
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer, GenericMask
import matplotlib.pyplot as plt
import pdb
# Note: preprocess_input was imported but never used - removed

def match_mask(pred_instance, pred_box, pred_mass, scores, dataset_dicts, im, iter_):
    """
    Match each predicted instance mask with a ground truth mask based solely on the visible segmentation.
    (A form of non-maximum suppression based on overlap.)
    
    Return:
        gts: list of GT masks
        gts_box: corresponding bounding boxes
        num_gt: number of GT annotations in the current image
        gt_mass: list of GT mass values
        gt_id: ids for GT; annotation index 
        final_preds: list of matched predicted masks
        final_box: list of corresponding predicted bounding boxes
        final_mass: list of predicted mass values
        final_scores: list of scores for the matched detections 
    """
    # Initialize lists to store results.
    used_ids_gt = []   # We'll still record IDs, but not check for duplicates.
    all_sum = []       # Stores the maximum overlap for each match.
    gts = []
    gts_box = []
    gt_mass = []
    gt_id = []
    final_preds = []
    final_box = []
    final_scores = []
    final_mass = []
    num_gt = len(dataset_dicts[0][iter_]['annotations'])
    
    # Loop over each predicted instance mask.
    for i in range(len(pred_instance)):
        print(len(pred_instance))
        pred_mask = pred_instance[i]
        max_sum = 0
        if num_gt >= 1:
            # For each prediction, loop over all ground-truth annotations.
            for j, gt_ann in enumerate(dataset_dicts[0][iter_]['annotations']):
                # Extract the segmentation polygon for the GT.
                gt_poly = gt_ann['segmentation']
                Gmask = GenericMask(gt_poly, im.shape[0], im.shape[1])
                gt_mask = Gmask.polygons_to_mask(gt_poly)
                overlap = np.sum(gt_mask * pred_mask)
                if overlap >= max_sum:
                    idx_m = j
                    idx_p = i
                    max_sum = overlap
            
            # Save the prediction with maximum overlap.
            final_preds.append(pred_instance[idx_p])
            final_box.append(pred_box[idx_p])
            final_mass.append(pred_mass[idx_p])
            final_scores.append(scores[idx_p])
            #print(dataset_dicts[0][iter_]['annotations'][idx_m].get('mass'))
            
            # Retrieve ground-truth mass (from your annotations) and assign an ID.
            gt_m = dataset_dicts[0][iter_]['annotations'][idx_m].get('mass', None)
            #print('got mass:', gt_m)
            gt_id_val = idx_m  # Using the annotation index as the ID.
            
            # Re-create the ground truth mask and bounding box.
            gt_poly_f = dataset_dicts[0][iter_]['annotations'][idx_m]['segmentation']
            Gmask = GenericMask(gt_poly_f, im.shape[0], im.shape[1])
            gt_mask_sing = Gmask.polygons_to_mask(gt_poly_f)
            gt_box_sing = dataset_dicts[0][iter_]['annotations'][idx_m]['bbox']
            
            # Temporarily remove the check for duplicate ground-truth matches.
            used_ids_gt.append(idx_m)
            all_sum.append(max_sum)
            gts.append(gt_mask_sing)
            gts_box.append(gt_box_sing)
            gt_mass.append([gt_m])  # Wrapped in a list for consistency.
            gt_id.append(gt_id_val)
        else:
            gts.append(None)
            gts_box.append(None)
            gt_mass.append(None)
            gt_id.append(None)
    
    #print('returning: ', gt_mass)
    gt_dict = dataset_dicts[0][iter_]
    return gts, gts_box, num_gt, gt_mass, gt_id, final_preds, final_box, final_mass, final_scores, gt_dict


def match_mask2(pred_instance, pred_box, pred_mass, scores, dataset_dicts, im, iter_):
    """
    Match each predicted instance mask with a ground truth mask based solely on the visible segmentation.
    (A form of non-maximum suppression based on overlap.)
    Return:
        gts: list of GT masks
        gts_box: corresponding bboxes
        num_gt: number of GT annotations in the current image
        gt_mass: list of GT mass values
        gt_id: ids for GT; annotation idex 
        final_preds: list of matched predicted masks
        final_box: list of corresponding predicted bboxes
        final_mass: list of predicted mass values
        final_scores: list of scores for the matched detections 
    """
    used_ids_gt = []
    all_sum = []
    gts = []
    gts_box = []
    gt_mass = []
    gt_id = []
    final_preds = []
    final_box = []
    final_scores = []
    final_mass = []
    num_gt = len(dataset_dicts[0][iter_]['annotations'])
    
    # Loop over each predicted instance mask.
    for i in range(len(pred_instance)):
        print(len(pred_instance))
        pred_mask = pred_instance[i]
        max_sum = 0
        if num_gt >= 1:
            # For each prediction, loop over all ground-truth annotations.
            for j, gt_ann in enumerate(dataset_dicts[0][iter_]['annotations']):
                # Extract the segmentation polygon for the GT.
                gt_poly = gt_ann['segmentation']
                Gmask = GenericMask(gt_poly, im.shape[0], im.shape[1])
                gt_mask = Gmask.polygons_to_mask(gt_poly)
                overlap = np.sum(gt_mask * pred_mask)
                if overlap >= max_sum:
                    idx_m = j
                    idx_p = i
                    max_sum = overlap
            # Save the prediction with maximum overlap.
            final_preds.append(pred_instance[idx_p])
            final_box.append(pred_box[idx_p])
            final_mass.append(pred_mass[idx_p])
            final_scores.append(scores[idx_p])
            print(dataset_dicts[0][iter_]['annotations'][idx_m].get('mass'))
            
            # Retrieve ground-truth mass (from your annotations) and a simple ID.
            gt_m = dataset_dicts[0][iter_]['annotations'][idx_m].get('mass', None)
            print('got mass:', gt_m)
            gt_id_val = idx_m  # using the annotation index as the ID
            
            # Re-create the ground truth mask and bounding box.
            gt_poly_f = dataset_dicts[0][iter_]['annotations'][idx_m]['segmentation']
            Gmask = GenericMask(gt_poly_f, im.shape[0], im.shape[1])
            gt_mask_sing = Gmask.polygons_to_mask(gt_poly_f)
            gt_box_sing = dataset_dicts[0][iter_]['annotations'][idx_m]['bbox']
            
            # If this ground truth has been matched before, check if this match is better.
            if idx_m in used_ids_gt:
                a = np.where(np.array(used_ids_gt) == idx_m)[0][0]
                if all_sum[a] < max_sum:
                    gts[a] = None
                    used_ids_gt[a] = None
                    gt_mass[a] = None
                    gt_id[a] = None
                    gts_box[a] = None
                else:
                    idx_m = None
                    gt_mask_sing = None
                    gt_box_sing = None
                    gt_m = None
                    gt_id_val = None

            used_ids_gt.append(idx_m)
            all_sum.append(max_sum)
            gts.append(gt_mask_sing)
            gts_box.append(gt_box_sing)
            gt_mass.append([gt_m])  # wrapped as list, similar to previous mass output
            gt_id.append(gt_id_val)
        else:
            gts.append(None)
            gts_box.append(None)
            gt_mass.append(None)
            gt_id.append(None)
    print('returning: ', gt_mass)
    
    return gts, gts_box, num_gt, gt_mass, gt_id, final_preds, final_box, final_mass, final_scores

def plot_masks(mask_plot,gt_mask,im):
    for i in range(len(gt_mask)):
        if gt_mask[i] is not None:
            # mask_plot[:, :, 0] = gt_mask_sing*255 + mask_plot[:, :, 0]
            # mask_plot[:, :, 1] = gt_mask_sing*255 + mask_plot[:, :, 1]
            mask_plot[:, :, 2] = gt_mask[i]*255 + mask_plot[:, :, 2]

    return mask_plot

#https://github.com/facebookresearch/detectron2/blob/master/detectron2/evaluation/pascal_voc_evaluation.py
def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap



def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """

    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))


    # intersection over union
    intersection = np.logical_and(masks1, masks2)
    union = np.logical_or(masks1, masks2)
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score


def iou_bbox(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def extract_polys(mask):
    # mm,contours, h = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, h = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #eddited by JGM
    poly = {}
    if len(contours) > 1:
        # base = np.load('base_cont.npy', allow_pickle=True)
        for j, cont in enumerate(contours):
            if j > 0:
                og = np.concatenate((og, cont), axis=0)
            else:
                og = cont
        # base[0] = og
        # contours= base
        contours = (og,)
    for i,cont in enumerate(contours):
        poly[i] = {}
        poly[i]['all_points_x'] = cont[:, 0, 0]
        poly[i]['all_points_y'] = cont[:, 0, 1]
    
    return poly