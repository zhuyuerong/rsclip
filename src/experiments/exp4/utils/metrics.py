# -*- coding: utf-8 -*-
"""
Detection Metrics
实现mAP计算（COCO格式），支持不同IoU阈值
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes
    
    Args:
        boxes1: [N, 4] (xmin, ymin, xmax, ymax)
        boxes2: [M, 4] (xmin, ymin, xmax, ymax)
    
    Returns:
        iou: [N, M] IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    inter = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter_area = inter[:, :, 0] * inter[:, :, 1]  # [N, M]
    
    union = area1[:, None] + area2 - inter_area
    iou = inter_area / (union + 1e-6)
    
    return iou


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Compute Average Precision (AP) from precision-recall curve
    
    Args:
        recall: Recall values
        precision: Precision values
    
    Returns:
        AP value
    """
    # Add sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    
    # Compute precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    # Sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap


class DetectionMetrics:
    """
    Detection metrics calculator
    支持mAP@0.5和mAP@0.5:0.95
    """
    
    def __init__(self, num_classes: int, iou_thresholds: List[float] = None):
        """
        Args:
            num_classes: Number of classes
            iou_thresholds: List of IoU thresholds for mAP calculation
                           If None, uses [0.5, 0.55, ..., 0.95] for COCO mAP
        """
        self.num_classes = num_classes
        
        if iou_thresholds is None:
            # COCO standard: 0.5:0.05:0.95
            self.iou_thresholds = np.arange(0.5, 1.0, 0.05)
        else:
            self.iou_thresholds = np.array(iou_thresholds)
        
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        # Store all predictions and ground truths
        self.pred_boxes = defaultdict(list)  # class_id -> List[boxes]
        self.pred_scores = defaultdict(list)   # class_id -> List[scores]
        self.pred_image_ids = defaultdict(list)  # class_id -> List[image_ids]
        
        self.gt_boxes = defaultdict(list)     # class_id -> List[boxes]
        self.gt_image_ids = defaultdict(list)  # class_id -> List[image_ids]
        self.gt_difficult = defaultdict(list)  # class_id -> List[is_difficult]
    
    def update(self, pred_boxes: List[torch.Tensor], 
               pred_labels: List[torch.Tensor],
               pred_scores: List[torch.Tensor],
               gt_boxes: List[torch.Tensor],
               gt_labels: List[torch.Tensor],
               image_ids: List[str]):
        """
        Update metrics with a batch of predictions
        
        Args:
            pred_boxes: List of [N, 4] predicted boxes (normalized)
            pred_labels: List of [N] predicted labels
            pred_scores: List of [N] predicted scores
            gt_boxes: List of [M, 4] ground truth boxes (normalized)
            gt_labels: List of [M] ground truth labels
            image_ids: List of image IDs
        """
        for i, (pb, pl, ps, gb, gl, img_id) in enumerate(
            zip(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, image_ids)
        ):
            # Process predictions
            for box, label, score in zip(pb, pl, ps):
                label = label.item()
                self.pred_boxes[label].append(box.cpu().numpy())
                self.pred_scores[label].append(score.item())
                self.pred_image_ids[label].append(img_id)
            
            # Process ground truths
            for box, label in zip(gb, gl):
                label = label.item()
                self.gt_boxes[label].append(box.cpu().numpy())
                self.gt_image_ids[label].append(img_id)
                self.gt_difficult[label].append(False)  # DIOR doesn't have difficult flag
    
    def compute_ap_per_class(self, class_id: int, iou_threshold: float) -> float:
        """
        Compute AP for a single class at a specific IoU threshold
        
        Args:
            class_id: Class index
            iou_threshold: IoU threshold
        
        Returns:
            AP value
        """
        pred_boxes = np.array(self.pred_boxes[class_id])
        pred_scores = np.array(self.pred_scores[class_id])
        pred_image_ids = self.pred_image_ids[class_id]
        
        gt_boxes = np.array(self.gt_boxes[class_id])
        gt_image_ids = self.gt_image_ids[class_id]
        gt_difficult = np.array(self.gt_difficult[class_id])
        
        if len(pred_boxes) == 0:
            if len(gt_boxes) == 0:
                return 1.0  # Perfect if no predictions and no GTs
            return 0.0  # No predictions but has GTs
        
        if len(gt_boxes) == 0:
            return 0.0  # Has predictions but no GTs
        
        # Sort predictions by score (descending)
        sorted_indices = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[sorted_indices]
        pred_scores = pred_scores[sorted_indices]
        pred_image_ids = [pred_image_ids[i] for i in sorted_indices]
        
        # Track which GT boxes have been matched
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        
        # Compute TP and FP
        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        
        for i, (pred_box, img_id) in enumerate(zip(pred_boxes, pred_image_ids)):
            # Find GT boxes in the same image
            gt_mask = np.array([gid == img_id for gid in gt_image_ids])
            if not gt_mask.any():
                fp[i] = 1
                continue
            
            gt_boxes_img = gt_boxes[gt_mask]
            gt_matched_img = gt_matched[gt_mask]
            gt_indices = np.where(gt_mask)[0]
            
            # Compute IoU with all GT boxes in this image
            ious = []
            for gt_box in gt_boxes_img:
                # Compute IoU
                inter_xmin = max(pred_box[0], gt_box[0])
                inter_ymin = max(pred_box[1], gt_box[1])
                inter_xmax = min(pred_box[2], gt_box[2])
                inter_ymax = min(pred_box[3], gt_box[3])
                
                if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
                    ious.append(0.0)
                else:
                    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
                    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                    union_area = pred_area + gt_area - inter_area
                    iou = inter_area / (union_area + 1e-6)
                    ious.append(iou)
            
            if len(ious) > 0:
                max_iou = max(ious)
                max_idx = np.argmax(ious)
                
                if max_iou >= iou_threshold and not gt_matched_img[max_idx]:
                    tp[i] = 1
                    gt_matched[gt_indices[max_idx]] = True
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / (len(gt_boxes) + 1e-6)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Compute AP
        ap = compute_ap(recalls, precisions)
        
        return ap
    
    def compute_map(self) -> Dict[str, float]:
        """
        Compute mAP at different IoU thresholds
        
        Returns:
            Dictionary with mAP metrics
        """
        results = {}
        
        # Compute AP for each IoU threshold
        aps_per_threshold = []
        
        for iou_thresh in self.iou_thresholds:
            aps_per_class = []
            
            for class_id in range(self.num_classes):
                ap = self.compute_ap_per_class(class_id, iou_thresh)
                aps_per_class.append(ap)
            
            mean_ap = np.mean(aps_per_class) if len(aps_per_class) > 0 else 0.0
            aps_per_threshold.append(mean_ap)
            
            # Store mAP@0.5 separately
            if abs(iou_thresh - 0.5) < 1e-6:
                results['mAP@0.5'] = mean_ap
        
        # mAP@0.5:0.95 (COCO standard)
        if len(aps_per_threshold) > 0:
            results['mAP@0.5:0.95'] = np.mean(aps_per_threshold)
        
        # Per-class APs at 0.5
        class_aps = []
        for class_id in range(self.num_classes):
            ap = self.compute_ap_per_class(class_id, 0.5)
            class_aps.append(ap)
        results['per_class_AP@0.5'] = class_aps
        
        return results


