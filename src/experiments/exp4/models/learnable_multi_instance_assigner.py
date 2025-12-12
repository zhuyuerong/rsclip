# -*- coding: utf-8 -*-
"""
可学习的多实例分配器
使用可学习的峰值检测器替代固定阈值方法
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from models.learnable_peak_detector import LearnableMultiPeakDetector
from models.multi_instance_assigner import PeakToGTMatcher, FallbackAssigner
from losses.detection_loss import generalized_box_iou
from scipy.optimize import linear_sum_assignment


class LearnableMultiInstanceAssigner(nn.Module):
    """
    可学习的多实例正样本分配器
    
    关键改进:
    1. 使用可学习的峰值检测器（替代固定阈值）
    2. 结合objectness score
    3. 端到端训练
    """
    
    def __init__(self, num_classes: int,
                 min_peak_distance: int = 2,
                 init_threshold: float = 0.3,
                 match_iou_threshold: float = 0.3,
                 use_objectness: bool = True):
        """
        Args:
            num_classes: 类别数
            min_peak_distance: 峰之间的最小距离
            init_threshold: 初始阈值（可学习）
            match_iou_threshold: 匹配的IoU阈值
            use_objectness: 是否使用objectness score
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.match_iou_threshold = match_iou_threshold
        
        # 可学习的峰值检测器
        self.peak_detector = LearnableMultiPeakDetector(
            num_classes=num_classes,
            min_peak_distance=min_peak_distance,
            init_threshold=init_threshold,
            use_objectness=use_objectness
        )
        
        self.matcher = PeakToGTMatcher(match_iou_threshold)
        self.fallback = FallbackAssigner()
    
    def match_peaks_to_gts_with_iou(self, peaks, pred_boxes_peaks, gt_boxes, 
                                     match_iou_threshold=0.3):
        """使用预测框IoU匹配峰值和GT boxes"""
        n_peaks = len(peaks)
        n_gts = len(gt_boxes)
        
        if n_peaks == 0 or n_gts == 0:
            return [], list(range(n_peaks)), list(range(n_gts))
        
        iou_matrix = generalized_box_iou(pred_boxes_peaks, gt_boxes)
        cost_matrix = 1.0 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
        
        matches = []
        matched_peaks = set()
        matched_gts = set()
        
        for p_idx, g_idx in zip(row_ind, col_ind):
            iou = iou_matrix[p_idx, g_idx].item()
            if iou >= match_iou_threshold:
                matches.append((p_idx, g_idx))
                matched_peaks.add(p_idx)
                matched_gts.add(g_idx)
        
        unmatched_peaks = [i for i in range(n_peaks) if i not in matched_peaks]
        unmatched_gts = [i for i in range(n_gts) if i not in matched_gts]
        
        return matches, unmatched_peaks, unmatched_gts
    
    def forward(self, cam: torch.Tensor, 
                pred_boxes: torch.Tensor,
                objectness: torch.Tensor,
                gt_boxes: torch.Tensor,
                gt_labels: torch.Tensor) -> Dict:
        """
        分配正样本（训练时，可微分）
        
        Args:
            cam: [C, H, W] CAM
            pred_boxes: [C, H, W, 4] 预测框
            objectness: [C, H, W] objectness score
            gt_boxes: [K, 4] GT框
            gt_labels: [K] GT标签
        
        Returns:
            pos_samples: List[dict] 正样本列表
            peak_masks: [C, H, W] 峰值mask（用于损失计算）
        """
        C, H, W = cam.shape
        
        # 使用可学习的峰值检测器
        peak_masks = self.peak_detector(cam.unsqueeze(0), objectness.unsqueeze(0))
        peak_masks = peak_masks.squeeze(0)  # [C, H, W]
        
        # 提取峰值位置（用于匹配）
        pos_samples = []
        
        unique_classes = torch.unique(gt_labels)
        
        for class_id in unique_classes:
            class_id = class_id.item()
            
            mask = (gt_labels == class_id)
            class_gt_boxes = gt_boxes[mask]
            class_gt_indices = torch.where(mask)[0].tolist()
            
            if len(class_gt_boxes) == 0:
                continue
            
            # 从peak_mask提取峰值位置
            peak_mask_class = peak_masks[class_id]  # [H, W]
            peak_indices = peak_mask_class.nonzero(as_tuple=False)
            
            if len(peak_indices) == 0:
                # Fallback策略
                for gt_idx in class_gt_indices:
                    gt_box = gt_boxes[gt_idx]
                    xmin, ymin, xmax, ymax = gt_box
                    i_center = int((ymin + ymax) / 2 * H)
                    j_center = int((xmin + xmax) / 2 * W)
                    i_center = max(0, min(H-1, i_center))
                    j_center = max(0, min(W-1, j_center))
                    
                    pos_samples.append({
                        'i': i_center,
                        'j': j_center,
                        'class': class_id,
                        'gt_idx': gt_idx,
                        'match_type': 'fallback',
                        'iou': 0.0
                    })
                continue
            
            # 提取峰值位置的预测框
            peak_pred_boxes = []
            peak_positions = []
            for idx in peak_indices:
                i, j = idx[0].item(), idx[1].item()
                peak_pred_boxes.append(pred_boxes[class_id, i, j])
                peak_positions.append((i, j))
            
            if len(peak_pred_boxes) > 0:
                peak_pred_boxes = torch.stack(peak_pred_boxes)
                
                # 匹配
                matches, unmatched_peaks, unmatched_gts = self.match_peaks_to_gts_with_iou(
                    peak_positions, peak_pred_boxes, class_gt_boxes, self.match_iou_threshold
                )
                
                # 添加匹配的样本
                for peak_idx, gt_local_idx in matches:
                    i, j = peak_positions[peak_idx]
                    gt_global_idx = class_gt_indices[gt_local_idx]
                    pred_box = peak_pred_boxes[peak_idx]
                    gt_box = class_gt_boxes[gt_local_idx]
                    
                    iou = generalized_box_iou(
                        pred_box.unsqueeze(0),
                        gt_box.unsqueeze(0)
                    )[0, 0].item()
                    
                    pos_samples.append({
                        'i': i,
                        'j': j,
                        'class': class_id,
                        'gt_idx': gt_global_idx,
                        'match_type': 'peak',
                        'iou': iou
                    })
                
                # Fallback未匹配的GT
                for gt_local_idx in unmatched_gts:
                    gt_global_idx = class_gt_indices[gt_local_idx]
                    gt_box = class_gt_boxes[gt_local_idx]
                    xmin, ymin, xmax, ymax = gt_box
                    i_center = int((ymin + ymax) / 2 * H)
                    j_center = int((xmin + xmax) / 2 * W)
                    i_center = max(0, min(H-1, i_center))
                    j_center = max(0, min(W-1, j_center))
                    
                    pos_samples.append({
                        'i': i_center,
                        'j': j_center,
                        'class': class_id,
                        'gt_idx': gt_global_idx,
                        'match_type': 'fallback',
                        'iou': 0.0
                    })
        
        return {
            'pos_samples': pos_samples,
            'peak_masks': peak_masks  # 用于损失计算
        }

