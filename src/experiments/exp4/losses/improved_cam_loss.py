# -*- coding: utf-8 -*-
"""
改进的CAM损失函数 - 实验3.1
使用更敏感的损失函数来改善CAM质量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from losses.improved_detection_loss import ImprovedDetectionLoss
from losses.detection_loss import generalized_box_iou


class ImprovedCAMDetectionLoss(ImprovedDetectionLoss):
    """
    改进的检测损失函数（实验3.1）
    
    关键改进:
    1. 使用Focal Loss风格的CAM损失: (1 - cam_in)^2
    2. 框外损失使用平方: cam_out^2
    3. 添加峰值鼓励项: 如果框内最大CAM < 0.3，增加惩罚
    """
    
    def __init__(self, 
                 lambda_l1: float = 1.0,
                 lambda_giou: float = 2.0,
                 lambda_cam: float = 0.5,
                 min_peak_distance: int = 2,
                 min_peak_value: float = 0.3,
                 match_iou_threshold: float = 0.3,
                 cam_peak_threshold: float = 0.3):
        """
        Args:
            cam_peak_threshold: 框内最大CAM的阈值，低于此值会增加惩罚
        """
        super().__init__(
            lambda_l1=lambda_l1,
            lambda_giou=lambda_giou,
            lambda_cam=lambda_cam,
            min_peak_distance=min_peak_distance,
            min_peak_value=min_peak_value,
            match_iou_threshold=match_iou_threshold
        )
        self.cam_peak_threshold = cam_peak_threshold
    
    def forward(self, outputs: Dict, targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        计算改进的检测损失（包含改进的CAM损失）
        """
        cam = outputs['cam']
        pred_boxes = outputs['pred_boxes']
        B, C, H, W, _ = pred_boxes.shape
        
        # ===== Step 1: 改进的正样本分配（继承自ImprovedDetectionLoss） =====
        loss_l1 = 0.0
        loss_giou = 0.0
        num_pos_samples = 0
        
        match_stats = {
            'total_peaks': 0,
            'total_gts': 0,
            'peak_matches': 0,
            'fallback_matches': 0,
            'unmatched_gts': 0,
            'avg_match_iou': []
        }
        
        for b in range(B):
            gt_boxes = targets[b]['boxes']
            gt_labels = targets[b]['labels']
            
            if len(gt_boxes) == 0:
                continue
            
            match_stats['total_gts'] += len(gt_boxes)
            
            # 按类别处理（使用父类的匹配逻辑）
            unique_classes = torch.unique(gt_labels)
            pos_samples = []
            
            for class_id in unique_classes:
                class_id = class_id.item()
                
                mask = (gt_labels == class_id)
                class_gt_boxes = gt_boxes[mask]
                class_gt_indices = torch.where(mask)[0].tolist()
                
                if len(class_gt_boxes) == 0:
                    continue
                
                cam_class = cam[b, class_id]
                peaks = self.peak_detector.detect_peaks(cam_class)
                match_stats['total_peaks'] += len(peaks)
                
                if len(peaks) == 0:
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
                        match_stats['fallback_matches'] += 1
                    continue
                
                peak_pred_boxes = []
                for i, j, score in peaks:
                    peak_pred_boxes.append(pred_boxes[b, class_id, i, j])
                
                if len(peak_pred_boxes) > 0:
                    peak_pred_boxes = torch.stack(peak_pred_boxes)
                    
                    matches, unmatched_peaks, unmatched_gts = self.match_peaks_to_gts_with_iou(
                        peaks, peak_pred_boxes, class_gt_boxes, self.match_iou_threshold
                    )
                    
                    for peak_idx, gt_local_idx in matches:
                        i, j, score = peaks[peak_idx]
                        gt_global_idx = class_gt_indices[gt_local_idx]
                        pred_box = peak_pred_boxes[peak_idx]
                        gt_box = class_gt_boxes[gt_local_idx]
                        
                        iou = generalized_box_iou(
                            pred_box.unsqueeze(0),
                            gt_box.unsqueeze(0)
                        )[0, 0].item()
                        match_stats['avg_match_iou'].append(iou)
                        
                        pos_samples.append({
                            'i': i,
                            'j': j,
                            'class': class_id,
                            'gt_idx': gt_global_idx,
                            'match_type': 'peak',
                            'iou': iou
                        })
                        match_stats['peak_matches'] += 1
                    
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
                        match_stats['fallback_matches'] += 1
            
            matched_gt_indices = set(s['gt_idx'] for s in pos_samples)
            match_stats['unmatched_gts'] += len(gt_boxes) - len(matched_gt_indices)
            
            # 计算Box回归损失
            if len(pos_samples) > 0:
                for sample in pos_samples:
                    i, j = sample['i'], sample['j']
                    class_id = sample['class']
                    gt_idx = sample['gt_idx']
                    
                    pred_box = pred_boxes[b, class_id, i, j]
                    gt_box = gt_boxes[gt_idx]
                    
                    loss_l1 += F.l1_loss(pred_box, gt_box)
                    
                    giou = generalized_box_iou(
                        pred_box.unsqueeze(0),
                        gt_box.unsqueeze(0)
                    )[0, 0]
                    loss_giou += (1 - giou)
                    
                    num_pos_samples += 1
        
        if num_pos_samples > 0:
            loss_l1 = loss_l1 / num_pos_samples
            loss_giou = loss_giou / num_pos_samples
        else:
            loss_l1 = torch.tensor(0.0, device=cam.device, requires_grad=True)
            loss_giou = torch.tensor(0.0, device=cam.device, requires_grad=True)
        
        # ===== Step 2: 改进的CAM监督损失 =====
        loss_cam = 0.0
        loss_cam_peak = 0.0  # 峰值鼓励项
        num_cam_samples = 0
        
        for b in range(B):
            gt_boxes = targets[b]['boxes']
            gt_labels = targets[b]['labels']
            
            if len(gt_boxes) == 0:
                continue
            
            for k, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
                label = label.item()
                xmin, ymin, xmax, ymax = box
                cam_c = cam[b, label]
                
                # 框内mask
                mask_in = torch.zeros(H, W, device=cam.device)
                i_min = max(0, int(ymin * H))
                i_max = min(H - 1, int(ymax * H))
                j_min = max(0, int(xmin * W))
                j_max = min(W - 1, int(xmax * W))
                
                if i_max >= i_min and j_max >= j_min:
                    mask_in[i_min:i_max+1, j_min:j_max+1] = 1
                
                mask_out = 1 - mask_in
                
                mask_in_sum = mask_in.sum()
                mask_out_sum = mask_out.sum()
                
                if mask_in_sum > 0:
                    # 改进1: 使用Focal Loss风格的损失
                    cam_in = (cam_c * mask_in).sum() / mask_in_sum
                    
                    # 使用sigmoid确保值在[0,1]
                    cam_in_normalized = torch.sigmoid(cam_in)
                    
                    # Focal loss风格: (1 - cam_in)^2
                    loss_cam_in = (1 - cam_in_normalized) ** 2
                    loss_cam += loss_cam_in
                
                if mask_out_sum > 0:
                    # 改进2: 框外损失使用平方
                    cam_out = (cam_c * mask_out).sum() / mask_out_sum
                    
                    # 使用ReLU确保只惩罚正值
                    cam_out_positive = F.relu(cam_out)
                    
                    # 平方损失: 鼓励接近0
                    loss_cam_out = cam_out_positive ** 2
                    loss_cam += loss_cam_out
                
                # 改进3: 峰值鼓励项
                if mask_in_sum > 0:
                    roi_cam = cam_c[i_min:i_max+1, j_min:j_max+1]
                    if roi_cam.numel() > 0:
                        roi_max = roi_cam.max()
                        
                        # 如果最大值太低，增加惩罚
                        if roi_max < self.cam_peak_threshold:
                            loss_cam_peak += (self.cam_peak_threshold - roi_max) ** 2
                
                num_cam_samples += 1
        
        if num_cam_samples > 0:
            loss_cam = loss_cam / num_cam_samples
            loss_cam_peak = loss_cam_peak / num_cam_samples
        else:
            loss_cam = torch.tensor(0.0, device=cam.device, requires_grad=True)
            loss_cam_peak = torch.tensor(0.0, device=cam.device, requires_grad=True)
        
        # 计算匹配统计
        if match_stats['avg_match_iou']:
            match_stats['avg_match_iou'] = sum(match_stats['avg_match_iou']) / len(match_stats['avg_match_iou'])
        else:
            match_stats['avg_match_iou'] = 0.0
        
        # ===== Step 3: 总损失 =====
        loss_total = (
            self.lambda_l1 * loss_l1 +
            self.lambda_giou * loss_giou +
            self.lambda_cam * loss_cam +
            self.lambda_cam * loss_cam_peak  # 峰值鼓励项使用相同的权重
        )
        
        return {
            'loss_box_l1': loss_l1,
            'loss_box_giou': loss_giou,
            'loss_cam': loss_cam,
            'loss_cam_peak': loss_cam_peak,  # 新增
            'loss_total': loss_total,
            'match_stats': match_stats
        }


