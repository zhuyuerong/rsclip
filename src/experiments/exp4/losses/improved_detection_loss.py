# -*- coding: utf-8 -*-
"""
改进的检测损失函数 - 实验2.2
使用预测框IoU进行更准确的正样本匹配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import sys
from pathlib import Path
from scipy.optimize import linear_sum_assignment

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.multi_instance_assigner import MultiPeakDetector, FallbackAssigner
from losses.detection_loss import generalized_box_iou


class ImprovedDetectionLoss(nn.Module):
    """
    改进的检测损失函数
    
    关键改进:
    1. 使用预测框IoU进行匹配，而不是仅基于峰值位置
    2. 为未匹配的GT添加fallback策略（使用GT框中心）
    3. 记录匹配质量统计
    """
    
    def __init__(self, 
                 lambda_l1: float = 1.0,
                 lambda_giou: float = 2.0,
                 lambda_cam: float = 0.5,
                 min_peak_distance: int = 2,
                 min_peak_value: float = 0.3,
                 match_iou_threshold: float = 0.3):
        """
        Args:
            lambda_l1: L1损失权重
            lambda_giou: GIoU损失权重
            lambda_cam: CAM监督损失权重
            min_peak_distance: 多峰检测的最小距离
            min_peak_value: 多峰检测的最小值
            match_iou_threshold: 匹配的IoU阈值
        """
        super().__init__()
        
        self.lambda_l1 = lambda_l1
        self.lambda_giou = lambda_giou
        self.lambda_cam = lambda_cam
        self.match_iou_threshold = match_iou_threshold
        
        # 峰值检测器
        self.peak_detector = MultiPeakDetector(
            min_peak_distance=min_peak_distance,
            min_peak_value=min_peak_value
        )
        
        # Fallback分配器
        self.fallback = FallbackAssigner()
    
    def match_peaks_to_gts_with_iou(self, peaks, pred_boxes_peaks, gt_boxes, 
                                     match_iou_threshold=0.3):
        """
        使用预测框IoU匹配峰值和GT boxes
        
        Args:
            peaks: List[(i, j, score)] CAM上的峰值
            pred_boxes_peaks: [N_peaks, 4] 峰值位置的预测框
            gt_boxes: [K, 4] GT框
            match_iou_threshold: IoU阈值
        
        Returns:
            matches: List[(peak_idx, gt_idx)]
            unmatched_peaks: List[peak_idx]
            unmatched_gts: List[gt_idx]
        """
        n_peaks = len(peaks)
        n_gts = len(gt_boxes)
        
        if n_peaks == 0 or n_gts == 0:
            return [], list(range(n_peaks)), list(range(n_gts))
        
        # 计算IoU矩阵
        iou_matrix = generalized_box_iou(
            pred_boxes_peaks, gt_boxes
        )  # [N_peaks, K]
        
        # 匈牙利算法匹配
        cost_matrix = 1.0 - iou_matrix  # cost = 1 - IoU
        row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
        
        # 过滤低质量匹配
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
    
    def forward(self, outputs: Dict, targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        计算改进的检测损失
        
        Args:
            outputs: 模型输出
                - cam: [B, C, H, W]
                - pred_boxes: [B, C, H, W, 4]
                - scores: [B, C, H, W]
            
            targets: List[dict] (per image)
                - boxes: [K, 4] 归一化坐标 [xmin, ymin, xmax, ymax]
                - labels: [K] 类别id
        
        Returns:
            loss_dict: {
                'loss_box_l1': ...,
                'loss_box_giou': ...,
                'loss_cam': ...,
                'loss_total': ...,
                'match_stats': {...}  # 新增：匹配统计
            }
        """
        cam = outputs['cam']
        pred_boxes = outputs['pred_boxes']
        B, C, H, W, _ = pred_boxes.shape
        
        # ===== Step 1: 改进的正样本分配 =====
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
            gt_boxes = targets[b]['boxes']  # [K, 4]
            gt_labels = targets[b]['labels']  # [K]
            
            if len(gt_boxes) == 0:
                continue
            
            match_stats['total_gts'] += len(gt_boxes)
            
            # 按类别处理
            unique_classes = torch.unique(gt_labels)
            pos_samples = []
            
            for class_id in unique_classes:
                class_id = class_id.item()
                
                # 该类的GT
                mask = (gt_labels == class_id)
                class_gt_boxes = gt_boxes[mask]
                class_gt_indices = torch.where(mask)[0].tolist()
                
                if len(class_gt_boxes) == 0:
                    continue
                
                # 检测峰值
                cam_class = cam[b, class_id]
                peaks = self.peak_detector.detect_peaks(cam_class)
                match_stats['total_peaks'] += len(peaks)
                
                if len(peaks) == 0:
                    # 如果没有峰值，使用fallback策略
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
                
                # 提取峰值位置的预测框
                peak_pred_boxes = []
                for i, j, score in peaks:
                    peak_pred_boxes.append(pred_boxes[b, class_id, i, j])
                
                if len(peak_pred_boxes) > 0:
                    peak_pred_boxes = torch.stack(peak_pred_boxes)  # [N_peaks, 4]
                    
                    # 使用预测框IoU进行匹配
                    matches, unmatched_peaks, unmatched_gts = self.match_peaks_to_gts_with_iou(
                        peaks, peak_pred_boxes, class_gt_boxes, self.match_iou_threshold
                    )
                    
                    # 添加匹配的样本
                    for peak_idx, gt_local_idx in matches:
                        i, j, score = peaks[peak_idx]
                        gt_global_idx = class_gt_indices[gt_local_idx]
                        pred_box = peak_pred_boxes[peak_idx]
                        gt_box = class_gt_boxes[gt_local_idx]
                        
                        # 计算IoU
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
                    
                    # 处理未匹配的GT（fallback）
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
            
            # 统计未匹配的GT
            matched_gt_indices = set(s['gt_idx'] for s in pos_samples)
            match_stats['unmatched_gts'] += len(gt_boxes) - len(matched_gt_indices)
            
            # ===== Step 2: 计算Box回归损失 =====
            if len(pos_samples) > 0:
                for sample in pos_samples:
                    i, j = sample['i'], sample['j']
                    class_id = sample['class']
                    gt_idx = sample['gt_idx']
                    
                    pred_box = pred_boxes[b, class_id, i, j]  # [4]
                    gt_box = gt_boxes[gt_idx]  # [4]
                    
                    # L1 loss
                    loss_l1 += F.l1_loss(pred_box, gt_box)
                    
                    # GIoU loss
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
            # 创建需要梯度的零tensor
            loss_l1 = torch.tensor(0.0, device=cam.device, requires_grad=True)
            loss_giou = torch.tensor(0.0, device=cam.device, requires_grad=True)
        
        # ===== Step 3: CAM监督损失（保持不变） =====
        loss_cam = 0.0
        num_cam_samples = 0
        
        for b in range(B):
            gt_boxes = targets[b]['boxes']
            gt_labels = targets[b]['labels']
            
            if len(gt_boxes) == 0:
                continue
            
            for k, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
                label = label.item()
                xmin, ymin, xmax, ymax = box
                
                # 框内mask
                mask_in = torch.zeros(H, W, device=cam.device)
                i_min = max(0, int(ymin * H))
                i_max = min(H - 1, int(ymax * H))
                j_min = max(0, int(xmin * W))
                j_max = min(W - 1, int(xmax * W))
                
                if i_max >= i_min and j_max >= j_min:
                    mask_in[i_min:i_max+1, j_min:j_max+1] = 1
                
                # 框外mask
                mask_out = 1 - mask_in
                
                # CAM响应
                cam_c = cam[b, label]
                
                # Loss: 框内应该高,框外应该低
                mask_in_sum = mask_in.sum()
                mask_out_sum = mask_out.sum()
                
                if mask_in_sum > 0:
                    cam_in = (cam_c * mask_in).sum() / mask_in_sum
                    loss_cam += (1 - cam_in)
                
                if mask_out_sum > 0:
                    cam_out = (cam_c * mask_out).sum() / mask_out_sum
                    loss_cam += cam_out
                
                num_cam_samples += 1
        
        if num_cam_samples > 0:
            loss_cam = loss_cam / num_cam_samples
        else:
            # 创建需要梯度的零tensor
            loss_cam = torch.tensor(0.0, device=cam.device, requires_grad=True)
        
        # 计算匹配统计
        if match_stats['avg_match_iou']:
            match_stats['avg_match_iou'] = sum(match_stats['avg_match_iou']) / len(match_stats['avg_match_iou'])
        else:
            match_stats['avg_match_iou'] = 0.0
        
        # ===== Step 4: 总损失 =====
        loss_total = (
            self.lambda_l1 * loss_l1 +
            self.lambda_giou * loss_giou +
            self.lambda_cam * loss_cam
        )
        
        return {
            'loss_box_l1': loss_l1,
            'loss_box_giou': loss_giou,
            'loss_cam': loss_cam,
            'loss_total': loss_total,
            'match_stats': match_stats  # 新增：匹配统计
        }


