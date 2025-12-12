# -*- coding: utf-8 -*-
"""
改进的多实例分配器 - 实验3.3
使用改进的峰值检测器
"""

import torch
from typing import List, Dict, Tuple
from models.improved_peak_detector import ImprovedMultiPeakDetector
from models.multi_instance_assigner import PeakToGTMatcher, FallbackAssigner


class ImprovedMultiInstanceAssigner:
    """
    改进的多实例正样本分配器
    
    使用改进的峰值检测器（自适应阈值、NMS去重）
    """
    
    def __init__(self,
                 min_peak_distance: int = 2,
                 min_peak_value: float = 0.3,
                 match_iou_threshold: float = 0.3,
                 adaptive_threshold: bool = True,
                 use_multi_scale: bool = False):
        """
        Args:
            min_peak_distance: 峰之间的最小距离
            min_peak_value: 最小峰值阈值（基础值）
            match_iou_threshold: 匹配的IoU阈值
            adaptive_threshold: 是否使用自适应阈值
            use_multi_scale: 是否使用多尺度峰值检测
        """
        self.peak_detector = ImprovedMultiPeakDetector(
            min_peak_distance=min_peak_distance,
            min_peak_value=min_peak_value,
            adaptive_threshold=adaptive_threshold,
            use_multi_scale=use_multi_scale
        )
        self.matcher = PeakToGTMatcher(match_iou_threshold)
        self.fallback = FallbackAssigner()
    
    def assign(self, cam: torch.Tensor, 
               gt_boxes: torch.Tensor,
               gt_labels: torch.Tensor) -> List[Dict]:
        """
        分配正样本
        
        Args:
            cam: [C, H, W]
            gt_boxes: [K, 4] 归一化坐标
            gt_labels: [K] 类别id
        
        Returns:
            pos_samples: List[dict]
                - i, j: CAM位置
                - class: 类别id
                - gt_idx: 对应的GT索引
                - match_type: 'peak' or 'fallback'
                - confidence: CAM激活值
        """
        C, H, W = cam.shape
        pos_samples = []
        
        # 按类别处理
        unique_classes = torch.unique(gt_labels)
        
        for class_id in unique_classes:
            class_id = class_id.item()
            
            # 该类的GT
            mask = (gt_labels == class_id)
            class_gt_boxes = gt_boxes[mask]
            class_gt_indices = torch.where(mask)[0].tolist()
            
            if len(class_gt_boxes) == 0:
                continue
            
            # 1. 使用改进的峰值检测器检测峰
            cam_class = cam[class_id]
            peaks = self.peak_detector.detect_peaks(cam_class)
            
            # 2. 匹配
            matches, unmatched_peaks, unmatched_gts = self.matcher.match(
                peaks, class_gt_boxes, H, W
            )
            
            # 3. 添加匹配的样本
            for peak_idx, gt_local_idx in matches:
                i, j, score = peaks[peak_idx]
                gt_global_idx = class_gt_indices[gt_local_idx]
                
                pos_samples.append({
                    'i': i,
                    'j': j,
                    'class': class_id,
                    'gt_idx': gt_global_idx,
                    'match_type': 'peak',
                    'confidence': score
                })
            
            # 4. 处理未匹配的GT (备用策略)
            if len(unmatched_gts) > 0:
                unmatched_global_indices = [
                    class_gt_indices[local_idx] 
                    for local_idx in unmatched_gts
                ]
                
                fallback_samples = self.fallback.assign_unmatched_gts(
                    unmatched_global_indices,
                    gt_boxes,
                    cam_class,
                    H, W
                )
                
                for i, j, gt_idx in fallback_samples:
                    pos_samples.append({
                        'i': i,
                        'j': j,
                        'class': class_id,
                        'gt_idx': gt_idx,
                        'match_type': 'fallback',
                        'confidence': cam_class[i, j].item()
                    })
        
        return pos_samples


