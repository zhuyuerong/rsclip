# -*- coding: utf-8 -*-
"""
Multi-Instance Assigner
多峰匹配分配器：检测CAM上的多个峰，用匈牙利算法匹配峰和GT
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict, Optional


class MultiPeakDetector:
    """
    在CAM上检测多个局部极大值
    """
    
    def __init__(self, min_peak_distance: int = 2, min_peak_value: float = 0.3):
        """
        Args:
            min_peak_distance: CAM grid上的最小间距
            min_peak_value: 最小峰值阈值
        """
        self.min_distance = min_peak_distance
        self.min_value = min_peak_value
    
    def detect_peaks(self, cam_class: torch.Tensor) -> List[Tuple[int, int, float]]:
        """
        检测CAM上的局部极大值
        
        Args:
            cam_class: [H, W] 单个类别的CAM
        
        Returns:
            peaks: List[(i, j, score)] 峰值位置和分数
        """
        H, W = cam_class.shape
        
        # 找所有局部极大值
        # 用max pooling找local maxima
        kernel_size = self.min_distance * 2 + 1
        pooled = F.max_pool2d(
            cam_class.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=self.min_distance
        ).squeeze()
        
        # 局部极大值: 和pooling结果相等的点
        is_peak = (cam_class == pooled) & (cam_class > self.min_value)
        
        # 提取峰值位置
        peak_indices = is_peak.nonzero(as_tuple=False)
        peaks = []
        
        for idx in peak_indices:
            i, j = idx[0].item(), idx[1].item()
            score = cam_class[i, j].item()
            peaks.append((i, j, score))
        
        # 按分数排序
        peaks.sort(key=lambda x: x[2], reverse=True)
        
        return peaks


class PeakToGTMatcher:
    """
    用匈牙利算法匹配检测到的峰和GT boxes
    """
    
    def __init__(self, iou_threshold: float = 0.3):
        """
        Args:
            iou_threshold: 匹配的IoU阈值
        """
        self.iou_threshold = iou_threshold
    
    def match(self, peaks: List[Tuple[int, int, float]], 
              gt_boxes: torch.Tensor,
              cam_height: int, cam_width: int) -> Tuple[List[Tuple[int, int]], 
                                                         List[int], 
                                                         List[int]]:
        """
        匹配峰和GT boxes
        
        Args:
            peaks: List[(i, j, score)] CAM上的峰
            gt_boxes: [K, 4] GT框(归一化坐标)
            cam_height, cam_width: CAM分辨率
        
        Returns:
            matches: List[(peak_idx, gt_idx)]
            unmatched_peaks: List[peak_idx]
            unmatched_gts: List[gt_idx]
        """
        n_peaks = len(peaks)
        n_gts = len(gt_boxes)
        
        if n_peaks == 0 or n_gts == 0:
            return [], list(range(n_peaks)), list(range(n_gts))
        
        # 计算cost matrix
        cost_matrix = torch.zeros(n_peaks, n_gts)
        
        for p_idx, (i, j, score) in enumerate(peaks):
            # 峰的位置(归一化)
            peak_x = (j + 0.5) / cam_width
            peak_y = (i + 0.5) / cam_height
            
            for g_idx, gt_box in enumerate(gt_boxes):
                xmin, ymin, xmax, ymax = gt_box
                
                # Cost = 峰是否在框内 + 距离
                in_box = (xmin <= peak_x <= xmax) and (ymin <= peak_y <= ymax)
                
                if in_box:
                    # 峰在框内 → cost低 (负数表示好)
                    # 距离框中心越近越好
                    cx_gt = (xmin + xmax) / 2
                    cy_gt = (ymin + ymax) / 2
                    dist = ((peak_x - cx_gt)**2 + (peak_y - cy_gt)**2)**0.5
                    cost_matrix[p_idx, g_idx] = dist - 1.0  # 减1使其为负
                else:
                    # 峰在框外 → cost高
                    # 计算到框的最短距离
                    dist_to_box = self._point_to_box_distance(
                        peak_x, peak_y, gt_box
                    )
                    cost_matrix[p_idx, g_idx] = dist_to_box
        
        # 匈牙利算法
        row_ind, col_ind = linear_sum_assignment(cost_matrix.numpy())
        
        # 过滤低质量匹配
        matches = []
        matched_peaks = set()
        matched_gts = set()
        
        for p_idx, g_idx in zip(row_ind, col_ind):
            cost = cost_matrix[p_idx, g_idx].item()
            
            # 只保留合理的匹配(峰在框内或很近)
            if cost < 0.5:  # 阈值可调
                matches.append((p_idx, g_idx))
                matched_peaks.add(p_idx)
                matched_gts.add(g_idx)
        
        unmatched_peaks = [i for i in range(n_peaks) if i not in matched_peaks]
        unmatched_gts = [i for i in range(n_gts) if i not in matched_gts]
        
        return matches, unmatched_peaks, unmatched_gts
    
    def _point_to_box_distance(self, px: float, py: float, 
                              box: torch.Tensor) -> float:
        """点到框的最短距离"""
        xmin, ymin, xmax, ymax = box
        
        # 点在框内
        if xmin <= px <= xmax and ymin <= py <= ymax:
            return 0.0
        
        # 点在框外,计算到边界的距离
        dx = max(xmin - px, 0, px - xmax)
        dy = max(ymin - py, 0, py - ymax)
        
        return (dx**2 + dy**2)**0.5


class FallbackAssigner:
    """
    为未匹配的GT分配正样本(备用策略)
    """
    
    def assign_unmatched_gts(self, unmatched_gt_indices: List[int],
                            gt_boxes: torch.Tensor,
                            cam_class: torch.Tensor,
                            cam_height: int, cam_width: int) -> List[Tuple[int, int, int]]:
        """
        为未匹配的GT分配正样本
        
        Args:
            unmatched_gt_indices: List[int] 未匹配的GT索引
            gt_boxes: [K, 4] 所有GT框
            cam_class: [H, W] CAM
        
        Returns:
            fallback_samples: List[(i, j, gt_idx)]
        """
        samples = []
        
        for gt_idx in unmatched_gt_indices:
            box = gt_boxes[gt_idx]
            xmin, ymin, xmax, ymax = box
            
            # 映射到CAM grid
            i_min = max(0, int(ymin * cam_height))
            i_max = min(cam_height - 1, int(ymax * cam_height))
            j_min = max(0, int(xmin * cam_width))
            j_max = min(cam_width - 1, int(xmax * cam_width))
            
            # 策略1: 框内CAM响应最大的点
            roi_cam = cam_class[i_min:i_max+1, j_min:j_max+1]
            
            if roi_cam.numel() > 0:
                flat_idx = roi_cam.argmax()
                i_local = flat_idx // roi_cam.shape[1]
                j_local = flat_idx % roi_cam.shape[1]
                
                i_star = i_min + i_local
                j_star = j_min + j_local
                
                samples.append((i_star, j_star, gt_idx))
            else:
                # 策略2: 框太小,用框中心
                i_center = int((ymin + ymax) / 2 * cam_height)
                j_center = int((xmin + xmax) / 2 * cam_width)
                
                i_center = max(0, min(cam_height - 1, i_center))
                j_center = max(0, min(cam_width - 1, j_center))
                
                samples.append((i_center, j_center, gt_idx))
        
        return samples


class MultiInstanceAssigner:
    """
    完整的多实例正样本分配器
    
    处理流程:
    1. 每个类别检测多个峰
    2. 峰与该类的GT匹配
    3. 未匹配的GT用备用策略
    """
    
    def __init__(self,
                 min_peak_distance: int = 2,
                 min_peak_value: float = 0.3,
                 match_iou_threshold: float = 0.3):
        """
        Args:
            min_peak_distance: 峰之间的最小距离
            min_peak_value: 最小峰值阈值
            match_iou_threshold: 匹配的IoU阈值
        """
        self.peak_detector = MultiPeakDetector(
            min_peak_distance, min_peak_value
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
            
            # 1. 检测峰
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


