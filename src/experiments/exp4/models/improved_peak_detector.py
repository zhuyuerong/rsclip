# -*- coding: utf-8 -*-
"""
改进的峰值检测器 - 实验3.3
更鲁棒的峰值检测算法
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np


class ImprovedMultiPeakDetector:
    """
    改进的多峰值检测器
    
    改进:
    1. 自适应阈值: threshold = max(min_value, cam_max * 0.5)
    2. 多尺度峰值检测（可选）
    3. NMS去重（移除距离太近的峰值）
    """
    
    def __init__(self, min_peak_distance: int = 2, min_peak_value: float = 0.3,
                 adaptive_threshold: bool = True, use_multi_scale: bool = False,
                 nms_threshold: float = None):
        """
        Args:
            min_peak_distance: CAM grid上的最小间距
            min_peak_value: 最小峰值阈值（基础值）
            adaptive_threshold: 是否使用自适应阈值
            use_multi_scale: 是否使用多尺度峰值检测
            nms_threshold: NMS阈值（如果None，使用min_peak_distance）
        """
        self.min_distance = min_peak_distance
        self.min_value = min_peak_value
        self.adaptive_threshold = adaptive_threshold
        self.use_multi_scale = use_multi_scale
        self.nms_threshold = nms_threshold if nms_threshold is not None else min_peak_distance
    
    def detect_peaks(self, cam_class: torch.Tensor) -> List[Tuple[int, int, float]]:
        """
        检测CAM上的局部极大值（改进版）
        
        Args:
            cam_class: [H, W] 单个类别的CAM
        
        Returns:
            peaks: List[(i, j, score)] 峰值位置和分数
        """
        H, W = cam_class.shape
        
        # 改进1: 自适应阈值
        if self.adaptive_threshold:
            cam_max = cam_class.max().item()
            cam_mean = cam_class.mean().item()
            # 动态调整阈值：如果最大值不高，降低阈值
            # threshold = max(min_value, cam_max * 0.5)
            threshold = max(self.min_value, cam_max * 0.5, cam_mean * 2.0)
        else:
            threshold = self.min_value
        
        peaks_multi_scale = []
        
        if self.use_multi_scale:
            # 改进2: 多尺度峰值检测
            for scale in [1, 2]:
                kernel_size = (self.min_distance * scale) * 2 + 1
                pooled = F.max_pool2d(
                    cam_class.unsqueeze(0).unsqueeze(0),
                    kernel_size=kernel_size,
                    stride=1,
                    padding=self.min_distance * scale
                ).squeeze()
                
                is_peak = (cam_class == pooled) & (cam_class > threshold)
                peak_indices = is_peak.nonzero(as_tuple=False)
                
                for idx in peak_indices:
                    i, j = idx[0].item(), idx[1].item()
                    score = cam_class[i, j].item()
                    peaks_multi_scale.append((i, j, score))
        else:
            # 单尺度检测
            kernel_size = self.min_distance * 2 + 1
            pooled = F.max_pool2d(
                cam_class.unsqueeze(0).unsqueeze(0),
                kernel_size=kernel_size,
                stride=1,
                padding=self.min_distance
            ).squeeze()
            
            is_peak = (cam_class == pooled) & (cam_class > threshold)
            peak_indices = is_peak.nonzero(as_tuple=False)
            
            for idx in peak_indices:
                i, j = idx[0].item(), idx[1].item()
                score = cam_class[i, j].item()
                peaks_multi_scale.append((i, j, score))
        
        # 改进3: NMS去重（移除距离太近的峰值）
        peaks = self._nms_peaks(peaks_multi_scale)
        
        # 按分数排序
        peaks.sort(key=lambda x: x[2], reverse=True)
        
        return peaks
    
    def _nms_peaks(self, peaks: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        """
        对峰值进行NMS去重
        
        Args:
            peaks: List[(i, j, score)] 原始峰值列表
        
        Returns:
            peaks_nms: List[(i, j, score)] 去重后的峰值列表
        """
        if len(peaks) == 0:
            return []
        
        # 转换为numpy数组便于计算
        peaks_array = np.array([[p[0], p[1], p[2]] for p in peaks])
        
        keep = []
        remaining = list(range(len(peaks)))
        
        while len(remaining) > 0:
            # 选择分数最高的
            best_idx = remaining[0]
            best_score = peaks_array[remaining[0], 2]
            
            for idx in remaining:
                if peaks_array[idx, 2] > best_score:
                    best_idx = idx
                    best_score = peaks_array[idx, 2]
            
            keep.append(best_idx)
            
            # 移除距离太近的峰值
            best_pos = peaks_array[best_idx, :2]
            remaining_new = []
            
            for idx in remaining:
                if idx == best_idx:
                    continue
                
                other_pos = peaks_array[idx, :2]
                distance = np.sqrt(np.sum((best_pos - other_pos) ** 2))
                
                if distance > self.nms_threshold:
                    remaining_new.append(idx)
            
            remaining = remaining_new
        
        # 返回保留的峰值
        return [(int(peaks_array[idx, 0]), int(peaks_array[idx, 1]), float(peaks_array[idx, 2])) 
                for idx in keep]


