# -*- coding: utf-8 -*-
"""
可学习的峰值检测器
使用objectness score和可学习的阈值替代固定阈值
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class LearnablePeakDetector(nn.Module):
    """
    可学习的峰值检测器
    
    改进:
    1. 使用objectness score替代固定阈值
    2. 可学习的阈值参数
    3. 端到端训练
    """
    
    def __init__(self, min_peak_distance: int = 2, 
                 init_threshold: float = 0.3,
                 use_objectness: bool = True):
        """
        Args:
            min_peak_distance: 峰之间的最小距离
            init_threshold: 初始阈值（可学习）
            use_objectness: 是否使用objectness score
        """
        super().__init__()
        
        self.min_distance = min_peak_distance
        self.use_objectness = use_objectness
        
        # 可学习的阈值参数（每个类别一个）
        # 初始化为init_threshold，通过sigmoid映射到[0,1]
        self.register_parameter(
            'threshold_logit',
            nn.Parameter(torch.ones(1) * self._threshold_to_logit(init_threshold))
        )
        
        # 可学习的自适应阈值权重
        self.adaptive_weight = nn.Parameter(torch.tensor(0.5))  # 平衡固定阈值和自适应阈值
    
    def _threshold_to_logit(self, threshold: float) -> float:
        """将阈值转换为logit空间"""
        return torch.logit(torch.tensor(threshold)).item()
    
    def _logit_to_threshold(self, logit: torch.Tensor) -> torch.Tensor:
        """将logit转换为阈值"""
        return torch.sigmoid(logit)
    
    def forward(self, cam_class: torch.Tensor, 
                objectness: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        检测峰值（训练时）
        
        Args:
            cam_class: [B, H, W] 或 [H, W] 单个类别的CAM
            objectness: [B, H, W] 或 [H, W] objectness score（可选）
        
        Returns:
            peak_mask: [B, H, W] 或 [H, W] 峰值mask（1表示峰值，0表示非峰值）
        """
        if cam_class.dim() == 2:
            cam_class = cam_class.unsqueeze(0)
            batch_mode = False
        else:
            batch_mode = True
        
        B, H, W = cam_class.shape
        
        # 获取可学习阈值
        learnable_threshold = self._logit_to_threshold(self.threshold_logit)
        
        # 计算自适应阈值（基于CAM的最大值和平均值）
        cam_max = cam_class.view(B, -1).max(dim=1)[0].view(B, 1, 1)  # [B, 1, 1]
        cam_mean = cam_class.view(B, -1).mean(dim=1).view(B, 1, 1)  # [B, 1, 1]
        
        # 自适应阈值：max * 0.5 或 mean * 2.0，取较大值
        adaptive_threshold = torch.max(
            cam_max * 0.5,
            cam_mean * 2.0
        )
        
        # 混合阈值：可学习阈值 + 自适应阈值
        # adaptive_weight控制两者的平衡
        mixed_threshold = (
            self.adaptive_weight * learnable_threshold +
            (1 - self.adaptive_weight) * adaptive_threshold
        )
        
        # 如果使用objectness，结合objectness score
        if self.use_objectness and objectness is not None:
            if objectness.dim() == 2:
                objectness = objectness.unsqueeze(0)
            
            # CAM * objectness作为最终分数
            combined_score = cam_class * objectness
            
            # 阈值也考虑objectness
            threshold = mixed_threshold * (0.5 + 0.5 * objectness.mean(dim=(1,2), keepdim=True))
        else:
            combined_score = cam_class
            threshold = mixed_threshold
        
        # 使用max pooling找局部极大值
        kernel_size = self.min_distance * 2 + 1
        pooled = F.max_pool2d(
            combined_score.unsqueeze(1),
            kernel_size=kernel_size,
            stride=1,
            padding=self.min_distance
        ).squeeze(1)
        
        # 局部极大值且超过阈值
        is_peak = (combined_score == pooled) & (combined_score > threshold)
        
        if not batch_mode:
            is_peak = is_peak.squeeze(0)
        
        return is_peak.float()
    
    def detect_peaks(self, cam_class: torch.Tensor,
                     objectness: Optional[torch.Tensor] = None) -> List[Tuple[int, int, float]]:
        """
        检测峰值（推理时）
        
        Args:
            cam_class: [H, W] 单个类别的CAM
            objectness: [H, W] objectness score（可选）
        
        Returns:
            peaks: List[(i, j, score)] 峰值位置和分数
        """
        self.eval()
        with torch.no_grad():
            peak_mask = self.forward(cam_class, objectness)
            
            # 提取峰值位置
            peak_indices = peak_mask.nonzero(as_tuple=False)
            peaks = []
            
            if objectness is not None:
                score = (cam_class * objectness)
            else:
                score = cam_class
            
            for idx in peak_indices:
                i, j = idx[0].item(), idx[1].item()
                peak_score = score[i, j].item()
                peaks.append((i, j, peak_score))
            
            # 按分数排序
            peaks.sort(key=lambda x: x[2], reverse=True)
            
            return peaks


class LearnableMultiPeakDetector(nn.Module):
    """
    可学习的多峰值检测器（支持多类别）
    """
    
    def __init__(self, num_classes: int, min_peak_distance: int = 2,
                 init_threshold: float = 0.3, use_objectness: bool = True):
        """
        Args:
            num_classes: 类别数
            min_peak_distance: 峰之间的最小距离
            init_threshold: 初始阈值
            use_objectness: 是否使用objectness score
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # 为每个类别创建可学习的峰值检测器
        self.detectors = nn.ModuleList([
            LearnablePeakDetector(
                min_peak_distance=min_peak_distance,
                init_threshold=init_threshold,
                use_objectness=use_objectness
            )
            for _ in range(num_classes)
        ])
    
    def forward(self, cam: torch.Tensor, 
                objectness: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        检测所有类别的峰值
        
        Args:
            cam: [B, C, H, W] CAM
            objectness: [B, C, H, W] objectness score（可选）
        
        Returns:
            peak_masks: [B, C, H, W] 峰值mask
        """
        B, C, H, W = cam.shape
        peak_masks = []
        
        for c in range(C):
            cam_class = cam[:, c]  # [B, H, W]
            obj_class = objectness[:, c] if objectness is not None else None
            
            peak_mask = self.detectors[c](cam_class, obj_class)  # [B, H, W]
            peak_masks.append(peak_mask)
        
        return torch.stack(peak_masks, dim=1)  # [B, C, H, W]
    
    def detect_peaks(self, cam: torch.Tensor,
                     objectness: Optional[torch.Tensor] = None) -> List[List[Tuple[int, int, float]]]:
        """
        检测所有类别的峰值（推理时）
        
        Args:
            cam: [C, H, W] 或 [B, C, H, W] CAM
            objectness: [C, H, W] 或 [B, C, H, W] objectness score
        
        Returns:
            peaks_per_class: List[List[(i, j, score)]] 每个类别的峰值列表
        """
        if cam.dim() == 3:
            cam = cam.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False
        
        B, C, H, W = cam.shape
        
        if objectness is not None and objectness.dim() == 3:
            objectness = objectness.unsqueeze(0)
        
        peaks_per_class = []
        
        for c in range(C):
            cam_class = cam[0, c] if squeeze_batch else cam[:, c]
            obj_class = objectness[0, c] if (objectness is not None and squeeze_batch) else \
                       (objectness[:, c] if objectness is not None else None)
            
            peaks = self.detectors[c].detect_peaks(cam_class, obj_class)
            peaks_per_class.append(peaks)
        
        return peaks_per_class


