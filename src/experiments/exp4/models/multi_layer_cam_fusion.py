# -*- coding: utf-8 -*-
"""
多层CAM融合模块
使用可学习加权平均融合多层CAM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class MultiLayerCAMFusion(nn.Module):
    """
    多层CAM融合模块
    
    输入: List[Tensor[B, C, H, W]] (3层CAM)
    输出: Tensor[B, C, H, W] (融合后的CAM)
    
    方法：可学习加权平均
    - layer_weights: Parameter(shape=[3])
    - 使用softmax归一化权重
    - 加权求和: fused_cam = sum(cam_i * weight_i)
    """
    
    def __init__(self, num_layers: int = 3):
        """
        Args:
            num_layers: CAM层数（默认3）
        """
        super().__init__()
        self.num_layers = num_layers
        
        # 可学习权重（初始化为均匀分布）
        self.layer_weights = nn.Parameter(
            torch.ones(num_layers) / num_layers
        )
    
    def forward(self, multi_layer_cams: List[torch.Tensor]) -> torch.Tensor:
        """
        融合多层CAM
        
        Args:
            multi_layer_cams: List[Tensor[B, C, H, W]] (L个)
        
        Returns:
            fused_cam: [B, C, H, W]
        """
        if len(multi_layer_cams) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} CAMs, got {len(multi_layer_cams)}"
            )
        
        # Stack: [B, L, C, H, W]
        cams_stack = torch.stack(multi_layer_cams, dim=1)
        
        # 归一化权重（softmax确保权重和为1）
        weights = F.softmax(self.layer_weights, dim=0)
        weights = weights.view(1, -1, 1, 1, 1)  # [1, L, 1, 1, 1]
        
        # 加权平均
        fused_cam = (cams_stack * weights).sum(dim=1)  # [B, C, H, W]
        
        return fused_cam
    
    def get_layer_weights(self) -> torch.Tensor:
        """
        获取归一化后的层权重（用于监控）
        
        Returns:
            weights: [L] 归一化权重
        """
        with torch.no_grad():
            weights = F.softmax(self.layer_weights, dim=0)
        return weights.cpu().numpy()


