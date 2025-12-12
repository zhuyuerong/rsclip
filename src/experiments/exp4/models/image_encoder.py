# -*- coding: utf-8 -*-
"""
原图编码器
极简CNN，将原图编码到CAM分辨率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleImageEncoder(nn.Module):
    """
    极简原图编码器
    
    架构:
    - Conv2d(3->64, k=3, p=1) + BN + ReLU
    - Conv2d(64->128, k=3, s=2, p=1) + BN + ReLU  # 下采样
    - Conv2d(128->128, k=3, s=2, p=1) + BN + ReLU # 下采样
    - AdaptiveAvgPool2d(7, 7)  # 到CAM尺寸
    
    输入: [B, 3, 224, 224]
    输出: [B, 128, 7, 7]
    参数量: ~75K
    """
    
    def __init__(self, output_dim: int = 128, output_size: int = 7):
        """
        Args:
            output_dim: 输出特征维度（默认128）
            output_size: 输出空间尺寸（默认7，CAM分辨率）
        """
        super().__init__()
        
        self.output_dim = output_dim
        self.output_size = output_size
        
        self.encoder = nn.Sequential(
            # 第一层：提取基础特征
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 第二层：下采样（224 -> 112）
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 第三层：进一步下采样（112 -> 56）
            nn.Conv2d(128, output_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            
            # 自适应池化到目标尺寸（56 -> 7）
            nn.AdaptiveAvgPool2d((output_size, output_size))
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [B, 3, 224, 224] 原图
        
        Returns:
            features: [B, output_dim, output_size, output_size]
        """
        return self.encoder(x)
    
    def get_param_count(self) -> int:
        """获取参数量"""
        return sum(p.numel() for p in self.parameters())


