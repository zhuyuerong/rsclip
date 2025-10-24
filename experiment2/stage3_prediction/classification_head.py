#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分类头

功能：
将解码器输出映射到 CLIP 空间

数学：
f_m = Normalize(W_f · z_m)  # 局部特征 ∈ ℝ^d_clip
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """
    分类头
    
    将解码器输出映射到CLIP空间，用于与文本进行对比
    """
    
    def __init__(
        self,
        d_model: int = 256,
        d_clip: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 2
    ):
        """
        参数:
            d_model: 解码器输出维度
            d_clip: CLIP空间维度
            hidden_dim: 隐藏层维度
            num_layers: MLP层数
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_clip = d_clip
        
        # 构建MLP
        layers = []
        in_dim = d_model
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        
        # 最后一层投影到CLIP空间
        layers.append(nn.Linear(in_dim, d_clip))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, decoder_output: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            decoder_output: 解码器输出 z_m，形状 (B, M, d_model)
        
        返回:
            local_features: 局部特征 f_m，形状 (B, M, d_clip)，已归一化
        """
        # 投影到CLIP空间
        local_features = self.mlp(decoder_output)  # (B, M, d_clip)
        
        # L2归一化到单位球
        local_features = F.normalize(local_features, p=2, dim=-1)
        
        return local_features


if __name__ == "__main__":
    head = ClassificationHead(d_model=256, d_clip=512)
    head = head.cuda()
    
    # 测试
    batch_size = 2
    num_queries = 100
    decoder_output = torch.randn(batch_size, num_queries, 256).cuda()
    
    local_features = head(decoder_output)
    
    print(f"局部特征形状: {local_features.shape}")
    print(f"特征范数: {local_features.norm(dim=-1).mean().item():.4f} (应该≈1.0)")
    print("✅ 分类头测试完成！")

