#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回归头

功能：
预测边界框 b_m

输出：
b_m = (cx, cy, w, h)，归一化坐标
"""

import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    """
    回归头
    
    预测边界框的中心坐标和宽高
    """
    
    def __init__(
        self,
        d_model: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3
    ):
        """
        参数:
            d_model: 解码器输出维度
            hidden_dim: 隐藏层维度
            num_layers: MLP层数
        """
        super().__init__()
        
        # 构建MLP
        layers = []
        in_dim = d_model
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ])
            in_dim = hidden_dim
        
        # 最后一层输出4个值 (cx, cy, w, h)
        layers.append(nn.Linear(in_dim, 4))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, decoder_output: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            decoder_output: 解码器输出 z_m，形状 (B, M, d_model)
        
        返回:
            bboxes: 预测的边界框，形状 (B, M, 4)
                   格式：(cx, cy, w, h)，归一化到 [0, 1]
        """
        # 预测边界框
        bboxes = self.mlp(decoder_output)  # (B, M, 4)
        
        # Sigmoid归一化到 [0, 1]
        bboxes = torch.sigmoid(bboxes)
        
        return bboxes


if __name__ == "__main__":
    head = RegressionHead(d_model=256)
    head = head.cuda()
    
    # 测试
    batch_size = 2
    num_queries = 100
    decoder_output = torch.randn(batch_size, num_queries, 256).cuda()
    
    bboxes = head(decoder_output)
    
    print(f"边界框形状: {bboxes.shape}")
    print(f"边界框范围: [{bboxes.min().item():.3f}, {bboxes.max().item():.3f}] (应该在[0,1])")
    print("✅ 回归头测试完成！")

