#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征金字塔网络 (FPN)

功能：
1. 融合多层级特征
2. 生成多尺度特征图
3. 支持自顶向下的特征传播
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class FPN(nn.Module):
    """
    特征金字塔网络
    
    从RemoteCLIP的多层级特征构建特征金字塔
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        num_outs: int = 4
    ):
        """
        参数:
            in_channels: 输入特征的通道数列表 (从低层到高层)
            out_channels: 输出特征的通道数
            num_outs: 输出特征图的数量
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        
        # 侧向连接（1x1卷积降维）
        self.lateral_convs = nn.ModuleList()
        for in_channel in in_channels:
            lateral_conv = nn.Conv2d(
                in_channel,
                out_channels,
                kernel_size=1
            )
            self.lateral_convs.append(lateral_conv)
        
        # 输出卷积（3x3卷积平滑）
        self.fpn_convs = nn.ModuleList()
        for _ in range(self.num_ins):
            fpn_conv = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            )
            self.fpn_convs.append(fpn_conv)
        
        # 额外层（如果需要更多输出）
        self.extra_convs = nn.ModuleList()
        if num_outs > self.num_ins:
            for i in range(num_outs - self.num_ins):
                extra_conv = nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1
                )
                self.extra_convs.append(extra_conv)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        前向传播
        
        参数:
            inputs: 输入特征列表 [(B, C1, H1, W1), (B, C2, H2, W2), ...]
        
        返回:
            outputs: 输出特征列表 [(B, C, H1, W1), (B, C, H2, W2), ...]
        """
        assert len(inputs) == self.num_ins
        
        # 自底向上传播（侧向连接）
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # 自顶向下传播 + 特征融合
        for i in range(self.num_ins - 1, 0, -1):
            # 上采样高层特征
            upsampled = F.interpolate(
                laterals[i],
                size=laterals[i-1].shape[-2:],
                mode='nearest'
            )
            # 融合
            laterals[i-1] = laterals[i-1] + upsampled
        
        # 输出卷积
        outputs = [
            fpn_conv(laterals[i])
            for i, fpn_conv in enumerate(self.fpn_convs)
        ]
        
        # 额外层
        if self.num_outs > self.num_ins:
            for i, extra_conv in enumerate(self.extra_convs):
                if i == 0:
                    outputs.append(extra_conv(outputs[-1]))
                else:
                    outputs.append(extra_conv(F.relu(outputs[-1])))
        
        return outputs


if __name__ == "__main__":
    print("=" * 70)
    print("测试FPN模块")
    print("=" * 70)
    
    # 创建FPN
    in_channels = [512, 1024, 2048]  # ResNet-50的输出通道
    fpn = FPN(
        in_channels=in_channels,
        out_channels=256,
        num_outs=4
    )
    
    # 测试数据
    batch_size = 2
    inputs = [
        torch.randn(batch_size, 512, 100, 100),
        torch.randn(batch_size, 1024, 50, 50),
        torch.randn(batch_size, 2048, 25, 25)
    ]
    
    # 前向传播
    outputs = fpn(inputs)
    
    print("\n输入特征:")
    for i, feat in enumerate(inputs):
        print(f"  层{i}: {feat.shape}")
    
    print("\n输出特征:")
    for i, feat in enumerate(outputs):
        print(f"  层{i}: {feat.shape}")
    
    print("\n✅ FPN模块测试完成！")

