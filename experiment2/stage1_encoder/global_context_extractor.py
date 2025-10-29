#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局上下文提取器

功能：
提取全局图像嵌入 I_g，用作自动负样本

数学：
I_g = Projection(CLS_token)
I_g = Normalize(I_g)  # 归一化到单位球
"""

import torch
import torch.nn as nn


class GlobalContextExtractor(nn.Module):
    """
    全局上下文提取器
    
    从CLIP图像编码器的输出中提取全局上下文
    """
    
    def __init__(self, d_clip: int = 512):
        super().__init__()
        self.d_clip = d_clip
    
    def forward(self, global_embedding: torch.Tensor) -> torch.Tensor:
        """
        提取全局上下文
        
        参数:
            global_embedding: CLIP的全局嵌入 (B, d_clip)
        
        返回:
            I_g: 全局上下文 (B, d_clip)
        """
        # 已经是归一化的，直接返回
        return global_embedding


if __name__ == "__main__":
    extractor = GlobalContextExtractor()
    
    # 测试
    batch_size = 2
    d_clip = 512
    global_emb = torch.randn(batch_size, d_clip)
    global_emb = global_emb / global_emb.norm(dim=-1, keepdim=True)
    
    I_g = extractor(global_emb)
    
    print(f"全局上下文形状: {I_g.shape}")
    print("✅ 全局上下文提取器测试完成！")

