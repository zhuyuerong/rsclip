#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本调制模块

功能：
使用目标文本调制查询特征

数学：
q̃_m = LayerNorm(feat_m + W_t · t_c)
"""

import torch
import torch.nn as nn


class TextConditioner(nn.Module):
    """
    文本调制模块
    
    将文本信息融入查询特征
    """
    
    def __init__(
        self,
        d_model: int = 256,
        d_text: int = 512,
        method: str = 'add'
    ):
        """
        参数:
            d_model: 查询维度
            d_text: 文本嵌入维度
            method: 融合方法 ('add', 'concat')
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_text = d_text
        self.method = method
        
        if method == 'add':
            # 将文本投影到查询空间
            self.text_proj = nn.Linear(d_text, d_model)
            self.norm = nn.LayerNorm(d_model)
        
        elif method == 'concat':
            # 拼接后投影
            self.text_proj = nn.Linear(d_text, d_model)
            self.fusion = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model)
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def forward(
        self,
        query_feat: torch.Tensor,     # (B, M, d_model)
        text_embed: torch.Tensor      # (B, d_text) 或 (B, num_classes, d_text)
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            query_feat: 查询特征 (B, M, d_model)
            text_embed: 文本嵌入 (B, d_text)
        
        返回:
            conditioned_query: 调制后的查询 (B, M, d_model)
        """
        batch_size, num_queries, _ = query_feat.shape
        
        # 处理文本嵌入维度
        if len(text_embed.shape) == 2:
            # (B, d_text) -> (B, 1, d_text) -> (B, M, d_text)
            text_embed = text_embed.unsqueeze(1).expand(-1, num_queries, -1)
        elif len(text_embed.shape) == 3 and text_embed.size(1) == 1:
            # (B, 1, d_text) -> (B, M, d_text)
            text_embed = text_embed.expand(-1, num_queries, -1)
        
        # 投影文本
        text_feat = self.text_proj(text_embed)  # (B, M, d_model)
        
        if self.method == 'add':
            # 加法融合
            conditioned_query = self.norm(query_feat + text_feat)
        
        elif self.method == 'concat':
            # 拼接融合
            concat_feat = torch.cat([query_feat, text_feat], dim=-1)  # (B, M, 2*d_model)
            conditioned_query = self.fusion(concat_feat)  # (B, M, d_model)
        
        return conditioned_query


if __name__ == "__main__":
    conditioner = TextConditioner(d_model=256, d_text=512)
    conditioner = conditioner.cuda()
    
    # 测试
    batch_size = 2
    num_queries = 100
    query_feat = torch.randn(batch_size, num_queries, 256).cuda()
    text_embed = torch.randn(batch_size, 512).cuda()
    
    conditioned = conditioner(query_feat, text_embed)
    
    print(f"调制后查询形状: {conditioned.shape}")
    print("✅ 文本调制模块测试完成！")

