#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
位置查询初始化

功能：
生成 M 个连续位置查询 q_m = [pos_m, feat_m]

数学：
pos_m: 可学习的位置嵌入 ∈ ℝ^d_pos
feat_m: 可学习的内容嵌入 ∈ ℝ^d_feat
"""

import torch
import torch.nn as nn
import math


class QueryInitializer(nn.Module):
    """
    位置查询初始化器
    
    生成M个可学习的查询，包含位置信息和内容信息
    """
    
    def __init__(
        self,
        num_queries: int = 100,
        d_model: int = 256,
        init_method: str = 'learned'
    ):
        """
        参数:
            num_queries: 查询数量 M
            d_model: 查询维度
            init_method: 初始化方法 ('learned', 'sinusoidal')
        """
        super().__init__()
        
        self.num_queries = num_queries
        self.d_model = d_model
        self.init_method = init_method
        
        if init_method == 'learned':
            # 可学习的查询嵌入
            self.query_embed = nn.Embedding(num_queries, d_model)
            self.query_pos = nn.Embedding(num_queries, d_model)
            
            # 初始化
            nn.init.normal_(self.query_embed.weight)
            nn.init.normal_(self.query_pos.weight)
        
        elif init_method == 'sinusoidal':
            # 正弦位置编码
            self.query_embed = nn.Parameter(torch.randn(num_queries, d_model))
            self.query_pos = self._get_sinusoidal_embeddings(num_queries, d_model)
            self.register_buffer('fixed_pos', self.query_pos)
        
        else:
            raise ValueError(f"Unknown init_method: {init_method}")
    
    def _get_sinusoidal_embeddings(self, num_pos: int, d_model: int) -> torch.Tensor:
        """
        生成正弦位置编码
        
        参数:
            num_pos: 位置数量
            d_model: 嵌入维度
        
        返回:
            pos_embed: (num_pos, d_model)
        """
        position = torch.arange(num_pos).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        pos_embed = torch.zeros(num_pos, d_model)
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        
        return pos_embed
    
    def forward(self, batch_size: int) -> tuple:
        """
        初始化查询
        
        参数:
            batch_size: 批次大小
        
        返回:
            query_embed: 内容嵌入 (M, d_model)
            query_pos: 位置嵌入 (M, d_model)
        """
        if self.init_method == 'learned':
            # 可学习嵌入
            indices = torch.arange(self.num_queries).to(next(self.parameters()).device)
            query_embed = self.query_embed(indices)  # (M, d_model)
            query_pos = self.query_pos(indices)      # (M, d_model)
        
        else:
            # 正弦编码
            query_embed = self.query_embed  # (M, d_model)
            query_pos = self.fixed_pos      # (M, d_model)
        
        return query_embed, query_pos


if __name__ == "__main__":
    initializer = QueryInitializer(num_queries=100, d_model=256)
    initializer = initializer.cuda()
    
    query_embed, query_pos = initializer(batch_size=2)
    
    print(f"查询内容嵌入形状: {query_embed.shape}")
    print(f"查询位置嵌入形状: {query_pos.shape}")
    print("✅ 查询初始化器测试完成！")

