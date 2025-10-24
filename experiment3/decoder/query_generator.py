#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查询生成器

功能：
1. 生成目标查询（object queries）
2. 支持可学习的查询嵌入
3. 支持位置查询和内容查询的分离
"""

import torch
import torch.nn as nn


class QueryGenerator(nn.Module):
    """
    查询生成器
    
    生成用于DETR解码器的目标查询
    """
    
    def __init__(
        self,
        num_queries: int = 300,
        d_model: int = 256,
        separate_pos_content: bool = True
    ):
        """
        参数:
            num_queries: 查询数量
            d_model: 模型维度
            separate_pos_content: 是否分离位置和内容查询
        """
        super().__init__()
        
        self.num_queries = num_queries
        self.d_model = d_model
        self.separate_pos_content = separate_pos_content
        
        if separate_pos_content:
            # 位置查询和内容查询分离
            self.query_pos = nn.Embedding(num_queries, d_model)
            self.query_content = nn.Embedding(num_queries, d_model)
        else:
            # 统一查询
            self.query_embed = nn.Embedding(num_queries, d_model * 2)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
    
    def forward(self, batch_size: int) -> tuple:
        """
        前向传播
        
        参数:
            batch_size: 批次大小
        
        返回:
            query_content: 内容查询 (B, num_queries, d_model)
            query_pos: 位置查询 (B, num_queries, d_model)
        """
        if self.separate_pos_content:
            # 分离的位置和内容查询
            query_content = self.query_content.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            query_pos = self.query_pos.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            # 统一查询，分割为两部分
            query_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            query_content, query_pos = torch.split(query_embed, self.d_model, dim=-1)
        
        return query_content, query_pos


if __name__ == "__main__":
    print("=" * 70)
    print("测试查询生成器")
    print("=" * 70)
    
    # 创建查询生成器
    query_gen = QueryGenerator(
        num_queries=300,
        d_model=256,
        separate_pos_content=True
    )
    
    # 生成查询
    batch_size = 2
    query_content, query_pos = query_gen(batch_size)
    
    print(f"\n内容查询: {query_content.shape}")
    print(f"位置查询: {query_pos.shape}")
    
    print("\n✅ 查询生成器测试完成！")

