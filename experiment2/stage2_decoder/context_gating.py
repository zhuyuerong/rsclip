#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上下文门控模块（核心创新）⭐

功能:
使用全局上下文 I_g 调制查询 q_m，生成上下文感知查询 q'_m

核心思想:
全局上下文 I_g 包含了整张图像的场景信息（例如 "天空+跑道+建筑"），
通过将其作为门控信号调制局部查询，可以让模型：
1. 增强与场景相关的特征（例如飞机在机场场景中）
2. 抑制与场景无关的背景噪声

两种实现方式:
1. FiLM (Feature-wise Linear Modulation):
   γ, β = MLP(I_g)
   q'_m = γ ⊙ q̃_m + β
   
2. Concat + MLP:
   q'_m = MLP([q̃_m, I_g])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FiLMGating(nn.Module):
    """
    FiLM 门控 (Feature-wise Linear Modulation)
    
    优点: 参数少，计算高效
    数学: q'_m = γ ⊙ q̃_m + β
         其中 γ, β = MLP(I_g)
    
    参数:
        d_model: 查询维度
        d_context: 全局上下文维度
        hidden_dim: 隐藏层维度（可选）
    """
    
    def __init__(
        self,
        d_model: int,
        d_context: int,
        hidden_dim: Optional[int] = None
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = d_model * 2
        
        # MLP 生成 γ 和 β
        self.gamma_beta_net = nn.Sequential(
            nn.Linear(d_context, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, d_model * 2)  # γ 和 β
        )
        
        # 初始化：让 γ 初始为 1，β 初始为 0（恒等变换）
        nn.init.zeros_(self.gamma_beta_net[-1].weight)
        nn.init.zeros_(self.gamma_beta_net[-1].bias)
        self.gamma_beta_net[-1].bias.data[:d_model] = 1.0  # γ 初始为 1
    
    def forward(
        self,
        query: torch.Tensor,           # q̃_m: (N, M, d_model)
        global_context: torch.Tensor   # I_g: (N, d_context)
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            query: 文本调制后的查询，形状 (batch_size, num_queries, d_model)
            global_context: 全局上下文，形状 (batch_size, d_context)
        
        返回:
            gated_query: 上下文感知查询，形状 (batch_size, num_queries, d_model)
        """
        # 生成 γ 和 β
        gamma_beta = self.gamma_beta_net(global_context)  # (N, d_model * 2)
        
        d_model = query.size(-1)
        gamma = gamma_beta[:, :d_model].unsqueeze(1)  # (N, 1, d_model)
        beta = gamma_beta[:, d_model:].unsqueeze(1)   # (N, 1, d_model)
        
        # FiLM 调制
        gated_query = gamma * query + beta  # (N, M, d_model)
        
        return gated_query


class ConcatMLPGating(nn.Module):
    """
    Concat + MLP 门控
    
    优点: 表达能力强，更灵活
    数学: q'_m = MLP([q̃_m, I_g])
    
    参数:
        d_model: 查询维度
        d_context: 全局上下文维度
        hidden_dim: 隐藏层维度
        dropout: Dropout 比例
    """
    
    def __init__(
        self,
        d_model: int,
        d_context: int,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model + d_context, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model)
        )
        
        # 残差连接
        self.use_residual = True
    
    def forward(
        self,
        query: torch.Tensor,           # q̃_m: (N, M, d_model)
        global_context: torch.Tensor   # I_g: (N, d_context)
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            query: 文本调制后的查询，形状 (batch_size, num_queries, d_model)
            global_context: 全局上下文，形状 (batch_size, d_context)
        
        返回:
            gated_query: 上下文感知查询，形状 (batch_size, num_queries, d_model)
        """
        batch_size, num_queries, d_model = query.shape
        
        # 扩展全局上下文到每个查询
        global_context_expanded = global_context.unsqueeze(1).expand(
            batch_size, num_queries, -1
        )  # (N, M, d_context)
        
        # Concatenate
        concat_features = torch.cat([query, global_context_expanded], dim=-1)  # (N, M, d_model + d_context)
        
        # MLP
        gated_query = self.mlp(concat_features)  # (N, M, d_model)
        
        # 残差连接
        if self.use_residual:
            gated_query = gated_query + query
        
        return gated_query


class ContextGating(nn.Module):
    """
    上下文门控（统一接口）
    
    支持两种实现:
    - "film": FiLM 门控
    - "concat_mlp": Concat + MLP 门控
    
    参数:
        d_model: 查询维度
        d_context: 全局上下文维度
        gating_type: 门控类型，"film" 或 "concat_mlp"
        hidden_dim: 隐藏层维度
        dropout: Dropout 比例
    """
    
    def __init__(
        self,
        d_model: int,
        d_context: int,
        gating_type: str = "film",
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.gating_type = gating_type
        
        if gating_type == "film":
            self.gating = FiLMGating(
                d_model=d_model,
                d_context=d_context,
                hidden_dim=hidden_dim
            )
        elif gating_type == "concat_mlp":
            self.gating = ConcatMLPGating(
                d_model=d_model,
                d_context=d_context,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown gating type: {gating_type}")
    
    def forward(
        self,
        query: torch.Tensor,
        global_context: torch.Tensor
    ) -> torch.Tensor:
        """前向传播"""
        return self.gating(query, global_context)


def test_context_gating():
    """测试上下文门控"""
    print("=" * 70)
    print("测试上下文门控")
    print("=" * 70)
    
    batch_size = 2
    num_queries = 100
    d_model = 256
    d_context = 512
    
    # 创建测试数据
    query = torch.randn(batch_size, num_queries, d_model)
    global_context = torch.randn(batch_size, d_context)
    
    # 测试 FiLM 门控
    print("\n测试 FiLM 门控:")
    film_gating = ContextGating(d_model, d_context, gating_type="film")
    gated_query_film = film_gating(query, global_context)
    print(f"  输入形状: {query.shape}")
    print(f"  全局上下文形状: {global_context.shape}")
    print(f"  输出形状: {gated_query_film.shape}")
    
    # 测试 Concat+MLP 门控
    print("\n测试 Concat+MLP 门控:")
    concat_gating = ContextGating(d_model, d_context, gating_type="concat_mlp")
    gated_query_concat = concat_gating(query, global_context)
    print(f"  输入形状: {query.shape}")
    print(f"  全局上下文形状: {global_context.shape}")
    print(f"  输出形状: {gated_query_concat.shape}")
    
    # 参数统计
    print("\n参数统计:")
    print(f"  FiLM 门控参数: {sum(p.numel() for p in film_gating.parameters()):,}")
    print(f"  Concat+MLP 门控参数: {sum(p.numel() for p in concat_gating.parameters()):,}")
    
    print("\n✅ 上下文门控测试完成！")


if __name__ == "__main__":
    test_context_gating()

