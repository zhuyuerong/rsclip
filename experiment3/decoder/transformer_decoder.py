#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer解码器

功能：
1. 多层Transformer解码器
2. 支持文本引导的交叉注意力
3. 支持多层级特征融合
"""

import torch
import torch.nn as nn
from typing import List, Optional


class TransformerDecoderLayer(nn.Module):
    """
    Transformer解码器层
    
    包括：
    1. 自注意力
    2. 交叉注意力（文本引导）
    3. 前馈网络
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        text_guided: bool = True
    ):
        """
        参数:
            d_model: 模型维度
            num_heads: 注意力头数
            dim_feedforward: FFN维度
            dropout: Dropout率
            text_guided: 是否使用文本引导
        """
        super().__init__()
        
        self.d_model = d_model
        self.text_guided = text_guided
        
        # 自注意力
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # 交叉注意力（视觉特征）
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # 文本引导的交叉注意力
        if text_guided:
            self.text_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.norm_text = nn.LayerNorm(d_model)
            self.dropout_text = nn.Dropout(dropout)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_pos: Optional[torch.Tensor] = None,
        memory_pos: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            tgt: 目标查询 (B, num_queries, d_model)
            memory: 编码器输出（视觉特征） (B, N, d_model)
            tgt_pos: 查询位置编码 (B, num_queries, d_model)
            memory_pos: 记忆位置编码 (B, N, d_model)
            text_features: 文本特征 (B, num_classes, d_model)
        
        返回:
            output: (B, num_queries, d_model)
        """
        # 自注意力
        q = k = tgt + tgt_pos if tgt_pos is not None else tgt
        attn_out, _ = self.self_attn(q, k, tgt)
        tgt = self.norm1(tgt + self.dropout1(attn_out))
        
        # 交叉注意力（视觉特征）
        query = tgt + tgt_pos if tgt_pos is not None else tgt
        key = memory + memory_pos if memory_pos is not None else memory
        attn_out, _ = self.cross_attn(query, key, memory)
        tgt = self.norm2(tgt + self.dropout2(attn_out))
        
        # 文本引导的交叉注意力
        if self.text_guided and text_features is not None:
            attn_out, _ = self.text_attn(tgt, text_features, text_features)
            tgt = self.norm_text(tgt + self.dropout_text(attn_out))
        
        # 前馈网络
        ffn_out = self.ffn(tgt)
        tgt = self.norm3(tgt + ffn_out)
        
        return tgt


class TransformerDecoder(nn.Module):
    """
    多层Transformer解码器
    
    支持：
    1. 多层解码
    2. 文本引导
    3. 多尺度特征融合
    """
    
    def __init__(
        self,
        num_layers: int = 6,
        d_model: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        text_guided: bool = True,
        return_intermediate: bool = True
    ):
        """
        参数:
            num_layers: 解码器层数
            d_model: 模型维度
            num_heads: 注意力头数
            dim_feedforward: FFN维度
            dropout: Dropout率
            text_guided: 是否使用文本引导
            return_intermediate: 是否返回中间层输出
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        
        # 解码器层
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                text_guided=text_guided
            )
            for _ in range(num_layers)
        ])
        
        # 归一化
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_pos: Optional[torch.Tensor] = None,
        memory_pos: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            tgt: 目标查询 (B, num_queries, d_model)
            memory: 编码器输出 (B, N, d_model)
            tgt_pos: 查询位置编码 (B, num_queries, d_model)
            memory_pos: 记忆位置编码 (B, N, d_model)
            text_features: 文本特征 (B, num_classes, d_model)
        
        返回:
            output: (B, num_queries, d_model) 或 (num_layers, B, num_queries, d_model)
        """
        output = tgt
        intermediate = []
        
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_pos=tgt_pos,
                memory_pos=memory_pos,
                text_features=text_features
            )
            
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        
        if self.return_intermediate:
            return torch.stack(intermediate)  # (num_layers, B, num_queries, d_model)
        
        return self.norm(output)


if __name__ == "__main__":
    print("=" * 70)
    print("测试Transformer解码器")
    print("=" * 70)
    
    # 创建解码器
    decoder = TransformerDecoder(
        num_layers=6,
        d_model=256,
        num_heads=8,
        dim_feedforward=2048,
        text_guided=True,
        return_intermediate=True
    )
    
    # 测试数据
    batch_size = 2
    num_queries = 300
    num_tokens = 5000
    num_classes = 20
    
    tgt = torch.randn(batch_size, num_queries, 256)
    memory = torch.randn(batch_size, num_tokens, 256)
    tgt_pos = torch.randn(batch_size, num_queries, 256)
    memory_pos = torch.randn(batch_size, num_tokens, 256)
    text_features = torch.randn(batch_size, num_classes, 256)
    
    # 前向传播
    output = decoder(tgt, memory, tgt_pos, memory_pos, text_features)
    
    print(f"\n输入:")
    print(f"  目标查询: {tgt.shape}")
    print(f"  记忆: {memory.shape}")
    print(f"  文本特征: {text_features.shape}")
    
    print(f"\n输出: {output.shape}")
    
    print("\n✅ Transformer解码器测试完成！")

