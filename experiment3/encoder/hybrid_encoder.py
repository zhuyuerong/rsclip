#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合编码器

功能：
1. 结合FPN和Transformer编码器
2. 多尺度特征提取
3. 支持位置编码
"""

import torch
import torch.nn as nn
import math
from typing import List, Optional


class PositionEmbeddingSine(nn.Module):
    """
    正弦位置编码
    """
    
    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (B, C, H, W)
        
        返回:
            pos: (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # 创建网格
        y_embed = torch.arange(H, dtype=torch.float32, device=x.device)
        x_embed = torch.arange(W, dtype=torch.float32, device=x.device)
        
        # 归一化到[0, 1]
        y_embed = y_embed / (H - 1) if H > 1 else y_embed
        x_embed = x_embed / (W - 1) if W > 1 else x_embed
        
        # 扩展维度
        y_embed = y_embed.unsqueeze(1).repeat(1, W)  # (H, W)
        x_embed = x_embed.unsqueeze(0).repeat(H, 1)  # (H, W)
        
        # 创建位置编码
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        pos_x = x_embed[:, :, None] / dim_t  # (H, W, num_pos_feats)
        pos_y = y_embed[:, :, None] / dim_t  # (H, W, num_pos_feats)
        
        # 应用sin/cos
        pos_x = torch.stack([pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()], dim=3).flatten(2)
        pos_y = torch.stack([pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()], dim=3).flatten(2)
        
        # 连接x和y的位置编码
        pos = torch.cat([pos_y, pos_x], dim=2).permute(2, 0, 1)  # (C, H, W)
        
        # 扩展batch维度
        pos = pos.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, C, H, W)
        
        return pos


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 多头自注意力
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # 归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        src: torch.Tensor,
        pos_embed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        参数:
            src: (B, N, d_model)
            pos_embed: (B, N, d_model)
        
        返回:
            output: (B, N, d_model)
        """
        # 添加位置编码
        q = k = src + pos_embed if pos_embed is not None else src
        
        # 自注意力
        attn_out, _ = self.self_attn(q, k, src)
        src = self.norm1(src + self.dropout(attn_out))
        
        # 前馈网络
        ffn_out = self.ffn(src)
        src = self.norm2(src + ffn_out)
        
        return src


class HybridEncoder(nn.Module):
    """
    混合编码器
    
    结合FPN的多尺度特征和Transformer的全局建模能力
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_encoder_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_feature_levels: int = 4
    ):
        """
        参数:
            d_model: 模型维度
            num_encoder_layers: 编码器层数
            num_heads: 注意力头数
            dim_feedforward: FFN维度
            dropout: Dropout率
            num_feature_levels: 特征层级数
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_feature_levels = num_feature_levels
        
        # Transformer编码器层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_encoder_layers)
        ])
        
        # 位置编码
        self.pos_embed = PositionEmbeddingSine(num_pos_feats=d_model // 2)
        
        # 层级编码（用于区分不同层级的特征）
        self.level_embed = nn.Parameter(torch.randn(num_feature_levels, d_model))
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        前向传播
        
        参数:
            features: 多层级特征列表 [(B, C, H1, W1), (B, C, H2, W2), ...]
        
        返回:
            encoded_features: 编码后的特征列表
        """
        # 为每个层级添加位置编码和层级编码
        src_flatten = []
        pos_flatten = []
        spatial_shapes = []
        
        for i, feat in enumerate(features):
            B, C, H, W = feat.shape
            spatial_shapes.append((H, W))
            
            # 位置编码
            pos = self.pos_embed(feat)  # (B, C, H, W)
            
            # 展平
            feat_flat = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
            pos_flat = pos.flatten(2).transpose(1, 2)  # (B, H*W, C)
            
            # 添加层级编码
            level_emb = self.level_embed[i].view(1, 1, -1)  # (1, 1, C)
            feat_flat = feat_flat + level_emb
            
            src_flatten.append(feat_flat)
            pos_flatten.append(pos_flat)
        
        # 连接所有层级
        src = torch.cat(src_flatten, dim=1)  # (B, N_total, C)
        pos = torch.cat(pos_flatten, dim=1)  # (B, N_total, C)
        
        # Transformer编码
        for layer in self.encoder_layers:
            src = layer(src, pos)
        
        # 分割回各个层级
        encoded_features = []
        start_idx = 0
        for i, (H, W) in enumerate(spatial_shapes):
            num_tokens = H * W
            feat_tokens = src[:, start_idx:start_idx+num_tokens, :]
            feat = feat_tokens.transpose(1, 2).reshape(B, C, H, W)
            encoded_features.append(feat)
            start_idx += num_tokens
        
        return encoded_features


if __name__ == "__main__":
    print("=" * 70)
    print("测试混合编码器")
    print("=" * 70)
    
    # 创建编码器
    encoder = HybridEncoder(
        d_model=256,
        num_encoder_layers=6,
        num_heads=8,
        dim_feedforward=2048,
        num_feature_levels=4
    )
    
    # 测试数据
    batch_size = 2
    features = [
        torch.randn(batch_size, 256, 100, 100),
        torch.randn(batch_size, 256, 50, 50),
        torch.randn(batch_size, 256, 25, 25),
        torch.randn(batch_size, 256, 13, 13)
    ]
    
    # 前向传播
    encoded_features = encoder(features)
    
    print("\n输入特征:")
    for i, feat in enumerate(features):
        print(f"  层{i}: {feat.shape}")
    
    print("\n编码后特征:")
    for i, feat in enumerate(encoded_features):
        print(f"  层{i}: {feat.shape}")
    
    print("\n✅ 混合编码器测试完成！")

