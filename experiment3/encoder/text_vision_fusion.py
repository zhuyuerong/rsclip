#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本-视觉特征融合模块

功能：
1. 使用视觉特征增强文本特征（Vision-Augmented Text）
2. 使用文本特征引导视觉特征
3. 多层级特征融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class TextVisionFusion(nn.Module):
    """
    文本-视觉融合模块
    
    基于OVA-DETR的设计，支持：
    1. 视觉增强文本 (VAT)
    2. 文本引导视觉
    """
    
    def __init__(
        self,
        text_dim: int = 1024,
        vision_dim: int = 256,
        output_dim: int = 256,
        num_levels: int = 4,
        enable_vat: bool = True
    ):
        """
        参数:
            text_dim: 文本特征维度
            vision_dim: 视觉特征维度
            output_dim: 输出维度
            num_levels: 特征层级数
            enable_vat: 是否启用视觉增强文本
        """
        super().__init__()
        
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.output_dim = output_dim
        self.num_levels = num_levels
        self.enable_vat = enable_vat
        
        # 文本特征投影
        self.text_proj = nn.Linear(text_dim, output_dim)
        
        # 视觉特征投影
        self.vision_projs = nn.ModuleList([
            nn.Conv2d(vision_dim, output_dim, kernel_size=1)
            for _ in range(num_levels)
        ])
        
        if enable_vat:
            # 视觉增强文本模块
            self.vat_attn = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            self.vat_norm = nn.LayerNorm(output_dim)
            self.vat_ffn = nn.Sequential(
                nn.Linear(output_dim, output_dim * 4),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(output_dim * 4, output_dim),
                nn.Dropout(0.1)
            )
            self.vat_ffn_norm = nn.LayerNorm(output_dim)
        
        # 文本引导视觉的注意力模块
        self.tvg_attns = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(num_levels)
        ])
        
        self.tvg_norms = nn.ModuleList([
            nn.LayerNorm(output_dim)
            for _ in range(num_levels)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def vision_augmented_text(
        self,
        text_features: torch.Tensor,
        vision_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        视觉增强文本 (VAT)
        
        使用多层级视觉特征增强文本特征
        
        参数:
            text_features: (B, num_classes, text_dim)
            vision_features: 多层级视觉特征列表
        
        返回:
            enhanced_text: (B, num_classes, output_dim)
        """
        # 投影文本特征
        text_feat = self.text_proj(text_features)  # (B, num_classes, output_dim)
        
        # 聚合所有层级的视觉特征
        vision_tokens = []
        for i, vision_feat in enumerate(vision_features):
            # (B, vision_dim, H, W) -> (B, output_dim, H, W)
            proj_feat = self.vision_projs[i](vision_feat)
            B, C, H, W = proj_feat.shape
            # (B, output_dim, H, W) -> (B, H*W, output_dim)
            tokens = proj_feat.flatten(2).transpose(1, 2)
            vision_tokens.append(tokens)
        
        # 连接所有层级的视觉token
        vision_tokens = torch.cat(vision_tokens, dim=1)  # (B, N_total, output_dim)
        
        # 交叉注意力：文本作为query，视觉作为key/value
        attn_out, _ = self.vat_attn(
            query=text_feat,
            key=vision_tokens,
            value=vision_tokens
        )
        
        # 残差连接 + 归一化
        text_feat = self.vat_norm(text_feat + attn_out)
        
        # FFN
        ffn_out = self.vat_ffn(text_feat)
        text_feat = self.vat_ffn_norm(text_feat + ffn_out)
        
        return text_feat
    
    def text_guided_vision(
        self,
        text_features: torch.Tensor,
        vision_features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        文本引导视觉
        
        使用文本特征增强每个层级的视觉特征
        
        参数:
            text_features: (B, num_classes, output_dim)
            vision_features: 多层级视觉特征列表
        
        返回:
            enhanced_vision: 增强后的多层级视觉特征列表
        """
        enhanced_features = []
        
        for i, vision_feat in enumerate(vision_features):
            # 投影视觉特征
            proj_feat = self.vision_projs[i](vision_feat)  # (B, output_dim, H, W)
            B, C, H, W = proj_feat.shape
            
            # 展平为序列
            vision_seq = proj_feat.flatten(2).transpose(1, 2)  # (B, H*W, output_dim)
            
            # 交叉注意力：视觉作为query，文本作为key/value
            attn_out, _ = self.tvg_attns[i](
                query=vision_seq,
                key=text_features,
                value=text_features
            )
            
            # 残差连接 + 归一化
            enhanced_seq = self.tvg_norms[i](vision_seq + attn_out)
            
            # 重塑为特征图
            enhanced_feat = enhanced_seq.transpose(1, 2).reshape(B, C, H, W)
            enhanced_features.append(enhanced_feat)
        
        return enhanced_features
    
    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: List[torch.Tensor]
    ) -> tuple:
        """
        前向传播
        
        参数:
            text_features: (B, num_classes, text_dim) 或 (num_classes, text_dim)
            vision_features: 多层级视觉特征列表
        
        返回:
            enhanced_text: 增强后的文本特征
            enhanced_vision: 增强后的视觉特征列表
        """
        # 处理维度
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(0)  # (1, num_classes, text_dim)
        
        # 视觉增强文本
        if self.enable_vat:
            enhanced_text = self.vision_augmented_text(text_features, vision_features)
        else:
            enhanced_text = self.text_proj(text_features)
        
        # 文本引导视觉
        enhanced_vision = self.text_guided_vision(enhanced_text, vision_features)
        
        return enhanced_text, enhanced_vision


if __name__ == "__main__":
    print("=" * 70)
    print("测试文本-视觉融合模块")
    print("=" * 70)
    
    # 创建融合模块
    fusion = TextVisionFusion(
        text_dim=1024,
        vision_dim=256,
        output_dim=256,
        num_levels=4,
        enable_vat=True
    )
    
    # 测试数据
    batch_size = 2
    num_classes = 20
    text_features = torch.randn(batch_size, num_classes, 1024)
    vision_features = [
        torch.randn(batch_size, 256, 100, 100),
        torch.randn(batch_size, 256, 50, 50),
        torch.randn(batch_size, 256, 25, 25),
        torch.randn(batch_size, 256, 13, 13)
    ]
    
    # 前向传播
    enhanced_text, enhanced_vision = fusion(text_features, vision_features)
    
    print(f"\n文本特征: {text_features.shape} -> {enhanced_text.shape}")
    print("\n视觉特征:")
    for i, (orig, enh) in enumerate(zip(vision_features, enhanced_vision)):
        print(f"  层{i}: {orig.shape} -> {enh.shape}")
    
    print("\n✅ 文本-视觉融合模块测试完成！")

