# -*- coding: utf-8 -*-
"""
VV自注意力机制（双路径设计）
基于CLIP Surgery论文实现

核心思想：
- 路径1（原始）: Attention(Q, K, V) = softmax(QK^T / √d) V
- 路径2（VV）:   Attention(V, V, V) = softmax(VV^T / √d) V

融合策略:
- CLS token: 使用原始路径（保持全局语义）
- Image tokens: 使用VV路径（增强局部特征）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VVAttention(nn.Module):
    """
    VV自注意力机制（双路径设计）
    
    同时维护两个注意力路径：
    1. 原始QK路径：保持全局语义理解
    2. VV路径：基于value相似度，增强局部特征一致性
    """
    
    def __init__(self, dim, num_heads=8, scale_multiplier=1.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        # 缩放因子
        self.scale = (self.head_dim ** -0.5)
        self.scale_vv = self.scale * scale_multiplier
        
        # QKV投影层（从原始attention复制权重）
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        # 输出投影层
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.0)  # 默认不dropout（用于inference）
        
    def forward(self, query, key=None, value=None, need_weights=False, attn_mask=None):
        """
        兼容CLIP的MultiheadAttention接口
        
        Args:
            query: [N, B, D] (CLIP格式，序列优先)，或者作为单个输入x
            key: 未使用（为了兼容性）
            value: 未使用（为了兼容性）
            need_weights: 是否返回注意力权重
            attn_mask: 未使用（保持接口兼容性）
        
        Returns:
            out: [N, B, D] 或 (out, attn_weights) if need_weights=True
        """
        x = query  # 使用query作为输入
        N, B, D = x.shape
        x_batch = x.permute(1, 0, 2)  # [B, N, D]
        
        # 确保dtype匹配
        if x_batch.dtype != self.qkv.weight.dtype:
            x_batch = x_batch.to(self.qkv.weight.dtype)
        
        # 计算Q, K, V
        qkv = self.qkv(x_batch).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个都是 [B, num_heads, N, head_dim]
        
        # ========== 路径1: 标准QK自注意力 ==========
        attn_ori = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn_ori = attn_ori.softmax(dim=-1)
        attn_ori = self.dropout(attn_ori)
        out_ori = attn_ori @ v  # [B, num_heads, N, head_dim]
        
        # ========== 路径2: VV自注意力 ==========
        # 使用V替换Q和K
        q_vv = v.clone()
        k_vv = v.clone()
        
        # L2归一化（提高稳定性，避免数值问题）
        q_vv = F.normalize(q_vv, p=2, dim=-1, eps=1e-6)
        k_vv = F.normalize(k_vv, p=2, dim=-1, eps=1e-6)
        
        # 计算VV注意力：Attention(V, V, V) = softmax(VV^T / scale) V
        attn_vv = (q_vv @ k_vv.transpose(-2, -1)) * self.scale_vv  # [B, num_heads, N, N]
        attn_vv = attn_vv.softmax(dim=-1)
        attn_vv = self.dropout(attn_vv)
        out_vv = attn_vv @ v  # [B, num_heads, N, head_dim]
        
        # ========== 融合策略 ==========
        # CLS token使用原始路径（保持全局语义）
        # Image patches使用VV路径（增强局部特征）
        out = torch.zeros_like(out_ori)
        if N > 0:  # 确保有CLS token
            out[:, :, 0:1, :] = out_ori[:, :, 0:1, :]  # CLS token (索引0)
        if N > 1:  # 确保有patches
            out[:, :, 1:, :] = out_vv[:, :, 1:, :]  # Image tokens (索引1:)
        
        # 计算融合后的注意力权重（用于热图生成）
        attn_mixed = torch.zeros_like(attn_ori)
        if N > 0:
            attn_mixed[:, :, 0:1, :] = attn_ori[:, :, 0:1, :]  # CLS使用QK
        if N > 1:
            attn_mixed[:, :, 1:, :] = attn_vv[:, :, 1:, :]  # patches使用VV
        
        # Reshape并投影
        out = out.transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        out = self.proj(out)
        
        # 转换回CLIP格式 [N, B, D]
        out = out.permute(1, 0, 2)
        
        if need_weights:
            # 返回三种注意力权重用于热图生成
            attn_weights = {
                'attn_qk': attn_ori.mean(dim=1),     # 原始QK路径 [B, N, N]
                'attn_vv': attn_vv.mean(dim=1),      # VV路径 [B, N, N]
                'attn_mixed': attn_mixed.mean(dim=1) # 混合路径 [B, N, N]
            }
            return out, attn_weights
        else:
            return out
    
    def extra_repr(self):
        return f'dim={self.dim}, num_heads={self.num_heads}, scale_vv={self.scale_vv:.4f}'

