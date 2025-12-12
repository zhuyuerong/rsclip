# -*- coding: utf-8 -*-
"""
增强的CAM生成器 - 实验3.2
使用多层MLP替代单层投影，增加学习能力
"""

import torch
import torch.nn as nn
import math


class EnhancedCAMGenerator(nn.Module):
    """
    增强的CAM生成器
    - 使用多层MLP替代单层投影
    - 增加学习能力
    """
    
    def __init__(self, use_clip_projection=True, unfreeze_last_layer=False):
        """
        Args:
            use_clip_projection: 是否使用CLIP的视觉投影层
            unfreeze_last_layer: 是否解冻最后一层（用于微调）
        """
        super().__init__()
        self.use_clip_projection = use_clip_projection
        self.unfreeze_last_layer = unfreeze_last_layer
        
        # 增强的投影层：多层MLP
        if unfreeze_last_layer:
            # 使用多层MLP替代单层
            self.learnable_proj = nn.Sequential(
                nn.Linear(512, 512, bias=False),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 512, bias=False),
                nn.LayerNorm(512)
            )
            
            # 初始化：第一层初始化为单位矩阵，第二层初始化为小的随机值
            with torch.no_grad():
                # 第一层：接近单位矩阵
                nn.init.eye_(self.learnable_proj[0].weight)
                # 第二层：小的随机值
                nn.init.normal_(self.learnable_proj[3].weight, std=0.01)
        else:
            self.learnable_proj = None
    
    def forward(self, patch_features, text_features, clip_visual_proj=None):
        """
        直接计算patch-text相似度作为CAM（不使用p2p传播）
        
        Args:
            patch_features: [B, N², D_img] - patch tokens
            text_features: [C, D_text] - text features
            clip_visual_proj: CLIP的视觉投影层权重 [D_img, D_text]
        
        Returns:
            cam: [B, C, N, N] - Class Activation Maps
        """
        B, N_sq, D_img = patch_features.shape
        C, D_text = text_features.shape
        N = int(math.sqrt(N_sq))
        
        # 归一化
        patch_features = patch_features / (patch_features.norm(dim=-1, keepdim=True) + 1e-8)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
        
        # 处理维度不匹配：使用CLIP的视觉投影层
        if D_img != D_text:
            if self.use_clip_projection and clip_visual_proj is not None:
                # 使用CLIP的视觉投影层将图像特征投影到文本特征维度
                patch_features = torch.matmul(patch_features, clip_visual_proj)  # [B, N², D_text]
                D_img = D_text
            else:
                # 降级方案：截断
                if D_img > D_text:
                    patch_features = patch_features[:, :, :D_text]
                else:
                    text_features = text_features[:, :D_img]
        
        # 可选：应用增强的可学习投影层（如果启用）
        if self.learnable_proj is not None:
            # 对text_features应用多层MLP投影
            text_features = self.learnable_proj(text_features)  # [C, D_text]
            # 重新归一化
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
        
        # 计算相似度: [B, N², D] @ [D, C] → [B, N², C]
        similarity = torch.matmul(patch_features, text_features.T)
        
        # 重塑为空间维度: [B, N², C] → [B, C, N²] → [B, C, N, N]
        cam = similarity.permute(0, 2, 1).reshape(B, C, N, N)
        
        return cam


