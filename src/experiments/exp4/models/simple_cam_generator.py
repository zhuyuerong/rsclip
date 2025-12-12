# -*- coding: utf-8 -*-
"""
简化的CAM生成器
直接使用SurgeryCLIP的热图生成逻辑，不使用p2p和AAF
"""

import torch
import torch.nn as nn
import math


class SimpleCAMGenerator(nn.Module):
    """
    简化的CAM生成器
    直接使用patch-text相似度，不使用p2p传播和AAF
    沿用SurgeryCLIP的热图生成逻辑
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, patch_features, text_features):
        """
        直接计算patch-text相似度作为CAM
        
        Args:
            patch_features: [B, N², D_img] - patch tokens (图像特征)
            text_features: [C, D_text] - text features (文本特征)
        
        Returns:
            cam: [B, C, N, N] - Class Activation Maps
        """
        B, N_sq, D_img = patch_features.shape
        C, D_text = text_features.shape
        N = int(math.sqrt(N_sq))
        
        # 归一化（沿用SurgeryCLIP的逻辑）
        patch_features = patch_features / (patch_features.norm(dim=-1, keepdim=True) + 1e-8)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
        
        # 处理维度不匹配
        # SurgeryCLIP中，图像特征和文本特征应该已经投影到相同的embed_dim
        # 但这里可能维度不匹配，需要检查
        if D_img != D_text:
            # 如果图像特征维度更大，使用投影层或截断
            # 先尝试截断图像特征到文本特征维度
            if D_img > D_text:
                patch_features = patch_features[:, :, :D_text]
                D_img = D_text
            else:
                # 如果文本特征维度更大，截断文本特征
                text_features = text_features[:, :D_img]
                D_text = D_img
        
        # 计算相似度: [B, N², D] @ [D, C] → [B, N², C]
        similarity = torch.matmul(patch_features, text_features.T)
        
        # 重塑为空间维度: [B, N², C] → [B, C, N²] → [B, C, N, N]
        cam = similarity.permute(0, 2, 1).reshape(B, C, N, N)
        
        # 可选：应用sigmoid（SurgeryCLIP中可能没有，但可以尝试）
        # cam = torch.sigmoid(cam)
        
        return cam

