# -*- coding: utf-8 -*-
"""
CAM Generator with Patch-to-Patch Propagation

Generates Class Activation Maps using:
1. Initial CAM from patch-text similarity
2. Patch-to-patch propagation using fused attention
"""

import torch
import torch.nn as nn
import math


class CAMGenerator(nn.Module):
    """
    CAM Generator
    
    Process:
    1. Initial CAM = patch_features @ text_features^T
    2. p2p propagation = attn_p2p @ CAM_init
    """
    
    def __init__(self, use_clip_projection=True):
        """
        Args:
            use_clip_projection: 是否使用CLIP的视觉投影层处理维度不匹配
        """
        super().__init__()
        self.use_clip_projection = use_clip_projection
        
    def forward(self, patch_features, text_features, attn_p2p, clip_visual_proj=None):
        """
        Args:
            patch_features: [B, N², D_img] - patch tokens (image dimension)
            text_features: [C, D_text] - text features (text dimension)
            attn_p2p: [B, N², N²] - patch affinity matrix from AAF
            clip_visual_proj: CLIP的视觉投影层权重 [D_img, D_text] (可选)
        
        Returns:
            cam: [B, C, N, N] - Class Activation Maps
        """
        B, N_sq, D_img = patch_features.shape
        C, D_text = text_features.shape
        N = int(math.sqrt(N_sq))
        
        # Handle dimension mismatch between image and text features
        if D_img != D_text:
            if self.use_clip_projection and clip_visual_proj is not None:
                # 使用CLIP的视觉投影层将图像特征投影到文本特征维度
                # patch_features: [B, N², D_img]
                # clip_visual_proj: [D_img, D_text]
                # 投影后: [B, N², D_text]
                patch_features = torch.matmul(patch_features, clip_visual_proj)
                D_img = D_text
            else:
                # 旧方法：使用零填充或截断（不推荐）
                if D_img > D_text:
                    # Pad text features with zeros
                    text_features_padded = torch.zeros(C, D_img, device=text_features.device, dtype=text_features.dtype)
                    text_features_padded[:, :D_text] = text_features
                    text_features = text_features_padded
                else:
                    # Truncate text features
                    text_features = text_features[:, :D_img]
        
        # ===== Step 1: Compute initial CAM (patch-text similarity) =====
        # [B, N², D] @ [D, C] → [B, N², C]
        similarity = torch.matmul(patch_features, text_features.T)
        
        # Reshape to 2D spatial: [B, C, N, N]
        cam_init = similarity.permute(0, 2, 1).reshape(B, C, N, N)
        cam_init = torch.sigmoid(cam_init)
        
        # ===== Step 2: Patch-to-Patch propagation =====
        # Flatten spatial dimensions
        cam_flat = cam_init.reshape(B, C, N_sq)  # [B, C, N²]
        
        # Apply affinity propagation
        # [B, N², N²] @ [B, N², C] → [B, N², C]
        cam_propagated = torch.bmm(
            attn_p2p,
            cam_flat.permute(0, 2, 1)  # [B, N², C]
        )
        
        # Reshape back to spatial
        # [B, N², C] → [B, C, N²] → [B, C, N, N]
        cam_final = cam_propagated.permute(0, 2, 1).reshape(B, C, N, N)
        
        return cam_final

