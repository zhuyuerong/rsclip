# -*- coding: utf-8 -*-
"""
Adaptive Attention Fusion (AAF) Module

Fuses attention from multiple layers of SurgeryCLIP's dual-path (VV and original).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AAF(nn.Module):
    """
    Adaptive Attention Fusion
    
    Functionality:
    - Fuses VV and original path attentions from SurgeryCLIP's last 6 layers
    - Learns per-layer weights for both paths
    - Outputs fused attention for patch-to-patch propagation
    """
    
    def __init__(self, num_layers=6):
        """
        Args:
            num_layers: Number of GLAM layers (default 6, corresponding to SurgeryCLIP's last 6 layers)
        """
        super().__init__()
        self.num_layers = num_layers
        
        # Learnable weights for VV path's each layer
        self.vv_layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # Learnable weights for original path's each layer
        self.ori_layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # Mixing coefficient between VV path and original path
        self.alpha = nn.Parameter(torch.tensor(0.7))
        
    def forward(self, vv_attentions, ori_attentions):
        """
        Args:
            vv_attentions: List of [B, num_heads, 1+N², 1+N²] (VV path attention)
            ori_attentions: List of [B, num_heads, 1+N², 1+N²] (original path attention)
            
        Returns:
            attn_p2p: [B, N², N²] - patch-to-patch affinity matrix
        """
        # ===== Step 1: Fuse VV path's multi-layer attention =====
        vv_weights = F.softmax(self.vv_layer_weights, dim=0)  # [num_layers]
        vv_fused = 0
        
        for i, attn in enumerate(vv_attentions):
            # attn: [B, num_heads, 1+N², 1+N²]
            # Average across heads
            attn_avg = attn.mean(dim=1)  # [B, 1+N², 1+N²]
            # Weighted accumulation
            vv_fused = vv_fused + vv_weights[i] * attn_avg
        
        # ===== Step 2: Fuse original path's multi-layer attention =====
        ori_weights = F.softmax(self.ori_layer_weights, dim=0)
        ori_fused = 0
        
        for i, attn in enumerate(ori_attentions):
            attn_avg = attn.mean(dim=1)
            ori_fused = ori_fused + ori_weights[i] * attn_avg
        
        # ===== Step 3: Mix two paths =====
        alpha = torch.sigmoid(self.alpha)  # Constrain to [0, 1]
        attn_fused = alpha * vv_fused + (1 - alpha) * ori_fused
        # attn_fused: [B, 1+N², 1+N²]
        
        # ===== Step 4: Extract patch-to-patch attention =====
        # Position 0 is cls token, positions 1+ are patches
        attn_p2p = attn_fused[:, 1:, 1:]  # [B, N², N²]
        
        return attn_p2p





