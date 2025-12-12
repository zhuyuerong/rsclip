# -*- coding: utf-8 -*-
"""
增强的SimpleSurgeryCAM模型 - 实验3.2
使用增强的CAM生成器（多层MLP）
"""

import torch
import torch.nn as nn
import sys
import os
from typing import List, Tuple, Dict
from pathlib import Path

# Add path to surgeryclip
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.competitors.clip_methods.surgeryclip.clip import tokenize
from models.enhanced_cam_generator import EnhancedCAMGenerator


class EnhancedSimpleSurgeryCAM(nn.Module):
    """
    增强的SimpleSurgeryCAM模型
    - 使用SurgeryCLIP提取特征
    - 不使用AAF和p2p传播
    - 使用增强的CAM生成器（多层MLP）
    """
    
    def __init__(self, surgery_clip_model, unfreeze_cam_last_layer=False):
        """
        Args:
            surgery_clip_model: 预加载的SurgeryCLIP模型
            unfreeze_cam_last_layer: 是否解冻CAM生成器的最后一层
        """
        super().__init__()
        
        # SurgeryCLIP模型（冻结）
        self.clip = surgery_clip_model
        for param in self.clip.parameters():
            param.requires_grad = False
        
        # 增强的CAM生成器（可训练）
        self.cam_generator = EnhancedCAMGenerator(
            use_clip_projection=True,
            unfreeze_last_layer=unfreeze_cam_last_layer
        )
        
        # 如果解冻最后一层，确保参数可训练
        if unfreeze_cam_last_layer:
            for param in self.cam_generator.learnable_proj.parameters():
                param.requires_grad = True
    
    def forward(self, images, text_queries):
        """
        Args:
            images: [B, 3, H, W]
            text_queries: List[str] - class names
        
        Returns:
            cam: [B, C, N, N]
            aux_outputs: dict
        """
        # ===== Step 1: Encode text =====
        with torch.no_grad():
            text_tokens = tokenize(text_queries).to(images.device)
            text_features = self.clip.encode_text(text_tokens)
            text_features = text_features.float()  # [C, D]
        
        # ===== Step 2: Encode image =====
        with torch.no_grad():
            if hasattr(self.clip.visual, 'encode_image_with_all_tokens'):
                image_features_all = self.clip.visual.encode_image_with_all_tokens(images)
                # image_features_all: [B, 1+N², D]
            else:
                raise NotImplementedError("encode_image_with_all_tokens not available")
        
        # Extract patch tokens (remove cls token)
        patch_features = image_features_all[:, 1:, :]  # [B, N², D]
        # 允许梯度流到CAM生成器
        patch_features = patch_features.detach().requires_grad_(True)
        
        # ===== Step 3: Generate CAM (不使用p2p和AAF) =====
        # 获取CLIP的视觉投影层
        clip_visual_proj = None
        if hasattr(self.clip.visual, 'proj'):
            clip_visual_proj = self.clip.visual.proj.data  # [D_img, D_text]
        
        cam = self.cam_generator(
            patch_features,
            text_features,
            clip_visual_proj=clip_visual_proj
        )
        # cam: [B, C, N, N]
        
        # Auxiliary outputs
        aux_outputs = {
            'patch_features': patch_features,
            'text_features': text_features,
        }
        
        return cam, aux_outputs


def create_enhanced_simple_surgery_cam_model(checkpoint_path, device='cuda', unfreeze_cam_last_layer=False):
    """
    创建增强的SimpleSurgeryCAM模型
    
    Args:
        checkpoint_path: SurgeryCLIP checkpoint路径
        device: 设备
        unfreeze_cam_last_layer: 是否解冻CAM生成器的最后一层
    
    Returns:
        EnhancedSimpleSurgeryCAM模型实例
    """
    # Import build_model
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.competitors.clip_methods.surgeryclip import build_model
    
    # Load SurgeryCLIP model
    surgery_clip, preprocess = build_model(
        model_name='surgeryclip',
        checkpoint_path=checkpoint_path,
        device=device
    )
    
    # Create EnhancedSimpleSurgeryCAM
    model = EnhancedSimpleSurgeryCAM(surgery_clip, unfreeze_cam_last_layer=unfreeze_cam_last_layer)
    model = model.to(device)
    
    return model, preprocess


