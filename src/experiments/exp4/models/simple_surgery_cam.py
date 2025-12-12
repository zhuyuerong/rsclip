# -*- coding: utf-8 -*-
"""
简化的SurgeryCAM模型
- 使用单纯的SurgeryCLIP
- 移除p2p传播和AAF
- CAM生成器可训练（解冻一层）
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


class SimpleCAMGenerator(nn.Module):
    """
    简化的CAM生成器
    - 不使用p2p传播
    - 直接计算patch-text相似度
    - 可训练（解冻最后一层）
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
        
        # 可选的投影层（如果需要微调）
        if unfreeze_last_layer:
            # 创建一个可学习的投影层，初始化为单位矩阵
            self.learnable_proj = nn.Linear(512, 512, bias=False)
            # 初始化为单位矩阵
            nn.init.eye_(self.learnable_proj.weight)
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
        import math
        
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
        
        # 可选：应用可学习的投影层（如果启用）
        if self.learnable_proj is not None:
            # 对text_features应用可学习投影
            text_features = self.learnable_proj(text_features)  # [C, D_text]
            # 重新归一化
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
        
        # 计算相似度: [B, N², D] @ [D, C] → [B, N², C]
        similarity = torch.matmul(patch_features, text_features.T)
        
        # 重塑为空间维度: [B, N², C] → [B, C, N²] → [B, C, N, N]
        cam = similarity.permute(0, 2, 1).reshape(B, C, N, N)
        
        # 可选：应用sigmoid（根据SurgeryCLIP的习惯，可能不需要）
        # cam = torch.sigmoid(cam)
        
        return cam


class SimpleSurgeryCAM(nn.Module):
    """
    简化的SurgeryCAM模型
    - 使用SurgeryCLIP提取特征
    - 不使用AAF和p2p传播
    - CAM生成器可训练
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
        
        # 简化的CAM生成器（可训练）
        self.cam_generator = SimpleCAMGenerator(
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


def create_simple_surgery_cam_model(checkpoint_path, device='cuda', unfreeze_cam_last_layer=False):
    """
    创建简化的SurgeryCAM模型
    
    Args:
        checkpoint_path: SurgeryCLIP checkpoint路径
        device: 设备
        unfreeze_cam_last_layer: 是否解冻CAM生成器的最后一层
    
    Returns:
        SimpleSurgeryCAM模型实例
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
    
    # Create SimpleSurgeryCAM
    model = SimpleSurgeryCAM(surgery_clip, unfreeze_cam_last_layer=unfreeze_cam_last_layer)
    model = model.to(device)
    
    return model, preprocess


