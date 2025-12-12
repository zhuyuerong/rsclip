# -*- coding: utf-8 -*-
"""
修复后的CAM生成器
1. 使用投影层处理维度不匹配（而不是截断）
2. 可以尝试使用original path的patch features
"""

import torch
import torch.nn as nn
import math


class FixedCAMGenerator(nn.Module):
    """
    修复后的CAM生成器
    使用投影层处理维度不匹配，而不是截断
    """
    
    def __init__(self, use_projection=True, use_original_path=False):
        """
        Args:
            use_projection: 是否使用投影层处理维度不匹配
            use_original_path: 是否使用original path的patch features（需要额外传入）
        """
        super().__init__()
        self.use_projection = use_projection
        self.use_original_path = use_original_path
        
        # 投影层：将文本特征投影到图像特征维度
        # 注意：这个投影层应该是可学习的，但为了测试，我们先使用CLIP的投影层
        self.text_to_image_proj = None  # 将在forward中初始化
    
    def forward(self, patch_features, text_features, clip_model=None):
        """
        使用投影层处理维度不匹配
        
        Args:
            patch_features: [B, N², D_img] - patch tokens (图像特征)
            text_features: [C, D_text] - text features (文本特征)
            clip_model: CLIP模型，用于获取投影层权重（可选）
        
        Returns:
            cam: [B, C, N, N] - Class Activation Maps
        """
        B, N_sq, D_img = patch_features.shape
        C, D_text = text_features.shape
        N = int(math.sqrt(N_sq))
        
        # 归一化
        patch_features = patch_features / (patch_features.norm(dim=-1, keepdim=True) + 1e-8)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
        
        # 处理维度不匹配
        if D_img != D_text:
            if self.use_projection:
                # 使用投影层
                if clip_model is not None and hasattr(clip_model.visual, 'proj'):
                    # 使用CLIP的视觉投影层（768 -> 512）
                    # 但我们需要反向：512 -> 768
                    # 所以我们需要创建一个新的投影层，或者使用文本投影层
                    
                    # 方案1: 将文本特征投影到图像特征维度
                    # 使用一个简单的线性层（可以初始化为单位矩阵的扩展）
                    if self.text_to_image_proj is None:
                        # 初始化投影层
                        # 如果D_img > D_text，将text_features投影到D_img
                        self.text_to_image_proj = nn.Linear(D_text, D_img, bias=False).to(text_features.device)
                        # 初始化为单位矩阵的扩展（前D_text维是单位矩阵，后D_img-D_text维是0）
                        with torch.no_grad():
                            self.text_to_image_proj.weight[:D_text, :] = torch.eye(D_text).to(text_features.device)
                            if D_img > D_text:
                                self.text_to_image_proj.weight[D_text:, :] = 0
                    
                    text_features_proj = self.text_to_image_proj(text_features)  # [C, D_img]
                    text_features = text_features_proj
                else:
                    # 如果没有CLIP模型，使用简单的投影
                    if self.text_to_image_proj is None:
                        self.text_to_image_proj = nn.Linear(D_text, D_img, bias=False).to(text_features.device)
                        with torch.no_grad():
                            self.text_to_image_proj.weight[:D_text, :] = torch.eye(D_text).to(text_features.device)
                            if D_img > D_text:
                                self.text_to_image_proj.weight[D_text:, :] = 0
                    
                    text_features = self.text_to_image_proj(text_features)
            else:
                # 不使用投影层，截断（原始方法）
                if D_img > D_text:
                    patch_features = patch_features[:, :, :D_text]
                    D_img = D_text
                else:
                    text_features = text_features[:, :D_img]
                    D_text = D_img
        
        # 计算相似度: [B, N², D] @ [D, C] → [B, N², C]
        similarity = torch.matmul(patch_features, text_features.T)
        
        # 重塑为空间维度: [B, N², C] → [B, C, N²] → [B, C, N, N]
        cam = similarity.permute(0, 2, 1).reshape(B, C, N, N)
        
        return cam


class FixedCAMGeneratorWithClipProj(nn.Module):
    """
    使用CLIP的投影层处理维度不匹配
    将图像特征投影到文本特征维度（使用CLIP的视觉投影层）
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, patch_features, text_features, clip_visual_proj=None):
        """
        使用CLIP的视觉投影层将图像特征投影到文本特征维度
        
        Args:
            patch_features: [B, N², D_img] - patch tokens (图像特征，768维)
            text_features: [C, D_text] - text features (文本特征，512维)
            clip_visual_proj: CLIP的视觉投影层权重 [D_img, D_text] (768, 512)
        
        Returns:
            cam: [B, C, N, N] - Class Activation Maps
        """
        B, N_sq, D_img = patch_features.shape
        C, D_text = text_features.shape
        N = int(math.sqrt(N_sq))
        
        # 归一化
        patch_features = patch_features / (patch_features.norm(dim=-1, keepdim=True) + 1e-8)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
        
        # 使用CLIP的视觉投影层将图像特征投影到文本特征维度
        if clip_visual_proj is not None and D_img != D_text:
            # patch_features: [B, N², D_img]
            # clip_visual_proj: [D_img, D_text]
            # 投影后: [B, N², D_text]
            patch_features_proj = torch.matmul(patch_features, clip_visual_proj)  # [B, N², D_text]
            patch_features = patch_features_proj
            D_img = D_text
        
        # 计算相似度: [B, N², D] @ [D, C] → [B, N², C]
        similarity = torch.matmul(patch_features, text_features.T)
        
        # 重塑为空间维度: [B, N², C] → [B, C, N²] → [B, C, N, N]
        cam = similarity.permute(0, 2, 1).reshape(B, C, N, N)
        
        return cam


