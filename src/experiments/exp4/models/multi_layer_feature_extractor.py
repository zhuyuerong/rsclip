# -*- coding: utf-8 -*-
"""
多层特征和CAM提取器
提取ViT最后3层的特征和CAM
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models.simple_surgery_cam import SimpleSurgeryCAM


class MultiLayerFeatureExtractor(nn.Module):
    """
    多层特征和CAM提取器
    
    功能:
    - 注册hook到ViT最后3层（layers [-3, -2, -1]）
    - 提取每层的patch特征
    - 为每层生成CAM
    """
    
    def __init__(self, simple_surgery_cam: SimpleSurgeryCAM):
        """
        Args:
            simple_surgery_cam: SimpleSurgeryCAM模型实例
        """
        super().__init__()
        self.simple_surgery_cam = simple_surgery_cam
        self.clip = simple_surgery_cam.clip
        self.cam_generator = simple_surgery_cam.cam_generator
        
        # 确定要提取的层（最后3层）
        self.layers_to_extract = [-3, -2, -1]  # 第10、11、12层
    
    def extract_multi_layer_features_and_cams(
        self, 
        images: torch.Tensor, 
        text_queries: List[str]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        提取多层特征和CAM
        
        Args:
            images: [B, 3, H, W]
            text_queries: List[str] 类别名称
        
        Returns:
            multi_layer_features: List[Tensor[B, N², D]] (3个)
            multi_layer_cams: List[Tensor[B, C, H, W]] (3个)
        """
        B = images.shape[0]
        device = images.device
        
        # ===== Step 1: 文本编码 =====
        with torch.no_grad():
            # Import tokenize (使用与simple_surgery_cam.py相同的方式)
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from src.competitors.clip_methods.surgeryclip.clip import tokenize
            text_tokens = tokenize(text_queries).to(device)
            text_features = self.clip.encode_text(text_tokens)
            text_features = text_features.float()  # [C, D]
        
        # ===== Step 2: 注册hook提取中间层输出 =====
        layer_outputs = []
        
        def hook_fn(module, input, output):
            """Hook函数：捕获层输出"""
            # ResidualAttentionBlock的forward可能返回list [x, x_ori] 或 tensor x
            # 根据clip_surgery_model.py，forward可能返回list
            if isinstance(output, (list, tuple)):
                # 如果是list/tuple，取第一个元素（主输出）
                out_tensor = output[0]
            else:
                # 直接是tensor
                out_tensor = output
            
            # 确保是tensor并clone
            if isinstance(out_tensor, torch.Tensor):
                layer_outputs.append(out_tensor.clone())
            else:
                # 如果不是tensor，尝试转换
                layer_outputs.append(torch.tensor(out_tensor) if out_tensor is not None else None)
        
        # 获取ViT的transformer blocks
        if not hasattr(self.clip.visual, 'transformer'):
            raise AttributeError("CLIP visual encoder does not have transformer attribute")
        
        transformer_blocks = self.clip.visual.transformer.resblocks
        num_layers = len(transformer_blocks)
        
        # 确定要提取的层索引（处理负数索引）
        layer_indices = []
        for idx in self.layers_to_extract:
            if idx < 0:
                layer_indices.append(num_layers + idx)
            else:
                layer_indices.append(idx)
        
        # 注册hooks
        hooks = []
        for layer_idx in layer_indices:
            if layer_idx >= num_layers:
                raise ValueError(f"Layer index {layer_idx} out of range (total: {num_layers})")
            hook = transformer_blocks[layer_idx].register_forward_hook(hook_fn)
            hooks.append(hook)
        
        # ===== Step 3: Forward pass触发hooks =====
        with torch.no_grad():
            if hasattr(self.clip.visual, 'encode_image_with_all_tokens'):
                image_features_all = self.clip.visual.encode_image_with_all_tokens(images)
                # image_features_all: [B, 1+N², D]
            else:
                raise NotImplementedError("encode_image_with_all_tokens not available")
        
        # 移除hooks
        for hook in hooks:
            hook.remove()
        
        # ===== Step 4: 提取多层特征和生成CAM =====
        multi_layer_features = []
        multi_layer_cams = []
        
        # 获取CLIP的视觉投影层
        clip_visual_proj = None
        if hasattr(self.clip.visual, 'proj'):
            clip_visual_proj = self.clip.visual.proj.data  # [D_img, D_text]
        
        for layer_output in layer_outputs:
            # 检查layer_output是否有效
            if layer_output is None:
                raise ValueError("Layer output is None, hook may have failed")
            
            # 确保是tensor
            if not isinstance(layer_output, torch.Tensor):
                raise TypeError(f"Layer output is not a tensor: {type(layer_output)}")
            
            # 提取patch tokens（去掉cls token）
            # layer_output应该是 [B, 1+N², D] 或 [1+N², B, D] (transposed)
            if layer_output.dim() == 3:
                # 检查是否是transposed格式 [seq_len, B, D]
                if layer_output.shape[0] > layer_output.shape[1]:
                    # 可能是transposed，需要转置回来
                    layer_output = layer_output.transpose(0, 1)  # [B, seq_len, D]
            
            patch_features = layer_output[:, 1:, :]  # [B, N², D]
            
            # 允许梯度流到CAM生成器
            patch_features = patch_features.detach().requires_grad_(True)
            
            # 为每层生成CAM
            cam_layer = self.cam_generator(
                patch_features,
                text_features,
                clip_visual_proj=clip_visual_proj
            )
            # cam_layer: [B, C, N, N]
            
            multi_layer_features.append(patch_features)
            multi_layer_cams.append(cam_layer)
        
        # 注意：multi_layer_features是List[[B, N², D]]，需要转换为空间格式用于检测头
        # 检测头会处理这个转换
        return multi_layer_features, multi_layer_cams
    
    def forward(self, images: torch.Tensor, text_queries: List[str]) -> Dict:
        """
        Forward pass（兼容接口）
        
        Returns:
            Dict with:
                - multi_layer_features: List[Tensor[B, N², D]]
                - multi_layer_cams: List[Tensor[B, C, H, W]]
        """
        multi_features, multi_cams = self.extract_multi_layer_features_and_cams(
            images, text_queries
        )
        
        return {
            'multi_layer_features': multi_features,
            'multi_layer_cams': multi_cams
        }

