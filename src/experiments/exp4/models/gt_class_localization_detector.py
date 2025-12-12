# -*- coding: utf-8 -*-
"""
GT类别定位检测器
用于解耦分类和定位问题，验证定位头的能力

核心特点：
1. 使用Surgery CLIP原始冻结权重
2. 假设分类是对的（使用GT类别文本）
3. 只训练检测头
4. CAM完全冻结，只作为输入特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.simple_surgery_cam import SimpleSurgeryCAM, create_simple_surgery_cam_model
from models.multi_layer_feature_extractor import MultiLayerFeatureExtractor
from models.image_encoder import SimpleImageEncoder
from models.multi_input_detection_head import MultiInputDetectionHead


class GTClassLocalizationDetector(nn.Module):
    """
    GT类别定位检测器
    
    架构:
    1. SurgeryCLIP（完全冻结）提取多层特征和生成CAM
    2. 编码原图
    3. 检测头（可训练）预测框坐标
    
    关键特点：
    - Surgery CLIP完全冻结（所有参数requires_grad=False）
    - 只使用GT类别文本生成CAM
    - 检测头只输出框坐标（不输出置信度）
    - 只训练检测头参数
    """
    
    def __init__(
        self, 
        surgery_clip_checkpoint, 
        num_classes=20,
        cam_resolution=7, 
        device='cuda'
    ):
        """
        Args:
            surgery_clip_checkpoint: SurgeryCLIP checkpoint路径
            num_classes: 类别数
            cam_resolution: CAM分辨率
            device: 设备
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.cam_resolution = cam_resolution
        self.device = device
        
        # ===== Surgery CLIP（完全冻结）=====
        print("Loading Surgery CLIP (frozen)...")
        self.simple_surgery_cam, _ = create_simple_surgery_cam_model(
            checkpoint_path=surgery_clip_checkpoint,
            device=device,
            unfreeze_cam_last_layer=False  # 完全冻结，不解冻任何层
        )
        
        # 冻结Surgery CLIP所有参数
        for param in self.simple_surgery_cam.parameters():
            param.requires_grad = False
        
        print("✅ Surgery CLIP完全冻结")
        
        # ===== 多层特征提取器（冻结，因为依赖Surgery CLIP）=====
        print("创建多层特征提取器（冻结）...")
        self.multi_layer_extractor = MultiLayerFeatureExtractor(self.simple_surgery_cam)
        # 多层特征提取器本身不包含参数，只是工具类
        # 注意：不需要CAM融合模块，因为CAM完全冻结，使用简单平均即可
        
        # ===== 原图编码器（可训练）=====
        print("创建原图编码器（可训练）...")
        self.image_encoder = SimpleImageEncoder(
            output_dim=128,
            output_size=cam_resolution
        )
        
        # ===== 检测头（可训练）=====
        print("创建检测头（可训练）...")
        self.detection_head = MultiInputDetectionHead(
            num_classes=num_classes,
            img_feat_dim=128,
            cam_dim=num_classes,
            layer_feat_dim=768,  # SurgeryCLIP的patch feature维度
            num_layers=3,
            hidden_dim=256,
            cam_resolution=cam_resolution
        )
        
        # 统计可训练参数
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        print(f"冻结参数: {total_params - trainable_params:,}")
        
        # 确保模型在正确的设备上
        self.image_encoder = self.image_encoder.to(device)
        self.detection_head = self.detection_head.to(device)
    
    def forward(self, images: torch.Tensor, text_queries: List[str] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            images: [B, 3, H, W]
            text_queries: List[str] 所有类别文本列表（用于生成CAM）
                         如果为None，使用所有DIOR类别
        
        Returns:
            Dict with:
                - pred_boxes: [B, C, H, W, 4] 预测框坐标
                - fused_cam: [B, C, H, W] 融合后的CAM（用于监控）
                - multi_layer_cams: List[Tensor[B, C, H, W]] 多层CAM（用于监控）
        """
        B = images.shape[0]
        
        # ===== Step 1: 提取多层特征和CAM =====
        # 使用所有类别生成CAM（后续损失函数会只使用GT类别通道）
        if text_queries is None:
            text_queries = [
                "airplane", "airport", "baseball field", "basketball court",
                "bridge", "chimney", "dam", "expressway service area",
                "expressway toll station", "golf course", "ground track field",
                "harbor", "overpass", "ship", "stadium", "storage tank",
                "tennis court", "train station", "vehicle", "wind mill"
            ]
        
        multi_features, multi_cams = self.multi_layer_extractor.extract_multi_layer_features_and_cams(
            images, text_queries
        )
        # multi_features: List[[B, N², D]] (3个)
        # multi_cams: List[[B, C, N, N]] (3个)，其中N可能不等于cam_resolution
        
        # 融合多层CAM（简单平均，因为不需要学习）
        # 需要上采样到cam_resolution
        fused_cam = sum(multi_cams) / len(multi_cams)  # [B, C, N, N]
        # 上采样到cam_resolution
        if fused_cam.shape[2] != self.cam_resolution or fused_cam.shape[3] != self.cam_resolution:
            fused_cam = F.interpolate(
                fused_cam, 
                size=(self.cam_resolution, self.cam_resolution),
                mode='bilinear', 
                align_corners=False
            )  # [B, C, H, W]
        
        # ===== Step 2: 编码原图 =====
        img_features = self.image_encoder(images)  # [B, 128, 7, 7]
        
        # ===== Step 3: 检测头预测 =====
        detection_outputs = self.detection_head(
            img_features, fused_cam, multi_features
        )
        # detection_outputs包含: boxes, confidences等
        
        return {
            'pred_boxes': detection_outputs['boxes'],  # [B, C, H, W, 4]
            'fused_cam': fused_cam,  # [B, C, H, W] 用于监控
            'multi_layer_cams': multi_cams,  # 用于监控
            'image_features': img_features,  # 用于监控
        }


def create_gt_class_localization_detector(
    surgery_clip_checkpoint, 
    num_classes=20,
    cam_resolution=7,
    device='cuda'
):
    """
    创建GT类别定位检测器
    
    Args:
        surgery_clip_checkpoint: SurgeryCLIP checkpoint路径
        num_classes: 类别数
        cam_resolution: CAM分辨率
        device: 设备
    
    Returns:
        GTClassLocalizationDetector实例
    """
    return GTClassLocalizationDetector(
        surgery_clip_checkpoint=surgery_clip_checkpoint,
        num_classes=num_classes,
        cam_resolution=cam_resolution,
        device=device
    )

