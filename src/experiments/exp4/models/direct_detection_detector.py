# -*- coding: utf-8 -*-
"""
直接检测器
使用CAM + 图像特征直接预测框，无需阈值检测
"""

import torch
import torch.nn as nn
from typing import List, Dict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.simple_surgery_cam import SimpleSurgeryCAM, create_simple_surgery_cam_model
from models.direct_detection_head import DirectDetectionHead


class DirectDetectionDetector(nn.Module):
    """
    直接检测器
    
    架构:
    1. SurgeryCLIP提取图像特征和生成CAM
    2. DirectDetectionHead融合CAM和特征，直接预测框
    3. 无需阈值检测，端到端训练
    """
    
    def __init__(self, surgery_clip_checkpoint, num_classes=20,
                 cam_resolution=7, upsample_cam=False, device='cuda',
                 unfreeze_cam_last_layer=True,
                 use_image_features=True):
        """
        Args:
            surgery_clip_checkpoint: SurgeryCLIP checkpoint路径
            num_classes: 类别数
            cam_resolution: CAM分辨率
            upsample_cam: 是否上采样CAM
            device: 设备
            unfreeze_cam_last_layer: 是否解冻CAM生成器的最后一层
            use_image_features: 是否使用图像特征
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.cam_resolution = cam_resolution
        self.upsample_cam = upsample_cam
        self.device = device
        self.use_image_features = use_image_features
        
        # ===== 冻结部分 =====
        # Load SimpleSurgeryCAM model
        print("Loading SimpleSurgeryCAM...")
        self.simple_surgery_cam, _ = create_simple_surgery_cam_model(
            checkpoint_path=surgery_clip_checkpoint,
            device=device,
            unfreeze_cam_last_layer=unfreeze_cam_last_layer
        )
        
        # 冻结SurgeryCLIP部分
        for param in self.simple_surgery_cam.clip.parameters():
            param.requires_grad = False
        
        # CAM生成器：如果unfreeze_cam_last_layer=True，最后一层可训练
        if unfreeze_cam_last_layer:
            if hasattr(self.simple_surgery_cam.cam_generator, 'learnable_proj'):
                for param in self.simple_surgery_cam.cam_generator.learnable_proj.parameters():
                    param.requires_grad = True
                print(f"✅ CAM生成器的可学习投影层已解冻")
        
        # ===== 可训练部分 =====
        # 直接检测头：融合CAM和图像特征，直接预测框
        print("创建直接检测头（CAM + 图像特征 → 框）...")
        self.detection_head = DirectDetectionHead(
            num_classes=num_classes,
            feature_dim=768,  # SurgeryCLIP的patch feature维度
            hidden_dim=256,
            cam_resolution=cam_resolution,
            use_image_features=use_image_features
        )
        
        # 统计可训练参数
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")
        
        # 确保模型在正确的设备上
        self.detection_head = self.detection_head.to(device)
    
    def forward(self, images: torch.Tensor, text_queries: List[str]) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            images: [B, 3, H, W]
            text_queries: List[str]
        
        Returns:
            Dict with:
                - cam: [B, C, H_cam, W_cam]
                - boxes: [B, C, H_cam, W_cam, 4] 预测框
                - confidences: [B, C, H_cam, W_cam] 置信度
                - image_features: [B, N², D] 图像特征（如果使用）
        """
        B = images.shape[0]
        
        # ===== Step 1: 生成CAM和提取图像特征 =====
        cam, aux = self.simple_surgery_cam(images, text_queries)
        # cam: [B, C, N, N]
        
        # 获取图像特征（patch features）
        image_features = None
        if self.use_image_features and 'patch_features' in aux:
            image_features = aux['patch_features']  # [B, N², D]
        
        # ===== Step 2: 直接检测头预测框 =====
        detection_outputs = self.detection_head(cam, image_features)
        # detection_outputs包含: boxes, confidences等
        
        return {
            'cam': cam,
            'pred_boxes': detection_outputs['boxes'],
            'confidences': detection_outputs['confidences'],
            'raw_confidences': detection_outputs.get('raw_confidences'),
            'image_features': image_features,
            **detection_outputs
        }
    
    def inference(self, images: torch.Tensor, text_queries: List[str],
                 conf_threshold: float = 0.3, nms_threshold: float = 0.5,
                 topk: int = 100) -> Dict:
        """
        推理接口
        
        Args:
            images: [B, 3, H, W]
            text_queries: List[str]
            conf_threshold: 置信度阈值
            nms_threshold: NMS阈值
            topk: 保留的top-k检测
        
        Returns:
            detections: List[List[dict]] 每个图像的检测结果
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(images, text_queries)
            cam = outputs['cam']
            image_features = outputs.get('image_features')
            
            # 使用检测头的推理接口
            detections = self.detection_head.inference(
                cam, image_features,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold,
                topk=topk
            )
            
            # 添加类别名称
            for img_detections in detections:
                for det in img_detections:
                    class_idx = det['class']
                    if class_idx < len(text_queries):
                        det['class_name'] = text_queries[class_idx]
                    else:
                        det['class_name'] = f"class_{class_idx}"
            
            return detections


def create_direct_detection_detector(surgery_clip_checkpoint, num_classes=20,
                                    cam_resolution=7, upsample_cam=False,
                                    device='cuda', unfreeze_cam_last_layer=True,
                                    use_image_features=True):
    """
    创建直接检测器
    
    Args:
        surgery_clip_checkpoint: SurgeryCLIP checkpoint路径
        num_classes: 类别数
        cam_resolution: CAM分辨率
        upsample_cam: 是否上采样CAM
        device: 设备
        unfreeze_cam_last_layer: 是否解冻CAM生成器的最后一层
        use_image_features: 是否使用图像特征
    
    Returns:
        DirectDetectionDetector实例
    """
    return DirectDetectionDetector(
        surgery_clip_checkpoint=surgery_clip_checkpoint,
        num_classes=num_classes,
        cam_resolution=cam_resolution,
        upsample_cam=upsample_cam,
        device=device,
        unfreeze_cam_last_layer=unfreeze_cam_last_layer,
        use_image_features=use_image_features
    )


