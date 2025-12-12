# -*- coding: utf-8 -*-
"""
改进的直接检测器
整合原图+多层特征+多层CAM的完整架构
"""

import torch
import torch.nn as nn
from typing import List, Dict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.simple_surgery_cam import SimpleSurgeryCAM, create_simple_surgery_cam_model
from models.multi_layer_feature_extractor import MultiLayerFeatureExtractor
from models.multi_layer_cam_fusion import MultiLayerCAMFusion
from models.image_encoder import SimpleImageEncoder
from models.multi_input_detection_head import MultiInputDetectionHead


class ImprovedDirectDetectionDetector(nn.Module):
    """
    改进的直接检测器
    
    架构:
    1. SurgeryCLIP提取多层特征和生成多层CAM
    2. 融合多层CAM
    3. 编码原图
    4. 多输入检测头预测框
    """
    
    def __init__(
        self, 
        surgery_clip_checkpoint, 
        num_classes=20,
        cam_resolution=7, 
        device='cuda',
        unfreeze_cam_last_layer=True
    ):
        """
        Args:
            surgery_clip_checkpoint: SurgeryCLIP checkpoint路径
            num_classes: 类别数
            cam_resolution: CAM分辨率
            device: 设备
            unfreeze_cam_last_layer: 是否解冻CAM生成器的最后一层
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.cam_resolution = cam_resolution
        self.device = device
        
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
        
        # ===== 可训练部分 =====
        
        # 多层特征和CAM提取器
        print("创建多层特征提取器...")
        self.multi_layer_extractor = MultiLayerFeatureExtractor(self.simple_surgery_cam)
        
        # 多层CAM融合
        print("创建多层CAM融合模块...")
        self.cam_fusion = MultiLayerCAMFusion(num_layers=3)
        
        # 原图编码器
        print("创建原图编码器...")
        self.image_encoder = SimpleImageEncoder(
            output_dim=128,
            output_size=cam_resolution
        )
        
        # 多输入检测头
        print("创建多输入检测头...")
        self.detection_head = MultiInputDetectionHead(
            num_classes=num_classes,
            img_feat_dim=128,
            cam_dim=num_classes,
            layer_feat_dim=768,  # SurgeryCLIP的patch feature维度
            num_layers=3,
            hidden_dim=256,
            cam_resolution=cam_resolution
        )
        
        # CAM生成器：如果unfreeze_cam_last_layer=True，最后一层可训练
        if unfreeze_cam_last_layer:
            self._unfreeze_cam_last_layer()
        
        # 统计可训练参数
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        
        # 确保模型在正确的设备上
        self.cam_fusion = self.cam_fusion.to(device)
        self.image_encoder = self.image_encoder.to(device)
        self.detection_head = self.detection_head.to(device)
    
    def _unfreeze_cam_last_layer(self):
        """解冻CAM生成器的最后一层"""
        if hasattr(self.simple_surgery_cam.cam_generator, 'learnable_proj'):
            for param in self.simple_surgery_cam.cam_generator.learnable_proj.parameters():
                param.requires_grad = True
            print(f"✅ CAM生成器的可学习投影层已解冻")
    
    def forward(self, images: torch.Tensor, text_queries: List[str]) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            images: [B, 3, H, W]
            text_queries: List[str]
        
        Returns:
            Dict with:
                - boxes: [B, C, H, W, 4] 预测框坐标
                - confidences: [B, C, H, W] 置信度分数
                - fused_cam: [B, C, H, W] 融合后的CAM
                - multi_layer_cams: List[Tensor[B, C, H, W]] 多层CAM（用于监控）
                - layer_weights: [3] 层权重（用于监控）
        """
        B = images.shape[0]
        
        # ===== Step 1: 提取多层特征和CAM =====
        multi_features, multi_cams = self.multi_layer_extractor.extract_multi_layer_features_and_cams(
            images, text_queries
        )
        # multi_features: List[[B, N², D]] (3个)
        # multi_cams: List[[B, C, N, N]] (3个)
        
        # ===== Step 2: 融合多层CAM =====
        fused_cam = self.cam_fusion(multi_cams)  # [B, C, H, W]
        
        # ===== Step 3: 编码原图 =====
        img_features = self.image_encoder(images)  # [B, 128, 7, 7]
        
        # ===== Step 4: 检测头预测 =====
        detection_outputs = self.detection_head(
            img_features, fused_cam, multi_features
        )
        # detection_outputs包含: boxes, confidences等
        
        # 获取层权重（用于监控）
        layer_weights = self.cam_fusion.get_layer_weights()
        
        return {
            'pred_boxes': detection_outputs['boxes'],
            'confidences': detection_outputs['confidences'],
            'raw_confidences': detection_outputs.get('raw_confidences'),
            'fused_cam': fused_cam,
            'multi_layer_cams': multi_cams,
            'layer_weights': layer_weights,
            'image_features': img_features,
            **detection_outputs
        }
    
    def inference(
        self, 
        images: torch.Tensor, 
        text_queries: List[str],
        conf_threshold: float = 0.3, 
        nms_threshold: float = 0.5,
        topk: int = 100
    ) -> Dict:
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
            boxes = outputs['pred_boxes']  # [B, C, H, W, 4]
            confidences = outputs['confidences']  # [B, C, H, W]
            
            B, C, H, W = confidences.shape
            all_detections = []
            
            for b in range(B):
                detections = []
                
                # 收集所有位置的检测
                for c in range(C):
                    conf_class = confidences[b, c]  # [H, W]
                    boxes_class = boxes[b, c]  # [H, W, 4]
                    
                    # 找到置信度超过阈值的位置
                    mask = conf_class > conf_threshold
                    if mask.sum() == 0:
                        continue
                    
                    # 提取检测
                    conf_values = conf_class[mask]  # [N]
                    box_values = boxes_class[mask]  # [N, 4]
                    positions = mask.nonzero(as_tuple=False)  # [N, 2]
                    
                    for pos, conf, box in zip(positions, conf_values, box_values):
                        detections.append({
                            'box': box.cpu(),
                            'confidence': conf.item(),
                            'class': c,
                            'position': (pos[0].item(), pos[1].item())
                        })
                
                # 按置信度排序
                detections.sort(key=lambda x: x['confidence'], reverse=True)
                
                # NMS（简化版，按类别分别处理）
                if len(detections) > 0:
                    import torchvision.ops
                    
                    boxes_tensor = torch.stack([d['box'] for d in detections])
                    scores_tensor = torch.tensor([d['confidence'] for d in detections])
                    
                    keep = torchvision.ops.nms(boxes_tensor, scores_tensor, nms_threshold)
                    detections = [detections[i] for i in keep.tolist()]
                    
                    # Top-k
                    detections = detections[:topk]
                
                # 添加类别名称
                for det in detections:
                    class_idx = det['class']
                    if class_idx < len(text_queries):
                        det['class_name'] = text_queries[class_idx]
                    else:
                        det['class_name'] = f"class_{class_idx}"
                
                all_detections.append(detections)
            
            return all_detections


def create_improved_direct_detection_detector(
    surgery_clip_checkpoint, 
    num_classes=20,
    cam_resolution=7,
    device='cuda', 
    unfreeze_cam_last_layer=True
):
    """
    创建改进的直接检测器
    
    Args:
        surgery_clip_checkpoint: SurgeryCLIP checkpoint路径
        num_classes: 类别数
        cam_resolution: CAM分辨率
        device: 设备
        unfreeze_cam_last_layer: 是否解冻CAM生成器的最后一层
    
    Returns:
        ImprovedDirectDetectionDetector实例
    """
    return ImprovedDirectDetectionDetector(
        surgery_clip_checkpoint=surgery_clip_checkpoint,
        num_classes=num_classes,
        cam_resolution=cam_resolution,
        device=device,
        unfreeze_cam_last_layer=unfreeze_cam_last_layer
    )


