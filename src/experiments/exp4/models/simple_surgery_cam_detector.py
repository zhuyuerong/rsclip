# -*- coding: utf-8 -*-
"""
简化的SurgeryCAM Detector
- 使用SimpleSurgeryCAM（无p2p和AAF）
- CAM生成器可训练（解冻一层）
"""

import torch
import torch.nn as nn
from typing import List, Dict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.simple_surgery_cam import SimpleSurgeryCAM, create_simple_surgery_cam_model
from models.box_head import BoxHead


class SimpleSurgeryCAMDetector(nn.Module):
    """
    简化的SurgeryCAM Detector
    - SurgeryCLIP冻结
    - SimpleCAMGenerator可训练（解冻一层）
    - BoxHead可训练
    """
    
    def __init__(self, surgery_clip_checkpoint, num_classes=20, 
                 cam_resolution=7, upsample_cam=False, device='cuda',
                 unfreeze_cam_last_layer=True):
        """
        Args:
            surgery_clip_checkpoint: SurgeryCLIP checkpoint路径
            num_classes: 类别数
            cam_resolution: CAM分辨率
            upsample_cam: 是否上采样CAM
            device: 设备
            unfreeze_cam_last_layer: 是否解冻CAM生成器的最后一层
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.cam_resolution = cam_resolution
        self.upsample_cam = upsample_cam
        self.device = device
        
        # ===== 冻结部分 =====
        # Load SimpleSurgeryCAM model
        print("Loading SimpleSurgeryCAM (no p2p, no AAF)...")
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
            # 解冻CAM生成器的可学习投影层
            if hasattr(self.simple_surgery_cam.cam_generator, 'learnable_proj'):
                for param in self.simple_surgery_cam.cam_generator.learnable_proj.parameters():
                    param.requires_grad = True
                print(f"✅ CAM生成器的可学习投影层已解冻")
        
        # ===== 可训练部分 =====
        # BoxHead: 从CAM预测框参数
        final_resolution = cam_resolution * 2 if upsample_cam else cam_resolution
        self.box_head = BoxHead(
            num_classes=num_classes,
            hidden_dim=256,
            cam_resolution=cam_resolution,
            upsample=upsample_cam
        )
        
        # 统计可训练参数
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")
        
        # 确保模型在正确的设备上
        self.box_head = self.box_head.to(device)
    
    def forward(self, images: torch.Tensor, text_queries: List[str]) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            images: [B, 3, H, W]
            text_queries: List[str], 如 ["ship", "plane", "harbor"]
        
        Returns:
            Dict with:
                - cam: [B, C, H_cam, W_cam]
                - pred_boxes: [B, C, H_cam, W_cam, 4]
                - scores: [B, C, H_cam, W_cam]
        """
        B = images.shape[0]
        
        # ===== Step 1: 生成CAM =====
        cam, aux = self.simple_surgery_cam(images, text_queries)
        # cam: [B, C, N, N] where N is patch resolution (e.g., 7)
        
        # ===== Step 2: BoxHead回归 =====
        box_outputs = self.box_head(cam)
        
        # 解码框参数为实际框坐标
        pred_boxes = self.box_head.decode_boxes(
            box_outputs['box_params'],
            cam_resolution=self.cam_resolution
        )
        # pred_boxes: [B, C, H, W, 4]
        
        # ===== Step 3: 置信度得分 =====
        scores = cam  # [B,C,H,W] - 直接用CAM激活
        
        return {
            'cam': cam,
            'pred_boxes': pred_boxes,
            'scores': scores,
            'box_params': box_outputs['box_params'],
            'objectness': box_outputs.get('objectness', None)
        }
    
    def inference(self, images: torch.Tensor, text_queries: List[str],
                 conf_threshold: float = 0.5, nms_threshold: float = 0.5,
                 topk: int = 50, min_peak_distance: int = 2,
                 max_peaks_per_class: int = 10) -> Dict:
        """
        推理接口（与原SurgeryCAMDetector兼容）
        """
        # 复用原SurgeryCAMDetector的inference逻辑
        from models.multi_instance_assigner import MultiPeakDetector
        from losses.detection_loss import generalized_box_iou
        import torch.nn.functional as F
        
        outputs = self.forward(images, text_queries)
        cam = outputs['cam']
        pred_boxes = outputs['pred_boxes']
        
        B, C, H, W = cam.shape
        final_detections = []
        
        for b in range(B):
            detections = []
            for c in range(C):
                cam_class = cam[b, c]  # [H, W]
                
                # 峰值检测
                peak_detector = MultiPeakDetector(
                    min_peak_distance=min_peak_distance,
                    min_peak_value=conf_threshold
                )
                peaks = peak_detector.detect_peaks(cam_class)
                
                # 限制每个类别的峰值数量
                peaks = sorted(peaks, key=lambda x: x[2], reverse=True)[:max_peaks_per_class]
                
                for i, j, score in peaks:
                    box = pred_boxes[b, c, i, j]  # [4]
                    detections.append({
                        'box': box.cpu(),
                        'score': score,
                        'class': c,
                        'class_name': text_queries[c] if c < len(text_queries) else f"class_{c}"
                    })
            
            final_detections.append(detections)
        
        return final_detections


def create_simple_surgery_cam_detector(surgery_clip_checkpoint, num_classes=20,
                                      cam_resolution=7, upsample_cam=False,
                                      device='cuda', unfreeze_cam_last_layer=True):
    """
    创建简化的SurgeryCAM Detector
    
    Args:
        surgery_clip_checkpoint: SurgeryCLIP checkpoint路径
        num_classes: 类别数
        cam_resolution: CAM分辨率
        upsample_cam: 是否上采样CAM
        device: 设备
        unfreeze_cam_last_layer: 是否解冻CAM生成器的最后一层
    
    Returns:
        SimpleSurgeryCAMDetector实例
    """
    return SimpleSurgeryCAMDetector(
        surgery_clip_checkpoint=surgery_clip_checkpoint,
        num_classes=num_classes,
        cam_resolution=cam_resolution,
        upsample_cam=upsample_cam,
        device=device,
        unfreeze_cam_last_layer=unfreeze_cam_last_layer
    )

