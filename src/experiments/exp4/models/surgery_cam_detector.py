# -*- coding: utf-8 -*-
"""
SurgeryCAMDetector
核心检测模型：集成SurgeryCLIP + AAF + CAMGenerator（冻结）+ BoxHead（可训练）
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import sys
import os
from pathlib import Path

# Import SurgeryAAF from exp1
# We'll import it lazily when needed, with proper path setup
def _import_surgery_aaf():
    """Lazy import of SurgeryAAF to handle path issues"""
    import importlib.util
    
    # Find exp1 directory
    exp_dir = Path(__file__).parent.parent.parent
    exp1_dir = exp_dir / 'exp1'
    exp1_models_dir = exp1_dir / 'models'
    
    # Import dependencies first
    aaf_path = exp1_models_dir / 'aaf.py'
    cam_gen_path = exp1_models_dir / 'cam_generator.py'
    surgery_aaf_path = exp1_models_dir / 'surgery_aaf.py'
    
    # Load AAF
    spec_aaf = importlib.util.spec_from_file_location("exp1_models_aaf", aaf_path)
    aaf_module = importlib.util.module_from_spec(spec_aaf)
    spec_aaf.loader.exec_module(aaf_module)
    sys.modules['exp1_models_aaf'] = aaf_module
    
    # Load CAMGenerator
    spec_cam = importlib.util.spec_from_file_location("exp1_models_cam_generator", cam_gen_path)
    cam_gen_module = importlib.util.module_from_spec(spec_cam)
    spec_cam.loader.exec_module(cam_gen_module)
    sys.modules['exp1_models_cam_generator'] = cam_gen_module
    
    # Read and modify surgery_aaf.py to replace relative imports
    with open(surgery_aaf_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Replace relative imports with module references
    code = code.replace('from .aaf import AAF', 'AAF = sys.modules["exp1_models_aaf"].AAF')
    code = code.replace('from .cam_generator import CAMGenerator', 'CAMGenerator = sys.modules["exp1_models_cam_generator"].CAMGenerator')
    
    # Create a new module with modified code
    surgery_aaf_module = type(sys)('exp1_models_surgery_aaf')
    surgery_aaf_module.__dict__.update({
        'sys': sys,
        'torch': torch,
        'nn': nn,
        'os': os,
        'Path': Path,
        '__file__': str(surgery_aaf_path),
        '__name__': 'exp1_models_surgery_aaf'
    })
    
    # Execute the modified code
    exec(compile(code, str(surgery_aaf_path), 'exec'), surgery_aaf_module.__dict__)
    
    return surgery_aaf_module.SurgeryAAF, surgery_aaf_module.create_surgery_aaf_model

# Store the import function for later use
_surgery_aaf_import = _import_surgery_aaf

# Import BoxHead
from .box_head import BoxHead


class SurgeryCAMDetector(nn.Module):
    """
    遥感开放词汇检测器
    
    Pipeline:
    Image → [Frozen] SurgeryCLIP+AAF+CAM → CAM[B,C,H,W]
                                              ↓
                                        [Trainable] BoxHead
                                              ↓
                                      pred_boxes[B,C,H,W,4]
    """
    
    def __init__(self, 
                 surgery_clip_checkpoint: str,
                 num_classes: int = 20,
                 cam_resolution: int = 7,
                 upsample_cam: bool = False,
                 device: str = 'cuda'):
        """
        Args:
            surgery_clip_checkpoint: Path to SurgeryCLIP checkpoint
            num_classes: Number of classes
            cam_resolution: Original CAM resolution (e.g., 7 for ViT-B/32)
            upsample_cam: Whether to upsample CAM before BoxHead
            device: Device to load model on
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.cam_resolution = cam_resolution
        self.upsample_cam = upsample_cam
        self.device = device
        
        # ===== 冻结部分 =====
        # Load SurgeryAAF model
        print("Loading SurgeryCLIP + AAF + CAMGenerator...")
        # Import SurgeryAAF lazily
        SurgeryAAF, create_surgery_aaf_model = _surgery_aaf_import()
        
        self.surgery_aaf, _ = create_surgery_aaf_model(
            checkpoint_path=surgery_clip_checkpoint,
            device=device,
            num_layers=6
        )
        
        # 冻结所有参数
        for param in self.surgery_aaf.parameters():
            param.requires_grad = False
        
        # Set to eval mode (but allow forward pass)
        self.surgery_aaf.eval()
        
        # ===== 可训练部分 =====
        # BoxHead: 从CAM预测框参数
        final_resolution = cam_resolution * 2 if upsample_cam else cam_resolution
        self.box_head = BoxHead(
            num_classes=num_classes,
            hidden_dim=256,
            cam_resolution=cam_resolution,
            upsample=upsample_cam
        )
        
        print(f"Trainable parameters: {sum(p.numel() for p in self.box_head.parameters() if p.requires_grad):,}")
        
        # 确保BoxHead在正确的设备上
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
                - pred_boxes: [B, C, H_cam, W_cam, 4]  # 归一化坐标
                - scores: [B, C, H_cam, W_cam]
        """
        B = images.shape[0]
        
        # ===== Step 1: 生成CAM (冻结) =====
        # 虽然SurgeryAAF是冻结的，但我们需要CAM的梯度用于BoxHead训练
        # 所以不使用no_grad()，而是确保SurgeryAAF的参数不更新
        cam, aux = self.surgery_aaf(images, text_queries)
        # cam: [B, C, N, N] where N is patch resolution (e.g., 7)
        # 虽然SurgeryAAF参数冻结，但CAM tensor本身可以有梯度用于BoxHead
        
        # ===== Step 2: BoxHead回归 (可训练) =====
        # 输入CAM,输出框参数
        box_outputs = self.box_head(cam)
        # box_outputs包含: box_params, delta_cx, delta_cy, w, h, objectness
        
        # 解码框参数为实际框坐标
        pred_boxes = self.box_head.decode_boxes(
            box_outputs['box_params'],
            cam_resolution=self.cam_resolution
        )
        # pred_boxes: [B, C, H, W, 4]
        
        # ===== Step 3: 置信度得分 =====
        scores = cam  # [B,C,H,W] - 直接用CAM激活
        # 或: scores = cam * box_outputs['objectness']  # 加入objectness
        
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
        推理阶段
        
        Args:
            images: [B, 3, H, W] 或单个图像
            text_queries: List[str], 如 ["ship", "airplane", "harbor"]
            conf_threshold: CAM置信度阈值（提高以减少误检）
            nms_threshold: NMS IoU阈值
            topk: 保留分数最高的k个检测结果（全局topk）
        
        Returns:
            detections: List[dict] per image
                - boxes: [N, 4] 归一化坐标
                - labels: [N]
                - scores: [N]
                - class_names: List[str]
        """
        self.eval()
        
        # Handle single image
        if images.dim() == 3:
            images = images.unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.forward(images, text_queries)
        
        cam = outputs['cam']  # [B, C, H, W]
        pred_boxes = outputs['pred_boxes']  # [B, C, H, W, 4]
        scores = outputs['scores']  # [B, C, H, W]
        
        B, C, H, W = cam.shape
        
        # 导入峰值检测器（用于多实例检测）
        from .multi_instance_assigner import MultiPeakDetector
        peak_detector = MultiPeakDetector(
            min_peak_distance=min_peak_distance,
            min_peak_value=conf_threshold  # 使用conf_threshold作为峰值阈值
        )
        
        # 收集所有检测
        all_detections = []
        
        for b in range(B):
            detections_per_image = []
            
            # ===== 正确的多类别多实例检测流程 =====
            # 对每个类别独立检测峰值（支持多类别，每个类别多个实例）
            for c in range(C):
                cam_class = cam[b, c]  # [H, W] - 该类别的CAM
                
                # 1. 检测该类别的峰值（多实例）
                peaks = peak_detector.detect_peaks(cam_class)
                
                # 2. 限制每个类别的峰值数量（避免过多检测）
                # 峰值已按分数排序，取前max_peaks_per_class个
                peaks = peaks[:max_peaks_per_class]
                
                # 3. 为每个峰值位置生成检测框
                for i, j, score in peaks:
                    # 使用该类别在该位置预测的框
                    box = pred_boxes[b, c, i, j]  # [4]
                    
                    detections_per_image.append({
                        'box': box.cpu(),
                        'label': c,
                        'score': score,
                        'class_name': text_queries[c]
                    })
            
            all_detections.append(detections_per_image)
        
        # 3. Per-class NMS
        from torchvision.ops import nms
        
        final_detections = []
        
        for b, detections_per_image in enumerate(all_detections):
            if len(detections_per_image) == 0:
                final_detections.append({
                    'boxes': torch.zeros((0, 4)),
                    'labels': torch.zeros((0,), dtype=torch.int64),
                    'scores': torch.zeros((0,)),
                    'class_names': []
                })
                continue
            
            # Group by class
            boxes_list = []
            scores_list = []
            labels_list = []
            class_names_list = []
            
            for c in range(C):
                # 该类的所有候选
                class_dets = [d for d in detections_per_image if d['label'] == c]
                if len(class_dets) == 0:
                    continue
                
                # 转tensor
                boxes = torch.stack([d['box'] for d in class_dets])
                scores_c = torch.tensor([d['score'] for d in class_dets])
                
                # NMS
                keep = nms(boxes, scores_c, nms_threshold)
                
                for k in keep:
                    boxes_list.append(boxes[k])
                    scores_list.append(scores_c[k].item())
                    labels_list.append(c)
                    class_names_list.append(text_queries[c])
            
            if len(boxes_list) > 0:
                boxes_tensor = torch.stack(boxes_list)
                scores_tensor = torch.tensor(scores_list)
                labels_tensor = torch.tensor(labels_list, dtype=torch.int64)
                
                # 全局TopK过滤：保留分数最高的k个检测结果
                if len(scores_tensor) > topk:
                    topk_scores, topk_indices = torch.topk(scores_tensor, topk, largest=True)
                    boxes_tensor = boxes_tensor[topk_indices]
                    scores_tensor = topk_scores
                    labels_tensor = labels_tensor[topk_indices]
                    class_names_list = [class_names_list[i] for i in topk_indices.tolist()]
                
                final_detections.append({
                    'boxes': boxes_tensor,
                    'labels': labels_tensor,
                    'scores': scores_tensor,
                    'class_names': class_names_list
                })
            else:
                final_detections.append({
                    'boxes': torch.zeros((0, 4)),
                    'labels': torch.zeros((0,), dtype=torch.int64),
                    'scores': torch.zeros((0,)),
                    'class_names': []
                })
        
        return final_detections


def create_surgery_cam_detector(surgery_clip_checkpoint: str,
                               num_classes: int = 20,
                               cam_resolution: int = 7,
                               upsample_cam: bool = False,
                               device: str = 'cuda') -> SurgeryCAMDetector:
    """
    Factory function to create SurgeryCAMDetector
    
    Args:
        surgery_clip_checkpoint: Path to SurgeryCLIP checkpoint
        num_classes: Number of classes
        cam_resolution: Original CAM resolution
        upsample_cam: Whether to upsample CAM
        device: Device to load model on
    
    Returns:
        SurgeryCAMDetector instance
    """
    return SurgeryCAMDetector(
        surgery_clip_checkpoint=surgery_clip_checkpoint,
        num_classes=num_classes,
        cam_resolution=cam_resolution,
        upsample_cam=upsample_cam,
        device=device
    )

