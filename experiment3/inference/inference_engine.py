#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理引擎

功能：
1. 加载训练好的模型
2. 对单张或批量图像进行推理
3. 后处理和可视化
"""

import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import List, Dict, Union
import sys
sys.path.append('..')

from config.default_config import DefaultConfig
from models.ova_detr import OVADETR
from utils.data_loader import DIOR_CLASSES
from utils.transforms import get_transforms


class InferenceEngine:
    """
    推理引擎
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        score_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        max_detections: int = 100
    ):
        """
        参数:
            checkpoint_path: 模型检查点路径
            device: 'cuda' 或 'cpu'
            score_threshold: 分数阈值
            nms_threshold: NMS阈值
            max_detections: 最大检测数
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        
        print("=" * 70)
        print("初始化推理引擎")
        print("=" * 70)
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint.get('config', DefaultConfig())
        
        # 创建模型
        self.model = OVADETR(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ 模型加载成功 (epoch {checkpoint.get('epoch', 'unknown')})")
        
        # 提取文本特征
        with torch.no_grad():
            self.text_features = self.model.backbone.forward_text(DIOR_CLASSES)
            self.text_features = self.text_features.to(self.device)
        
        print(f"✅ 文本特征提取完成: {self.text_features.shape}")
        
        # 转换
        self.transforms = get_transforms(mode='val', image_size=self.config.image_size)
        
        # 类别
        self.classes = DIOR_CLASSES
        
        print("=" * 70)
    
    @torch.no_grad()
    def predict_single(
        self,
        image: Union[str, Path, Image.Image]
    ) -> Dict:
        """
        预测单张图像
        
        参数:
            image: 图像路径或PIL Image
        
        返回:
            result: {
                'boxes': (N, 4) [x1, y1, x2, y2],
                'scores': (N,),
                'labels': (N,),
                'class_names': List[str]
            }
        """
        # 加载图像
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        
        orig_w, orig_h = image.size
        
        # 转换
        image_tensor, _ = self.transforms(image, None)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # 推理
        outputs = self.model(image_tensor, self.text_features)
        
        # 使用最后一层的输出
        pred_logits = outputs['pred_logits'][-1]  # (1, num_queries, num_classes)
        pred_boxes = outputs['pred_boxes'][-1]    # (1, num_queries, 4)
        
        # 转换为numpy
        pred_logits = pred_logits[0].cpu()  # (num_queries, num_classes)
        pred_boxes = pred_boxes[0].cpu()    # (num_queries, 4)
        
        # 计算分数和标签
        scores = pred_logits.sigmoid()  # (num_queries, num_classes)
        max_scores, labels = scores.max(dim=-1)  # (num_queries,)
        
        # 过滤低分数
        keep = max_scores > self.score_threshold
        boxes = pred_boxes[keep]
        scores_keep = max_scores[keep]
        labels_keep = labels[keep]
        
        if len(boxes) > 0:
            # 转换边界框：cxcywh (归一化) -> xyxy (像素)
            cx, cy, w, h = boxes.unbind(-1)
            x1 = (cx - w / 2) * orig_w
            y1 = (cy - h / 2) * orig_h
            x2 = (cx + w / 2) * orig_w
            y2 = (cy + h / 2) * orig_h
            boxes = torch.stack([x1, y1, x2, y2], dim=-1)
            
            # NMS
            keep_nms = self._nms(boxes, scores_keep, self.nms_threshold)
            boxes = boxes[keep_nms]
            scores_keep = scores_keep[keep_nms]
            labels_keep = labels_keep[keep_nms]
            
            # 限制数量
            if len(boxes) > self.max_detections:
                sorted_indices = scores_keep.argsort(descending=True)[:self.max_detections]
                boxes = boxes[sorted_indices]
                scores_keep = scores_keep[sorted_indices]
                labels_keep = labels_keep[sorted_indices]
        
        # 获取类别名称
        class_names = [self.classes[label] for label in labels_keep]
        
        return {
            'boxes': boxes.numpy(),
            'scores': scores_keep.numpy(),
            'labels': labels_keep.numpy(),
            'class_names': class_names
        }
    
    def _nms(self, boxes, scores, threshold):
        """简单的NMS实现"""
        if len(boxes) == 0:
            return torch.tensor([], dtype=torch.long)
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        _, order = scores.sort(descending=True)
        
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
            
            i = order[0].item()
            keep.append(i)
            
            # 计算IoU
            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # 保留IoU小于阈值的框
            mask = iou <= threshold
            order = order[1:][mask]
        
        return torch.tensor(keep, dtype=torch.long)
    
    def visualize(
        self,
        image: Union[str, Path, Image.Image],
        result: Dict,
        output_path: str = None,
        show_scores: bool = True
    ) -> Image.Image:
        """
        可视化检测结果
        
        参数:
            image: 图像路径或PIL Image
            result: 检测结果
            output_path: 输出路径
            show_scores: 是否显示分数
        
        返回:
            vis_image: 可视化图像
        """
        # 加载图像
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        
        # 创建绘图对象
        draw = ImageDraw.Draw(image)
        
        # 尝试加载字体
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # 颜色映射
        colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
            '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B195', '#C06C84'
        ]
        
        # 绘制边界框
        boxes = result['boxes']
        scores = result['scores']
        class_names = result['class_names']
        
        for i, (box, score, cls_name) in enumerate(zip(boxes, scores, class_names)):
            x1, y1, x2, y2 = box
            color = colors[i % len(colors)]
            
            # 绘制矩形
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 绘制标签
            if show_scores:
                label = f"{cls_name}: {score:.2f}"
            else:
                label = cls_name
            
            # 计算文本大小
            bbox = draw.textbbox((x1, y1), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            # 绘制文本背景
            draw.rectangle(
                [x1, y1 - text_h - 4, x1 + text_w + 4, y1],
                fill=color
            )
            
            # 绘制文本
            draw.text((x1 + 2, y1 - text_h - 2), label, fill='white', font=font)
        
        # 保存
        if output_path:
            image.save(output_path)
            print(f"✅ 结果保存到: {output_path}")
        
        return image


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='OVA-DETR推理')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--image', type=str, required=True,
                       help='输入图像路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出图像路径')
    parser.add_argument('--score_threshold', type=float, default=0.5,
                       help='分数阈值')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 创建推理引擎
    engine = InferenceEngine(
        checkpoint_path=args.checkpoint,
        device=args.device,
        score_threshold=args.score_threshold
    )
    
    # 推理
    print(f"\n推理图像: {args.image}")
    result = engine.predict_single(args.image)
    
    print(f"\n检测结果:")
    print(f"  检测到 {len(result['boxes'])} 个目标")
    for i, (score, cls_name) in enumerate(zip(result['scores'], result['class_names'])):
        print(f"  {i+1}. {cls_name}: {score:.3f}")
    
    # 可视化
    if args.output:
        engine.visualize(args.image, result, args.output)
    else:
        vis_image = engine.visualize(args.image, result)
        vis_image.show()

