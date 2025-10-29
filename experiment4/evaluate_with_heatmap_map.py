# -*- coding: utf-8 -*-
"""
热图评估脚本
使用热图生成检测框并计算mAP
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment4.utils.heatmap_generator import (
    generate_similarity_heatmap,
    generate_bboxes_from_heatmap,
    compute_bbox_score
)
from experiment4.utils.map_calculator import calculate_map
from experiment4.utils.visualization import visualize_heatmap_and_boxes


def evaluate_model_with_heatmap(model, val_loader, device, attn_type='vv', max_visualizations=10):
    """
    使用热图生成检测框并计算mAP
    
    Args:
        model: CLIPSurgeryVV或CLIPSurgeryWrapper
        val_loader: DIOR验证集或mini_dataset验证集
        device: torch.device
        attn_type: 'vv', 'qk', 'mixed', 或 'standard'
        max_visualizations: 最大可视化样本数
    
    Returns:
        results: {
            'mAP': float,
            'per_class_ap': dict,
            'visualizations': list of matplotlib figures,
            'num_samples': int,
            'num_classes': int
        }
    """
    model_obj = model.model if hasattr(model, 'model') else model
    if hasattr(model_obj, 'eval'):
        model_obj.eval()
    
    all_predictions = defaultdict(list)
    all_ground_truths = defaultdict(list)
    visualizations = []
    
    print(f"\n生成热图并提取检测框（attn_type={attn_type}）...")
    
    for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"评估({attn_type})")):
        images = batch['image'].to(device)
        labels = batch['label']
        class_names = batch['class_name']
        gt_bboxes = batch['bbox']  # [B, 4] 归一化坐标
        has_bbox = batch['has_bbox']
        
        # 1. 提取特征（包含注意力权重）
        if hasattr(model, 'encode_image_with_attn') and attn_type != 'standard':
            try:
                image_features, attn_weights = model.encode_image_with_attn(images)
            except:
                # 如果失败，使用标准方法
                image_features = model.get_all_features(images)
                attn_weights = None
        else:
            image_features = model.get_all_features(images)
            attn_weights = None
        
        # 2. 获取文本特征
        unique_classes = list(set(class_names))
        text_features = model.encode_text(unique_classes)
        
        # 3. 生成相似度热图
        similarity_map = generate_similarity_heatmap(
            image_features, text_features, attn_type
        )  # [B, 7, 7, K]
        
        # 4. 对每个样本生成检测框
        for i in range(len(images)):
            class_name = class_names[i]
            class_idx = unique_classes.index(class_name)
            
            # 检查是否有有效的bbox标注
            if not has_bbox[i]:
                continue
            
            # 该类别的热图
            heatmap = similarity_map[i, :, :, class_idx].cpu().numpy()
            
            # 生成检测框
            pred_bboxes = generate_bboxes_from_heatmap(
                heatmap, 
                threshold_percentile=90, 
                image_size=224
            )
            
            # 计算每个框的分数（该区域的平均相似度）
            for bbox in pred_bboxes:
                score = compute_bbox_score(heatmap, bbox, image_size=224)
                all_predictions[class_name].append({
                    'bbox': bbox,
                    'score': score
                })
            
            # 保存GT（转换归一化坐标到像素坐标）
            gt_bbox_norm = gt_bboxes[i].cpu().numpy()
            gt_bbox_pixel = [
                int(gt_bbox_norm[0] * 224),  # x_min
                int(gt_bbox_norm[1] * 224),  # y_min
                int(gt_bbox_norm[2] * 224),  # x_max
                int(gt_bbox_norm[3] * 224)   # y_max
            ]
            all_ground_truths[class_name].append(gt_bbox_pixel)
            
            # 5. 可视化（保存前N个样本）
            if len(visualizations) < max_visualizations:
                vis_fig = visualize_heatmap_and_boxes(
                    images[i], heatmap, pred_bboxes, gt_bbox_pixel, class_name
                )
                visualizations.append(vis_fig)
    
    # 6. 计算mAP
    print(f"\n计算mAP...")
    mAP, per_class_ap = calculate_map(
        all_predictions, 
        all_ground_truths, 
        iou_threshold=0.5
    )
    
    # 7. 统计信息
    num_samples = sum(len(bboxes) for bboxes in all_ground_truths.values())
    num_classes = len(all_ground_truths)
    
    print(f"\n评估完成:")
    print(f"  样本数: {num_samples}")
    print(f"  类别数: {num_classes}")
    print(f"  mAP@0.5: {mAP:.4f}")
    
    return {
        'mAP': mAP,
        'per_class_ap': per_class_ap,
        'visualizations': visualizations,
        'num_samples': num_samples,
        'num_classes': num_classes
    }

