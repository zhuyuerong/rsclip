#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment3 检测性能评估
计算mAP@50, mAP@75, mAP@[.5:.95]
"""

import torch
import numpy as np
import sys
from pathlib import Path
from PIL import Image
import json

sys.path.append('..')

from config.default_config import DefaultConfig
from models.ova_detr import OVADETR
from utils.data_loader import DIOR_CLASSES
from datasets.mini_dataset.mini_dataset_loader import MiniDataset
import torchvision.transforms as T


def calculate_iou(box1, box2):
    """计算两个框的IoU (boxes格式: [x1, y1, x2, y2])"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def calculate_ap_per_class(predictions, ground_truths, class_id, iou_threshold=0.5):
    """计算单个类别的AP"""
    class_preds = [p for p in predictions if p['category_id'] == class_id]
    class_gts = [gt for gt in ground_truths if gt['category_id'] == class_id]
    
    if len(class_gts) == 0:
        return None
    
    if len(class_preds) == 0:
        return 0.0
    
    # 按置信度排序
    class_preds = sorted(class_preds, key=lambda x: x['score'], reverse=True)
    
    # 标记GT是否被匹配
    gt_matched = [False] * len(class_gts)
    
    tp = []
    fp = []
    
    for pred in class_preds:
        max_iou = 0
        max_gt_idx = -1
        
        # 找到同一图像的GT
        for gt_idx, gt in enumerate(class_gts):
            if gt['image_id'] == pred['image_id']:
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
        
        if max_iou >= iou_threshold and not gt_matched[max_gt_idx]:
            tp.append(1)
            fp.append(0)
            gt_matched[max_gt_idx] = True
        else:
            tp.append(0)
            fp.append(1)
    
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / len(class_gts)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    
    # 计算AP (11点插值)
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    
    return ap


def evaluate():
    """评估检测性能"""
    
    print("=" * 70)
    print("Experiment3: OVA-DETR 检测性能评估")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    
    # 配置
    config = DefaultConfig()
    
    print("\n创建模型...")
    model = OVADETR(config).to(device)
    model.eval()
    
    print("准备文本特征...")
    with torch.no_grad():
        text_features = model.backbone.forward_text(DIOR_CLASSES).to(device)
    
    print(f"文本特征: {text_features.shape}")
    
    print("\n加载mini_dataset...")
    
    # 不使用transforms，手动处理
    dataset = MiniDataset(
        root_dir='../datasets/mini_dataset',
        split='test',
        transforms=None
    )
    
    # 定义图像转换
    img_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"测试集大小: {len(dataset)} 张图")
    
    if len(dataset) == 0:
        print("❌ 测试集为空，无法评估！")
        return
    
    predictions = []
    ground_truths = []
    
    print("\n开始推理...")
    with torch.no_grad():
        for idx in range(len(dataset)):
            if idx % 5 == 0:
                print(f"  处理: {idx+1}/{len(dataset)}")
            
            # 加载数据
            image_pil, target = dataset[idx]
            
            # 转换图像
            image = img_transform(image_pil).unsqueeze(0).to(device)  # [1, 3, H, W]
            
            # 模型推理
            outputs = model(image, text_features)
            
            # 解析输出
            pred_boxes = outputs['pred_boxes'][0]  # [num_queries, 4] cxcywh normalized
            pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
            
            # 计算置信度
            pred_scores = pred_logits.softmax(-1)
            max_scores, pred_labels = pred_scores.max(-1)
            
            # 过滤低置信度
            keep = max_scores > config.score_threshold
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            max_scores = max_scores[keep]
            
            # 转换box格式 (cxcywh normalized -> xyxy pixel)
            H, W = image.shape[2], image.shape[3]  # [1, 3, H, W]
            boxes_xyxy = torch.zeros_like(pred_boxes)
            boxes_xyxy[:, 0] = (pred_boxes[:, 0] - pred_boxes[:, 2] / 2) * W
            boxes_xyxy[:, 1] = (pred_boxes[:, 1] - pred_boxes[:, 3] / 2) * H
            boxes_xyxy[:, 2] = (pred_boxes[:, 0] + pred_boxes[:, 2] / 2) * W
            boxes_xyxy[:, 3] = (pred_boxes[:, 1] + pred_boxes[:, 3] / 2) * H
            
            # 收集预测
            for box, label, score in zip(boxes_xyxy, pred_labels, max_scores):
                predictions.append({
                    'image_id': idx,
                    'category_id': int(label),
                    'bbox': box.cpu().tolist(),
                    'score': float(score)
                })
            
            # 收集GT
            for box, label in zip(target['boxes'], target['labels']):
                ground_truths.append({
                    'image_id': idx,
                    'category_id': int(label),
                    'bbox': box.tolist()
                })
    
    print(f"\n收集完成:")
    print(f"  预测框数量: {len(predictions)}")
    print(f"  GT框数量: {len(ground_truths)}")
    
    # 计算所有类别的mAP
    print("\n计算检测指标...")
    
    all_classes = set([p['category_id'] for p in predictions] + [gt['category_id'] for gt in ground_truths])
    
    # mAP@50
    aps_50 = {}
    for class_id in all_classes:
        ap = calculate_ap_per_class(predictions, ground_truths, class_id, iou_threshold=0.5)
        if ap is not None:
            aps_50[class_id] = ap
    
    mAP_50 = np.mean(list(aps_50.values())) if aps_50 else 0.0
    
    # mAP@75
    aps_75 = {}
    for class_id in all_classes:
        ap = calculate_ap_per_class(predictions, ground_truths, class_id, iou_threshold=0.75)
        if ap is not None:
            aps_75[class_id] = ap
    
    mAP_75 = np.mean(list(aps_75.values())) if aps_75 else 0.0
    
    # mAP@[.5:.95] (COCO风格)
    mAPs_coco = []
    for iou_thr in np.arange(0.5, 1.0, 0.05):
        aps = {}
        for class_id in all_classes:
            ap = calculate_ap_per_class(predictions, ground_truths, class_id, iou_threshold=iou_thr)
            if ap is not None:
                aps[class_id] = ap
        if aps:
            mAPs_coco.append(np.mean(list(aps.values())))
    
    mAP_coco = np.mean(mAPs_coco) if mAPs_coco else 0.0
    
    # 打印结果
    print("\n" + "=" * 70)
    print("检测性能指标")
    print("=" * 70)
    
    print(f"\n📊 总体指标:")
    print(f"  mAP@50:       {mAP_50:.4f}")
    print(f"  mAP@75:       {mAP_75:.4f}")
    print(f"  mAP@[.5:.95]: {mAP_coco:.4f}")
    print(f"  检测类别数:   {len(aps_50)}/{len(DIOR_CLASSES)}")
    
    print(f"\n📋 各类别AP@50:")
    for class_id in sorted(aps_50.keys()):
        if class_id < len(DIOR_CLASSES):
            class_name = DIOR_CLASSES[class_id]
            ap = aps_50[class_id]
            print(f"  {class_name:20s}: {ap:.4f}")
    
    # 保存结果
    results = {
        'dataset': 'mini_dataset',
        'split': 'test',
        'num_images': len(dataset),
        'num_predictions': len(predictions),
        'num_ground_truths': len(ground_truths),
        'metrics': {
            'mAP@50': float(mAP_50),
            'mAP@75': float(mAP_75),
            'mAP@[.5:.95]': float(mAP_coco),
            'num_classes_detected': len(aps_50)
        },
        'AP_per_class_@50': {
            DIOR_CLASSES[k]: float(v) 
            for k, v in aps_50.items() 
            if k < len(DIOR_CLASSES)
        }
    }
    
    with open('detection_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ 结果保存到: detection_results.json")
    print("\n" + "=" * 70)
    
    return results


if __name__ == '__main__':
    evaluate()

