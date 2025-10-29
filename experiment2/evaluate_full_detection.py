#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment2 完整检测评估
计算mAP@50, mAP@75, mAP@[.5:.95]
使用全局-局部对比学习的检测模型
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

sys.path.append('..')

from config.default_config import DefaultConfig
from stage1_encoder.clip_text_encoder import CLIPTextEncoder
from stage1_encoder.clip_image_encoder import CLIPImageEncoder
from datasets.mini_dataset.mini_dataset_loader import MiniDataset
import torchvision.transforms as T

# 从新的训练脚本加载类定义
with open('train_full_DIOR.py', 'r') as f:
    train_code = f.read()
    # 提取类定义部分
    class_code = train_code.split('def collate_fn')[0]
    exec(class_code)


def calculate_iou_matrix(boxes1, boxes2):
    """
    计算IoU矩阵
    boxes: [N, 4] xyxy格式
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    
    return iou


def calculate_ap(precisions, recalls):
    """计算AP（11点插值）"""
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return ap


def evaluate_detection(predictions, ground_truths, iou_threshold=0.5):
    """
    评估检测性能
    
    Args:
        predictions: list of dicts with 'image_id', 'category_id', 'bbox', 'score'
        ground_truths: list of dicts with 'image_id', 'category_id', 'bbox'
        iou_threshold: IoU阈值
    
    Returns:
        dict with mAP and AP per class
    """
    from collections import defaultdict
    
    gt_by_class = defaultdict(list)
    pred_by_class = defaultdict(list)
    
    for gt in ground_truths:
        gt_by_class[gt['category_id']].append(gt)
    
    for pred in predictions:
        pred_by_class[pred['category_id']].append(pred)
    
    aps = {}
    all_classes = set(list(gt_by_class.keys()) + list(pred_by_class.keys()))
    
    for class_id in all_classes:
        class_gts = gt_by_class[class_id]
        class_preds = pred_by_class[class_id]
        
        if len(class_gts) == 0:
            continue
        
        if len(class_preds) == 0:
            aps[class_id] = 0.0
            continue
        
        # 按置信度排序
        class_preds = sorted(class_preds, key=lambda x: x['score'], reverse=True)
        
        gt_matched = [False] * len(class_gts)
        
        tp = []
        fp = []
        
        for pred in class_preds:
            max_iou = 0
            max_gt_idx = -1
            
            for gt_idx, gt in enumerate(class_gts):
                if gt['image_id'] == pred['image_id']:
                    # 计算IoU
                    iou = calculate_iou_single(pred['bbox'], gt['bbox'])
                    if iou > max_iou:
                        max_iou = iou
                        max_gt_idx = gt_idx
            
            if max_iou >= iou_threshold and max_gt_idx >= 0 and not gt_matched[max_gt_idx]:
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
        
        ap = calculate_ap(precisions, recalls)
        aps[class_id] = ap
    
    mAP = np.mean(list(aps.values())) if aps else 0.0
    
    return {
        'mAP': mAP,
        'AP_per_class': aps,
        'num_classes': len(aps)
    }


def calculate_iou_single(box1, box2):
    """计算单个IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-6)


def evaluate():
    print("=" * 70)
    print("Experiment2 完整检测评估")
    print("自适应全局-局部对比学习")
    print("=" * 70)
    
    device = torch.device('cuda')
    
    # 加载类别
    from utils.dataloader import DIOR_CLASSES
    
    # 加载checkpoint
    checkpoint_path = 'outputs/checkpoints/DIOR_best_model.pth'
    print(f"\n加载checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    
    # 创建模型
    print("\n创建模型组件...")
    text_encoder = CLIPTextEncoder('RN50', '../checkpoints/RemoteCLIP-RN50.pt').cuda()
    image_encoder = CLIPImageEncoder('RN50', '../checkpoints/RemoteCLIP-RN50.pt', freeze=False).cuda()
    query_extractor = SimpleDeformableQueryExtractor(d_model=1024).cuda()
    box_regressor = BoxRegressor(d_model=1024).cuda()
    
    # 加载权重
    text_encoder.load_state_dict(checkpoint['text_encoder'])
    image_encoder.load_state_dict(checkpoint['image_encoder'])
    query_extractor.load_state_dict(checkpoint['query_extractor'])
    box_regressor.load_state_dict(checkpoint['box_regressor'])
    
    text_encoder.eval()
    image_encoder.eval()
    query_extractor.eval()
    box_regressor.eval()
    
    print(f"✅ 模型加载成功 (Epoch {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f})")
    
    # 提取文本特征
    print("\n提取文本特征...")
    with torch.no_grad():
        text_features = text_encoder(DIOR_CLASSES).cuda()
    
    # 加载测试数据
    print("\n加载测试数据...")
    test_dataset = MiniDataset('../datasets/mini_dataset', 'test', transforms=None)
    print(f"  测试集: {len(test_dataset)} 张图")
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 推理
    print("\n开始推理...")
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="检测"):
            image_pil, target = test_dataset[idx]
            image = transform(image_pil).unsqueeze(0).cuda()
            
            # 提取全局特征
            _, global_features = image_encoder(image)  # [1, 1024]
            
            # 归一化
            global_features_norm = global_features / global_features.norm(dim=-1, keepdim=True)
            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 使用多个query进行检测（网格采样）
            num_queries = 25  # 5x5网格
            queries = []
            for i in range(5):
                for j in range(5):
                    cx = (i + 0.5) / 5
                    cy = (j + 0.5) / 5
                    w = 0.3
                    h = 0.3
                    queries.append([cx, cy, w, h])
            
            init_boxes = torch.tensor(queries, device=device)  # [25, 4]
            
            # 提取局部特征
            global_feat_expanded = global_features.expand(num_queries, -1)
            local_features = query_extractor(global_feat_expanded, init_boxes)  # [25, 1024]
            
            # 回归边界框
            pred_boxes_cxcywh = box_regressor(local_features)  # [25, 4]
            
            # 归一化局部特征并计算与文本的相似度
            local_features_norm = local_features / (local_features.norm(dim=-1, keepdim=True) + 1e-8)
            scores = local_features_norm @ text_features_norm.T  # [25, 20]
            
            # 对每个query找最佳类别
            max_scores, pred_labels = scores.max(dim=-1)  # [25]
            
            # 过滤低置信度
            keep = max_scores > 0.15  # 降低阈值
            
            pred_boxes_cxcywh = pred_boxes_cxcywh[keep]
            pred_labels = pred_labels[keep]
            max_scores = max_scores[keep]
            
            # 转换为xyxy
            pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes_cxcywh) * 224  # 像素坐标
            
            for box, label, score in zip(pred_boxes_xyxy, pred_labels, max_scores):
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
                    'bbox': box.cpu().tolist()
                })
    
    print(f"\n收集完成:")
    print(f"  预测框数量: {len(predictions)}")
    print(f"  GT框数量: {len(ground_truths)}")
    
    # 计算mAP
    print("\n计算检测指标...")
    
    results_50 = evaluate_detection(predictions, ground_truths, iou_threshold=0.5)
    results_75 = evaluate_detection(predictions, ground_truths, iou_threshold=0.75)
    
    # COCO-style mAP
    mAPs = []
    for iou_thr in np.arange(0.5, 1.0, 0.05):
        res = evaluate_detection(predictions, ground_truths, iou_threshold=iou_thr)
        mAPs.append(res['mAP'])
    mAP_coco = np.mean(mAPs)
    
    print("\n" + "=" * 70)
    print("检测性能指标")
    print("=" * 70)
    
    print(f"\n📊 总体指标:")
    print(f"  mAP@50:       {results_50['mAP']:.4f}")
    print(f"  mAP@75:       {results_75['mAP']:.4f}")
    print(f"  mAP@[.5:.95]: {mAP_coco:.4f}")
    print(f"  检测类别数:   {results_50['num_classes']}/{len(DIOR_CLASSES)}")
    
    print(f"\n📋 各类别AP@50:")
    for class_id in sorted(results_50['AP_per_class'].keys()):
        if class_id < len(DIOR_CLASSES):
            class_name = DIOR_CLASSES[class_id]
            ap = results_50['AP_per_class'][class_id]
            print(f"  {class_name:25s}: {ap:.4f}")
    
    # 保存结果
    results = {
        'checkpoint': checkpoint_path,
        'epoch': checkpoint['epoch'],
        'train_loss': checkpoint['loss'],
        'test_metrics': {
            'mAP@50': float(results_50['mAP']),
            'mAP@75': float(results_75['mAP']),
            'mAP@[.5:.95]': float(mAP_coco),
            'num_classes_detected': results_50['num_classes']
        },
        'AP_per_class': {
            DIOR_CLASSES[k]: float(v)
            for k, v in results_50['AP_per_class'].items()
            if k < len(DIOR_CLASSES)
        },
        'num_predictions': len(predictions),
        'num_ground_truths': len(ground_truths)
    }
    
    with open('outputs/full_detection_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果保存到: outputs/full_detection_results.json")
    
    # 可视化几个检测结果
    print(f"\n可视化检测结果...")
    vis_dir = Path('outputs/full_detection_vis')
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    for img_id in range(min(5, len(test_dataset))):
        image_pil, target = test_dataset[img_id]
        
        # 获取该图的预测
        img_preds = [p for p in predictions if p['image_id'] == img_id]
        img_gts = [gt for gt in ground_truths if gt['image_id'] == img_id]
        
        # 绘制
        draw = ImageDraw.Draw(image_pil)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # GT (绿色)
        for gt in img_gts:
            bbox = gt['bbox']
            label = gt['category_id']
            draw.rectangle(bbox, outline='green', width=3)
            draw.text((bbox[0], bbox[1]-20), f"GT:{DIOR_CLASSES[label]}", fill='green', font=font)
        
        # 预测 (红色)
        for pred in img_preds:
            bbox = pred['bbox']
            label = pred['category_id']
            score = pred['score']
            draw.rectangle(bbox, outline='red', width=2)
            draw.text((bbox[0], bbox[1]-20), f"{DIOR_CLASSES[label]}:{score:.2f}", fill='red', font=font)
        
        image_pil.save(vis_dir / f'detection_{img_id:03d}.jpg')
    
    print(f"✅ 可视化保存到: {vis_dir}/")
    
    print("\n" + "=" * 70)
    print("评估完成！")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    evaluate()

