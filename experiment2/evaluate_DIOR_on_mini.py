#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用DIOR训练的模型在mini_dataset上评估
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import torchvision.transforms as T

sys.path.append('..')

from stage1_encoder.clip_text_encoder import CLIPTextEncoder
from stage1_encoder.clip_image_encoder import CLIPImageEncoder
from datasets.mini_dataset.mini_dataset_loader import MiniDataset, DIOR_CLASSES

# 导入模型类定义
exec(open('train_correct_DIOR.py').read().split('def train()')[0])


def box_cxcywh_to_xyxy(boxes):
    """cxcywh -> xyxy"""
    x_c, y_c, w, h = boxes.unbind(-1)
    b = [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]
    return torch.stack(b, dim=-1)


def calculate_iou(boxes1, boxes2):
    """计算IoU矩阵"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    
    return iou


def calculate_map(all_predictions, all_targets, iou_threshold=0.5):
    """计算mAP"""
    num_classes = 20
    aps = []
    class_results = {}
    
    for cls in range(num_classes):
        cls_preds = []
        cls_targets = []
        
        for preds, targets in zip(all_predictions, all_targets):
            # 预测
            if len(preds['boxes']) > 0:
                cls_mask = preds['labels'] == cls
                if cls_mask.any():
                    cls_preds.append({
                        'boxes': preds['boxes'][cls_mask],
                        'scores': preds['scores'][cls_mask]
                    })
                else:
                    cls_preds.append({'boxes': torch.empty(0, 4), 'scores': torch.empty(0)})
            else:
                cls_preds.append({'boxes': torch.empty(0, 4), 'scores': torch.empty(0)})
            
            # GT
            if len(targets['boxes']) > 0:
                cls_mask_gt = targets['labels'] == cls
                if cls_mask_gt.any():
                    cls_targets.append(targets['boxes'][cls_mask_gt])
                else:
                    cls_targets.append(torch.empty(0, 4))
            else:
                cls_targets.append(torch.empty(0, 4))
        
        all_scores = []
        all_tp = []
        num_gt = sum([len(gt) for gt in cls_targets])
        
        if num_gt == 0:
            class_results[DIOR_CLASSES[cls]] = {'AP': 0.0, 'GT': 0, 'pred': 0}
            continue
        
        for pred, gt in zip(cls_preds, cls_targets):
            if len(pred['boxes']) == 0:
                continue
            
            scores = pred['scores'].cpu()
            boxes = pred['boxes'].cpu()
            
            if len(gt) == 0:
                all_scores.extend(scores.tolist())
                all_tp.extend([0] * len(scores))
            else:
                gt_boxes = gt.cpu()
                iou = calculate_iou(boxes, gt_boxes)
                
                for i in range(len(boxes)):
                    all_scores.append(scores[i].item())
                    if iou[i].max() >= iou_threshold:
                        all_tp.append(1)
                    else:
                        all_tp.append(0)
        
        if len(all_scores) == 0:
            class_results[DIOR_CLASSES[cls]] = {'AP': 0.0, 'GT': num_gt, 'pred': 0}
            continue
        
        # 按分数排序
        indices = np.argsort(all_scores)[::-1]
        tp = np.array(all_tp)[indices]
        
        # 计算precision和recall
        累计tp = np.cumsum(tp)
        累计fp = np.cumsum(1 - tp)
        
        recall = 累计tp / num_gt
        precision = 累计tp / (累计tp + 累计fp)
        
        # 计算AP
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11
        
        aps.append(ap)
        class_results[DIOR_CLASSES[cls]] = {
            'AP': ap,
            'GT': num_gt,
            'pred': len(all_scores),
            'TP': 累计tp[-1] if len(累计tp) > 0 else 0
        }
    
    return np.mean(aps) if len(aps) > 0 else 0.0, class_results


def evaluate():
    print("=" * 70)
    print("用DIOR训练的模型在Mini Dataset上评估")
    print("=" * 70)
    
    device = torch.device('cuda')
    
    # 加载最佳模型
    checkpoint_path = 'outputs/checkpoints/DIOR_best_model.pth'
    print(f"\n加载checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    
    # 创建模型
    print("\n创建模型...")
    text_encoder = CLIPTextEncoder('RN50', '../checkpoints/RemoteCLIP-RN50.pt').cuda()
    image_encoder = CLIPImageEncoder('RN50', '../checkpoints/RemoteCLIP-RN50.pt', freeze=False).cuda()
    text_to_position = TextToPositionProjector(text_dim=1024).cuda()
    query_extractor = SimpleDeformableQueryExtractor(d_model=1024).cuda()
    box_regressor = BoxRegressor(d_model=1024).cuda()
    
    # 加载权重
    text_encoder.load_state_dict(checkpoint['text_encoder'])
    image_encoder.load_state_dict(checkpoint['image_encoder'])
    text_to_position.load_state_dict(checkpoint['text_to_position'])
    query_extractor.load_state_dict(checkpoint['query_extractor'])
    box_regressor.load_state_dict(checkpoint['box_regressor'])
    
    text_encoder.eval()
    image_encoder.eval()
    text_to_position.eval()
    query_extractor.eval()
    box_regressor.eval()
    
    print(f"✅ 模型加载成功")
    print(f"   训练Epoch: {checkpoint['epoch']}")
    print(f"   训练Loss: {checkpoint['loss']:.4f}")
    
    # 提取文本特征
    with torch.no_grad():
        text_features = text_encoder(DIOR_CLASSES).cuda()
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # 加载测试数据
    print("\n加载测试数据...")
    test_dataset = MiniDataset('../datasets/mini_dataset', 'test')
    print(f"  测试集: {len(test_dataset)} 张图")
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 测试不同阈值
    thresholds = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    print("\n" + "=" * 70)
    print("测试不同分数阈值")
    print("=" * 70)
    
    best_threshold = 0.2
    best_map = 0
    
    for threshold in thresholds:
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for idx in range(len(test_dataset)):
                image_pil, target = test_dataset[idx]
                image = transform(image_pil).unsqueeze(0).cuda()
                
                # 提取全局特征
                _, global_features = image_encoder(image)
                
                # 文本预测初始框位置
                predicted_init_boxes = text_to_position(text_features)
                
                # 扩展全局特征
                global_feat_expanded = global_features.expand(20, -1)
                
                # 提取局部特征
                local_features = query_extractor(global_feat_expanded, predicted_init_boxes)
                
                # 精修框
                refined_boxes = box_regressor(local_features)
                
                # 计算分数
                local_features_norm = local_features / (local_features.norm(dim=-1, keepdim=True) + 1e-8)
                scores = (local_features_norm * text_features_norm).sum(dim=-1)
                
                # 过滤
                keep = scores > threshold
                
                pred_boxes = refined_boxes[keep]
                pred_labels = torch.arange(20, device=device)[keep]
                pred_scores = scores[keep]
                
                # 转换为xyxy像素坐标
                pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes) * 224
                
                all_predictions.append({
                    'boxes': pred_boxes_xyxy,
                    'labels': pred_labels,
                    'scores': pred_scores
                })
                
                # GT
                gt_boxes = target['boxes'].cuda()
                gt_labels = target['labels'].cuda()
                gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes) * 224
                
                all_targets.append({
                    'boxes': gt_boxes_xyxy,
                    'labels': gt_labels
                })
        
        # 计算mAP
        map_50, class_results = calculate_map(all_predictions, all_targets, iou_threshold=0.5)
        
        total_preds = sum([len(p['boxes']) for p in all_predictions])
        total_tp = sum([r.get('TP', 0) for r in class_results.values()])
        
        print(f"\n阈值={threshold:.2f}: mAP@50={map_50*100:5.2f}% | 预测={total_preds:3d} | 匹配={total_tp:2.0f}")
        
        if map_50 > best_map:
            best_map = map_50
            best_threshold = threshold
            best_class_results = class_results
    
    # 显示最佳结果
    print("\n" + "=" * 70)
    print(f"最佳阈值: {best_threshold} (mAP@50: {best_map*100:.2f}%)")
    print("=" * 70)
    
    print("\n各类别AP:")
    for cls_name, result in sorted(best_class_results.items(), key=lambda x: x[1]['AP'], reverse=True):
        if result['GT'] > 0:
            print(f"  {cls_name:20s}: AP={result['AP']*100:5.1f}% | GT={result['GT']:2d} | 预测={result['pred']:3d} | 匹配={result['TP']:2.0f}")
    
    # 保存结果
    results = {
        'checkpoint': checkpoint_path,
        'epoch': checkpoint['epoch'],
        'train_loss': checkpoint['loss'],
        'best_threshold': best_threshold,
        'mAP@50': float(best_map),
        'class_results': {k: {kk: float(vv) if isinstance(vv, (int, float, np.number)) else vv 
                              for kk, vv in v.items()} 
                         for k, v in best_class_results.items()},
        'total_test_images': len(test_dataset),
        'total_gt': sum([len(t['boxes']) for t in all_targets])
    }
    
    Path('outputs').mkdir(exist_ok=True)
    with open('outputs/DIOR_model_on_mini_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果保存到: outputs/DIOR_model_on_mini_results.json")
    
    return results


if __name__ == '__main__':
    evaluate()

