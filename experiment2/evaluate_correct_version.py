#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估正确版本的Experiment2
使用文本驱动的位置预测
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

from datasets.mini_dataset.mini_dataset_loader import MiniDataset, DIOR_CLASSES


# 从训练脚本导入模型类
class TextToPositionProjector(nn.Module):
    def __init__(self, text_dim=1024, hidden_dim=512):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Sigmoid()
        )
    
    def forward(self, text_features):
        return self.projector(text_features)


class SimpleDeformableQueryExtractor(nn.Module):
    def __init__(self, d_model=1024):
        super().__init__()
        self.position_embed = nn.Linear(4, d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, global_features, boxes):
        pos_embed = self.position_embed(boxes)
        combined = torch.cat([global_features, pos_embed], dim=-1)
        return self.fusion(combined)


class BoxRegressor(nn.Module):
    def __init__(self, d_model=1024):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Sigmoid()
        )
    
    def forward(self, local_features):
        return self.regressor(local_features)


def box_cxcywh_to_xyxy(boxes):
    x_c, y_c, w, h = boxes.unbind(-1)
    b = [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]
    return torch.stack(b, dim=-1)


def calculate_iou(boxes1, boxes2):
    """计算IoU矩阵 (boxes1: [N, 4], boxes2: [M, 4])"""
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
    
    for cls in range(num_classes):
        # 收集该类的所有预测和GT
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
        
        # 计算该类的AP
        all_scores = []
        all_tp = []
        num_gt = sum([len(gt) for gt in cls_targets])
        
        if num_gt == 0:
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
            continue
        
        # 按分数排序
        indices = np.argsort(all_scores)[::-1]
        tp = np.array(all_tp)[indices]
        
        # 计算precision和recall
        累计tp = np.cumsum(tp)
        累计fp = np.cumsum(1 - tp)
        
        recall = 累计tp / num_gt
        precision = 累计tp / (累计tp + 累计fp)
        
        # 计算AP (11-point interpolation)
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11
        
        aps.append(ap)
        print(f"  {DIOR_CLASSES[cls]:15s} AP: {ap*100:.2f}% (GT: {num_gt})")
    
    if len(aps) == 0:
        return 0.0
    
    return np.mean(aps)


def evaluate():
    print("=" * 70)
    print("Experiment2 正确版本评估")
    print("文本驱动位置预测 + 全局-局部对比学习")
    print("=" * 70)
    
    device = torch.device('cuda')
    
    # 加载最佳模型
    checkpoint_path = 'outputs/checkpoints/correct_best_model.pth'
    print(f"\n加载checkpoint: {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint不存在，请先完成训练")
        return
    
    checkpoint = torch.load(checkpoint_path)
    
    # 导入编码器
    from stage1_encoder.clip_text_encoder import CLIPTextEncoder
    from stage1_encoder.clip_image_encoder import CLIPImageEncoder
    
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
    
    print(f"✅ 模型加载成功 (Epoch {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f})")
    
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
    
    print("\n开始评估...")
    print("  使用文本预测初始位置 → Query → 精修框")
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="评估"):
            image_pil, target = test_dataset[idx]
            image = transform(image_pil).unsqueeze(0).cuda()
            
            # 提取全局特征
            _, global_features = image_encoder(image)
            
            # 关键：用文本预测初始框位置
            predicted_init_boxes = text_to_position(text_features)  # [20, 4]
            
            # 扩展全局特征
            global_feat_expanded = global_features.expand(20, -1)
            
            # 提取局部特征
            local_features = query_extractor(global_feat_expanded, predicted_init_boxes)
            
            # 精修框
            refined_boxes = box_regressor(local_features)  # [20, 4]
            
            # 计算分数
            local_features_norm = local_features / (local_features.norm(dim=-1, keepdim=True) + 1e-8)
            scores = (local_features_norm * text_features_norm).sum(dim=-1)
            
            # 过滤低分
            score_threshold = 0.3
            keep = scores > score_threshold
            
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
            
            # GT也转换为xyxy像素坐标
            gt_boxes = target['boxes'].cuda()
            gt_labels = target['labels'].cuda()
            gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes) * 224
            
            all_targets.append({
                'boxes': gt_boxes_xyxy,
                'labels': gt_labels
            })
    
    # 计算mAP
    print("\n" + "=" * 70)
    print("计算mAP@50...")
    print("=" * 70)
    
    map_50 = calculate_map(all_predictions, all_targets, iou_threshold=0.5)
    
    print("\n" + "=" * 70)
    print(f"📊 最终结果")
    print("=" * 70)
    print(f"  mAP@50:    {map_50*100:.2f}%")
    print(f"  测试图片:   {len(test_dataset)}")
    print(f"  总预测框:   {sum([len(p['boxes']) for p in all_predictions])}")
    print(f"  总GT框:     {sum([len(t['boxes']) for t in all_targets])}")
    
    # 保存结果
    results = {
        'mAP@50': float(map_50),
        'num_test_images': len(test_dataset),
        'total_predictions': sum([len(p['boxes']) for p in all_predictions]),
        'total_gt': sum([len(t['boxes']) for t in all_targets]),
        'epoch': checkpoint['epoch'],
        'training_loss': checkpoint['loss']
    }
    
    Path('outputs').mkdir(exist_ok=True)
    with open('outputs/correct_version_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果保存到: outputs/correct_version_results.json")


if __name__ == '__main__':
    evaluate()

