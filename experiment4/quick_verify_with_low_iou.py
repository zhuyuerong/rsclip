# -*- coding: utf-8 -*-
"""
快速验证：使用低IoU阈值（0.1, 0.2, 0.3）验证框架正确性
"""

import os
import sys
import json
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment4.config import Config
from experiment4.models.clip_surgery import CLIPSurgeryWrapper
from experiment4.data.dataset import get_dataloaders
from experiment4.utils.heatmap_generator import (
    generate_similarity_heatmap,
    generate_bboxes_from_heatmap,
    compute_bbox_score
)
from experiment4.utils.map_calculator import calculate_map


def evaluate_with_iou_threshold(model, val_loader, device, iou_threshold=0.1):
    """
    使用指定IoU阈值评估
    """
    all_predictions = defaultdict(list)
    all_ground_truths = defaultdict(list)
    
    print(f"\n评估（IoU阈值={iou_threshold}）...")
    
    for batch in tqdm(val_loader, desc=f"IoU={iou_threshold}"):
        images = batch['image'].to(device)
        class_names = batch['class_name']
        gt_bboxes = batch['bbox']
        has_bbox = batch['has_bbox']
        
        # 提取特征
        image_features = model.get_all_features(images)
        
        # 获取文本特征
        unique_classes = list(set(class_names))
        text_features = model.encode_text(unique_classes)
        
        # 生成热图
        similarity_map = generate_similarity_heatmap(image_features, text_features)
        
        # 对每个样本生成检测框
        for i in range(len(images)):
            if not has_bbox[i]:
                continue
            
            class_name = class_names[i]
            class_idx = unique_classes.index(class_name)
            
            # 该类别的热图
            heatmap = similarity_map[i, :, :, class_idx].cpu().numpy()
            
            # 生成检测框
            pred_bboxes = generate_bboxes_from_heatmap(heatmap, threshold_percentile=75)
            
            # 计算每个框的分数
            for bbox in pred_bboxes:
                score = compute_bbox_score(heatmap, bbox)
                all_predictions[class_name].append({
                    'bbox': bbox,
                    'score': score
                })
            
            # 保存GT
            gt_bbox_norm = gt_bboxes[i].cpu().numpy()
            gt_bbox_pixel = [
                int(gt_bbox_norm[0] * 224),
                int(gt_bbox_norm[1] * 224),
                int(gt_bbox_norm[2] * 224),
                int(gt_bbox_norm[3] * 224)
            ]
            all_ground_truths[class_name].append(gt_bbox_pixel)
    
    # 计算mAP
    mAP, per_class_ap = calculate_map(all_predictions, all_ground_truths, iou_threshold=iou_threshold)
    
    return mAP, per_class_ap


def main():
    """测试多个IoU阈值"""
    print("="*70)
    print("多IoU阈值快速验证")
    print("="*70)
    
    config = Config()
    device = config.device
    
    # 加载数据
    print("\n加载数据...")
    train_loader, val_loader, test_loader, dataset = get_dataloaders(config)
    print(f"验证集: {len(val_loader.dataset)} 样本")
    
    # 加载模型
    print("\n加载标准Surgery模型...")
    model = CLIPSurgeryWrapper(config)
    
    # 测试多个IoU阈值
    iou_thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    
    results = {}
    
    print("\n" + "="*70)
    print("测试不同IoU阈值")
    print("="*70)
    
    for iou_th in iou_thresholds:
        mAP, per_class_ap = evaluate_with_iou_threshold(model, val_loader, device, iou_th)
        results[f'iou_{iou_th}'] = {
            'threshold': iou_th,
            'mAP': mAP,
            'per_class_ap': per_class_ap
        }
        print(f"  IoU={iou_th:.2f}: mAP={mAP:.4f}")
    
    # 打印详细表格
    print("\n" + "="*70)
    print("详细结果")
    print("="*70)
    
    # 表头
    print(f"\n{'IoU阈值':<10}", end="")
    print(f"{'mAP':<10}", end="")
    
    all_classes = sorted(list(results['iou_0.05']['per_class_ap'].keys()))
    for cls in all_classes[:5]:  # 只显示前5个类
        print(f"{cls[:10]:<12}", end="")
    print()
    print("-" * 80)
    
    # 数据行
    for iou_th in iou_thresholds:
        key = f'iou_{iou_th}'
        print(f"{iou_th:<10.2f}", end="")
        print(f"{results[key]['mAP']:<10.4f}", end="")
        
        for cls in all_classes[:5]:
            ap = results[key]['per_class_ap'].get(cls, 0.0)
            print(f"{ap:<12.4f}", end="")
        print()
    
    # 保存结果
    output_dir = Path(config.output_dir) / "heatmap_evaluation"
    output_file = output_dir / "multi_iou_results.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存: {output_file}")
    
    # 推荐
    print("\n" + "="*70)
    print("推荐配置")
    print("="*70)
    
    # 找到最高mAP的阈值
    best_iou = max(results.items(), key=lambda x: x[1]['mAP'])
    print(f"推荐IoU阈值: {best_iou[1]['threshold']:.2f}")
    print(f"对应mAP: {best_iou[1]['mAP']:.4f}")
    
    # 列出有非零AP的类别
    print(f"\n有非零AP的类别（IoU={best_iou[1]['threshold']:.2f}）:")
    for cls, ap in best_iou[1]['per_class_ap'].items():
        if ap > 0:
            print(f"  {cls}: {ap:.4f}")


if __name__ == "__main__":
    main()

