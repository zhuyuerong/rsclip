#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证VV机制的效果
对比标准CLIP Surgery和VV机制CLIP Surgery在DIOR数据集上的性能
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment4.config import Config
from experiment4.models.clip_surgery import CLIPSurgeryWrapper
from experiment4.models.clip_surgery_vv import CLIPSurgeryVVWrapper
from experiment4.data.dataset import get_dataloaders


def evaluate_model(surgery_model, val_loader, device, model_name="model"):
    """
    评估模型性能
    
    Returns:
        results: dict - 包含各项评估指标
    """
    surgery_model.model.clip_model.eval()
    
    results = {
        'total_samples': 0,
        'correct_predictions': 0,
        'patch_text_similarities': [],
        'top1_hit_rate': 0.0,
        'top5_hit_rate': 0.0,
        'iou_scores': [],
    }
    
    all_features = []
    all_labels = []
    all_bboxes = []
    
    print(f"\n评估 {model_name}...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"评估{model_name}")):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            class_names = batch['class_name']
            has_bbox = batch.get('has_bbox', None)
            bboxes = batch.get('bbox', None)
            
            # 提取特征
            patch_features = surgery_model.get_patch_features(images)  # [B, N, 512]
            cls_features = surgery_model.get_cls_features(images)  # [B, 512]
            
            # 提取文本特征
            unique_classes = list(set(class_names))
            text_features = surgery_model.encode_text(unique_classes)  # [K, 512]
            
            # 简单分类：使用CLS特征
            cls_features_norm = cls_features / (cls_features.norm(dim=-1, keepdim=True) + 1e-8)
            text_features_norm = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
            
            # 计算相似度
            similarity = cls_features_norm @ text_features_norm.t()  # [B, K]
            
            # 预测类别
            pred_indices = similarity.argmax(dim=-1)  # [B]
            pred_classes = [unique_classes[idx] for idx in pred_indices.cpu().numpy()]
            
            # 计算准确率
            for pred_cls, true_cls in zip(pred_classes, class_names):
                results['total_samples'] += 1
                if pred_cls == true_cls:
                    results['correct_predictions'] += 1
            
            # 收集数据用于进一步分析
            all_features.append({
                'patch': patch_features.cpu(),
                'cls': cls_features.cpu()
            })
            all_labels.extend(class_names)
            if bboxes is not None:
                all_bboxes.append(bboxes)
    
    # 计算准确率
    if results['total_samples'] > 0:
        results['accuracy'] = results['correct_predictions'] / results['total_samples']
    else:
        results['accuracy'] = 0.0
    
    print(f"  {model_name} 准确率: {results['accuracy']*100:.2f}%")
    print(f"  总样本数: {results['total_samples']}")
    print(f"  正确预测: {results['correct_predictions']}")
    
    return results


def main():
    """主函数"""
    print("="*70)
    print("VV机制验证：标准Surgery vs VV机制Surgery")
    print("="*70)
    
    # 初始化配置
    config = Config()
    config.use_vv_mechanism = False  # 先评估标准版本
    device = config.device
    
    # 获取数据加载器
    print("\n加载数据集...")
    try:
        train_loader, val_loader, test_loader, dataset = get_dataloaders(config)
    except Exception as e:
        print(f"⚠️ 数据加载失败: {e}")
        print("尝试使用DIOR数据集...")
        # 如果mini_dataset失败，可以在这里添加DIOR数据集的加载逻辑
        return
    
    print(f"  验证集大小: {len(val_loader.dataset)}")
    
    # 评估标准Surgery
    print("\n" + "="*70)
    print("1. 评估标准CLIP Surgery")
    print("="*70)
    config.use_vv_mechanism = False
    standard_model = CLIPSurgeryWrapper(config)
    standard_results = evaluate_model(standard_model, val_loader, device, "标准Surgery")
    
    # 评估VV机制Surgery
    print("\n" + "="*70)
    print("2. 评估VV机制CLIP Surgery")
    print("="*70)
    config.use_vv_mechanism = True
    vv_model = CLIPSurgeryVVWrapper(config, num_vv_blocks=config.num_vv_blocks)
    vv_results = evaluate_model(vv_model, val_loader, device, "VV机制Surgery")
    
    # 对比结果
    print("\n" + "="*70)
    print("结果对比")
    print("="*70)
    
    comparison = {
        'standard': standard_results,
        'vv_mechanism': vv_results,
        'improvement': {
            'accuracy_delta': vv_results['accuracy'] - standard_results['accuracy'],
            'accuracy_relative': ((vv_results['accuracy'] - standard_results['accuracy']) / 
                                  (standard_results['accuracy'] + 1e-8)) * 100
        },
        'timestamp': datetime.now().isoformat(),
        'config': {
            'num_vv_blocks': config.num_vv_blocks,
            'vv_scale_multiplier': config.vv_scale_multiplier,
            'dataset_size': len(val_loader.dataset)
        }
    }
    
    print(f"\n【准确率对比】")
    print(f"  标准Surgery:    {standard_results['accuracy']*100:.2f}%")
    print(f"  VV机制Surgery:  {vv_results['accuracy']*100:.2f}%")
    print(f"  提升:           {comparison['improvement']['accuracy_delta']*100:+.2f}% "
          f"({comparison['improvement']['accuracy_relative']:+.2f}%)")
    
    # 保存结果
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "vv_mechanism_comparison.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 对比结果已保存: {output_file}")
    
    return comparison


if __name__ == "__main__":
    comparison_results = main()

