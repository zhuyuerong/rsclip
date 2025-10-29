# -*- coding: utf-8 -*-
"""
测试更低的阈值（30%, 50%, 75%）和反转热图
"""

import os
import sys
import torch
from pathlib import Path
import json
from tqdm import tqdm
from collections import defaultdict

# 添加路径
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.append(str(root_dir))

from experiment4.core.config import Config
from experiment4.core.models.clip_surgery import CLIPSurgeryWrapper
from experiment4.core.data.dataset import get_dataloaders
from experiment4.core.utils.heatmap_generator import generate_similarity_heatmap, compute_bbox_score
from experiment4.core.utils.heatmap_generator_v2 import generate_bboxes_from_heatmap_v2
from experiment4.core.utils.map_calculator import calculate_map
import torch.nn.functional as F


def generate_similarity_heatmap_inverted(image_features, text_features):
    """生成反转的相似度热图"""
    patch_features = image_features[:, 1:, :]
    patch_norm = F.normalize(patch_features, dim=-1, p=2)
    text_norm = F.normalize(text_features, dim=-1, p=2)
    
    similarity = torch.einsum('bnd,kd->bnk', patch_norm, text_norm)
    similarity_inverted = -similarity  # 反转
    
    B, N, K = similarity_inverted.shape
    H = W = int(N ** 0.5)
    similarity_map = similarity_inverted.reshape(B, H, W, K)
    
    return similarity_map


def evaluate_threshold(model, val_loader, threshold_pct, use_inverted, config):
    """评估特定阈值"""
    all_predictions = defaultdict(list)
    all_ground_truths = defaultdict(list)
    
    device = config.device
    
    method_name = f"{'反转+' if use_inverted else ''}阈值{threshold_pct}%"
    
    for batch in tqdm(val_loader, desc=f"评估({method_name})", leave=False):
        images = batch['image'].to(device)
        class_names = batch['class_name']
        gt_bboxes = batch['bbox']
        has_bbox = batch['has_bbox']
        
        with torch.no_grad():
            image_features = model.get_all_features(images)
            unique_classes = list(set(class_names))
            text_features = model.encode_text(unique_classes)
        
        # 生成热图
        if use_inverted:
            similarity_map = generate_similarity_heatmap_inverted(image_features, text_features)
        else:
            similarity_map = generate_similarity_heatmap(image_features, text_features)
        
        for i in range(len(images)):
            if not has_bbox[i]:
                continue
            
            class_name = class_names[i]
            class_idx = unique_classes.index(class_name)
            heatmap = similarity_map[i, :, :, class_idx].cpu().numpy()
            
            # 使用修复的阈值方法
            pred_bboxes = generate_bboxes_from_heatmap_v2(heatmap, threshold_percentile=threshold_pct)
            
            for bbox in pred_bboxes:
                score = compute_bbox_score(heatmap, bbox)
                all_predictions[class_name].append({'bbox': bbox, 'score': score})
            
            gt_bbox_norm = gt_bboxes[i].cpu().numpy()
            gt_bbox_pixel = [
                int(gt_bbox_norm[0] * 224),
                int(gt_bbox_norm[1] * 224),
                int(gt_bbox_norm[2] * 224),
                int(gt_bbox_norm[3] * 224)
            ]
            all_ground_truths[class_name].append(gt_bbox_pixel)
    
    # 计算mAP
    mAP_005, _ = calculate_map(all_predictions, all_ground_truths, iou_threshold=0.05)
    mAP_010, _ = calculate_map(all_predictions, all_ground_truths, iou_threshold=0.10)
    mAP_020, _ = calculate_map(all_predictions, all_ground_truths, iou_threshold=0.20)
    mAP_050, per_class = calculate_map(all_predictions, all_ground_truths, iou_threshold=0.50)
    
    return {
        'method': method_name,
        'threshold': threshold_pct,
        'inverted': use_inverted,
        'mAP_005': mAP_005,
        'mAP_010': mAP_010,
        'mAP_020': mAP_020,
        'mAP_050': mAP_050,
        'per_class': per_class
    }


def main():
    """主函数"""
    print("="*70)
    print("低阈值和反转热图实验")
    print("="*70)
    
    config = Config()
    
    print("\n加载数据...")
    train_loader, val_loader, test_loader, dataset = get_dataloaders(config)
    
    print("\n加载模型...")
    model = CLIPSurgeryWrapper(config)
    
    # 测试配置
    test_configs = [
        # (阈值%, 是否反转)
        (75, False),   # 原始
        (50, False),   # 降低阈值
        (30, False),   # 更低阈值
        (75, True),    # 反转+75%
        (50, True),    # 反转+50%
        (30, True),    # 反转+30%
    ]
    
    all_results = []
    
    print(f"\n测试{len(test_configs)}种配置...")
    
    for threshold_pct, use_inverted in test_configs:
        result = evaluate_threshold(model, val_loader, threshold_pct, use_inverted, config)
        all_results.append(result)
        
        print(f"\n{result['method']}: mAP@0.05={result['mAP_005']:.4f}, mAP@0.50={result['mAP_050']:.4f}")
    
    # 汇总
    print(f"\n{'='*70}")
    print(f"完整对比")
    print(f"{'='*70}")
    
    print(f"\n{'配置':<25} {'mAP@0.05':>10} {'mAP@0.10':>10} {'mAP@0.20':>10} {'mAP@0.50':>10}")
    print(f"-"*70)
    
    for r in all_results:
        print(f"{r['method']:<25} {r['mAP_005']:>10.4f} {r['mAP_010']:>10.4f} {r['mAP_020']:>10.4f} {r['mAP_050']:>10.4f}")
    
    # 找最佳
    best_005 = max(all_results, key=lambda x: x['mAP_005'])
    best_050 = max(all_results, key=lambda x: x['mAP_050'])
    
    print(f"\n最佳配置:")
    print(f"  mAP@0.05最优: {best_005['method']} → {best_005['mAP_005']:.4f}")
    print(f"  mAP@0.50最优: {best_050['method']} → {best_050['mAP_050']:.4f}")
    
    # 保存
    output_dir = Path("experiment4/experiments/exp1_threshold_fix/outputs")
    with open(output_dir / "low_threshold_results.json", 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存: {output_dir}/low_threshold_results.json")
    
    # 关键发现
    baseline_005 = all_results[0]['mAP_005']  # 75%, 不反转
    
    print(f"\n{'='*70}")
    print(f"关键发现:")
    print(f"{'='*70}")
    
    for r in all_results[1:]:
        improvement = r['mAP_005'] - baseline_005
        if improvement > 0.05:  # 提升>5%
            print(f"✅ {r['method']}: +{improvement:.4f} (+{improvement/baseline_005*100:.1f}%)")
    
    if best_005['mAP_005'] < baseline_005 * 1.2:
        print(f"\n⚠️ 所有修复方法提升都不明显（<20%）")
        print(f"   可能的原因:")
        print(f"   1. GT区域本身相似度就很低（平均0.17 vs 背景0.20）")
        print(f"   2. 相似度分布过于均匀（0.12-0.24范围内）")
        print(f"   3. 需要更本质的特征改进（如训练模型）")


if __name__ == "__main__":
    main()
