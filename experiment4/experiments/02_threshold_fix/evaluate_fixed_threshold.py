# -*- coding: utf-8 -*-
"""
实验1.1+1.2：评估修复后的阈值逻辑
对比：原始归一化 vs 修复版(原始值百分位) vs Top-k
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
from experiment4.core.utils.heatmap_generator_v2 import generate_bboxes_from_heatmap_v2, generate_bboxes_topk
from experiment4.core.utils.map_calculator import calculate_map
from experiment4.core.utils.visualization import visualize_heatmap_and_boxes, save_visualization


def evaluate_with_method(model, val_loader, bbox_method, method_name, config):
    """
    使用指定的bbox生成方法评估
    
    Args:
        bbox_method: "v1_normalized" / "v2_raw_percentile" / "v2_topk"
    """
    all_predictions = defaultdict(list)
    all_ground_truths = defaultdict(list)
    visualizations = []
    
    device = config.device
    max_visualizations = 10
    
    print(f"\n使用方法: {method_name}")
    print(f"生成热图并提取检测框...")
    
    for batch in tqdm(val_loader, desc=f"评估({method_name})"):
        images = batch['image'].to(device)
        class_names = batch['class_name']
        gt_bboxes = batch['bbox']
        has_bbox = batch['has_bbox']
        
        # 获取特征
        with torch.no_grad():
            image_features = model.get_all_features(images)
            
            # 获取文本特征
            unique_classes = list(set(class_names))
            text_features = model.encode_text(unique_classes)
        
        # 生成热图
        similarity_map = generate_similarity_heatmap(image_features, text_features)
        
        # 对每个样本
        for i in range(len(images)):
            if not has_bbox[i]:
                continue
            
            class_name = class_names[i]
            class_idx = unique_classes.index(class_name)
            
            # 热图
            heatmap = similarity_map[i, :, :, class_idx].cpu().numpy()
            
            # 根据方法生成检测框
            if bbox_method == "v1_normalized":
                # 原始方法（归一化+百分位）
                from experiment4.core.utils.heatmap_generator import generate_bboxes_from_heatmap
                pred_bboxes = generate_bboxes_from_heatmap(heatmap, threshold_percentile=75)
            elif bbox_method == "v2_raw_percentile":
                # 修复方法1（原始值百分位）
                pred_bboxes = generate_bboxes_from_heatmap_v2(heatmap, threshold_percentile=75)
            elif bbox_method == "v2_topk":
                # 修复方法2（top-k）
                pred_bboxes = generate_bboxes_topk(heatmap, top_k_ratio=0.25)  # 保留前25%
            else:
                raise ValueError(f"Unknown method: {bbox_method}")
            
            # 计算分数
            for bbox in pred_bboxes:
                score = compute_bbox_score(heatmap, bbox)
                all_predictions[class_name].append({
                    'bbox': bbox,
                    'score': score
                })
            
            # GT
            gt_bbox_norm = gt_bboxes[i].cpu().numpy()
            gt_bbox_pixel = [
                int(gt_bbox_norm[0] * 224),
                int(gt_bbox_norm[1] * 224),
                int(gt_bbox_norm[2] * 224),
                int(gt_bbox_norm[3] * 224)
            ]
            all_ground_truths[class_name].append(gt_bbox_pixel)
            
            # 可视化
            if len(visualizations) < max_visualizations:
                vis_fig = visualize_heatmap_and_boxes(
                    images[i], heatmap, pred_bboxes, gt_bbox_pixel, class_name
                )
                visualizations.append((vis_fig, class_name))
    
    # 计算mAP
    print(f"计算mAP...")
    mAP_005, per_class_ap_005 = calculate_map(all_predictions, all_ground_truths, iou_threshold=0.05)
    mAP_010, per_class_ap_010 = calculate_map(all_predictions, all_ground_truths, iou_threshold=0.10)
    mAP_020, per_class_ap_020 = calculate_map(all_predictions, all_ground_truths, iou_threshold=0.20)
    mAP_050, per_class_ap_050 = calculate_map(all_predictions, all_ground_truths, iou_threshold=0.50)
    
    print(f"评估完成:")
    print(f"  样本数: {sum(len(bboxes) for bboxes in all_ground_truths.values())}")
    print(f"  类别数: {len(all_ground_truths)}")
    print(f"  mAP@0.05: {mAP_005:.4f}")
    print(f"  mAP@0.10: {mAP_010:.4f}")
    print(f"  mAP@0.20: {mAP_020:.4f}")
    print(f"  mAP@0.50: {mAP_050:.4f}")
    
    return {
        'method': method_name,
        'mAP_005': mAP_005,
        'mAP_010': mAP_010,
        'mAP_020': mAP_020,
        'mAP_050': mAP_050,
        'per_class_ap_005': per_class_ap_005,
        'per_class_ap_010': per_class_ap_010,
        'per_class_ap_020': per_class_ap_020,
        'per_class_ap_050': per_class_ap_050,
        'visualizations': visualizations
    }


def main():
    """主函数"""
    print("="*70)
    print("阈值逻辑修复对比实验")
    print("="*70)
    
    config = Config()
    
    # 加载数据和模型
    print("\n加载数据...")
    train_loader, val_loader, test_loader, dataset = get_dataloaders(config)
    
    print("\n加载CLIP Surgery模型...")
    model = CLIPSurgeryWrapper(config)
    
    # 测试3种方法
    methods = [
        ("v1_normalized", "原始方法(归一化+百分位)"),
        ("v2_raw_percentile", "修复方法1(原始值百分位)"),
        ("v2_topk", "修复方法2(Top-25%)")
    ]
    
    all_results = {}
    
    for bbox_method, method_name in methods:
        print(f"\n{'='*70}")
        print(f"测试方法: {method_name}")
        print(f"{'='*70}")
        
        results = evaluate_with_method(model, val_loader, bbox_method, method_name, config)
        all_results[bbox_method] = results
    
    # 对比分析
    print(f"\n{'='*70}")
    print(f"对比结果")
    print(f"{'='*70}")
    
    print(f"\nmAP对比表:")
    print(f"{'方法':<30} {'mAP@0.05':>10} {'mAP@0.10':>10} {'mAP@0.20':>10} {'mAP@0.50':>10}")
    print(f"-"*70)
    
    for bbox_method, method_name in methods:
        r = all_results[bbox_method]
        print(f"{method_name:<30} {r['mAP_005']:>10.4f} {r['mAP_010']:>10.4f} {r['mAP_020']:>10.4f} {r['mAP_050']:>10.4f}")
    
    # 计算提升
    baseline = all_results['v1_normalized']
    
    print(f"\n提升幅度（相对原始方法）:")
    print(f"-"*70)
    
    for bbox_method, method_name in methods[1:]:  # 跳过baseline
        r = all_results[bbox_method]
        
        improvement_005 = r['mAP_005'] - baseline['mAP_005']
        improvement_050 = r['mAP_050'] - baseline['mAP_050']
        
        print(f"{method_name}:")
        print(f"  mAP@0.05: {improvement_005:+.4f} ({improvement_005/max(baseline['mAP_005'], 0.0001)*100:+.1f}%)")
        print(f"  mAP@0.50: {improvement_050:+.4f}")
    
    # 找出最佳方法
    best_method_005 = max(all_results.items(), key=lambda x: x[1]['mAP_005'])
    best_method_050 = max(all_results.items(), key=lambda x: x[1]['mAP_050'])
    
    print(f"\n最佳方法:")
    print(f"  mAP@0.05最优: {dict(methods)[best_method_005[0]]} ({best_method_005[1]['mAP_005']:.4f})")
    print(f"  mAP@0.50最优: {dict(methods)[best_method_050[0]]} ({best_method_050[1]['mAP_050']:.4f})")
    
    # 保存结果
    output_dir = Path("experiment4/experiments/exp1_threshold_fix/outputs")
    
    # 保存数值
    results_to_save = {
        method: {
            'mAP_005': r['mAP_005'],
            'mAP_010': r['mAP_010'],
            'mAP_020': r['mAP_020'],
            'mAP_050': r['mAP_050'],
            'per_class_ap_005': r['per_class_ap_005'],
            'per_class_ap_050': r['per_class_ap_050']
        }
        for method, r in all_results.items()
    }
    
    with open(output_dir / "threshold_fix_comparison.json", 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    
    # 保存可视化
    for bbox_method, method_name in methods:
        vis_dir = output_dir / bbox_method.replace('_', '-')
        vis_dir.mkdir(exist_ok=True)
        
        for idx, (fig, class_name) in enumerate(all_results[bbox_method]['visualizations']):
            save_path = vis_dir / f"sample_{idx:03d}_{class_name}.png"
            save_visualization(fig, save_path)
    
    print(f"\n✅ 结果已保存: {output_dir}")
    print(f"   数值: threshold_fix_comparison.json")
    print(f"   可视化: v1-normalized/, v2-raw-percentile/, v2-topk/")
    
    # 结论
    print(f"\n{'='*70}")
    print(f"结论:")
    print(f"{'='*70}")
    
    if best_method_005[1]['mAP_005'] > baseline['mAP_005'] * 1.1:
        print(f"✅ 阈值修复显著改善了mAP！")
        print(f"   推荐使用: {dict(methods)[best_method_005[0]]}")
    else:
        print(f"⚠️ 阈值修复未显著改善")
        print(f"   需要进一步分析其他因素")


if __name__ == "__main__":
    main()

