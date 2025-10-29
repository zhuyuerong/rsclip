# -*- coding: utf-8 -*-
"""
热图质量分析工具
深入分析为什么mAP为0，以及如何改进
"""

import os
import sys
import json
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment4.config import Config
from experiment4.models.clip_surgery import CLIPSurgeryWrapper
from experiment4.data.dataset import get_dataloaders
from experiment4.utils.heatmap_generator import generate_similarity_heatmap, generate_bboxes_from_heatmap
from experiment4.utils.map_calculator import calculate_iou


def analyze_single_sample(model, image, class_name, gt_bbox, device):
    """
    分析单个样本的热图质量
    
    Returns:
        analysis: dict包含详细分析结果
    """
    # 提取特征
    image = image.unsqueeze(0).to(device)
    image_features = model.get_all_features(image)  # [1, 50, 512]
    text_features = model.encode_text([class_name])  # [1, 512]
    
    # 生成热图
    similarity_map = generate_similarity_heatmap(image_features, text_features)  # [1, 7, 7, 1]
    heatmap = similarity_map[0, :, :, 0].cpu().numpy()  # [7, 7]
    
    # 上采样到224x224
    heatmap_224 = cv2.resize(heatmap.astype(np.float32), (224, 224), interpolation=cv2.INTER_LINEAR)
    
    # 归一化
    heatmap_norm = (heatmap_224 - heatmap_224.min()) / (heatmap_224.max() - heatmap_224.min() + 1e-8)
    
    # 分析不同阈值下的结果
    analysis = {
        'class_name': class_name,
        'heatmap_stats': {
            'min': float(heatmap.min()),
            'max': float(heatmap.max()),
            'mean': float(heatmap.mean()),
            'std': float(heatmap.std())
        },
        'gt_bbox': gt_bbox.tolist() if isinstance(gt_bbox, torch.Tensor) else gt_bbox,
        'threshold_analysis': {}
    }
    
    # GT框（像素坐标）
    gt_bbox_pixel = [
        int(gt_bbox[0] * 224),
        int(gt_bbox[1] * 224),
        int(gt_bbox[2] * 224),
        int(gt_bbox[3] * 224)
    ]
    
    # 测试不同阈值
    for percentile in [75, 80, 85, 90, 95]:
        threshold = np.percentile(heatmap_norm, percentile)
        mask = (heatmap_norm >= threshold).astype(np.uint8) * 255
        
        # 连通域分析
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 提取框
        pred_bboxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 5 and h > 5:
                pred_bboxes.append([x, y, x + w, y + h])
        
        # 计算最佳IoU
        best_iou = 0
        if len(pred_bboxes) > 0:
            ious = [calculate_iou(bbox, gt_bbox_pixel) for bbox in pred_bboxes]
            best_iou = max(ious)
        
        analysis['threshold_analysis'][f'p{percentile}'] = {
            'threshold_value': float(threshold),
            'num_boxes': len(pred_bboxes),
            'best_iou': float(best_iou),
            'activated_ratio': float((heatmap_norm >= threshold).mean())
        }
    
    # GT区域内的热图响应
    x_min, y_min, x_max, y_max = gt_bbox_pixel
    x_min = max(0, min(x_min, 223))
    y_min = max(0, min(y_min, 223))
    x_max = max(1, min(x_max, 224))
    y_max = max(1, min(y_max, 224))
    
    gt_region_response = heatmap_224[y_min:y_max, x_min:x_max]
    if gt_region_response.size > 0:
        analysis['gt_region'] = {
            'mean_response': float(gt_region_response.mean()),
            'max_response': float(gt_region_response.max()),
            'response_percentile': float(np.percentile(heatmap_norm.flatten(), 
                                                       100 * (heatmap_norm < gt_region_response.mean()).mean()))
        }
    
    return analysis


def main():
    """主函数：分析热图质量"""
    print("="*70)
    print("热图质量分析")
    print("="*70)
    
    config = Config()
    device = config.device
    
    # 加载数据
    print("\n加载数据...")
    train_loader, val_loader, test_loader, dataset = get_dataloaders(config)
    print(f"验证集: {len(val_loader.dataset)} 样本")
    
    # 加载模型
    print("\n加载标准CLIP Surgery...")
    model = CLIPSurgeryWrapper(config)
    
    # 分析样本
    print("\n分析样本热图质量...")
    all_analyses = []
    
    sample_count = 0
    max_samples = 20  # 分析前20个样本
    
    for batch in tqdm(val_loader, desc="分析"):
        images = batch['image']
        class_names = batch['class_name']
        bboxes = batch['bbox']
        has_bbox = batch['has_bbox']
        
        for i in range(len(images)):
            if not has_bbox[i]:
                continue
            
            analysis = analyze_single_sample(
                model, 
                images[i], 
                class_names[i], 
                bboxes[i], 
                device
            )
            all_analyses.append(analysis)
            
            sample_count += 1
            if sample_count >= max_samples:
                break
        
        if sample_count >= max_samples:
            break
    
    # 汇总分析
    print("\n" + "="*70)
    print("分析结果汇总")
    print("="*70)
    
    # 统计各阈值下的平均IoU
    threshold_summary = {}
    for percentile in [75, 80, 85, 90, 95]:
        key = f'p{percentile}'
        ious = [a['threshold_analysis'][key]['best_iou'] for a in all_analyses]
        num_boxes = [a['threshold_analysis'][key]['num_boxes'] for a in all_analyses]
        activated_ratios = [a['threshold_analysis'][key]['activated_ratio'] for a in all_analyses]
        
        threshold_summary[key] = {
            'percentile': percentile,
            'avg_iou': float(np.mean(ious)),
            'max_iou': float(np.max(ious)),
            'avg_num_boxes': float(np.mean(num_boxes)),
            'avg_activated_ratio': float(np.mean(activated_ratios)),
            'samples_with_iou_gt_0.5': sum(1 for iou in ious if iou >= 0.5)
        }
    
    print(f"\n{'阈值':<10} {'平均IoU':<12} {'最大IoU':<12} {'平均框数':<12} {'激活比例':<12} {'IoU≥0.5样本数':<15}")
    print("-" * 80)
    for key, stats in threshold_summary.items():
        print(f"{stats['percentile']}%ile    "
              f"{stats['avg_iou']:<12.4f} "
              f"{stats['max_iou']:<12.4f} "
              f"{stats['avg_num_boxes']:<12.2f} "
              f"{stats['avg_activated_ratio']:<12.2%} "
              f"{stats['samples_with_iou_gt_0.5']:<15}")
    
    # GT区域响应分析
    print(f"\n{'='*70}")
    print("GT区域热图响应分析")
    print(f"{'='*70}")
    
    gt_responses = [a['gt_region']['mean_response'] for a in all_analyses if 'gt_region' in a]
    gt_max_responses = [a['gt_region']['max_response'] for a in all_analyses if 'gt_region' in a]
    
    print(f"GT区域平均响应: {np.mean(gt_responses):.4f} ± {np.std(gt_responses):.4f}")
    print(f"GT区域最大响应: {np.mean(gt_max_responses):.4f} ± {np.std(gt_max_responses):.4f}")
    
    # 保存分析结果
    output_dir = Path(config.output_dir) / "heatmap_evaluation"
    output_file = output_dir / "heatmap_quality_analysis.json"
    
    results = {
        'threshold_summary': threshold_summary,
        'per_sample_analysis': all_analyses,
        'summary_stats': {
            'num_samples': len(all_analyses),
            'gt_mean_response_avg': float(np.mean(gt_responses)),
            'gt_mean_response_std': float(np.std(gt_responses))
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 分析结果已保存: {output_file}")
    
    # 推荐最佳阈值
    print(f"\n{'='*70}")
    print("推荐配置")
    print(f"{'='*70}")
    
    best_threshold = max(threshold_summary.items(), key=lambda x: x[1]['avg_iou'])
    print(f"推荐阈值: {best_threshold[1]['percentile']}% (平均IoU: {best_threshold[1]['avg_iou']:.4f})")
    print(f"该阈值下有 {best_threshold[1]['samples_with_iou_gt_0.5']}/{len(all_analyses)} 样本的IoU≥0.5")


if __name__ == "__main__":
    main()

