# -*- coding: utf-8 -*-
"""
多阈值mAP评估
测试不同IoU阈值和热图阈值的组合，找到最佳配置
"""

import os
import sys
import json
import torch
from pathlib import Path
from datetime import datetime

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment4.config import Config
from experiment4.models.clip_surgery import CLIPSurgeryWrapper
from experiment4.models.clip_surgery_vv import CLIPSurgeryVVWrapper
from experiment4.data.dataset import get_dataloaders
from experiment4.evaluate_with_heatmap_map import evaluate_model_with_heatmap_custom


def evaluate_with_custom_thresholds(model, val_loader, device, model_name, 
                                    iou_thresholds, percentile_thresholds):
    """
    使用自定义阈值组合评估
    
    Returns:
        results: dict[iou_threshold][percentile] = mAP
    """
    results = {}
    
    for iou_th in iou_thresholds:
        results[iou_th] = {}
        for percentile in percentile_thresholds:
            print(f"\n  IoU={iou_th:.1f}, Percentile={percentile}%")
            
            # 这里需要修改evaluate_model_with_heatmap以支持自定义阈值
            # 暂时使用固定实现
            mAP = 0.0  # 占位
            results[iou_th][percentile] = mAP
    
    return results


def main():
    """
    多阈值评估主函数
    """
    print("="*70)
    print("多阈值mAP评估")
    print("="*70)
    
    config = Config()
    device = config.device
    
    # 加载数据
    print("\n加载数据...")
    train_loader, val_loader, test_loader, dataset = get_dataloaders(config)
    
    # 测试的阈值组合
    iou_thresholds = [0.1, 0.3, 0.5, 0.7]
    percentile_thresholds = [75, 80, 85, 90, 95]
    
    print(f"\nIoU阈值: {iou_thresholds}")
    print(f"热图阈值百分位: {percentile_thresholds}")
    
    # 评估标准Surgery
    print("\n" + "="*70)
    print("评估标准CLIP Surgery")
    print("="*70)
    
    config.use_vv_mechanism = False
    standard_model = CLIPSurgeryWrapper(config)
    
    standard_results = evaluate_with_custom_thresholds(
        standard_model, val_loader, device, "标准Surgery",
        iou_thresholds, percentile_thresholds
    )
    
    # 保存结果
    output_dir = Path(config.output_dir) / "heatmap_evaluation"
    output_file = output_dir / "multi_threshold_results.json"
    
    results = {
        'standard': standard_results,
        'config': {
            'iou_thresholds': iou_thresholds,
            'percentile_thresholds': percentile_thresholds
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ 结果已保存: {output_file}")


if __name__ == "__main__":
    main()

