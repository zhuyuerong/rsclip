#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 mini_dataset 上运行所有实验并生成性能报告

功能：
1. 运行 Experiment1 (两阶段检测)
2. 运行 Experiment2 (上下文引导检测) 
3. 运行 Experiment3 (OVA-DETR)
4. 收集性能指标
5. 生成对比报告
"""

import torch
import time
import json
from pathlib import Path
import numpy as np
from collections import defaultdict
import sys

# 添加路径
sys.path.append('experiment1')
sys.path.append('experiment2')
sys.path.append('experiment3')


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def end(self):
        """结束计时"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - self.start_time
        return elapsed
    
    def get_model_params(self, model):
        """获取模型参数量"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': total_params - trainable_params
        }
    
    def get_gpu_memory(self):
        """获取GPU内存使用"""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
                'reserved': torch.cuda.memory_reserved() / 1024**2,
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**2
            }
        return {}


def evaluate_experiment3_mini():
    """在 mini_dataset 上评估 Experiment3"""
    
    print("=" * 70)
    print("Experiment3: OVA-DETR 评估")
    print("=" * 70)
    
    from experiment3.config.default_config import DefaultConfig
    from experiment3.models.ova_detr import OVADETR
    from experiment3.utils.data_loader import create_data_loader, DIOR_CLASSES
    from experiment3.utils.transforms import get_transforms
    
    monitor = PerformanceMonitor()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 配置
    config = DefaultConfig()
    config.batch_size = 4
    config.image_size = (800, 800)
    
    # 创建模型
    print("\n创建模型...")
    model = OVADETR(config).to(device)
    model.eval()
    
    # 模型参数
    model_params = monitor.get_model_params(model)
    
    print(f"\n模型参数:")
    print(f"  总参数: {model_params['total_params']:,}")
    print(f"  可训练: {model_params['trainable_params']:,}")
    print(f"  冻结: {model_params['frozen_params']:,}")
    
    # 提取文本特征
    print("\n提取文本特征...")
    with torch.no_grad():
        text_features = model.backbone.forward_text(DIOR_CLASSES).to(device)
    
    # 创建数据加载器
    print("\n加载 mini_dataset...")
    val_transforms = get_transforms(mode='val', image_size=config.image_size)
    
    # 使用 mini_dataset
    from experiment3.utils.data_loader import DiorDataset
    
    dataset = DiorDataset(
        root_dir='datasets/mini_dataset',
        split='train',  # mini_dataset 只有train分割
        transforms=val_transforms
    )
    
    from torch.utils.data import DataLoader
    from experiment3.utils.data_loader import collate_fn
    
    data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 推理
    print("\n开始推理...")
    monitor.start()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            
            # 前向传播
            outputs = model(images, text_features)
            
            # 使用最后一层
            pred_logits = outputs['pred_logits'][-1]
            pred_boxes = outputs['pred_boxes'][-1]
            
            # 处理每张图
            for i in range(pred_logits.shape[0]):
                logits = pred_logits[i]
                boxes = pred_boxes[i]
                
                scores = logits.sigmoid()
                max_scores, labels = scores.max(dim=-1)
                
                # 过滤
                keep = max_scores > 0.3
                
                all_predictions.append({
                    'boxes': boxes[keep].cpu(),
                    'scores': max_scores[keep].cpu(),
                    'labels': labels[keep].cpu()
                })
                
                all_targets.append({
                    'boxes': targets[i]['boxes'].cpu(),
                    'labels': targets[i]['labels'].cpu()
                })
    
    inference_time = monitor.end()
    
    # 计算指标
    print("\n计算性能指标...")
    from experiment3.evaluate import evaluate_detections
    from experiment3.losses.bbox_loss import box_cxcywh_to_xyxy
    
    # 转换坐标
    for i, (pred, target) in enumerate(zip(all_predictions, all_targets)):
        if len(pred['boxes']) > 0:
            pred['boxes'] = box_cxcywh_to_xyxy(pred['boxes']) * 800
        if len(target['boxes']) > 0:
            target['boxes'] = box_cxcywh_to_xyxy(target['boxes']) * 800
    
    metrics = evaluate_detections(all_predictions, all_targets, len(DIOR_CLASSES))
    
    # GPU内存
    gpu_memory = monitor.get_gpu_memory()
    
    return {
        'experiment': 'Experiment3',
        'model': 'OVA-DETR',
        'params': model_params,
        'inference_time': inference_time,
        'fps': len(dataset) / inference_time,
        'metrics': metrics,
        'gpu_memory': gpu_memory
    }


def create_experiment1_evaluator():
    """为 Experiment1 创建评估脚本"""
    
    print("=" * 70)
    print("Experiment1: 两阶段检测 评估")  
    print("=" * 70)
    
    # Experiment1 主要是基于检索和区域的方法
    # 需要针对性地创建评估流程
    
    print("\n⚠️ Experiment1 需要创建专门的评估脚本")
    print("   Experiment1 使用两阶段方法：")
    print("   - Stage1: 提议生成 + 分类")
    print("   - Stage2: 目标检测 + 边界框细化")
    print("   需要创建适配 mini_dataset 的评估流程")
    
    return None


def create_experiment2_complete():
    """为 Experiment2 创建完整系统"""
    
    print("=" * 70)
    print("Experiment2: 上下文引导检测 评估")
    print("=" * 70)
    
    print("\n⚠️ Experiment2 缺少以下组件：")
    print("   ❌ 数据加载器")
    print("   ❌ 训练脚本")
    print("   ❌ 评估脚本")
    print("   需要补充完整系统后才能运行")
    
    return None


def generate_comparison_report(results):
    """生成对比报告"""
    
    report = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': 'mini_dataset',
        'dataset_size': 100,
        'experiments': results
    }
    
    # 保存报告
    report_path = Path('experiment_comparison_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 报告保存到: {report_path}")
    
    return report


def print_comparison_table(results):
    """打印对比表格"""
    
    print("\n" + "=" * 70)
    print("实验性能对比")
    print("=" * 70)
    
    # 表头
    print(f"\n{'实验':<15} {'模型':<15} {'参数量':<15} {'推理时间':<12} {'FPS':<8} {'mAP':<8}")
    print("-" * 70)
    
    for result in results:
        if result is None:
            continue
        
        exp_name = result['experiment']
        model_name = result['model']
        total_params = result['params']['total_params'] / 1e6  # M
        inference_time = result['inference_time']
        fps = result['fps']
        mAP = result['metrics'].get('mAP', 0.0)
        
        print(f"{exp_name:<15} {model_name:<15} {total_params:>8.2f}M {inference_time:>8.2f}s {fps:>6.2f} {mAP:>6.4f}")


def main():
    """主函数"""
    
    print("\n" + "🎯" * 35)
    print("在 Mini Dataset 上运行所有实验")
    print("🎯" * 35 + "\n")
    
    results = []
    
    # Experiment1
    print("\n" + "▶" * 35)
    result1 = create_experiment1_evaluator()
    if result1:
        results.append(result1)
    
    # Experiment2  
    print("\n" + "▶" * 35)
    result2 = create_experiment2_complete()
    if result2:
        results.append(result2)
    
    # Experiment3
    print("\n" + "▶" * 35)
    result3 = evaluate_experiment3_mini()
    if result3:
        results.append(result3)
    
    # 生成报告
    if len(results) > 0:
        print_comparison_table(results)
        generate_comparison_report(results)
    
    print("\n" + "=" * 70)
    print("评估完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()

