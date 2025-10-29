# -*- coding: utf-8 -*-
"""
主评估脚本
运行完整的热图生成和mAP计算流程
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
from experiment4.evaluate_with_heatmap_map import evaluate_model_with_heatmap
from experiment4.utils.visualization import save_visualization


def main():
    """
    完整评估流程：
    1. 加载数据集
    2. 评估标准Surgery（生成1种热图）
    3. 评估VV机制Surgery（生成3种热图：QK、VV、混合）
    4. 计算mAP
    5. 保存可视化结果
    """
    print("="*70)
    print("CLIP Surgery 热图生成与mAP评估")
    print("="*70)
    
    # 初始化配置
    config = Config()
    device = config.device
    
    print(f"\n设备: {device}")
    print(f"数据集: {config.dataset_root}")
    
    # 加载数据集
    print("\n" + "="*70)
    print("加载数据集")
    print("="*70)
    
    try:
        train_loader, val_loader, test_loader, dataset = get_dataloaders(config)
        print(f"✓ 验证集大小: {len(val_loader.dataset)}")
        print(f"✓ Seen类别: {len(dataset.seen_classes)}")
        print(f"✓ Unseen类别: {len(dataset.unseen_classes)}")
    except Exception as e:
        print(f"⚠️ 数据加载失败: {e}")
        print("请确保数据集路径正确")
        return
    
    # ===== 步骤1: 评估标准Surgery =====
    print("\n" + "="*70)
    print("步骤1: 评估标准CLIP Surgery")
    print("="*70)
    
    config.use_vv_mechanism = False
    standard_model = CLIPSurgeryWrapper(config)
    
    # 标准Surgery只有一种attention（V-V self-attention）
    standard_results = evaluate_model_with_heatmap(
        standard_model, 
        val_loader, 
        device, 
        attn_type='standard',
        max_visualizations=10
    )
    
    # ===== 步骤2: 评估VV机制Surgery（三种热图）=====
    print("\n" + "="*70)
    print("步骤2: 评估VV机制CLIP Surgery")
    print("="*70)
    
    config.use_vv_mechanism = True
    vv_model = CLIPSurgeryVVWrapper(config, num_vv_blocks=6)
    
    # 三种注意力类型
    vv_results = {}
    for attn_type in ['qk', 'vv', 'mixed']:
        print(f"\n--- 生成{attn_type.upper()}热图 ---")
        vv_results[attn_type] = evaluate_model_with_heatmap(
            vv_model, 
            val_loader, 
            device, 
            attn_type=attn_type,
            max_visualizations=10
        )
    
    # ===== 步骤3: 对比结果 =====
    print("\n" + "="*70)
    print("步骤3: mAP对比结果")
    print("="*70)
    
    print(f"\n{'模型':<20} {'mAP@0.5':<10} {'样本数':<10} {'类别数':<10}")
    print("-" * 50)
    print(f"{'标准Surgery':<20} {standard_results['mAP']:<10.4f} {standard_results['num_samples']:<10} {standard_results['num_classes']:<10}")
    print(f"{'VV机制 (QK路径)':<20} {vv_results['qk']['mAP']:<10.4f} {vv_results['qk']['num_samples']:<10} {vv_results['qk']['num_classes']:<10}")
    print(f"{'VV机制 (VV路径)':<20} {vv_results['vv']['mAP']:<10.4f} {vv_results['vv']['num_samples']:<10} {vv_results['vv']['num_classes']:<10}")
    print(f"{'VV机制 (混合)':<20} {vv_results['mixed']['mAP']:<10.4f} {vv_results['mixed']['num_samples']:<10} {vv_results['mixed']['num_classes']:<10}")
    
    # 打印每个类别的AP
    print(f"\n{'类别':<20} {'标准':<10} {'VV-QK':<10} {'VV-VV':<10} {'VV-混合':<10}")
    print("-" * 70)
    
    all_classes = sorted(standard_results['per_class_ap'].keys())
    for class_name in all_classes:
        ap_std = standard_results['per_class_ap'].get(class_name, 0.0)
        ap_qk = vv_results['qk']['per_class_ap'].get(class_name, 0.0)
        ap_vv = vv_results['vv']['per_class_ap'].get(class_name, 0.0)
        ap_mixed = vv_results['mixed']['per_class_ap'].get(class_name, 0.0)
        
        print(f"{class_name:<20} {ap_std:<10.4f} {ap_qk:<10.4f} {ap_vv:<10.4f} {ap_mixed:<10.4f}")
    
    # ===== 步骤4: 保存结果 =====
    print("\n" + "="*70)
    print("步骤4: 保存结果")
    print("="*70)
    
    output_dir = Path(config.output_dir) / "heatmap_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备保存的结果（去除matplotlib figures）
    results_to_save = {
        'standard': {
            'mAP': standard_results['mAP'],
            'per_class_ap': standard_results['per_class_ap'],
            'num_samples': standard_results['num_samples'],
            'num_classes': standard_results['num_classes']
        },
        'vv_qk': {
            'mAP': vv_results['qk']['mAP'],
            'per_class_ap': vv_results['qk']['per_class_ap'],
            'num_samples': vv_results['qk']['num_samples'],
            'num_classes': vv_results['qk']['num_classes']
        },
        'vv_vv': {
            'mAP': vv_results['vv']['mAP'],
            'per_class_ap': vv_results['vv']['per_class_ap'],
            'num_samples': vv_results['vv']['num_samples'],
            'num_classes': vv_results['vv']['num_classes']
        },
        'vv_mixed': {
            'mAP': vv_results['mixed']['mAP'],
            'per_class_ap': vv_results['mixed']['per_class_ap'],
            'num_samples': vv_results['mixed']['num_samples'],
            'num_classes': vv_results['mixed']['num_classes']
        },
        'config': {
            'dataset': config.dataset_root,
            'device': str(device),
            'num_vv_blocks': config.num_vv_blocks,
            'iou_threshold': 0.5,
            'threshold_percentile': 90
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # 保存数值结果
    results_file = output_dir / "map_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    print(f"✓ 数值结果已保存: {results_file}")
    
    # 保存可视化
    for name, results in [
        ('standard', standard_results),
        ('vv_qk', vv_results['qk']),
        ('vv_vv', vv_results['vv']),
        ('vv_mixed', vv_results['mixed'])
    ]:
        vis_dir = output_dir / name
        vis_dir.mkdir(exist_ok=True)
        
        for j, fig in enumerate(results['visualizations']):
            save_path = vis_dir / f"sample_{j:03d}.png"
            save_visualization(fig, save_path)
        
        print(f"✓ {name} 可视化已保存: {vis_dir} ({len(results['visualizations'])} 张)")
    
    print(f"\n{'='*70}")
    print(f"所有结果已保存至: {output_dir}")
    print(f"{'='*70}")
    
    # 生成Markdown报告
    report_file = output_dir / "evaluation_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# CLIP Surgery 热图评估报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 配置\n\n")
        f.write(f"- 数据集: `{config.dataset_root}`\n")
        f.write(f"- 设备: `{device}`\n")
        f.write(f"- VV机制层数: {config.num_vv_blocks}\n")
        f.write(f"- IoU阈值: 0.5\n")
        f.write(f"- 热图阈值百分位: 90 (top 10%)\n\n")
        
        f.write(f"## mAP对比\n\n")
        f.write(f"| 模型 | mAP@0.5 | 样本数 | 类别数 |\n")
        f.write(f"|------|---------|--------|--------|\n")
        f.write(f"| 标准Surgery | {standard_results['mAP']:.4f} | {standard_results['num_samples']} | {standard_results['num_classes']} |\n")
        f.write(f"| VV机制 (QK路径) | {vv_results['qk']['mAP']:.4f} | {vv_results['qk']['num_samples']} | {vv_results['qk']['num_classes']} |\n")
        f.write(f"| VV机制 (VV路径) | {vv_results['vv']['mAP']:.4f} | {vv_results['vv']['num_samples']} | {vv_results['vv']['num_classes']} |\n")
        f.write(f"| VV机制 (混合) | {vv_results['mixed']['mAP']:.4f} | {vv_results['mixed']['num_samples']} | {vv_results['mixed']['num_classes']} |\n\n")
        
        f.write(f"## 每个类别的AP\n\n")
        f.write(f"| 类别 | 标准 | VV-QK | VV-VV | VV-混合 |\n")
        f.write(f"|------|------|-------|-------|--------|\n")
        
        for class_name in all_classes:
            ap_std = standard_results['per_class_ap'].get(class_name, 0.0)
            ap_qk = vv_results['qk']['per_class_ap'].get(class_name, 0.0)
            ap_vv = vv_results['vv']['per_class_ap'].get(class_name, 0.0)
            ap_mixed = vv_results['mixed']['per_class_ap'].get(class_name, 0.0)
            f.write(f"| {class_name} | {ap_std:.4f} | {ap_qk:.4f} | {ap_vv:.4f} | {ap_mixed:.4f} |\n")
        
        f.write(f"\n## 可视化示例\n\n")
        f.write(f"每种方法的可视化结果保存在对应的子目录中：\n\n")
        f.write(f"- `standard/`: 标准Surgery热图\n")
        f.write(f"- `vv_qk/`: VV机制QK路径热图\n")
        f.write(f"- `vv_vv/`: VV机制VV路径热图\n")
        f.write(f"- `vv_mixed/`: VV机制混合路径热图\n")
    
    print(f"✓ Markdown报告已保存: {report_file}")


if __name__ == "__main__":
    main()

