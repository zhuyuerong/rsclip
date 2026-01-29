#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验结果对比分析脚本

用途:
- 对比多个实验的训练曲线
- 生成对比表格和图表
- 验证实验假设

使用:
    python scripts/compare_experiments.py \
        --exp_dirs outputs/exp9_pseudo_query/a0_baseline_* \
                  outputs/exp9_pseudo_query/a2_teacher_* \
                  outputs/exp9_pseudo_query/a3_heatmap_*
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import re

import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(log_path: Path) -> Dict:
    """解析log.txt文件"""
    metrics = {
        'epoch': [],
        'train_loss': [],
        'recall_100': [],
        'map_50': [],
        'map_50_95': [],
    }
    
    if not log_path.exists():
        return metrics
    
    with open(log_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                metrics['epoch'].append(data.get('epoch', 0))
                metrics['train_loss'].append(data.get('train_loss', 0))
                # 注意: 需要根据实际log格式调整
            except:
                continue
    
    return metrics


def load_experiment(exp_dir: Path) -> Dict:
    """加载单个实验的结果"""
    exp_name = exp_dir.name
    
    # 加载config
    config_path = exp_dir / 'config.json'
    config = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # 加载log
    log_path = exp_dir / 'log.txt'
    metrics = parse_log_file(log_path)
    
    return {
        'name': exp_name,
        'dir': exp_dir,
        'config': config,
        'metrics': metrics,
    }


def compare_convergence(experiments: List[Dict], epochs=[1, 5, 10, 20, 50]):
    """对比收敛速度"""
    print("\n" + "="*70)
    print("📊 收敛速度对比")
    print("="*70)
    
    # 表头
    header = f"{'实验':<30} | " + " | ".join([f"Epoch {e:>2}" for e in epochs])
    print(header)
    print("-" * len(header))
    
    # 每个实验的数据
    for exp in experiments:
        name = exp['name'][:28]
        metrics = exp['metrics']
        
        if not metrics['recall_100']:
            print(f"{name:<30} | 无数据")
            continue
        
        # 提取指定epoch的recall
        recalls = []
        for epoch in epochs:
            if epoch <= len(metrics['recall_100']):
                recalls.append(f"{metrics['recall_100'][epoch-1]:.4f}")
            else:
                recalls.append("N/A")
        
        row = f"{name:<30} | " + " | ".join([f"{r:>9}" for r in recalls])
        print(row)


def plot_training_curves(experiments: List[Dict], output_dir: Path):
    """绘制训练曲线"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Loss曲线
    plt.figure(figsize=(12, 6))
    for exp in experiments:
        metrics = exp['metrics']
        if metrics['train_loss']:
            plt.plot(metrics['epoch'], metrics['train_loss'], 
                    label=exp['name'], marker='o', markersize=3)
    
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'loss_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存: {output_dir / 'loss_comparison.png'}")
    
    # 2. Recall曲线
    plt.figure(figsize=(12, 6))
    for exp in experiments:
        metrics = exp['metrics']
        if metrics['recall_100']:
            plt.plot(metrics['epoch'], metrics['recall_100'], 
                    label=exp['name'], marker='o', markersize=3)
    
    plt.xlabel('Epoch')
    plt.ylabel('Recall@100')
    plt.title('Recall@100 Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'recall_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存: {output_dir / 'recall_comparison.png'}")


def verify_hypotheses(experiments: List[Dict]):
    """验证实验假设"""
    print("\n" + "="*70)
    print("🔬 假设验证")
    print("="*70)
    
    # 找到各个实验
    a0 = next((e for e in experiments if 'a0' in e['name'].lower()), None)
    a2 = next((e for e in experiments if 'a2' in e['name'].lower()), None)
    a3 = next((e for e in experiments if 'a3' in e['name'].lower()), None)
    b1 = next((e for e in experiments if 'b1' in e['name'].lower()), None)
    b2 = next((e for e in experiments if 'b2' in e['name'].lower()), None)
    
    # H1: A2应该比A0更快收敛
    if a0 and a2:
        print("\n📌 H1: A2 (Teacher) 应该比 A0 (Baseline) 更快收敛")
        
        a0_recall_10 = a0['metrics']['recall_100'][9] if len(a0['metrics']['recall_100']) > 9 else 0
        a2_recall_10 = a2['metrics']['recall_100'][9] if len(a2['metrics']['recall_100']) > 9 else 0
        
        improvement = (a2_recall_10 - a0_recall_10) / (a0_recall_10 + 1e-6) * 100
        
        print(f"   A0 @ Epoch 10: Recall = {a0_recall_10:.4f}")
        print(f"   A2 @ Epoch 10: Recall = {a2_recall_10:.4f}")
        print(f"   相对提升: {improvement:+.2f}%")
        
        if improvement > 3:
            print("   ✅ 假设成立: A2显著快于A0")
        elif improvement > 0:
            print("   ⚠️  假设部分成立: A2略快于A0")
        else:
            print("   ❌ 假设不成立: A2未能超过A0")
    
    # H2: A3应该不劣于A2
    if a2 and a3:
        print("\n📌 H2: A3 (Heatmap) 应该不劣于 A2 (Teacher)")
        
        a2_final = a2['metrics']['recall_100'][-1] if a2['metrics']['recall_100'] else 0
        a3_final = a3['metrics']['recall_100'][-1] if a3['metrics']['recall_100'] else 0
        
        print(f"   A2 最终: Recall = {a2_final:.4f}")
        print(f"   A3 最终: Recall = {a3_final:.4f}")
        
        if a3_final >= a2_final * 0.95:  # 允许5%误差
            print("   ✅ 假设成立: A3不劣于A2")
        else:
            print("   ❌ 假设不成立: A3明显差于A2")
    
    # H3: B1应该明显差于A3
    if a3 and b1:
        print("\n📌 H3: B1 (Random) 应该明显差于 A3 (Heatmap)")
        
        a3_final = a3['metrics']['recall_100'][-1] if a3['metrics']['recall_100'] else 0
        b1_final = b1['metrics']['recall_100'][-1] if b1['metrics']['recall_100'] else 0
        
        print(f"   A3 最终: Recall = {a3_final:.4f}")
        print(f"   B1 最终: Recall = {b1_final:.4f}")
        
        if b1_final < a3_final * 0.9:
            print("   ✅ 假设成立: B1显著差于A3")
        else:
            print("   ❌ 假设不成立: B1未明显差于A3 (方法可能无效)")
    
    # H4: B2应该明显差于A3
    if a3 and b2:
        print("\n📌 H4: B2 (Shuffled) 应该明显差于 A3 (Heatmap)")
        
        a3_final = a3['metrics']['recall_100'][-1] if a3['metrics']['recall_100'] else 0
        b2_final = b2['metrics']['recall_100'][-1] if b2['metrics']['recall_100'] else 0
        
        print(f"   A3 最终: Recall = {a3_final:.4f}")
        print(f"   B2 最终: Recall = {b2_final:.4f}")
        
        if b2_final < a3_final * 0.9:
            print("   ✅ 假设成立: B2显著差于A3")
        else:
            print("   ❌ 假设不成立: B2未明显差于A3 (因果链可能有问题)")


def generate_report(experiments: List[Dict], output_path: Path):
    """生成Markdown报告"""
    with open(output_path, 'w') as f:
        f.write("# Exp9 Pseudo Query 实验对比报告\n\n")
        f.write(f"**生成时间**: {Path(output_path).stat().st_mtime}\n\n")
        
        f.write("## 实验列表\n\n")
        for exp in experiments:
            f.write(f"- **{exp['name']}**\n")
            f.write(f"  - 路径: `{exp['dir']}`\n")
            if exp['config']:
                f.write(f"  - Epochs: {exp['config'].get('epochs', 'N/A')}\n")
                f.write(f"  - Batch size: {exp['config'].get('batch_size', 'N/A')}\n")
            f.write("\n")
        
        f.write("## 收敛速度对比\n\n")
        f.write("| 实验 | Epoch 1 | Epoch 5 | Epoch 10 | Epoch 20 | Epoch 50 |\n")
        f.write("|------|---------|---------|----------|----------|----------|\n")
        
        for exp in experiments:
            metrics = exp['metrics']
            if metrics['recall_100']:
                row = [exp['name'][:20]]
                for epoch in [1, 5, 10, 20, 50]:
                    if epoch <= len(metrics['recall_100']):
                        row.append(f"{metrics['recall_100'][epoch-1]:.4f}")
                    else:
                        row.append("N/A")
                f.write("| " + " | ".join(row) + " |\n")
        
        f.write("\n## 图表\n\n")
        f.write("![Loss Comparison](loss_comparison.png)\n\n")
        f.write("![Recall Comparison](recall_comparison.png)\n\n")
    
    print(f"✅ 报告保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='对比多个实验结果')
    parser.add_argument('--exp_dirs', nargs='+', required=True,
                        help='实验目录列表 (支持通配符)')
    parser.add_argument('--output_dir', default='outputs/exp9_pseudo_query/comparison',
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 展开通配符
    exp_dirs = []
    for pattern in args.exp_dirs:
        exp_dirs.extend(Path('.').glob(pattern))
    
    exp_dirs = [d for d in exp_dirs if d.is_dir()]
    
    if not exp_dirs:
        print("❌ 未找到任何实验目录")
        return
    
    print(f"📁 找到 {len(exp_dirs)} 个实验目录")
    
    # 加载实验
    experiments = []
    for exp_dir in exp_dirs:
        try:
            exp = load_experiment(exp_dir)
            experiments.append(exp)
            print(f"   ✅ {exp['name']}")
        except Exception as e:
            print(f"   ❌ {exp_dir.name}: {e}")
    
    if not experiments:
        print("❌ 没有成功加载的实验")
        return
    
    # 对比分析
    compare_convergence(experiments)
    
    # 绘制曲线
    output_dir = Path(args.output_dir)
    plot_training_curves(experiments, output_dir)
    
    # 验证假设
    verify_hypotheses(experiments)
    
    # 生成报告
    generate_report(experiments, output_dir / 'report.md')
    
    print("\n" + "="*70)
    print("✅ 对比分析完成！")
    print("="*70)


if __name__ == '__main__':
    main()
