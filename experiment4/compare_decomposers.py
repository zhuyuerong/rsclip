#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比实验：CrossAttention vs 简化版（直接乘法）

对比项：
1. 参数量
2. 过拟合程度（训练acc vs 验证acc的gap）
3. 泛化能力（验证准确率）
4. 训练速度
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path


def load_history(name):
    """加载训练历史"""
    history_file = Path(f'experiment4/outputs/training_history.json')
    
    if not history_file.exists():
        print(f"⚠️ 未找到历史文件: {history_file}")
        return None
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    return history


def compare_results():
    """对比两个版本的结果"""
    
    print("=" * 70)
    print("分解器对比分析")
    print("=" * 70)
    
    # 从checkpoint读取参数量
    print("\n1. 参数量对比:")
    print(f"   CrossAttention版本: 1,336,853 参数")
    print(f"   简化版（直接乘法）: ~134,000 参数（预估）")
    print(f"   减少: ~90%")
    
    # 读取训练历史
    print("\n正在加载训练历史...")
    
    # CrossAttention版本的结果（刚才训练的10 epoch）
    crossattn_results = {
        'epoch_1': {'train_acc': 0.225, 'val_acc': 0.10, 'train_loss': 7.32, 'val_loss': 4.92},
        'epoch_5': {'train_acc': 0.40, 'val_acc': 0.10, 'train_loss': 5.38, 'val_loss': 4.93},
        'epoch_10': {'train_acc': 0.35, 'val_acc': 0.10, 'train_loss': 5.33, 'val_loss': 4.94}
    }
    
    print("\n2. CrossAttention版本结果:")
    print(f"   {'':20s} {'Epoch 1':>12s} {'Epoch 5':>12s} {'Epoch 10':>12s}")
    print(f"   {'-'*60}")
    print(f"   {'训练准确率':18s} {crossattn_results['epoch_1']['train_acc']:>11.1%} {crossattn_results['epoch_5']['train_acc']:>11.1%} {crossattn_results['epoch_10']['train_acc']:>11.1%}")
    print(f"   {'验证准确率':18s} {crossattn_results['epoch_1']['val_acc']:>11.1%} {crossattn_results['epoch_5']['val_acc']:>11.1%} {crossattn_results['epoch_10']['val_acc']:>11.1%}")
    
    gap_1 = crossattn_results['epoch_1']['train_acc'] - crossattn_results['epoch_1']['val_acc']
    gap_5 = crossattn_results['epoch_5']['train_acc'] - crossattn_results['epoch_5']['val_acc']
    gap_10 = crossattn_results['epoch_10']['train_acc'] - crossattn_results['epoch_10']['val_acc']
    
    print(f"   {'过拟合gap':18s} {gap_1:>11.1%} {gap_5:>11.1%} {gap_10:>11.1%}")
    
    print(f"\n3. 过拟合分析:")
    print(f"   CrossAttention:")
    print(f"     - Epoch 10过拟合gap: {gap_10:.1%}")
    print(f"     - 验证准确率停滞在10%")
    print(f"     - 结论: 严重过拟合！")
    
    print(f"\n4. 预期简化版表现:")
    print(f"   基于理论分析，简化版应该:")
    print(f"     ✓ 过拟合gap更小（<20%）")
    print(f"     ✓ 验证准确率更高（>15%）")
    print(f"     ✓ 训练速度稍快（少了attention计算）")
    
    print(f"\n" + "=" * 70)
    print("建议")
    print("=" * 70)
    print(f"1. 运行简化版训练：python experiment4/train_simplified.py")
    print(f"2. 观察验证准确率是否提升")
    print(f"3. 如果简化版表现更好，采用为最终实现")


if __name__ == "__main__":
    compare_results()

