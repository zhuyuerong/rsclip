#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用简化版分解器训练（直接乘法）

对比CrossAttention版本，这个版本：
- 参数少~90%
- 更适合小数据集  
- 符合Surgery简洁思想
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment4.config import get_config
from experiment4.train_seen import Experiment4Trainer

# 临时替换分解器
from experiment4.models.decomposer_simple import SimplifiedTextGuidedDecomposer, SimplifiedImageOnlyDecomposer
import experiment4.models.decomposer as decomposer_module
decomposer_module.TextGuidedDecomposer = SimplifiedTextGuidedDecomposer
decomposer_module.ImageOnlyDecomposer = SimplifiedImageOnlyDecomposer


def main():
    print("=" * 70)
    print("简化版分解器训练（直接乘法）")
    print("=" * 70)
    
    config = get_config()
    config.epochs = 10
    config.batch_size = 2
    
    print(f"\n配置:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  分解器: 简化版（直接逐元素乘法）")
    
    # 创建训练器
    print("\n初始化训练器...")
    trainer = Experiment4Trainer(config)
    
    # 统计参数
    text_params = sum(p.numel() for p in trainer.text_decomposer.parameters() if p.requires_grad)
    img_params = sum(p.numel() for p in trainer.img_decomposer.parameters() if p.requires_grad)
    
    print(f"\n参数统计:")
    print(f"  Text分解器: {text_params:,} 参数")
    print(f"  Image分解器: {img_params:,} 参数")
    print(f"  总计: {(text_params + img_params):,} 参数")
    
    # 训练
    print("\n开始训练...")
    trainer.train()
    
    # 分析结果
    history = trainer.history
    
    print("\n" + "=" * 70)
    print("训练完成 - 简化版分解器")
    print("=" * 70)
    
    print(f"\nEpoch进度:")
    print(f"  {'Epoch':>6s} {'Train Acc':>12s} {'Val Acc':>12s} {'Gap':>12s}")
    print(f"  {'-'*45}")
    
    for i, h in enumerate(history, 1):
        gap = h['train_acc'] - h.get('val_acc_text', 0)
        print(f"  {i:>6d} {h['train_acc']:>11.1%} {h.get('val_acc_text', 0):>11.1%} {gap:>11.1%}")
    
    # 最终结果
    final = history[-1]
    print(f"\n最终结果 (Epoch 10):")
    print(f"  训练准确率: {final['train_acc']:.2%}")
    print(f"  验证准确率: {final.get('val_acc_text', 0):.2%}")
    print(f"  过拟合gap: {(final['train_acc'] - final.get('val_acc_text', 0)):.2%}")
    print(f"  训练Loss: {final['train_loss']:.4f}")
    print(f"  验证Loss: {final['val_loss']:.4f}")
    
    # 分析
    print(f"\n分析:")
    gap = final['train_acc'] - final.get('val_acc_text', 0)
    if gap < 0.20:
        print(f"  ✅ 过拟合较小 (gap={gap:.2%})")
    else:
        print(f"  ⚠️ 存在过拟合 (gap={gap:.2%})")
    
    if final.get('val_acc_text', 0) > 0.15:
        print(f"  ✅ 验证准确率良好 ({final.get('val_acc_text', 0):.2%})")
    else:
        print(f"  ⚠️ 验证准确率较低 ({final.get('val_acc_text', 0):.2%})")
    
    return history


if __name__ == "__main__":
    history = main()

