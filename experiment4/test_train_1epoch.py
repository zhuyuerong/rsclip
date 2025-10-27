#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最小训练测试 - 只跑1个epoch验证流程
"""
import os
import sys

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from experiment4.config import get_config
from experiment4.train_seen import Experiment4Trainer

def main():
    print("="*60)
    print("最小训练测试 - 1 Epoch")
    print("="*60)
    
    # 获取配置并修改为最小设置
    config = get_config()
    config.epochs = 1          # 只训练1个epoch
    config.batch_size = 2      # 小batch size（避免内存问题）
    config.num_workers = 0     # 避免多进程问题
    config.save_freq = 1       # 每次都保存
    config.eval_freq = 1       # 每次都评估
    
    print(f"\n配置:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Device: {config.device}")
    print(f"  Dataset: {config.dataset_root}")
    
    try:
        # 创建训练器
        print("\n初始化训练器...")
        trainer = Experiment4Trainer(config)
        print("✓ 训练器初始化成功！")
        
        # 开始训练
        print("\n开始训练...")
        trainer.train()
        
        print("\n" + "="*60)
        print("✓✓✓ 训练测试成功！实验4可以正常运行！")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())

