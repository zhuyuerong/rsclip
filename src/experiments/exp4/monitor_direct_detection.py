#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
监控直接检测训练进度
"""

import time
import os
from pathlib import Path

log_file = Path("checkpoints/direct_detection/training.log")

print("=" * 80)
print("直接检测训练监控")
print("=" * 80)
print(f"日志文件: {log_file}")
print()

if not log_file.exists():
    print("⚠️  训练日志尚未生成，训练可能还在初始化...")
    print("   请稍候片刻后再次运行此脚本")
else:
    # 读取最后几行
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    if len(lines) == 0:
        print("日志文件为空，训练可能还在初始化...")
    else:
        print("最新训练输出:")
        print("-" * 80)
        for line in lines[-10:]:
            print(line.rstrip())
        print("-" * 80)
        
        # 检查关键指标
        print("\n关键指标分析:")
        print("-" * 80)
        
        epoch_lines = [l for l in lines if "Epoch" in l]
        if len(epoch_lines) > 0:
            latest = epoch_lines[-1]
            print(f"最新epoch: {latest.rstrip()}")
            
            # 提取数值
            try:
                import re
                loss_match = re.search(r'Loss: ([\d.]+)', latest)
                pos_match = re.search(r'PosRatio: ([\d.]+)', latest)
                
                if loss_match:
                    loss = float(loss_match.group(1))
                    print(f"当前损失: {loss:.4f}")
                    
                    if len(epoch_lines) > 1:
                        prev_loss_match = re.search(r'Loss: ([\d.]+)', epoch_lines[-2])
                        if prev_loss_match:
                            prev_loss = float(prev_loss_match.group(1))
                            change = loss - prev_loss
                            if change < 0:
                                print(f"损失变化: {change:.4f} (下降 ✅)")
                            else:
                                print(f"损失变化: {change:.4f} (上升 ⚠️)")
                
                if pos_match:
                    pos_ratio = float(pos_match.group(1))
                    print(f"正样本比例: {pos_ratio:.4f}")
                    if pos_ratio > 0:
                        print("✅ 有正样本，训练正常")
                    else:
                        print("⚠️  无正样本，可能需要调整正样本分配策略")
            except:
                pass
        
        print(f"\n总epoch数: {len(epoch_lines)}")
        
        # 检查是否有错误
        error_lines = [l for l in lines if "Error" in l or "Traceback" in l or "Exception" in l]
        if error_lines:
            print("\n⚠️  发现错误:")
            for err in error_lines[-5:]:
                print(f"  {err.rstrip()}")

print("\n" + "=" * 80)
print("监控完成")
print("=" * 80)
print("\n实时查看训练日志:")
print(f"  tail -f {log_file}")


