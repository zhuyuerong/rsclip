#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控训练进度
"""

import json
import time
from pathlib import Path

def monitor():
    history_file = Path('outputs/logs/correct_train_history.json')
    
    if not history_file.exists():
        print("❌ 训练历史文件不存在")
        return
    
    with open(history_file) as f:
        history = json.load(f)
    
    if not history:
        print("❌ 训练历史为空")
        return
    
    print("=" * 80)
    print("Experiment2 正确版本 - 训练进度监控")
    print("=" * 80)
    
    latest = history[-1]
    first = history[0]
    
    print(f"\n📊 当前进度: Epoch {latest['epoch']}/50")
    print(f"  总损失:     {latest['loss']:.4f} (初始: {first['loss']:.4f}, ⬇️ {((first['loss']-latest['loss'])/first['loss']*100):.1f}%)")
    print(f"  位置损失:   {latest['position_loss']:.4f} (初始: {first['position_loss']:.4f})")
    print(f"  对比损失:   {latest['contrast_loss']:.4f} (初始: {first['contrast_loss']:.4f})")
    print(f"  精修损失:   {latest['bbox_loss']:.5f} (初始: {first['bbox_loss']:.5f})")
    print(f"  学习率:     {latest['lr']:.2e}")
    
    # 找最佳
    best_epoch = min(history, key=lambda x: x['loss'])
    print(f"\n🌟 最佳模型: Epoch {best_epoch['epoch']}, Loss: {best_epoch['loss']:.4f}")
    
    # 最近10个epochs的趋势
    if len(history) >= 10:
        recent_10 = history[-10:]
        avg_loss = sum([h['loss'] for h in recent_10]) / 10
        print(f"\n📈 最近10个epochs平均损失: {avg_loss:.4f}")
        
        if len(history) >= 20:
            prev_10 = history[-20:-10]
            prev_avg = sum([h['loss'] for h in prev_10]) / 10
            trend = "⬇️ 下降" if avg_loss < prev_avg else "⬆️ 上升"
            print(f"  趋势: {trend} (前10个epochs平均: {prev_avg:.4f})")
    
    # 预计剩余时间
    if len(history) >= 2:
        time_per_epoch = (latest['epoch'] - first['epoch']) / len(history) * 0.17  # 约0.17分钟/epoch
        remaining_epochs = 50 - latest['epoch']
        eta_minutes = remaining_epochs * time_per_epoch
        print(f"\n⏱️  预计剩余时间: {eta_minutes:.1f} 分钟")
    
    print(f"\n💾 Checkpoints保存位置: outputs/checkpoints/")
    print(f"  - correct_best_model.pth (最佳)")
    print(f"  - correct_epoch_*.pth (每10个epoch)")
    
    print(f"\n📝 训练完成后运行:")
    print(f"  python evaluate_correct_version.py")


if __name__ == '__main__':
    monitor()

