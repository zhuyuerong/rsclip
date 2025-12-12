#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
监控改进检测器训练进度
实时显示CAM对比度、正样本比例、层权重等关键指标
"""

import re
import sys
from pathlib import Path
from datetime import datetime

def parse_log_line(line):
    """解析日志行，提取关键指标"""
    metrics = {}
    
    # 提取损失
    loss_match = re.search(r'Loss: ([\d.]+)', line)
    if loss_match:
        metrics['loss'] = float(loss_match.group(1))
    
    l1_match = re.search(r'L1: ([\d.]+)', line)
    if l1_match:
        metrics['l1'] = float(l1_match.group(1))
    
    giou_match = re.search(r'GIoU: ([\d.]+)', line)
    if giou_match:
        metrics['giou'] = float(giou_match.group(1))
    
    conf_match = re.search(r'Conf: ([\d.]+)', line)
    if conf_match:
        metrics['conf'] = float(conf_match.group(1))
    
    pos_match = re.search(r'PosRatio: ([\d.]+)', line)
    if pos_match:
        metrics['pos_ratio'] = float(pos_match.group(1))
    
    cam_contrast_match = re.search(r'CAM_Contrast: ([\d.]+)', line)
    if cam_contrast_match:
        metrics['cam_contrast'] = float(cam_contrast_match.group(1))
    
    layer_weights_match = re.search(r'LayerWeights: \[([\d.]+), ([\d.]+), ([\d.]+)\]', line)
    if layer_weights_match:
        metrics['layer_weights'] = [
            float(layer_weights_match.group(1)),
            float(layer_weights_match.group(2)),
            float(layer_weights_match.group(3))
        ]
    
    epoch_match = re.search(r'Epoch \[(\d+)/(\d+)\]', line)
    if epoch_match:
        metrics['epoch'] = int(epoch_match.group(1))
        metrics['total_epochs'] = int(epoch_match.group(2))
    
    return metrics


def monitor_training(log_file):
    """监控训练日志"""
    print("=" * 80)
    print("改进检测器训练监控")
    print("=" * 80)
    print(f"日志文件: {log_file}")
    print()
    
    if not Path(log_file).exists():
        print("⚠️  训练日志尚未生成，训练可能还在初始化...")
        return
    
    # 读取所有日志行
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    if len(lines) == 0:
        print("日志文件为空，训练可能还在初始化...")
        return
    
    # 解析所有epoch
    epochs_data = []
    for line in lines:
        if "Epoch" in line:
            metrics = parse_log_line(line)
            if metrics:
                epochs_data.append(metrics)
    
    if len(epochs_data) == 0:
        print("未找到epoch数据")
        return
    
    # 显示最新epoch
    latest = epochs_data[-1]
    print("最新训练状态:")
    print("-" * 80)
    if 'epoch' in latest:
        print(f"Epoch: {latest['epoch']}/{latest.get('total_epochs', '?')}")
    
    if 'loss' in latest:
        print(f"总损失: {latest['loss']:.4f}")
    
    if 'l1' in latest:
        print(f"L1损失: {latest['l1']:.4f}")
    
    if 'giou' in latest:
        print(f"GIoU损失: {latest['giou']:.4f}")
    
    if 'conf' in latest:
        print(f"置信度损失: {latest['conf']:.4f}")
    
    if 'pos_ratio' in latest:
        pos_ratio = latest['pos_ratio']
        print(f"正样本比例: {pos_ratio:.4f} ({pos_ratio*100:.2f}%)")
        if pos_ratio > 0.01:
            print("  ✅ 正样本比例正常 (>1%)")
        else:
            print("  ⚠️  正样本比例偏低 (<1%)")
    
    if 'cam_contrast' in latest:
        contrast = latest['cam_contrast']
        print(f"CAM对比度: {contrast:.2f}")
        if contrast > 2.0:
            print("  ✅ CAM质量良好 (>2.0)")
        elif contrast > 1.5:
            print("  ⚠️  CAM质量中等 (1.5-2.0)")
        else:
            print("  ❌ CAM质量差 (<1.5)")
    
    if 'layer_weights' in latest:
        weights = latest['layer_weights']
        print(f"层权重: Layer10={weights[0]:.3f}, Layer11={weights[1]:.3f}, Layer12={weights[2]:.3f}")
        max_idx = weights.index(max(weights))
        print(f"  最重要层: Layer{10+max_idx} (权重={weights[max_idx]:.3f})")
    
    print("-" * 80)
    
    # 显示趋势
    if len(epochs_data) > 1:
        print("\n训练趋势:")
        print("-" * 80)
        
        prev = epochs_data[-2]
        curr = epochs_data[-1]
        
        if 'loss' in curr and 'loss' in prev:
            loss_change = curr['loss'] - prev['loss']
            if loss_change < 0:
                print(f"✅ 损失下降: {abs(loss_change):.4f}")
            else:
                print(f"⚠️  损失上升: {loss_change:.4f}")
        
        if 'giou' in curr and 'giou' in prev:
            giou_change = curr['giou'] - prev['giou']
            if giou_change < 0:
                print(f"✅ GIoU损失下降: {abs(giou_change):.4f}")
            else:
                print(f"⚠️  GIoU损失上升: {giou_change:.4f}")
        
        if 'pos_ratio' in curr and 'pos_ratio' in prev:
            pos_change = curr['pos_ratio'] - prev['pos_ratio']
            if pos_change > 0:
                print(f"✅ 正样本比例增加: {pos_change:.4f}")
            else:
                print(f"⚠️  正样本比例减少: {abs(pos_change):.4f}")
        
        if 'cam_contrast' in curr and 'cam_contrast' in prev:
            contrast_change = curr['cam_contrast'] - prev['cam_contrast']
            if contrast_change > 0:
                print(f"✅ CAM对比度提升: {contrast_change:.2f}")
            else:
                print(f"⚠️  CAM对比度下降: {abs(contrast_change):.2f}")
    
    print("\n" + "=" * 80)
    print("实时查看训练日志:")
    print(f"  tail -f {log_file}")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='监控改进检测器训练')
    parser.add_argument('--log', type=str, default=None,
                       help='日志文件路径（默认自动查找最新的）')
    
    args = parser.parse_args()
    
    if args.log:
        log_file = args.log
    else:
        # 自动查找最新的日志文件
        log_dir = Path("checkpoints/improved_detector")
        if log_dir.exists():
            log_files = list(log_dir.glob("training_improved_detector_*.log"))
            if log_files:
                log_file = str(max(log_files, key=lambda p: p.stat().st_mtime))
            else:
                log_file = str(log_dir / "training_improved_detector.log")
        else:
            log_file = "checkpoints/improved_detector/training_improved_detector.log"
    
    monitor_training(log_file)


