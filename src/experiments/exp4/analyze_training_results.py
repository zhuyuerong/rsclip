#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析训练结果，评估是否需要继续训练
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def parse_log(log_file):
    """解析训练日志"""
    epochs = []
    losses = []
    l1_losses = []
    giou_losses = []
    conf_losses = []
    pos_ratios = []
    cam_contrasts = []
    layer_weights = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            
            # 解析epoch信息
            epoch_match = re.search(r'Epoch \[(\d+)/(\d+)\]', line)
            if not epoch_match:
                continue
            
            epoch = int(epoch_match.group(1))
            epochs.append(epoch)
            
            # 解析损失
            loss_match = re.search(r'Loss: ([\d.]+)', line)
            if loss_match:
                losses.append(float(loss_match.group(1)))
            
            l1_match = re.search(r'L1: ([\d.]+)', line)
            if l1_match:
                l1_losses.append(float(l1_match.group(1)))
            
            giou_match = re.search(r'GIoU: ([\d.]+)', line)
            if giou_match:
                giou_losses.append(float(giou_match.group(1)))
            
            conf_match = re.search(r'Conf: ([\d.]+)', line)
            if conf_match:
                conf_losses.append(float(conf_match.group(1)))
            
            pos_match = re.search(r'PosRatio: ([\d.]+)', line)
            if pos_match:
                pos_ratios.append(float(pos_match.group(1)))
            
            cam_match = re.search(r'CAM_Contrast: ([\d.]+)', line)
            if cam_match:
                cam_contrasts.append(float(cam_match.group(1)))
            else:
                cam_contrasts.append(None)
            
            weights_match = re.search(r'LayerWeights: \[([\d.]+), ([\d.]+), ([\d.]+)\]', line)
            if weights_match:
                layer_weights.append([
                    float(weights_match.group(1)),
                    float(weights_match.group(2)),
                    float(weights_match.group(3))
                ])
    
    return {
        'epochs': epochs,
        'losses': losses,
        'l1_losses': l1_losses,
        'giou_losses': giou_losses,
        'conf_losses': conf_losses,
        'pos_ratios': pos_ratios,
        'cam_contrasts': cam_contrasts,
        'layer_weights': layer_weights
    }

def analyze_convergence(data):
    """分析收敛情况"""
    epochs = data['epochs']
    losses = data['losses']
    giou_losses = data['giou_losses']
    
    # 计算最后10个epoch的平均损失
    if len(losses) >= 10:
        recent_losses = losses[-10:]
        recent_giou = giou_losses[-10:]
        
        avg_recent_loss = np.mean(recent_losses)
        avg_recent_giou = np.mean(recent_giou)
        
        # 计算损失变化率
        loss_change = (losses[-1] - losses[-10]) / losses[-10] * 100
        giou_change = (giou_losses[-1] - giou_losses[-10]) / giou_losses[-10] * 100
        
        print("=" * 80)
        print("训练收敛分析")
        print("=" * 80)
        print(f"\n总epoch数: {len(epochs)}")
        print(f"初始损失: {losses[0]:.4f}")
        print(f"最终损失: {losses[-1]:.4f}")
        print(f"损失下降: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
        print(f"\n最后10个epoch平均损失: {avg_recent_loss:.4f}")
        print(f"最后10个epoch损失变化: {loss_change:.2f}%")
        
        print(f"\n初始GIoU损失: {giou_losses[0]:.4f}")
        print(f"最终GIoU损失: {giou_losses[-1]:.4f}")
        print(f"GIoU损失下降: {(giou_losses[0] - giou_losses[-1]) / giou_losses[0] * 100:.1f}%")
        print(f"最后10个epoch GIoU变化: {giou_change:.2f}%")
        
        # 判断是否收敛
        print("\n" + "=" * 80)
        print("收敛判断")
        print("=" * 80)
        
        is_converged = abs(loss_change) < 2.0  # 损失变化<2%认为收敛
        print(f"损失变化率: {loss_change:.2f}%")
        if is_converged:
            print("✅ 损失已基本收敛（变化<2%）")
        else:
            print("⚠️  损失仍在变化（变化>2%），可能还有收敛空间")
        
        # 评估关键指标
        print("\n" + "=" * 80)
        print("关键指标评估")
        print("=" * 80)
        
        # GIoU损失
        print(f"\n1. GIoU损失: {giou_losses[-1]:.4f}")
        if giou_losses[-1] < 0.3:
            print("   ✅ 达到目标（<0.3）")
        elif giou_losses[-1] < 0.5:
            print("   ⚠️  接近目标（0.3-0.5），可能需要继续训练")
        else:
            print("   ❌ 未达到目标（>0.5），建议继续训练")
        
        # 正样本比例
        pos_ratios = data['pos_ratios']
        avg_pos_ratio = np.mean(pos_ratios[-10:]) if len(pos_ratios) >= 10 else pos_ratios[-1]
        print(f"\n2. 正样本比例: {avg_pos_ratio:.4f} ({avg_pos_ratio*100:.2f}%)")
        if avg_pos_ratio > 0.01:
            print("   ✅ 正常（>1%）")
        elif avg_pos_ratio > 0.005:
            print("   ⚠️  偏低（0.5%-1%），可能影响训练")
        else:
            print("   ❌ 过低（<0.5%），匹配策略可能有问题")
        
        # CAM对比度
        cam_contrasts = [c for c in data['cam_contrasts'] if c is not None]
        if cam_contrasts:
            recent_cam = cam_contrasts[-10:] if len(cam_contrasts) >= 10 else cam_contrasts
            avg_cam_contrast = np.mean(recent_cam)
            std_cam_contrast = np.std(recent_cam)
            print(f"\n3. CAM对比度: 平均={avg_cam_contrast:.2f}, 标准差={std_cam_contrast:.2f}")
            if avg_cam_contrast > 2.0 and std_cam_contrast < 5.0:
                print("   ✅ 质量良好且稳定")
            elif avg_cam_contrast > 2.0:
                print("   ⚠️  质量良好但不稳定（波动大）")
            elif avg_cam_contrast > 1.5:
                print("   ⚠️  质量中等")
            else:
                print("   ❌ 质量差（<1.5）")
        
        # 层权重
        layer_weights = data['layer_weights']
        if layer_weights:
            recent_weights = layer_weights[-10:] if len(layer_weights) >= 10 else layer_weights
            avg_weights = np.mean(recent_weights, axis=0)
            print(f"\n4. 层权重: Layer10={avg_weights[0]:.3f}, Layer11={avg_weights[1]:.3f}, Layer12={avg_weights[2]:.3f}")
            max_idx = np.argmax(avg_weights)
            print(f"   最重要层: Layer{10+max_idx} (权重={avg_weights[max_idx]:.3f})")
            if np.max(avg_weights) - np.min(avg_weights) > 0.1:
                print("   ✅ 层权重有差异，说明学习到了不同层的重要性")
            else:
                print("   ⚠️  层权重差异小，可能未充分利用多层信息")
        
        # 建议
        print("\n" + "=" * 80)
        print("训练建议")
        print("=" * 80)
        
        recommendations = []
        
        if not is_converged:
            recommendations.append("✅ 损失仍在下降，建议继续训练50-100个epochs")
        
        if giou_losses[-1] > 0.5:
            recommendations.append("⚠️  GIoU损失仍然较高（>0.5），建议继续训练")
        
        if avg_pos_ratio < 0.005:
            recommendations.append("❌ 正样本比例过低，建议先检查匹配策略，再决定是否继续训练")
        
        if cam_contrasts and np.std([c for c in cam_contrasts[-10:] if c is not None]) > 5.0:
            recommendations.append("⚠️  CAM对比度波动大，建议检查CAM生成质量")
        
        if not recommendations:
            recommendations.append("✅ 训练基本收敛，建议先进行推理评估，再决定是否继续训练")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        print("\n" + "=" * 80)
        print("推荐方案")
        print("=" * 80)
        print("\n方案1: 先推理评估（推荐）")
        print("  - 评估当前模型在验证集上的mAP")
        print("  - 可视化检测结果和CAM")
        print("  - 根据实际检测效果决定是否继续训练")
        print("\n方案2: 继续训练50-100个epochs")
        print("  - 如果损失仍在下降且GIoU>0.5，可以继续训练")
        print("  - 建议降低学习率（当前LR已很小：1e-6）")
        print("  - 监控正样本比例和CAM对比度")
        
        return {
            'is_converged': is_converged,
            'final_loss': losses[-1],
            'final_giou': giou_losses[-1],
            'avg_pos_ratio': avg_pos_ratio,
            'recommendations': recommendations
        }

def plot_training_curves(data, output_dir):
    """绘制训练曲线"""
    epochs = data['epochs']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 总损失
    axes[0, 0].plot(epochs, data['losses'], 'b-', label='Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    # GIoU损失
    axes[0, 1].plot(epochs, data['giou_losses'], 'r-', label='GIoU Loss')
    axes[0, 1].axhline(y=0.3, color='g', linestyle='--', label='Target (0.3)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('GIoU Loss')
    axes[0, 1].set_title('GIoU Loss')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    
    # 正样本比例
    axes[1, 0].plot(epochs, data['pos_ratios'], 'g-', label='Pos Ratio')
    axes[1, 0].axhline(y=0.01, color='r', linestyle='--', label='Target (1%)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Positive Ratio')
    axes[1, 0].set_title('Positive Sample Ratio')
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    
    # CAM对比度
    cam_contrasts = data['cam_contrasts']
    valid_epochs = [e for e, c in zip(epochs, cam_contrasts) if c is not None]
    valid_contrasts = [c for c in cam_contrasts if c is not None]
    if valid_contrasts:
        axes[1, 1].plot(valid_epochs, valid_contrasts, 'm-', label='CAM Contrast', marker='o')
        axes[1, 1].axhline(y=2.0, color='g', linestyle='--', label='Target (2.0)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('CAM Contrast')
        axes[1, 1].set_title('CAM Contrast')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'training_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n训练曲线已保存: {output_path}")
    plt.close()

if __name__ == '__main__':
    import sys
    
    log_file = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/improved_detector/training_improved_detector_20251210_140315.log'
    
    if not Path(log_file).exists():
        print(f"错误: 日志文件不存在: {log_file}")
        sys.exit(1)
    
    print(f"分析日志文件: {log_file}\n")
    
    data = parse_log(log_file)
    results = analyze_convergence(data)
    
    # 绘制训练曲线
    output_dir = Path(log_file).parent
    plot_training_curves(data, output_dir)


