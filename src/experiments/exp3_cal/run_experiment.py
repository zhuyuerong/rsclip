#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CAL实验运行脚本
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.competitors.clip_methods.surgeryclip.model_wrapper import SurgeryCLIPWrapper
from .experiment_configs import ALL_CAL_CONFIGS
from PIL import Image
import argparse


def run_single_experiment(config_name: str, 
                         image_path: str,
                         class_name: str,
                         checkpoint_path: str = 'checkpoints/ViT-B-32.pt',
                         device: str = 'cuda',
                         output_dir: str = 'outputs/exp3_cal'):
    """
    运行单个CAL实验
    
    Args:
        config_name: 实验配置名称（如 'q1_exp1_fixed'）
        image_path: 测试图像路径
        class_name: 目标类别名称
        checkpoint_path: 模型权重路径
        device: 设备
        output_dir: 输出目录
    """
    print("=" * 80)
    print(f"🧪 运行CAL实验: {config_name}")
    print("=" * 80)
    
    # 获取实验配置
    if config_name not in ALL_CAL_CONFIGS:
        print(f"❌ 未知的实验配置: {config_name}")
        print(f"可用配置: {list(ALL_CAL_CONFIGS.keys())}")
        return
    
    cal_config = ALL_CAL_CONFIGS[config_name]
    print(f"\n📋 实验配置:")
    print(f"   实验ID: {cal_config.get_experiment_id()}")
    print(f"   负样本模式: {cal_config.negative_mode}")
    print(f"   加权系数: alpha={cal_config.alpha}")
    print(f"   操作位置: {cal_config.cal_space}")
    
    # 创建模型
    print(f"\n📦 创建模型...")
    model = SurgeryCLIPWrapper(
        model_name='surgeryclip',
        checkpoint_path=checkpoint_path,
        device=device,
        use_surgery_single='empty',
        use_surgery_multi=True,
        cal_config=cal_config
    )
    
    # 加载模型
    print(f"📥 加载模型...")
    model.load_model()
    
    # 加载图像
    print(f"\n🖼️  处理图像: {image_path}")
    image = Image.open(image_path).convert('RGB')
    
    # 生成热图
    print(f"🔥 生成热图（类别: {class_name}）...")
    heatmap = model.generate_heatmap(image, [class_name])
    
    # 保存结果
    output_path = Path(output_dir) / config_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 原图
    axes[0].imshow(image)
    axes[0].set_title(f'Original Image\nClass: {class_name}')
    axes[0].axis('off')
    
    # 热图
    im = axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title(f'CAL Heatmap\n{config_name}')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # 保存
    image_name = Path(image_path).stem
    save_path = output_path / f"{image_name}_{class_name}_cal.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 结果已保存: {save_path}")
    print(f"   热图统计: min={heatmap.min():.4f}, max={heatmap.max():.4f}, mean={heatmap.mean():.4f}")
    
    return {
        'config_name': config_name,
        'experiment_id': cal_config.get_experiment_id(),
        'heatmap_stats': {
            'min': float(heatmap.min()),
            'max': float(heatmap.max()),
            'mean': float(heatmap.mean()),
            'std': float(heatmap.std())
        },
        'output_path': str(save_path)
    }


def main():
    parser = argparse.ArgumentParser(description='运行CAL实验')
    parser.add_argument('--config', type=str, required=True,
                       help='实验配置名称（如 q1_exp1_fixed）')
    parser.add_argument('--image', type=str, required=True,
                       help='测试图像路径')
    parser.add_argument('--class', type=str, required=True, dest='class_name',
                       help='目标类别名称')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/ViT-B-32.pt',
                       help='模型权重路径')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备（cuda/cpu）')
    parser.add_argument('--output-dir', type=str, default='outputs/exp3_cal',
                       help='输出目录')
    
    args = parser.parse_args()
    
    result = run_single_experiment(
        config_name=args.config,
        image_path=args.image,
        class_name=args.class_name,
        checkpoint_path=args.checkpoint,
        device=args.device,
        output_dir=args.output_dir
    )
    
    print("\n" + "=" * 80)
    print("✅ 实验完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()






