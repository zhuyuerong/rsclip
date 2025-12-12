#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行所有CAL实验的批量脚本
"""
import sys
from pathlib import Path
import time
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.competitors.clip_methods.surgeryclip.model_wrapper import SurgeryCLIPWrapper
from .experiment_configs import ALL_CAL_CONFIGS
from PIL import Image
import argparse
import os


def run_all_experiments(image_paths: list,
                        class_names: list,
                        checkpoint_path: str = 'checkpoints/ViT-B-32.pt',
                        device: str = 'cuda',
                        output_dir: str = 'outputs/exp3_cal',
                        skip_existing: bool = True):
    """
    运行所有CAL实验
    
    Args:
        image_paths: 测试图像路径列表
        class_names: 对应的类别名称列表
        checkpoint_path: 模型权重路径
        device: 设备
        output_dir: 输出目录
        skip_existing: 是否跳过已存在的实验结果
    """
    print("=" * 80)
    print("🚀 开始运行所有CAL实验")
    print("=" * 80)
    print(f"📋 实验总数: {len(ALL_CAL_CONFIGS)}")
    print(f"🖼️  测试图像数: {len(image_paths)}")
    print(f"📁 输出目录: {output_dir}")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    results_summary = {
        'total': len(ALL_CAL_CONFIGS) * len(image_paths),
        'completed': 0,
        'failed': 0,
        'skipped': 0,
        'experiments': []
    }
    
    start_time = time.time()
    
    # 遍历所有实验配置
    for config_idx, (config_name, cal_config) in enumerate(ALL_CAL_CONFIGS.items(), 1):
        print(f"\n{'='*80}")
        print(f"📦 实验 {config_idx}/{len(ALL_CAL_CONFIGS)}: {config_name}")
        print(f"{'='*80}")
        print(f"   实验ID: {cal_config.get_experiment_id()}")
        print(f"   负样本模式: {cal_config.negative_mode}")
        print(f"   加权系数: alpha={cal_config.alpha}")
        print(f"   操作位置: {cal_config.cal_space}")
        
        # 为每个实验创建模型（可以复用）
        try:
            model = SurgeryCLIPWrapper(
                model_name='surgeryclip',
                checkpoint_path=checkpoint_path,
                device=device,
                use_surgery_single='empty',
                use_surgery_multi=True,
                cal_config=cal_config
            )
            model.load_model()
            print(f"   ✅ 模型加载成功")
        except Exception as e:
            print(f"   ❌ 模型加载失败: {e}")
            results_summary['failed'] += len(image_paths)
            continue
        
        # 处理每张图像
        for img_idx, (image_path, class_name) in enumerate(zip(image_paths, class_names), 1):
            print(f"\n   🖼️  图像 {img_idx}/{len(image_paths)}: {Path(image_path).name} (类别: {class_name})")
            
            # 检查是否已存在
            image_name = Path(image_path).stem
            save_path = output_path / config_name / f"{image_name}_{class_name}_cal.png"
            
            if skip_existing and save_path.exists():
                print(f"      ⏭️  跳过（已存在）: {save_path}")
                results_summary['skipped'] += 1
                continue
            
            try:
                # 加载图像
                if not os.path.exists(image_path):
                    print(f"      ⚠️  图像不存在: {image_path}")
                    results_summary['failed'] += 1
                    continue
                
                image = Image.open(image_path).convert('RGB')
                
                # 生成热图
                heatmap = model.generate_heatmap(image, [class_name])
                
                # 保存结果
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
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
                
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"      ✅ 保存成功: {save_path}")
                print(f"         统计: min={heatmap.min():.4f}, max={heatmap.max():.4f}, mean={heatmap.mean():.4f}")
                
                results_summary['completed'] += 1
                results_summary['experiments'].append({
                    'config': config_name,
                    'image': image_name,
                    'class': class_name,
                    'status': 'success',
                    'path': str(save_path)
                })
                
            except Exception as e:
                print(f"      ❌ 失败: {e}")
                import traceback
                traceback.print_exc()
                results_summary['failed'] += 1
                results_summary['experiments'].append({
                    'config': config_name,
                    'image': image_name,
                    'class': class_name,
                    'status': 'failed',
                    'error': str(e)
                })
    
    # 打印总结
    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("📊 实验总结")
    print(f"{'='*80}")
    print(f"   总实验数: {results_summary['total']}")
    print(f"   ✅ 完成: {results_summary['completed']}")
    print(f"   ⏭️  跳过: {results_summary['skipped']}")
    print(f"   ❌ 失败: {results_summary['failed']}")
    print(f"   ⏰ 总耗时: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)")
    print(f"   ⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # 保存总结到文件
    import json
    summary_file = output_path / 'experiments_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n📄 总结已保存: {summary_file}")
    
    return results_summary


def main():
    parser = argparse.ArgumentParser(description='运行所有CAL实验')
    parser.add_argument('--images', type=str, nargs='+', required=True,
                       help='测试图像路径列表')
    parser.add_argument('--classes', type=str, nargs='+', required=True,
                       help='对应的类别名称列表')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/ViT-B-32.pt',
                       help='模型权重路径')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备（cuda/cpu）')
    parser.add_argument('--output-dir', type=str, default='outputs/exp3_cal',
                       help='输出目录')
    parser.add_argument('--no-skip', action='store_true',
                       help='不跳过已存在的实验结果')
    
    args = parser.parse_args()
    
    if len(args.images) != len(args.classes):
        print("❌ 错误: 图像数量和类别数量必须相同")
        return
    
    results = run_all_experiments(
        image_paths=args.images,
        class_names=args.classes,
        checkpoint_path=args.checkpoint,
        device=args.device,
        output_dir=args.output_dir,
        skip_existing=not args.no_skip
    )
    
    print("\n✅ 所有实验完成！")


if __name__ == '__main__':
    main()






