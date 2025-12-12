#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验5.2: 消融实验
逐个移除改进，重新训练，对比性能差异
"""

import torch
import sys
import json
from pathlib import Path
import yaml
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluate_all_experiments import evaluate_model
from datasets.dior_detection import get_detection_dataloader


def run_ablation_study(checkpoints_dir, config_path, device='cuda'):
    """
    运行消融实验
    
    Args:
        checkpoints_dir: Checkpoint目录
        config_path: 配置文件路径
        device: 设备
    """
    checkpoints_dir = Path(checkpoints_dir)
    config_path = Path(config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 定义消融实验配置
    ablation_configs = {
        'baseline': {
            'checkpoint': 'checkpoints/best_simple_model.pth',
            'model_type': 'simple',
            'description': 'Baseline: 原始简化模型'
        },
        'exp2.2': {
            'checkpoint': 'checkpoints/exp2.2/best_exp2.2_model.pth',
            'model_type': 'simple',
            'description': '实验2.2: 改进正样本分配（预测框IoU匹配）'
        },
        'exp3.1': {
            'checkpoint': 'checkpoints/exp3.1/best_exp3.1_model.pth',
            'model_type': 'simple',
            'description': '实验3.1: 改进CAM损失函数'
        },
        'exp3.2': {
            'checkpoint': 'checkpoints/exp3.2/best_exp3.2_model.pth',
            'model_type': 'enhanced',
            'description': '实验3.2: 增强CAM生成器（多层MLP）'
        },
        'exp4.1': {
            'checkpoint': 'checkpoints/exp4.1/best_exp4.1_model.pth',
            'model_type': 'enhanced',
            'description': '实验4.1: 组合最佳方案（所有改进）'
        },
        'exp4.2': {
            'checkpoint': 'checkpoints/exp4.2/best_exp4.2_model.pth',
            'model_type': 'enhanced',
            'description': '实验4.2: 学习率调度优化'
        }
    }
    
    # 加载验证集
    val_loader = get_detection_dataloader(
        root=config.get('dataset_root'),
        split='test',
        batch_size=1,
        num_workers=0,
        image_size=config.get('image_size', 224),
        augment=False,
        train_only_seen=False
    )
    
    results = {}
    
    for exp_name, exp_config in ablation_configs.items():
        checkpoint_path = Path(exp_config['checkpoint'])
        
        if not checkpoint_path.exists():
            print(f"⚠️  Checkpoint不存在: {checkpoint_path}，跳过 {exp_name}")
            continue
        
        print(f"\n{'='*80}")
        print(f"评估: {exp_name} - {exp_config['description']}")
        print(f"{'='*80}")
        
        try:
            # 这里需要导入评估函数，但由于循环依赖，我们直接在这里实现评估逻辑
            # 或者调用evaluate_all_experiments.py作为子进程
            import subprocess
            
            result_file = Path(f'outputs/ablation_study/{exp_name}_results.json')
            result_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 调用评估脚本
            cmd = [
                sys.executable,
                str(Path(__file__).parent / 'evaluate_all_experiments.py'),
                '--checkpoint', str(checkpoint_path),
                '--model-type', exp_config['model_type'],
                '--config', str(config_path),
                '--output', str(result_file)
            ]
            
            subprocess.run(cmd, check=True)
            
            # 加载结果
            with open(result_file, 'r') as f:
                exp_results = json.load(f)
            
            results[exp_name] = {
                'description': exp_config['description'],
                'mAP@0.5': exp_results['mAP@0.5'],
                'mAP@0.5:0.95': exp_results['mAP@0.5:0.95'],
                'seen_mAP@0.5': exp_results['seen_mAP@0.5'],
                'unseen_mAP@0.5': exp_results['unseen_mAP@0.5']
            }
            
            print(f"✅ {exp_name} 评估完成")
            print(f"   mAP@0.5: {exp_results['mAP@0.5']:.4f}")
            print(f"   Seen mAP@0.5: {exp_results['seen_mAP@0.5']:.4f}")
            print(f"   Unseen mAP@0.5: {exp_results['unseen_mAP@0.5']:.4f}")
            
        except Exception as e:
            print(f"❌ {exp_name} 评估失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成对比报告
    print(f"\n{'='*80}")
    print("消融实验对比报告")
    print(f"{'='*80}")
    
    print("\n实验对比表:")
    print(f"{'实验':<15} {'mAP@0.5':<12} {'Seen mAP':<12} {'Unseen mAP':<12} {'改进':<20}")
    print("-" * 80)
    
    baseline_map = results.get('baseline', {}).get('mAP@0.5', 0.0)
    
    for exp_name, exp_results in results.items():
        map_05 = exp_results['mAP@0.5']
        seen_map = exp_results['seen_mAP@0.5']
        unseen_map = exp_results['unseen_mAP@0.5']
        
        improvement = map_05 - baseline_map
        improvement_str = f"+{improvement:.4f}" if improvement > 0 else f"{improvement:.4f}"
        
        print(f"{exp_name:<15} {map_05:<12.4f} {seen_map:<12.4f} {unseen_map:<12.4f} {improvement_str:<20}")
    
    # 保存结果
    output_path = Path('outputs/ablation_study/ablation_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ 消融实验结果已保存到: {output_path}")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='运行消融实验')
    parser.add_argument('--checkpoints-dir', type=str, default='checkpoints',
                       help='Checkpoint目录')
    parser.add_argument('--config', type=str, default='configs/surgery_cam_config.yaml',
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    run_ablation_study(args.checkpoints_dir, args.config)


if __name__ == '__main__':
    main()


