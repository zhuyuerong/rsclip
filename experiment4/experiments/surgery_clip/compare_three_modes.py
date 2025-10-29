# -*- coding: utf-8 -*-
"""
3种模式对比实验

对比:
1. 标准RemoteCLIP
2. Surgery去冗余
3. Surgery+VV机制

评估指标:
- Seen/Unseen数据集的mAP
- 热图可视化对比
"""

import sys
from pathlib import Path
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

# 添加项目根目录
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.append(str(root_dir))

# 添加实验目录
exp_dir = Path(__file__).parent
sys.path.append(str(exp_dir))

from experiment4.core.config import Config
from experiment4.core.models.clip_surgery import CLIPSurgeryWrapper, clip_feature_surgery, get_similarity_map
from experiment4.core.utils.map_calculator import calculate_map
from config_experiments import ExperimentConfig
from utils.seen_unseen_split import SeenUnseenDataset


def evaluate_with_heatmaps(model, dataloader, config, max_samples=50):
    """
    评估模型并生成热图
    
    Args:
        model: CLIPSurgeryWrapper
        dataloader: 数据加载器
        config: 配置对象
        max_samples: 最大样本数（用于可视化）
    
    Returns:
        (mAP, heatmaps): mAP分数和热图样本
    """
    all_predictions = {}
    all_ground_truths = {}
    heatmap_samples = []
    
    model_device = config.device
    sample_count = 0
    
    for batch in tqdm(dataloader, desc="评估"):
        if sample_count >= max_samples:
            break
        
        images = batch['image'].to(model_device)
        class_names = batch['class_name']
        image_ids = batch['image_id']
        bboxes_batch = batch['bboxes']
        
        batch_size = images.shape[0]
        
        for i in range(batch_size):
            if sample_count >= max_samples:
                break
            
            img_id = image_ids[i]
            class_name = class_names[i]
            
            # 提取特征
            image_features = model.encode_image(images[i:i+1])  # [1, N+1, C]
            text_features = model.encode_text([class_name])  # [1, C]
            text_features = F.normalize(text_features, dim=-1)
            
            # 使用Feature Surgery计算相似度
            if config.use_surgery:
                similarity = clip_feature_surgery(image_features, text_features, t=2)
            else:
                # 标准余弦相似度
                patch_features = image_features[:, 1:, :]  # [1, N_patches, C]
                patch_features_norm = F.normalize(patch_features, dim=-1)
                similarity = patch_features_norm @ text_features.t()  # [1, N_patches, 1]
            
            # 生成热图
            heatmap = get_similarity_map(similarity, (224, 224))  # [1, 1, 224, 224]
            
            # 保存样本
            if len(heatmap_samples) < 10:
                heatmap_samples.append({
                    'image': images[i].cpu(),
                    'heatmap': heatmap[0, 0].cpu(),
                    'class_name': class_name,
                    'image_id': img_id
                })
            
            sample_count += 1
    
    # 计算mAP（简化版，仅返回示例）
    mAP = 0.0  # 待实现完整mAP计算
    
    return mAP, heatmap_samples


def run_comparison_experiment(args):
    """运行3种模式对比实验"""
    print("="*60)
    print("3种模式对比实验")
    print("="*60)
    
    # 基础配置
    base_config = Config()
    base_config.dataset_root = args.dataset
    base_config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Unseen类别
    unseen_classes = ['airplane', 'bridge', 'storagetank', 'vehicle', 'windmill']
    
    results = {}
    mode_configs = ExperimentConfig.get_mode_configs()
    
    for mode_key, mode_info in mode_configs.items():
        print(f"\n{'='*60}")
        print(f"运行模式: {mode_info['name']}")
        print(f"描述: {mode_info['description']}")
        print(f"{'='*60}")
        
        # 创建模式配置
        config = ExperimentConfig.get_config_for_mode(mode_key, base_config)
        
        # 加载模型
        print(f"加载模型...")
        model = CLIPSurgeryWrapper(config)
        
        # Seen数据集评估
        print(f"\n评估Seen数据集...")
        seen_dataset = SeenUnseenDataset(
            config.dataset_root,
            split='seen',
            mode='val',
            unseen_classes=unseen_classes
        )
        seen_loader = torch.utils.data.DataLoader(
            seen_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        
        seen_map, seen_heatmaps = evaluate_with_heatmaps(model, seen_loader, config, max_samples=20)
        
        # Unseen数据集评估
        print(f"\n评估Unseen数据集...")
        unseen_dataset = SeenUnseenDataset(
            config.dataset_root,
            split='unseen',
            mode='val',
            unseen_classes=unseen_classes
        )
        unseen_loader = torch.utils.data.DataLoader(
            unseen_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        
        unseen_map, unseen_heatmaps = evaluate_with_heatmaps(model, unseen_loader, config, max_samples=20)
        
        # 保存结果
        results[mode_key] = {
            'name': mode_info['name'],
            'seen_map': seen_map,
            'unseen_map': unseen_map,
            'seen_heatmaps': seen_heatmaps,
            'unseen_heatmaps': unseen_heatmaps
        }
        
        print(f"\n{mode_info['name']}结果:")
        print(f"  Seen mAP: {seen_map:.4f}")
        print(f"  Unseen mAP: {unseen_map:.4f}")
    
    # 生成对比可视化
    output_dir = Path("experiment4/experiments/04_vv_multi_layer_analysis/outputs/mode_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_comparison_grid(results, output_dir)
    generate_map_comparison_table(results, output_dir)
    
    print(f"\n✓ 实验完成！结果保存至: {output_dir}")


def generate_comparison_grid(results, output_dir):
    """生成3种模式的热图对比网格"""
    print("\n生成热图对比网格...")
    
    # Seen数据集对比
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Seen数据集 - 3种模式热图对比', fontsize=16)
    
    mode_keys = ['mode1_baseline', 'mode2_surgery', 'mode3_surgery_vv']
    for row, mode_key in enumerate(mode_keys):
        mode_name = results[mode_key]['name']
        heatmaps = results[mode_key]['seen_heatmaps'][:4]
        
        for col, sample in enumerate(heatmaps):
            if col >= 4:
                break
            
            ax = axes[row, col]
            
            # 显示热图
            ax.imshow(sample['heatmap'], cmap='jet', alpha=0.6)
            ax.set_title(f"{mode_name}\n{sample['class_name']}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mode_comparison_seen.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Unseen数据集对比
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Unseen数据集 - 3种模式热图对比', fontsize=16)
    
    for row, mode_key in enumerate(mode_keys):
        mode_name = results[mode_key]['name']
        heatmaps = results[mode_key]['unseen_heatmaps'][:4]
        
        for col, sample in enumerate(heatmaps):
            if col >= 4:
                break
            
            ax = axes[row, col]
            ax.imshow(sample['heatmap'], cmap='jet', alpha=0.6)
            ax.set_title(f"{mode_name}\n{sample['class_name']}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mode_comparison_unseen.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 热图保存至: {output_dir}")


def generate_map_comparison_table(results, output_dir):
    """生成mAP对比表"""
    print("\n生成mAP对比表...")
    
    # 创建表格数据
    table_data = []
    for mode_key in ['mode1_baseline', 'mode2_surgery', 'mode3_surgery_vv']:
        row = {
            '模式': results[mode_key]['name'],
            'Seen mAP': f"{results[mode_key]['seen_map']:.4f}",
            'Unseen mAP': f"{results[mode_key]['unseen_map']:.4f}"
        }
        table_data.append(row)
    
    # 保存为JSON
    with open(output_dir / 'map_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(table_data, f, indent=4, ensure_ascii=False)
    
    # 打印表格
    print("\nmAP对比表:")
    print("-" * 60)
    print(f"{'模式':<20} {'Seen mAP':<15} {'Unseen mAP':<15}")
    print("-" * 60)
    for row in table_data:
        print(f"{row['模式']:<20} {row['Seen mAP']:<15} {row['Unseen mAP']:<15}")
    print("-" * 60)
    
    print(f"  ✓ 表格保存至: {output_dir / 'map_comparison.json'}")


def main():
    parser = argparse.ArgumentParser(description='3种模式对比实验')
    parser.add_argument('--dataset', type=str, default='datasets/mini_dataset',
                        help='数据集路径')
    parser.add_argument('--quick-test', action='store_true',
                        help='快速测试模式（使用mini_dataset）')
    parser.add_argument('--full-eval', action='store_true',
                        help='完整评估模式（使用完整DIOR数据集）')
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.dataset = 'datasets/mini_dataset'
        print("快速测试模式：使用mini_dataset")
    elif args.full_eval:
        args.dataset = 'datasets/DIOR'
        print("完整评估模式：使用完整DIOR数据集")
    
    run_comparison_experiment(args)


if __name__ == "__main__":
    main()

