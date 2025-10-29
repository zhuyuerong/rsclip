# -*- coding: utf-8 -*-
"""
Patch相似度矩阵分析

分析某层内部patch之间的相似度矩阵（49x49）
对比标准特征 vs Surgery去冗余后的特征
"""

import sys
from pathlib import Path
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.append(str(root_dir))

# 添加实验目录
exp_dir = Path(__file__).parent
sys.path.append(str(exp_dir))

from experiment4.core.config import Config
from experiment4.core.models.clip_surgery import CLIPSurgeryWrapper
from utils.seen_unseen_split import SeenUnseenDataset


def analyze_patch_similarity_matrix(model, images, layer_idx=12, use_surgery=False):
    """
    分析patch相似度矩阵
    
    Args:
        model: CLIPSurgeryWrapper
        images: [B, 3, H, W]
        layer_idx: 要分析的层
        use_surgery: 是否应用Surgery去冗余
    
    Returns:
        similarity_matrix: [B, N_patches, N_patches]
    """
    # 提取指定层特征
    layer_features = model.get_layer_features(images, layer_indices=[layer_idx])
    features = layer_features[layer_idx]  # [B, N+1, C]
    patch_features = features[:, 1:, :]  # [B, N_patches, C]
    
    if use_surgery:
        # Surgery去冗余: F - mean(F)
        redundant = patch_features.mean(dim=1, keepdim=True)
        patch_features = patch_features - redundant
    
    # 计算patch相似度矩阵
    patch_features_norm = F.normalize(patch_features, dim=-1)
    similarity_matrix = patch_features_norm @ patch_features_norm.transpose(-2, -1)
    # [B, N_patches, N_patches]
    
    return similarity_matrix


def visualize_similarity_matrix(similarity_matrix, title="Patch Similarity Matrix", output_path=None):
    """
    可视化相似度矩阵
    
    Args:
        similarity_matrix: [N, N] numpy array
        title: 标题
        output_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(similarity_matrix, cmap='viridis', vmin=-1, vmax=1)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Patch Index', fontsize=12)
    ax.set_ylabel('Patch Index', fontsize=12)
    
    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label('Cosine Similarity', fontsize=10)
    
    # 添加网格
    ax.set_xticks(np.arange(0, similarity_matrix.shape[0], 7))
    ax.set_yticks(np.arange(0, similarity_matrix.shape[0], 7))
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 相似度矩阵保存至: {output_path}")
    else:
        plt.show()
    
    plt.close()


def compare_surgery_impact(model, images, layer_idx=12, output_dir=None):
    """
    对比Surgery前后的patch相似度矩阵
    
    Args:
        model: CLIPSurgeryWrapper
        images: [B, 3, H, W]
        layer_idx: 要分析的层
        output_dir: 输出目录
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 标准特征的相似度矩阵
    sim_matrix_standard = analyze_patch_similarity_matrix(model, images, layer_idx, use_surgery=False)
    sim_matrix_standard_np = sim_matrix_standard[0].detach().cpu().numpy()
    
    # 2. Surgery特征的相似度矩阵
    sim_matrix_surgery = analyze_patch_similarity_matrix(model, images, layer_idx, use_surgery=True)
    sim_matrix_surgery_np = sim_matrix_surgery[0].detach().cpu().numpy()
    
    # 3. 可视化对比
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 标准特征
    im1 = axes[0].imshow(sim_matrix_standard_np, cmap='viridis', vmin=-1, vmax=1)
    axes[0].set_title(f'Layer {layer_idx} - 标准特征', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Patch Index')
    axes[0].set_ylabel('Patch Index')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    # Surgery特征
    im2 = axes[1].imshow(sim_matrix_surgery_np, cmap='viridis', vmin=-1, vmax=1)
    axes[1].set_title(f'Layer {layer_idx} - Surgery去冗余', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Patch Index')
    axes[1].set_ylabel('Patch Index')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    # 差异图
    diff = sim_matrix_surgery_np - sim_matrix_standard_np
    im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[2].set_title(f'差异 (Surgery - 标准)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Patch Index')
    axes[2].set_ylabel('Patch Index')
    plt.colorbar(im3, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    
    if output_dir:
        save_path = output_dir / f'surgery_comparison_layer{layer_idx}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Surgery对比图保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # 4. 打印统计信息
    print(f"\nLayer {layer_idx} 相似度矩阵统计:")
    print(f"  标准特征:")
    print(f"    均值: {sim_matrix_standard_np.mean():.4f}")
    print(f"    标准差: {sim_matrix_standard_np.std():.4f}")
    print(f"  Surgery特征:")
    print(f"    均值: {sim_matrix_surgery_np.mean():.4f}")
    print(f"    标准差: {sim_matrix_surgery_np.std():.4f}")
    print(f"  Surgery影响:")
    print(f"    均值变化: {sim_matrix_surgery_np.mean() - sim_matrix_standard_np.mean():.4f}")
    print(f"    标准差变化: {sim_matrix_surgery_np.std() - sim_matrix_standard_np.std():.4f}")


def main():
    parser = argparse.ArgumentParser(description='Patch相似度矩阵分析')
    parser.add_argument('--dataset', type=str, default='datasets/mini_dataset',
                        help='数据集路径')
    parser.add_argument('--layer', type=int, default=12,
                        help='要分析的层')
    parser.add_argument('--use-vv', action='store_true',
                        help='使用VV机制')
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"Patch相似度矩阵分析 - Layer {args.layer}")
    print("="*60)
    
    # 配置
    config = Config()
    config.dataset_root = args.dataset
    config.use_surgery = False  # 在函数内部控制
    config.use_vv_mechanism = args.use_vv
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型
    print(f"\n加载模型...")
    model = CLIPSurgeryWrapper(config)
    print(f"✓ 模型加载完成")
    
    # 加载一个样本
    print(f"\n加载数据集...")
    unseen_classes = ['airplane', 'bridge', 'storagetank', 'vehicle', 'windmill']
    dataset = SeenUnseenDataset(
        config.dataset_root,
        split='all',
        mode='val',
        unseen_classes=unseen_classes
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 获取第一个样本
    sample = next(iter(dataloader))
    images = sample['image'].to(config.device)
    print(f"✓ 图像形状: {images.shape}")
    
    # 分析并对比
    output_dir = Path("experiment4/experiments/04_vv_multi_layer_analysis/outputs/layer_analysis")
    compare_surgery_impact(model, images, layer_idx=args.layer, output_dir=output_dir)
    
    print(f"\n✓ 分析完成！")


if __name__ == "__main__":
    main()

