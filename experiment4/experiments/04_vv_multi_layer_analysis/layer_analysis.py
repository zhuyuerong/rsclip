# -*- coding: utf-8 -*-
"""
多层特征分析

分析第1/6/9/12层的特征：
1. patch-text相似度热图（余弦相似度/VV^T/QKV）
2. 每层的mAP对比
3. GT区域响应强度
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
from experiment4.core.models.clip_surgery import CLIPSurgeryWrapper, clip_feature_surgery
from utils.seen_unseen_split import SeenUnseenDataset


def compute_cosine_similarity(patch_features, text_features):
    """
    计算标准余弦相似度
    
    Args:
        patch_features: [B, N_patches, C]
        text_features: [N_classes, C]
    
    Returns:
        similarity: [B, N_patches, N_classes]
    """
    patch_features_norm = F.normalize(patch_features, dim=-1)
    text_features_norm = F.normalize(text_features, dim=-1)
    
    similarity = patch_features_norm @ text_features_norm.t()
    return similarity


def compute_vvt_similarity(patch_features):
    """
    计算VV^T patch自相似度
    
    Args:
        patch_features: [B, N_patches, C]
    
    Returns:
        similarity: [B, N_patches, N_patches]
    """
    patch_features_norm = F.normalize(patch_features, dim=-1)
    similarity = patch_features_norm @ patch_features_norm.transpose(-2, -1)
    return similarity


def generate_heatmap_from_similarity(similarity, image_size=224):
    """
    从相似度生成热图
    
    Args:
        similarity: [B, N_patches, K] 相似度（K可以是N_classes或N_patches）
        image_size: 图像尺寸
    
    Returns:
        heatmap: [B, K, H, W] 热图
    """
    B, N_patches, K = similarity.shape
    
    # Min-Max归一化
    sim_min = similarity.min(dim=1, keepdim=True)[0]
    sim_max = similarity.max(dim=1, keepdim=True)[0]
    sim_norm = (similarity - sim_min) / (sim_max - sim_min + 1e-8)
    
    # Reshape为空间特征图
    side = int(N_patches ** 0.5)
    sim_grid = sim_norm.reshape(B, side, side, K)  # [B, H, W, K]
    sim_grid = sim_grid.permute(0, 3, 1, 2)  # [B, K, H, W]
    
    # 上采样到原图尺寸
    heatmap = F.interpolate(sim_grid, size=(image_size, image_size), 
                            mode='bilinear', align_corners=False)
    
    return heatmap


def analyze_layer_features(model, dataloader, config, layers=[1, 6, 9, 12], max_samples=20):
    """
    分析不同层的patch-text相似度
    
    Args:
        model: CLIPSurgeryWrapper
        dataloader: 数据加载器
        config: 配置对象
        layers: 要分析的层
        max_samples: 最大样本数
    
    Returns:
        results: {layer_idx: {method: heatmaps}}
    """
    results = {layer_idx: {'cosine': [], 'surgery': [], 'vvt': []} for layer_idx in layers}
    sample_count = 0
    
    for batch in tqdm(dataloader, desc="分析多层特征"):
        if sample_count >= max_samples:
            break
        
        images = batch['image'].to(config.device)
        class_names = batch['class_name']
        
        batch_size = images.shape[0]
        
        for i in range(batch_size):
            if sample_count >= max_samples:
                break
            
            class_name = class_names[i]
            
            # 提取多层特征
            layer_features_dict = model.get_layer_features(images[i:i+1], layer_indices=layers)
            text_features = model.encode_text([class_name])  # [1, C]
            text_features = F.normalize(text_features, dim=-1)
            
            for layer_idx in layers:
                features = layer_features_dict[layer_idx]  # [B, N+1, C]
                patch_features = features[:, 1:, :]  # [B, N_patches, C]
                
                # 方法1: 标准余弦相似度
                sim_cosine = compute_cosine_similarity(patch_features, text_features)  # [1, N, 1]
                heatmap_cosine = generate_heatmap_from_similarity(sim_cosine, config.image_size)
                results[layer_idx]['cosine'].append(heatmap_cosine[0, 0].cpu().numpy())
                
                # 方法2: Feature Surgery相似度
                if config.use_surgery:
                    sim_surgery = clip_feature_surgery(features, text_features, t=2)  # [1, N, 1]
                    heatmap_surgery = generate_heatmap_from_similarity(sim_surgery, config.image_size)
                    results[layer_idx]['surgery'].append(heatmap_surgery[0, 0].cpu().numpy())
                
                # 方法3: VV^T patch自相似度（可视化patch关系）
                sim_vvt = compute_vvt_similarity(patch_features)  # [1, N, N]
                # 取平均值作为每个patch的"重要性"
                vvt_importance = sim_vvt.mean(dim=-1, keepdim=True)  # [1, N, 1]
                heatmap_vvt = generate_heatmap_from_similarity(vvt_importance, config.image_size)
                results[layer_idx]['vvt'].append(heatmap_vvt[0, 0].cpu().numpy())
            
            sample_count += 1
    
    return results


def visualize_layer_comparison(results, output_dir, layers=[1, 6, 9, 12]):
    """
    可视化不同层的热图对比
    
    生成网格：4层 x 3种方法
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    methods = ['cosine', 'surgery', 'vvt']
    method_names = {
        'cosine': '余弦相似度',
        'surgery': 'Feature Surgery',
        'vvt': 'VV^T重要性'
    }
    
    # 创建4x3网格
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    fig.suptitle('多层特征热图对比', fontsize=16, y=0.995)
    
    for row, layer_idx in enumerate(layers):
        for col, method in enumerate(methods):
            ax = axes[row, col]
            
            heatmaps = results[layer_idx][method]
            if len(heatmaps) > 0:
                # 显示第一个样本的热图
                heatmap = heatmaps[0]
                im = ax.imshow(heatmap, cmap='jet')
                
                # 标题
                if row == 0:
                    ax.set_title(method_names[method], fontsize=12, fontweight='bold')
                if col == 0:
                    ax.set_ylabel(f'Layer {layer_idx}', fontsize=12, fontweight='bold')
                
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)
            else:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'layer_comparison_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 层对比热图保存至: {output_dir / 'layer_comparison_heatmaps.png'}")


def compute_gt_region_response(heatmaps, bboxes_batch, image_size=224):
    """
    计算GT区域的平均响应强度
    
    Args:
        heatmaps: list of [H, W] numpy arrays
        bboxes_batch: list of bbox dicts
        image_size: 图像尺寸
    
    Returns:
        avg_response: 平均响应强度
    """
    responses = []
    
    for heatmap, bboxes in zip(heatmaps, bboxes_batch):
        for bbox in bboxes:
            xmin = int(bbox['xmin'] * image_size / image_size)  # 如果已是像素坐标
            ymin = int(bbox['ymin'] * image_size / image_size)
            xmax = int(bbox['xmax'] * image_size / image_size)
            ymax = int(bbox['ymax'] * image_size / image_size)
            
            # 提取GT区域
            gt_region = heatmap[ymin:ymax, xmin:xmax]
            if gt_region.size > 0:
                responses.append(gt_region.mean())
    
    return np.mean(responses) if responses else 0.0


def analyze_layer_statistics(results, layers=[1, 6, 9, 12]):
    """分析各层统计信息"""
    print("\n" + "="*60)
    print("各层统计信息")
    print("="*60)
    
    for layer_idx in layers:
        print(f"\nLayer {layer_idx}:")
        
        for method in ['cosine', 'surgery', 'vvt']:
            heatmaps = results[layer_idx][method]
            if len(heatmaps) > 0:
                heatmaps_array = np.array(heatmaps)
                print(f"  {method}:")
                print(f"    均值: {heatmaps_array.mean():.4f}")
                print(f"    标准差: {heatmaps_array.std():.4f}")
                print(f"    最大值: {heatmaps_array.max():.4f}")
                print(f"    最小值: {heatmaps_array.min():.4f}")


def main():
    parser = argparse.ArgumentParser(description='多层特征分析')
    parser.add_argument('--dataset', type=str, default='datasets/mini_dataset',
                        help='数据集路径')
    parser.add_argument('--layers', type=int, nargs='+', default=[1, 6, 9, 12],
                        help='要分析的层')
    parser.add_argument('--max-samples', type=int, default=20,
                        help='最大样本数')
    parser.add_argument('--use-surgery', action='store_true',
                        help='使用Feature Surgery')
    parser.add_argument('--use-vv', action='store_true',
                        help='使用VV机制')
    
    args = parser.parse_args()
    
    print("="*60)
    print("多层特征分析实验")
    print("="*60)
    print(f"数据集: {args.dataset}")
    print(f"分析层: {args.layers}")
    print(f"使用Surgery: {args.use_surgery}")
    print(f"使用VV机制: {args.use_vv}")
    
    # 配置
    config = Config()
    config.dataset_root = args.dataset
    config.use_surgery = args.use_surgery
    config.use_vv_mechanism = args.use_vv
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型
    print(f"\n加载模型...")
    model = CLIPSurgeryWrapper(config)
    print(f"✓ 模型加载完成")
    
    # 加载数据集
    print(f"\n加载数据集...")
    unseen_classes = ['airplane', 'bridge', 'storagetank', 'vehicle', 'windmill']
    dataset = SeenUnseenDataset(
        config.dataset_root,
        split='all',  # 使用所有类别
        mode='val',
        unseen_classes=unseen_classes
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    print(f"✓ 数据集加载完成: {len(dataset)}个样本")
    
    # 分析多层特征
    print(f"\n开始分析...")
    results = analyze_layer_features(model, dataloader, config, 
                                     layers=args.layers, max_samples=args.max_samples)
    
    # 生成可视化
    output_dir = Path("experiment4/experiments/04_vv_multi_layer_analysis/outputs/layer_analysis")
    visualize_layer_comparison(results, output_dir, layers=args.layers)
    
    # 分析统计信息
    analyze_layer_statistics(results, layers=args.layers)
    
    # 保存结果
    results_to_save = {}
    for layer_idx in args.layers:
        results_to_save[f'layer_{layer_idx}'] = {
            'cosine_mean': float(np.mean([h.mean() for h in results[layer_idx]['cosine']])),
            'cosine_std': float(np.mean([h.std() for h in results[layer_idx]['cosine']])),
            'num_samples': len(results[layer_idx]['cosine'])
        }
    
    with open(output_dir / 'layer_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=4, ensure_ascii=False)
    
    print(f"\n✓ 分析完成！结果保存至: {output_dir}")


if __name__ == "__main__":
    main()

