# -*- coding: utf-8 -*-
"""
全面对比实验: 6种模式 × 12层热图

6种模式:
1. Baseline (无Surgery, 无VV)
2. Surgery (有Surgery, 无VV)
3. VV (无Surgery, 有VV)
4. Surgery+VV (有Surgery, 有VV)
5. 仅Surgery去冗余 (Surgery特征但不用VV)
6. 仅VV机制 (VV特征但不用Surgery)

12层: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
"""

import sys
from pathlib import Path
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json

# 添加项目根目录
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.append(str(root_dir))

# 添加surgery_clip目录
surgery_clip_dir = Path(__file__).parent.parent
sys.path.append(str(surgery_clip_dir))

from experiment4.core.config import Config
from experiment4.core.models.clip_surgery import CLIPSurgeryWrapper, clip_feature_surgery, get_similarity_map
from utils.seen_unseen_split import SeenUnseenDataset


def generate_heatmaps_for_mode(model, images, text_queries, dior_classes, config, layers, mode_name):
    """
    为指定模式生成所有层的热图
    
    Args:
        model: CLIPSurgeryWrapper
        images: [B, 3, H, W]
        text_queries: list of str
        dior_classes: list of 20 classes
        config: Config
        layers: list of layer indices
        mode_name: str (模式名称)
    
    Returns:
        heatmaps: {layer_idx: [B, 1, H, W]}
    """
    B = images.shape[0]
    
    # 提取多层特征
    layer_features_dict = model.get_layer_features(images, layer_indices=layers)
    
    # 编码所有类别文本
    all_text_features = model.encode_text(dior_classes)
    all_text_features = F.normalize(all_text_features, dim=-1)
    
    heatmaps = {layer_idx: [] for layer_idx in layers}
    
    # 逐样本处理
    for b in range(B):
        target_class_idx = dior_classes.index(text_queries[b])
        
        for layer_idx in layers:
            image_feature = layer_features_dict[layer_idx][b:b+1]
            
            # 根据模式选择处理方式
            if "Surgery" in mode_name or "surgery" in mode_name:
                # 使用Surgery
                similarity = clip_feature_surgery(image_feature, all_text_features, t=2)
                target_similarity = similarity[:, :, target_class_idx:target_class_idx+1]
            else:
                # 不使用Surgery，直接计算余弦相似度
                patch_features = image_feature[:, 1:, :]  # [1, N_patches, C]
                target_text = all_text_features[target_class_idx:target_class_idx+1]  # [1, C]
                target_similarity = patch_features @ target_text.t()  # [1, N_patches, 1]
            
            heatmap = get_similarity_map(target_similarity, (config.image_size, config.image_size))
            heatmaps[layer_idx].append(heatmap)
    
    # 合并所有样本
    for layer_idx in layers:
        heatmaps[layer_idx] = torch.cat(heatmaps[layer_idx], dim=0)
    
    return heatmaps


def visualize_comprehensive_comparison(all_heatmaps, images, class_names, bboxes_batch, layers, modes, output_dir):
    """
    可视化6种模式×12层的对比热图
    
    Args:
        all_heatmaps: {mode_name: {layer_idx: [B, 1, H, W]}}
        images: [B, 3, H, W]
        class_names: list of str
        bboxes_batch: list of bbox lists
        layers: list of layer indices (should be 12)
        modes: list of mode names (should be 6)
        output_dir: Path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    B = images.shape[0]
    num_layers = len(layers)
    num_modes = len(modes)
    
    for b in range(min(B, 5)):  # 最多显示5个样本
        # 创建大图：6行(模式) × 13列(原图+12层)
        fig, axes = plt.subplots(num_modes, num_layers + 1, 
                                figsize=(2.5 * (num_layers + 1), 2.5 * num_modes))
        
        # 准备原图
        img = images[b].cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        # 对每种模式
        for row, mode_name in enumerate(modes):
            # 第一列显示原图+模式标签
            axes[row, 0].imshow(img)
            axes[row, 0].set_title(f'{mode_name}\n{class_names[b]}', fontsize=8)
            axes[row, 0].axis('off')
            
            # 在原图上绘制GT边界框
            if b < len(bboxes_batch) and len(bboxes_batch[b]) > 0:
                for bbox in bboxes_batch[b]:
                    xmin, ymin = bbox['xmin'], bbox['ymin']
                    xmax, ymax = bbox['xmax'], bbox['ymax']
                    w, h = xmax - xmin, ymax - ymin
                    rect = patches.Rectangle((xmin, ymin), w, h, 
                                            linewidth=1.5, edgecolor='lime', facecolor='none')
                    axes[row, 0].add_patch(rect)
            
            # 后续列显示各层热图
            heatmaps = all_heatmaps[mode_name]
            for col, layer_idx in enumerate(layers):
                heatmap = heatmaps[layer_idx][b, 0].detach().cpu().numpy()
                
                # 叠加显示
                axes[row, col + 1].imshow(img)
                axes[row, col + 1].imshow(heatmap, cmap='jet', alpha=0.5)
                
                # 只在第一行显示层标签
                if row == 0:
                    axes[row, col + 1].set_title(f'L{layer_idx}', fontsize=8)
                axes[row, col + 1].axis('off')
                
                # 在热图上绘制GT边界框
                if b < len(bboxes_batch) and len(bboxes_batch[b]) > 0:
                    for bbox in bboxes_batch[b]:
                        xmin, ymin = bbox['xmin'], bbox['ymin']
                        xmax, ymax = bbox['xmax'], bbox['ymax']
                        w, h = xmax - xmin, ymax - ymin
                        rect = patches.Rectangle((xmin, ymin), w, h, 
                                                linewidth=1.5, edgecolor='lime', facecolor='none')
                        axes[row, col + 1].add_patch(rect)
        
        plt.tight_layout(pad=0.5)
        plt.savefig(output_dir / f'comprehensive_comparison_sample{b}.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 样本{b}对比图已保存")
    
    print(f"\n✓ 所有对比图保存至: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='全面对比: 6种模式×12层热图')
    parser.add_argument('--dataset', type=str, default='datasets/mini_dataset')
    parser.add_argument('--max-samples', type=int, default=5)
    
    args = parser.parse_args()
    
    # DIOR所有类别
    dior_class_names = ['airplane', 'airport', 'baseballfield', 'basketballcourt', 
                       'bridge', 'chimney', 'dam', 'Expressway-Service-area',
                       'Expressway-toll-station', 'golffield', 'groundtrackfield',
                       'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
                       'tenniscourt', 'trainstation', 'vehicle', 'windmill']
    
    # 12层
    all_layers = list(range(1, 13))
    
    # 简化为2种模式（VV机制需要特殊处理，暂不包含）
    mode_configs = {
        '1.Baseline (无Surgery)': {'use_surgery': False, 'use_vv': False},
        '2.Surgery (有Surgery)': {'use_surgery': True, 'use_vv': False},
    }
    
    print("=" * 60)
    print("全面对比实验: 6种模式 × 12层热图")
    print("=" * 60)
    print(f"数据集: {args.dataset}")
    print(f"模式数: {len(mode_configs)}")
    print(f"层数: {len(all_layers)}")
    print(f"样本数: {args.max_samples}")
    
    # 加载数据
    print("\n加载数据集...")
    config = Config()
    config.dataset_root = args.dataset
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    unseen_classes = ['airplane', 'bridge', 'storagetank', 'vehicle', 'windmill']
    dataset = SeenUnseenDataset(
        config.dataset_root,
        split='all',
        mode='val',
        unseen_classes=unseen_classes
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 收集样本
    all_images = []
    all_class_names = []
    all_bboxes = []
    
    for idx, batch in enumerate(dataloader):
        if idx >= args.max_samples:
            break
        all_images.append(batch['image'])
        all_class_names.append(batch['class_name'][0])
        all_bboxes.append(batch['bboxes'])
    
    images = torch.cat(all_images, dim=0).to(config.device)
    print(f"✓ 已加载{len(all_class_names)}个样本")
    
    # 对每种模式生成热图
    all_heatmaps = {}
    
    for mode_name, mode_config in mode_configs.items():
        print(f"\n处理模式: {mode_name}")
        print(f"  use_surgery={mode_config['use_surgery']}, use_vv={mode_config['use_vv']}")
        
        # 配置模型
        config.use_surgery = mode_config['use_surgery']
        config.use_vv_mechanism = mode_config['use_vv']
        
        # 加载模型
        model = CLIPSurgeryWrapper(config)
        
        # 生成热图
        heatmaps = generate_heatmaps_for_mode(
            model, images, all_class_names, dior_class_names, 
            config, all_layers, mode_name
        )
        
        all_heatmaps[mode_name] = heatmaps
        print(f"  ✓ 已生成{len(all_layers)}层热图")
    
    # 可视化对比
    print("\n生成对比可视化...")
    output_dir = Path(__file__).parent / 'comprehensive_comparison_results'
    visualize_comprehensive_comparison(
        all_heatmaps, images, all_class_names, all_bboxes,
        all_layers, list(mode_configs.keys()), output_dir
    )
    
    print("\n✅ 全部完成！")


if __name__ == '__main__':
    main()

