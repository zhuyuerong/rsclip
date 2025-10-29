# -*- coding: utf-8 -*-
"""
文本引导的VV^T热图生成

使用Feature Surgery计算多层的文本引导热图
对比第1/3/6/9层的表现
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

# 添加surgery_clip目录
surgery_clip_dir = Path(__file__).parent.parent
sys.path.append(str(surgery_clip_dir))

from experiment4.core.config import Config
from experiment4.core.models.clip_surgery import CLIPSurgeryWrapper, clip_feature_surgery, get_similarity_map
from utils.seen_unseen_split import SeenUnseenDataset


def generate_text_guided_vvt_heatmaps(model, images, text_queries, all_class_names, config, layers=[1, 3, 6, 9]):
    """
    生成文本引导的多层VV^T热图
    
    Args:
        model: CLIPSurgeryWrapper
        images: [B, 3, H, W]
        text_queries: list of str（每个图像对应一个文本查询）
        all_class_names: list of str（所有20个类别，用于Surgery）
        config: Config对象
        layers: 要分析的层
    
    Returns:
        heatmaps: {layer_idx: [B, 1, H, W]}（每个样本一个热图）
    """
    B = images.shape[0]
    
    # 提取多层特征（一次性）
    layer_features_dict = model.get_layer_features(images, layer_indices=layers)
    
    # 编码所有类别文本（Surgery需要多个类别）
    all_text_features = model.encode_text(all_class_names)  # [N_classes, C]
    all_text_features = F.normalize(all_text_features, dim=-1)
    
    heatmaps = {layer_idx: [] for layer_idx in layers}
    
    # 逐样本处理（提取对应类别的热图）
    for b in range(B):
        # 找到目标类别的索引
        target_class_idx = all_class_names.index(text_queries[b])
        
        for layer_idx in layers:
            # 获取该层该样本的特征
            image_feature = layer_features_dict[layer_idx][b:b+1]  # [1, N+1, C]
            
            # 使用Feature Surgery计算相似度（所有类别）
            similarity = clip_feature_surgery(image_feature, all_text_features, t=2)
            # [1, N_patches, N_classes]
            
            # 只提取目标类别的相似度
            target_similarity = similarity[:, :, target_class_idx:target_class_idx+1]
            # [1, N_patches, 1]
            
            # 生成热图（归一化 + reshape + 上采样）
            heatmap = get_similarity_map(target_similarity, (config.image_size, config.image_size))
            # [1, 1, H, W]
            
            heatmaps[layer_idx].append(heatmap)
    
    # 合并所有样本
    for layer_idx in layers:
        heatmaps[layer_idx] = torch.cat(heatmaps[layer_idx], dim=0)  # [B, 1, H, W]
    
    return heatmaps


def visualize_multi_layer_heatmaps(heatmaps, images, class_names, bboxes_batch, layers, output_dir):
    """
    可视化多层热图（带GT边界框）
    
    Args:
        heatmaps: {layer_idx: [B, N_classes, H, W]}
        images: [B, 3, H, W]
        class_names: list of str
        bboxes_batch: list of bbox lists
        layers: list of layer indices
        output_dir: 输出目录
    """
    import matplotlib.patches as patches
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    B = images.shape[0]
    num_layers = len(layers)
    
    for b in range(min(B, 5)):  # 最多显示5个样本
        fig, axes = plt.subplots(1, num_layers + 1, figsize=(4 * (num_layers + 1), 4))
        
        # 显示原图
        img = images[b].cpu().permute(1, 2, 0).numpy()
        # 反归一化
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        axes[0].imshow(img)
        axes[0].set_title(f'原图\n{class_names[b]}', fontsize=10)
        axes[0].axis('off')
        
        # 在原图上绘制GT边界框
        if b < len(bboxes_batch) and len(bboxes_batch[b]) > 0:
            for bbox in bboxes_batch[b]:
                xmin, ymin = bbox['xmin'], bbox['ymin']
                xmax, ymax = bbox['xmax'], bbox['ymax']
                w, h = xmax - xmin, ymax - ymin
                rect = patches.Rectangle((xmin, ymin), w, h, 
                                        linewidth=2, edgecolor='lime', facecolor='none')
                axes[0].add_patch(rect)
        
        # 显示各层热图
        for i, layer_idx in enumerate(layers):
            heatmap = heatmaps[layer_idx][b, 0].detach().cpu().numpy()  # 取第一个类别
            
            # 叠加显示
            axes[i+1].imshow(img)
            axes[i+1].imshow(heatmap, cmap='jet', alpha=0.5)
            axes[i+1].set_title(f'Layer {layer_idx}', fontsize=10)
            axes[i+1].axis('off')
            
            # 在热图上也绘制GT边界框
            if b < len(bboxes_batch) and len(bboxes_batch[b]) > 0:
                for bbox in bboxes_batch[b]:
                    xmin, ymin = bbox['xmin'], bbox['ymin']
                    xmax, ymax = bbox['xmax'], bbox['ymax']
                    w, h = xmax - xmin, ymax - ymin
                    rect = patches.Rectangle((xmin, ymin), w, h, 
                                            linewidth=2, edgecolor='lime', facecolor='none')
                    axes[i+1].add_patch(rect)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'text_guided_vvt_sample{b}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✓ 热图保存至: {output_dir}")


def analyze_gt_response_by_layer(heatmaps, bboxes_batch, layers, image_size=224):
    """
    分析各层的GT区域响应强度
    
    Args:
        heatmaps: {layer_idx: [B, N_classes, H, W]}
        bboxes_batch: list of bbox lists
        layers: list of layer indices
        image_size: 图像尺寸
    
    Returns:
        gt_responses: {layer_idx: avg_response}
    """
    gt_responses = {}
    
    for layer_idx in layers:
        responses = []
        heatmap_tensor = heatmaps[layer_idx]  # [B, N_classes, H, W]
        
        for b, bboxes in enumerate(bboxes_batch):
            if b >= heatmap_tensor.shape[0]:
                break
            
            heatmap = heatmap_tensor[b, 0].detach().cpu().numpy()  # [H, W]
            
            for bbox in bboxes:
                xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
                
                # 提取GT区域
                gt_region = heatmap[ymin:ymax, xmin:xmax]
                if gt_region.size > 0:
                    responses.append(gt_region.mean())
        
        gt_responses[layer_idx] = np.mean(responses) if responses else 0.0
    
    return gt_responses


def main():
    parser = argparse.ArgumentParser(description='文本引导的VV^T热图生成')
    parser.add_argument('--dataset', type=str, default='datasets/mini_dataset',
                        help='数据集路径')
    parser.add_argument('--layers', type=int, nargs='+', default=[1, 3, 6, 9],
                        help='要分析的层')
    parser.add_argument('--max-samples', type=int, default=10,
                        help='最大样本数')
    parser.add_argument('--use-vv', action='store_true',
                        help='使用VV机制')
    
    args = parser.parse_args()
    
    print("="*60)
    print("文本引导的VV^T热图生成")
    print("="*60)
    print(f"数据集: {args.dataset}")
    print(f"分析层: {args.layers}")
    print(f"使用VV机制: {args.use_vv}")
    
    # 配置
    config = Config()
    config.dataset_root = args.dataset
    config.use_surgery = True  # Feature Surgery必须启用
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
        split='all',
        mode='val',
        unseen_classes=unseen_classes
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"✓ 数据集加载完成: {len(dataset)}个样本")
    
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
    
    images = torch.cat(all_images, dim=0).to(config.device)  # [B, 3, H, W]
    
    # DIOR数据集的20个类别
    dior_class_names = ['airplane', 'airport', 'baseballfield', 'basketballcourt', 
                       'bridge', 'chimney', 'dam', 'Expressway-Service-area',
                       'Expressway-toll-station', 'golffield', 'groundtrackfield',
                       'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
                       'tenniscourt', 'trainstation', 'vehicle', 'windmill']
    
    # 生成多层热图
    print(f"\n生成多层热图...")
    heatmaps = generate_text_guided_vvt_heatmaps(
        model, images, all_class_names, dior_class_names, config, layers=args.layers
    )
    
    # 可视化
    output_dir = Path(__file__).parent  # 保存在当前实验目录
    visualize_multi_layer_heatmaps(heatmaps, images, all_class_names, all_bboxes, args.layers, output_dir)
    
    # 分析GT区域响应
    print(f"\n分析GT区域响应强度...")
    gt_responses = analyze_gt_response_by_layer(heatmaps, all_bboxes, args.layers)
    
    print("\n各层GT区域响应强度:")
    print("-" * 40)
    for layer_idx in args.layers:
        print(f"  Layer {layer_idx}: {gt_responses[layer_idx]:.4f}")
    print("-" * 40)
    
    # 保存结果（转换float16为float）
    with open(output_dir / 'gt_responses.json', 'w', encoding='utf-8') as f:
        json.dump({f'layer_{k}': float(v) for k, v in gt_responses.items()}, f, indent=4)
    
    print(f"\n✓ 分析完成！结果保存至: {output_dir}")


if __name__ == "__main__":
    main()

