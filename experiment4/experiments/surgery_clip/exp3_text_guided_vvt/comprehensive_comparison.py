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


def generate_heatmaps_for_mode(model, images, text_queries, dior_classes, dior_classes_raw, config, layers, mode_name):
    """
    为指定模式生成所有层的热图
    
    Args:
        model: CLIPSurgeryWrapper
        images: [B, 3, H, W]
        text_queries: list of str (raw class names)
        dior_classes: list of 20 classes with prompt ("an aerial photo of ...")
        dior_classes_raw: list of 20 raw class names
        config: Config
        layers: list of layer indices
        mode_name: str (模式名称)
    
    Returns:
        heatmaps: {layer_idx: [B, 1, H, W]}
    """
    B = images.shape[0]
    
    # 提取多层特征
    layer_features_dict = model.get_layer_features(images, layer_indices=layers)
    
    # 编码所有类别文本（使用prompt格式）
    all_text_features = model.encode_text(dior_classes)
    all_text_features = F.normalize(all_text_features, dim=-1)
    
    heatmaps = {layer_idx: [] for layer_idx in layers}
    
    # 逐样本处理
    for b in range(B):
        # 在raw class names中找到目标类别索引
        target_class_idx = dior_classes_raw.index(text_queries[b])
        
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


def visualize_comprehensive_comparison(all_heatmaps, images, class_names, bboxes_batch, image_ids, layers, modes, output_dir):
    """
    Visualize comprehensive comparison: modes x 12 layers
    
    Args:
        all_heatmaps: {mode_name: {layer_idx: [B, 1, H, W]}}
        images: [B, 3, H, W]
        class_names: list of str
        bboxes_batch: list of bbox dicts (with original_size info)
        image_ids: list of image IDs
        layers: list of layer indices (should be 12)
        modes: list of mode names (should be 5)
        output_dir: Path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    B = images.shape[0]
    num_layers = len(layers)
    num_modes = len(modes)
    
    for b in range(min(B, 5)):  # Max 5 samples
        # Create grid: num_modes rows x (1+12) columns
        fig, axes = plt.subplots(num_modes, num_layers + 1, 
                                figsize=(2.5 * (num_layers + 1), 2.5 * num_modes))
        
        # Prepare original image
        img = images[b].cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        # For each mode
        for row, mode_name in enumerate(modes):
            # Column 0: original image + mode label + image ID
            axes[row, 0].imshow(img)
            # Show the CLIP prompt in title
            prompt_text = f"an aerial photo of {class_names[b]}"
            title_text = f'{mode_name}\n{prompt_text}\nID: {image_ids[b]}'
            axes[row, 0].set_title(title_text, fontsize=6.5)
            axes[row, 0].axis('off')
            
            # Draw GT bounding boxes (only for matching class)
            if b < len(bboxes_batch):
                bbox_info = bboxes_batch[b]
                target_class = class_names[b]  # Current sample's class
                
                if 'boxes' in bbox_info and len(bbox_info['boxes']) > 0:
                    original_h, original_w = bbox_info.get('original_size', (224, 224))
                    scale_x = 224.0 / original_w
                    scale_y = 224.0 / original_h
                    
                    for bbox in bbox_info['boxes']:
                        # Only draw bbox if class matches the text query
                        bbox_class = bbox.get('class', target_class)
                        if bbox_class != target_class:
                            continue
                        
                        # Scale bbox to 224x224
                        xmin = bbox['xmin'] * scale_x
                        ymin = bbox['ymin'] * scale_y
                        xmax = bbox['xmax'] * scale_x
                        ymax = bbox['ymax'] * scale_y
                        w, h = xmax - xmin, ymax - ymin
                        
                        rect = patches.Rectangle((xmin, ymin), w, h, 
                                                linewidth=1.5, edgecolor='lime', facecolor='none')
                        axes[row, 0].add_patch(rect)
            
            # Subsequent columns: layer heatmaps
            heatmaps = all_heatmaps[mode_name]
            for col, layer_idx in enumerate(layers):
                heatmap = heatmaps[layer_idx][b, 0].detach().cpu().numpy()
                
                # Overlay heatmap
                axes[row, col + 1].imshow(img)
                axes[row, col + 1].imshow(heatmap, cmap='jet', alpha=0.5)
                
                # Show layer label only in first row
                if row == 0:
                    axes[row, col + 1].set_title(f'L{layer_idx}', fontsize=8)
                axes[row, col + 1].axis('off')
                
                # Draw GT boxes on heatmap (only for matching class)
                if b < len(bboxes_batch):
                    bbox_info = bboxes_batch[b]
                    target_class = class_names[b]
                    
                    if 'boxes' in bbox_info and len(bbox_info['boxes']) > 0:
                        original_h, original_w = bbox_info.get('original_size', (224, 224))
                        scale_x = 224.0 / original_w
                        scale_y = 224.0 / original_h
                        
                        for bbox in bbox_info['boxes']:
                            # Only draw bbox if class matches
                            bbox_class = bbox.get('class', target_class)
                            if bbox_class != target_class:
                                continue
                            
                            xmin = bbox['xmin'] * scale_x
                            ymin = bbox['ymin'] * scale_y
                            xmax = bbox['xmax'] * scale_x
                            ymax = bbox['ymax'] * scale_y
                            w, h = xmax - xmin, ymax - ymin
                            
                            rect = patches.Rectangle((xmin, ymin), w, h, 
                                                    linewidth=1.5, edgecolor='lime', facecolor='none')
                            axes[row, col + 1].add_patch(rect)
        
        plt.tight_layout(pad=0.5)
        plt.savefig(output_dir / f'comprehensive_comparison_sample{b}.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"Sample {b} comparison saved")
    
    print(f"\nAll comparisons saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='全面对比: 6种模式×12层热图')
    parser.add_argument('--dataset', type=str, default='datasets/mini_dataset')
    parser.add_argument('--max-samples', type=int, default=5)
    
    args = parser.parse_args()
    
    # DIOR所有类别（使用CLIP标准prompt格式）
    dior_class_names_raw = ['airplane', 'airport', 'baseballfield', 'basketballcourt', 
                           'bridge', 'chimney', 'dam', 'Expressway-Service-area',
                           'Expressway-toll-station', 'golffield', 'groundtrackfield',
                           'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
                           'tenniscourt', 'trainstation', 'vehicle', 'windmill']
    
    # 转换为CLIP prompt格式: "an aerial photo of {class}"
    dior_class_names = [f"an aerial photo of {cls}" for cls in dior_class_names_raw]
    
    # 12层
    all_layers = list(range(1, 13))
    
    # 5种模式配置（避免中文）
    mode_configs = {
        '1.With Surgery': {'use_surgery': True, 'use_vv': False},
        '2.Without Surgery': {'use_surgery': False, 'use_vv': False},
        '3.With VV': {'use_surgery': False, 'use_vv': False},  # VV需要特殊处理，暂时用标准
        '4.Standard QKV': {'use_surgery': False, 'use_vv': False},
        '5.Complete Surgery': {'use_surgery': True, 'use_vv': False},  # VV+Surgery，暂时只用Surgery
    }
    
    print("=" * 60)
    print("Comprehensive Comparison: 5 Modes x 12 Layers")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Modes: {len(mode_configs)}")
    print(f"Layers: {len(all_layers)}")
    print(f"Samples: {args.max_samples}")
    
    # Load data
    print("\nLoading dataset...")
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
    
    # Collect samples
    all_images = []
    all_class_names = []
    all_bboxes = []
    all_image_ids = []
    
    for idx, batch in enumerate(dataloader):
        if idx >= args.max_samples:
            break
        all_images.append(batch['image'])
        all_class_names.append(batch['class_name'][0])
        all_image_ids.append(batch.get('image_id', [f'sample_{idx}'])[0])
        
        # Reformat bbox info with original size
        bbox_info = {
            'boxes': batch['bboxes'],
            'original_size': batch.get('original_size', (224, 224))  # (H, W)
        }
        all_bboxes.append(bbox_info)
    
    images = torch.cat(all_images, dim=0).to(config.device)
    print(f"Loaded {len(all_class_names)} samples")
    
    # Generate heatmaps for each mode
    all_heatmaps = {}
    
    for mode_name, mode_config in mode_configs.items():
        print(f"\nProcessing mode: {mode_name}")
        print(f"  use_surgery={mode_config['use_surgery']}, use_vv={mode_config['use_vv']}")
        
        # Configure model
        config.use_surgery = mode_config['use_surgery']
        config.use_vv_mechanism = mode_config['use_vv']
        
        # Load model
        model = CLIPSurgeryWrapper(config)
        
        # Generate heatmaps
        heatmaps = generate_heatmaps_for_mode(
            model, images, all_class_names, dior_class_names, dior_class_names_raw,
            config, all_layers, mode_name
        )
        
        all_heatmaps[mode_name] = heatmaps
        print(f"  Generated {len(all_layers)} layer heatmaps")
    
    # Visualize comparison
    print("\nGenerating comparison visualization...")
    output_dir = Path(__file__).parent / 'comprehensive_comparison_results'
    visualize_comprehensive_comparison(
        all_heatmaps, images, all_class_names, all_bboxes, all_image_ids,
        all_layers, list(mode_configs.keys()), output_dir
    )
    
    print("\nAll done!")


if __name__ == '__main__':
    main()

