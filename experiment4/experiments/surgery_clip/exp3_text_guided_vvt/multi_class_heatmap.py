# -*- coding: utf-8 -*-
"""
Multi-Class Heatmap Generator

For images with multiple object classes, generate separate heatmaps for each class.
Example: DIOR_05386 has overpass + vehicle
  - Query "overpass" → heatmap with overpass GT boxes
  - Query "vehicle" → heatmap with vehicle GT boxes
"""

import sys
from pathlib import Path
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add paths
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.append(str(root_dir))
surgery_clip_dir = Path(__file__).parent.parent
sys.path.append(str(surgery_clip_dir))

from experiment4.core.config import Config
from experiment4.core.models.clip_surgery import CLIPSurgeryWrapper, clip_feature_surgery, get_similarity_map
from utils.seen_unseen_split import SeenUnseenDataset


def generate_multi_mode_heatmaps(models, image, query_class, all_dior_classes, all_dior_prompts, config, layers):
    """
    为一个查询类别生成多种模式的热图
    
    Args:
        models: dict of {mode_name: CLIPSurgeryWrapper}
        image: [1, 3, H, W] single image
        query_class: str (query class name)
        all_dior_classes: list of 20 raw class names
        all_dior_prompts: list of 20 prompts
        config: Config
        layers: list of layer indices
    
    Returns:
        heatmaps_per_mode: {mode_name: {layer_idx: [1, 1, H, W]}}
    """
    class_idx = all_dior_classes.index(query_class)
    heatmaps_per_mode = {}
    
    for mode_name, model in models.items():
        # Extract multi-layer features
        layer_features_dict = model.get_layer_features(image, layer_indices=layers)
        
        # Encode all class texts
        all_text_features = model.encode_text(all_dior_prompts)
        all_text_features = F.normalize(all_text_features, dim=-1)
        
        heatmaps_per_mode[mode_name] = {}
        
        for layer_idx in layers:
            image_feature = layer_features_dict[layer_idx]  # [1, N+1, C]
            
            # Use Surgery or standard based on mode
            if "Surgery" in mode_name:
                similarity = clip_feature_surgery(image_feature, all_text_features, t=2)
            else:
                patch_features = image_feature[:, 1:, :]  # [1, N_patches, C]
                similarity = patch_features @ all_text_features.t()  # [1, N_patches, N_classes]
            
            # Extract similarity for this class
            target_similarity = similarity[:, :, class_idx:class_idx+1]  # [1, N_patches, 1]
            
            # Generate heatmap
            heatmap = get_similarity_map(target_similarity, (config.image_size, config.image_size))
            heatmaps_per_mode[mode_name][layer_idx] = heatmap
    
    return heatmaps_per_mode


def visualize_5mode_comparison(image_data, query_class, heatmaps_per_mode, bboxes, layers, modes, output_path):
    """
    Visualize 5-mode comparison for a single class query
    
    Layout: 5 modes x (1 original + 12 layers) columns
    """
    num_modes = len(modes)
    num_layers = len(layers)
    
    # Denormalize image
    img = image_data['image_tensor'].cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    img = img * std + mean
    img = np.clip(img, 0, 1)
    
    # Get scaling info
    original_h, original_w = image_data['original_size']
    scale_x = 224.0 / original_w
    scale_y = 224.0 / original_h
    
    # Create figure: 5 modes x (1 + num_layers) columns
    fig, axes = plt.subplots(num_modes, num_layers + 1, 
                            figsize=(2.5 * (num_layers + 1), 2.5 * num_modes))
    
    # For each mode
    for row, mode_name in enumerate(modes):
        # Column 0: Original image with query class GT boxes ONLY
        axes[row, 0].imshow(img)
        prompt = f"an aerial photo of {query_class}"
        title = f'{mode_name}\n{prompt}\nID: {image_data["image_id"]}'
        axes[row, 0].set_title(title, fontsize=6.5)
        axes[row, 0].axis('off')
        
        # Draw ONLY query class GT boxes (green)
        for bbox in bboxes:
            bbox_class = bbox.get('class', query_class)
            if bbox_class != query_class:
                continue  # Skip non-query class boxes
            
            xmin = bbox['xmin'] * scale_x
            ymin = bbox['ymin'] * scale_y
            xmax = bbox['xmax'] * scale_x
            ymax = bbox['ymax'] * scale_y
            w, h = xmax - xmin, ymax - ymin
            
            rect = patches.Rectangle((xmin, ymin), w, h, 
                                    linewidth=2.5, edgecolor='lime', facecolor='none')
            axes[row, 0].add_patch(rect)
        
        # Columns 1-N: Layer heatmaps
        for col, layer_idx in enumerate(layers):
            heatmap = heatmaps_per_mode[mode_name][layer_idx][0, 0].detach().cpu().numpy()
            
            axes[row, col + 1].imshow(img)
            axes[row, col + 1].imshow(heatmap, cmap='jet', alpha=0.5)
            
            if row == 0:
                axes[row, col + 1].set_title(f'L{layer_idx}', fontsize=8)
            axes[row, col + 1].axis('off')
            
            # Draw ONLY query class GT boxes on heatmap
            for bbox in bboxes:
                bbox_class = bbox.get('class', query_class)
                if bbox_class != query_class:
                    continue
                
                xmin = bbox['xmin'] * scale_x
                ymin = bbox['ymin'] * scale_y
                xmax = bbox['xmax'] * scale_x
                ymax = bbox['ymax'] * scale_y
                w, h = xmax - xmin, ymax - ymin
                
                rect = patches.Rectangle((xmin, ymin), w, h, 
                                        linewidth=2.5, edgecolor='lime', facecolor='none')
                axes[row, col + 1].add_patch(rect)
    
    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Multi-class heatmap generation')
    parser.add_argument('--dataset', type=str, default='datasets/mini_dataset')
    parser.add_argument('--max-samples', type=int, default=5)
    parser.add_argument('--layers', type=int, nargs='+', default=list(range(1, 13)))
    
    args = parser.parse_args()
    
    # DIOR classes
    dior_classes = ['airplane', 'airport', 'baseballfield', 'basketballcourt', 
                   'bridge', 'chimney', 'dam', 'Expressway-Service-area',
                   'Expressway-toll-station', 'golffield', 'groundtrackfield',
                   'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
                   'tenniscourt', 'trainstation', 'vehicle', 'windmill']
    
    dior_prompts = [f"an aerial photo of {cls}" for cls in dior_classes]
    
    # 5 modes
    mode_configs = {
        '1.With Surgery': {'use_surgery': True, 'use_vv': False},
        '2.Without Surgery': {'use_surgery': False, 'use_vv': False},
        '3.With VV': {'use_surgery': False, 'use_vv': False},  # Placeholder
        '4.Standard QKV': {'use_surgery': False, 'use_vv': False},
        '5.Complete Surgery': {'use_surgery': True, 'use_vv': False},  # Placeholder
    }
    
    print("=" * 70)
    print("Multi-Class Heatmap Generator (5-Mode Comparison)")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Layers: {args.layers}")
    print(f"Modes: {len(mode_configs)}")
    
    # Setup
    config = Config()
    config.dataset_root = args.dataset
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load all models
    print("\nLoading models for all modes...")
    models = {}
    for mode_name, mode_config in mode_configs.items():
        config.use_surgery = mode_config['use_surgery']
        config.use_vv_mechanism = mode_config['use_vv']
        models[mode_name] = CLIPSurgeryWrapper(config)
        print(f"  {mode_name}: loaded")
    
    # Load dataset
    unseen_classes = ['airplane', 'bridge', 'storagetank', 'vehicle', 'windmill']
    dataset = SeenUnseenDataset(args.dataset, split='all', mode='val', unseen_classes=unseen_classes)
    
    output_dir = Path(__file__).parent / 'multi_class_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process samples
    processed = 0
    for idx in range(len(dataset)):
        if processed >= args.max_samples:
            break
        
        sample = dataset[idx]
        classes_in_image = sample.get('classes', [sample['class_name']])
        
        # Get unique classes
        unique_classes = list(set(classes_in_image))
        
        # Only process images with multiple unique classes
        if len(unique_classes) < 2:
            continue
        
        print(f"\n{'='*70}")
        print(f"Sample {idx}: {sample['image_id']}")
        print(f"Unique classes: {unique_classes} (total {len(classes_in_image)} objects)")
        print(f"{'='*70}")
        
        # Prepare data
        image_tensor = sample['image'].unsqueeze(0).to(config.device)
        image_data = {
            'image_tensor': image_tensor[0],
            'image_id': sample['image_id'],
            'original_size': sample['original_size']
        }
        
        # Generate heatmaps for each unique class (5 modes per class)
        print(f"Generating 5-mode heatmaps for {len(unique_classes)} classes...")
        
        for query_class in unique_classes:
            print(f"  Query: {query_class}")
            
            # Generate heatmaps for all 5 modes
            heatmaps_per_mode = generate_multi_mode_heatmaps(
                models, image_tensor, query_class,
                dior_classes, dior_prompts, config, args.layers
            )
            
            # Visualize 5-mode comparison
            output_path = output_dir / f'{sample["image_id"]}_{query_class}.png'
            visualize_5mode_comparison(
                image_data, query_class, heatmaps_per_mode,
                sample['bboxes'], args.layers, list(mode_configs.keys()), output_path
            )
            
            print(f"    Saved: {output_path.name}")
        
        processed += 1
    
    print(f"\n{'='*70}")
    print(f"Total processed: {processed} multi-class images")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

