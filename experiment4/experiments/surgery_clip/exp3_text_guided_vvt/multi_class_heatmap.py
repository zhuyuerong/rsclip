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


def generate_multi_class_heatmaps(model, image, classes_in_image, all_dior_classes, all_dior_prompts, config, layers):
    """
    为一张图像的每个类别生成热图
    
    Args:
        model: CLIPSurgeryWrapper
        image: [1, 3, H, W] single image
        classes_in_image: list of class names in this image
        all_dior_classes: list of 20 raw class names
        all_dior_prompts: list of 20 prompts ("an aerial photo of ...")
        config: Config
        layers: list of layer indices
    
    Returns:
        heatmaps_per_class: {class_name: {layer_idx: [1, 1, H, W]}}
    """
    # Extract multi-layer features
    layer_features_dict = model.get_layer_features(image, layer_indices=layers)
    
    # Encode all class texts
    all_text_features = model.encode_text(all_dior_prompts)
    all_text_features = F.normalize(all_text_features, dim=-1)
    
    heatmaps_per_class = {}
    
    # For each class in this image
    for query_class in classes_in_image:
        class_idx = all_dior_classes.index(query_class)
        heatmaps_per_class[query_class] = {}
        
        for layer_idx in layers:
            image_feature = layer_features_dict[layer_idx]  # [1, N+1, C]
            
            # Use Surgery to compute similarity
            similarity = clip_feature_surgery(image_feature, all_text_features, t=2)
            # [1, N_patches, N_classes]
            
            # Extract similarity for this class
            target_similarity = similarity[:, :, class_idx:class_idx+1]  # [1, N_patches, 1]
            
            # Generate heatmap
            heatmap = get_similarity_map(target_similarity, (config.image_size, config.image_size))
            # [1, 1, H, W]
            
            heatmaps_per_class[query_class][layer_idx] = heatmap
    
    return heatmaps_per_class


def visualize_multi_class_comparison(image_data, heatmaps_per_class, bboxes, layers, output_path):
    """
    Visualize heatmaps for different classes in the same image
    
    Layout: N_classes rows x (1 + 12 layers) columns
    """
    classes = list(heatmaps_per_class.keys())
    num_classes = len(classes)
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
    
    # Create figure
    fig, axes = plt.subplots(num_classes, num_layers + 1, 
                            figsize=(2.5 * (num_layers + 1), 2.5 * num_classes))
    
    # Handle single row case
    if num_classes == 1:
        axes = axes.reshape(1, -1)
    
    # For each class
    for row, query_class in enumerate(classes):
        # Column 0: Original image with this class's GT boxes
        axes[row, 0].imshow(img)
        prompt = f"an aerial photo of {query_class}"
        title = f'{prompt}\nID: {image_data["image_id"]}'
        axes[row, 0].set_title(title, fontsize=7)
        axes[row, 0].axis('off')
        
        # Draw GT boxes for this class in green, others in yellow
        for bbox in bboxes:
            xmin = bbox['xmin'] * scale_x
            ymin = bbox['ymin'] * scale_y
            xmax = bbox['xmax'] * scale_x
            ymax = bbox['ymax'] * scale_y
            w, h = xmax - xmin, ymax - ymin
            
            bbox_class = bbox.get('class', query_class)
            if bbox_class == query_class:
                color = 'lime'
                linewidth = 2.5
            else:
                color = 'yellow'
                linewidth = 1.0
            
            rect = patches.Rectangle((xmin, ymin), w, h, 
                                    linewidth=linewidth, edgecolor=color, facecolor='none')
            axes[row, 0].add_patch(rect)
        
        # Columns 1-12: Layer heatmaps
        for col, layer_idx in enumerate(layers):
            heatmap = heatmaps_per_class[query_class][layer_idx][0, 0].detach().cpu().numpy()
            
            axes[row, col + 1].imshow(img)
            axes[row, col + 1].imshow(heatmap, cmap='jet', alpha=0.5)
            
            if row == 0:
                axes[row, col + 1].set_title(f'L{layer_idx}', fontsize=8)
            axes[row, col + 1].axis('off')
            
            # Draw GT boxes on heatmap
            for bbox in bboxes:
                xmin = bbox['xmin'] * scale_x
                ymin = bbox['ymin'] * scale_y
                xmax = bbox['xmax'] * scale_x
                ymax = bbox['ymax'] * scale_y
                w, h = xmax - xmin, ymax - ymin
                
                bbox_class = bbox.get('class', query_class)
                if bbox_class == query_class:
                    color = 'lime'
                    linewidth = 2.5
                else:
                    color = 'yellow'
                    linewidth = 1.0
                
                rect = patches.Rectangle((xmin, ymin), w, h, 
                                        linewidth=linewidth, edgecolor=color, facecolor='none')
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
    
    print("=" * 70)
    print("Multi-Class Heatmap Generator")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Layers: {args.layers}")
    
    # Setup
    config = Config()
    config.dataset_root = args.dataset
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.use_surgery = True  # Use Surgery
    config.use_vv_mechanism = False
    
    model = CLIPSurgeryWrapper(config)
    
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
        
        # Generate heatmaps for each unique class
        print(f"Generating heatmaps for {len(unique_classes)} classes...")
        heatmaps_per_class = generate_multi_class_heatmaps(
            model, image_tensor, unique_classes, 
            dior_classes, dior_prompts, config, args.layers
        )
        
        # Visualize
        output_path = output_dir / f'multi_class_{sample["image_id"]}.png'
        visualize_multi_class_comparison(
            image_data, heatmaps_per_class, sample['bboxes'], 
            args.layers, output_path
        )
        
        print(f"Saved: {output_path.name}")
        processed += 1
    
    print(f"\n{'='*70}")
    print(f"Total processed: {processed} multi-class images")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

