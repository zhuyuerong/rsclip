# -*- coding: utf-8 -*-
"""
统一热图生成器 - 整合所有功能

整合功能:
1. 文本引导VV^T热图生成 (text_guided_vvt.py)
2. 多类别4模式对比 (multi_class_heatmap.py) 
3. GT边界框调试可视化 (debug_gt_boxes.py)
4. 全面对比实验 (comprehensive_comparison.py)

支持模式:
- 4种模式对比: With Surgery, Without Surgery, With VV, Complete Surgery
- 12层热图分析: L1-L12
- 多类别图像处理: 每个类别独立查询
- GT边界框可视化: 精确坐标缩放和显示
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
from PIL import Image

# 添加项目根目录
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.append(str(root_dir))

# 添加surgery_clip目录
surgery_clip_dir = Path(__file__).parent.parent
sys.path.append(str(surgery_clip_dir))

from experiment4.core.config import Config
from experiment4.core.models.clip_surgery import CLIPSurgeryWrapper, clip_feature_surgery, get_similarity_map
from utils.seen_unseen_split import SeenUnseenDataset


class UnifiedHeatmapGenerator:
    """统一热图生成器"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # DIOR数据集20个类别
        self.dior_classes_raw = [
            'airplane', 'airport', 'baseballfield', 'basketballcourt', 
            'bridge', 'chimney', 'dam', 'Expressway-Service-area',
            'Expressway-toll-station', 'golffield', 'groundtrackfield',
            'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
            'tenniscourt', 'trainstation', 'vehicle', 'windmill'
        ]
        self.dior_prompts = [f"an aerial photo of {cls}" for cls in self.dior_classes_raw]
        
        # 4种模式配置
        self.mode_configs = {
            '1.With Surgery': {'use_surgery': True, 'use_vv': False},
            '2.Without Surgery': {'use_surgery': False, 'use_vv': False},
            '3.With VV': {'use_surgery': False, 'use_vv': True},
            '4.Complete Surgery': {'use_surgery': True, 'use_vv': True},
        }
        
        # 加载所有模式模型
        self.models = self._load_all_models()
    
    def _load_all_models(self):
        """加载所有4种模式的模型"""
        print("Loading models for all 4 modes...")
        models = {}
        
        for mode_name, mode_config in self.mode_configs.items():
            # 为每个模式创建独立的config实例
            mode_cfg = Config()
            mode_cfg.dataset_root = self.config.dataset_root
            mode_cfg.device = self.device
            mode_cfg.use_surgery = mode_config['use_surgery']
            mode_cfg.use_vv_mechanism = mode_config['use_vv']
            
            models[mode_name] = CLIPSurgeryWrapper(mode_cfg)
            print(f"  {mode_name}: loaded (surgery={mode_config['use_surgery']}, vv={mode_config['use_vv']})")
        
        return models
    
    def generate_multi_mode_heatmaps(self, image, query_class, layers):
        """
        为一个查询类别生成4种模式的热图
        
        Args:
            image: [1, 3, H, W] single image
            query_class: str (query class name)
            layers: list of layer indices
        
        Returns:
            heatmaps_per_mode: {mode_name: {layer_idx: [1, 1, H, W]}}
        """
        class_idx = self.dior_classes_raw.index(query_class)
        heatmaps_per_mode = {}
        
        for mode_name, model in self.models.items():
            # 提取多层特征
            layer_features_dict = model.get_layer_features(image, layer_indices=layers)
            
            # 编码所有类别文本
            all_text_features = model.encode_text(self.dior_prompts)
            all_text_features = F.normalize(all_text_features, dim=-1)
            
            heatmaps_per_mode[mode_name] = {}
            
            for layer_idx in layers:
                image_feature = layer_features_dict[layer_idx]  # [1, N+1, C]
                
                # 根据模式选择相似度计算方式
                if "Surgery" in mode_name:
                    similarity = clip_feature_surgery(image_feature, all_text_features, t=2)
                else:
                    patch_features = image_feature[:, 1:, :]  # [1, N_patches, C]
                    similarity = patch_features @ all_text_features.t()  # [1, N_patches, N_classes]
                
                # 提取目标类别的相似度
                target_similarity = similarity[:, :, class_idx:class_idx+1]  # [1, N_patches, 1]
                
                # 生成热图
                heatmap = get_similarity_map(target_similarity, (self.config.image_size, self.config.image_size))
                heatmaps_per_mode[mode_name][layer_idx] = heatmap
        
        return heatmaps_per_mode
    
    def visualize_4mode_comparison(self, image_data, query_class, heatmaps_per_mode, bboxes, layers, output_path):
        """
        可视化4模式对比
        
        Layout: 4 modes x (1 original + 12 layers) columns
        """
        num_modes = len(self.mode_configs)
        num_layers = len(layers)
        
        # 反归一化图像
        img = image_data['image_tensor'].cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        # 获取缩放信息
        original_h, original_w = image_data['original_size']
        scale_x = 224.0 / original_w
        scale_y = 224.0 / original_h
        
        # 创建图形: 4 modes x (1 + num_layers) columns
        fig, axes = plt.subplots(num_modes, num_layers + 1, 
                                figsize=(2.5 * (num_layers + 1), 2.5 * num_modes))
        
        # 对每个模式
        for row, mode_name in enumerate(self.mode_configs.keys()):
            # 第0列: 原图 + 查询类别GT框
            axes[row, 0].imshow(img)
            prompt = f"an aerial photo of {query_class}"
            title = f'{mode_name}\n{prompt}\nID: {image_data["image_id"]}'
            axes[row, 0].set_title(title, fontsize=6.5)
            axes[row, 0].axis('off')
            
            # 绘制查询类别的GT框
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
                axes[row, 0].add_patch(rect)
            
            # 第1-N列: 层热图
            for col, layer_idx in enumerate(layers):
                heatmap = heatmaps_per_mode[mode_name][layer_idx][0, 0].detach().cpu().numpy()
                
                axes[row, col + 1].imshow(img)
                axes[row, col + 1].imshow(heatmap, cmap='jet', alpha=0.5)
                
                if row == 0:
                    axes[row, col + 1].set_title(f'L{layer_idx}', fontsize=8)
                axes[row, col + 1].axis('off')
                
                # 在热图上绘制GT框
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
    
    def debug_gt_boxes(self, dataset, num_samples=5, output_dir=None):
        """
        GT边界框调试可视化
        
        Args:
            dataset: SeenUnseenDataset
            num_samples: 要可视化的样本数
            output_dir: 输出目录
        """
        if output_dir is None:
            output_dir = Path(__file__).parent / 'gt_box_debug'
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for idx in range(min(num_samples, len(dataset))):
            sample = dataset[idx]
            
            # 获取图像信息
            image_id = sample.get('image_id', f'sample_{idx}')
            class_name = sample['class_name']
            original_size = sample.get('original_size', (224, 224))  # (H, W)
            bboxes = sample['bboxes']
            
            # 加载原图
            image_path = sample.get('image_path', '')
            if image_path and Path(image_path).exists():
                img_pil = Image.open(image_path).convert('RGB')
                img_np = np.array(img_pil)
            else:
                # 使用变换后的图像并反归一化
                img_tensor = sample['image']
                img = img_tensor.permute(1, 2, 0).numpy()
                mean = np.array([0.48145466, 0.4578275, 0.40821073])
                std = np.array([0.26862954, 0.26130258, 0.27577711])
                img = img * std + mean
                img_np = np.clip(img, 0, 1)
            
            # 创建2个子图
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # 左图: 原图 + GT框 (224x224)
            axes[0].imshow(img_np)
            axes[0].set_title(f'Image ID: {image_id}\nClass: {class_name}\n224x224 (Scaled)', fontsize=10)
            
            # 计算缩放因子
            orig_h, orig_w = original_size
            scale_x = 224.0 / orig_w
            scale_y = 224.0 / orig_h
            
            # 绘制所有bbox
            for i, bbox in enumerate(bboxes):
                # 缩放bbox
                xmin = bbox['xmin'] * scale_x
                ymin = bbox['ymin'] * scale_y
                xmax = bbox['xmax'] * scale_x
                ymax = bbox['ymax'] * scale_y
                w = xmax - xmin
                h = ymax - ymin
                
                # 绘制矩形
                rect = patches.Rectangle((xmin, ymin), w, h, 
                                        linewidth=2, edgecolor='lime', facecolor='none')
                axes[0].add_patch(rect)
                
                # 添加bbox编号标签
                axes[0].text(xmin, ymin-5, f'Box{i}', color='lime', fontsize=8, 
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
            
            axes[0].axis('off')
            
            # 右图: 坐标信息文本
            axes[1].axis('off')
            
            # 构建信息文本
            info_lines = [
                f"IMAGE INFORMATION",
                f"=" * 50,
                f"Image ID: {image_id}",
                f"Class: {class_name}",
                f"Original Size: {orig_w}x{orig_h} pixels",
                f"Scaled Size: 224x224 pixels",
                f"Scale Factor: x={scale_x:.4f}, y={scale_y:.4f}",
                f"",
                f"BOUNDING BOXES ({len(bboxes)} total)",
                f"=" * 50,
            ]
            
            for i, bbox in enumerate(bboxes):
                xmin_orig = bbox['xmin']
                ymin_orig = bbox['ymin']
                xmax_orig = bbox['xmax']
                ymax_orig = bbox['ymax']
                
                xmin_scaled = xmin_orig * scale_x
                ymin_scaled = ymin_orig * scale_y
                xmax_scaled = xmax_orig * scale_x
                ymax_scaled = ymax_orig * scale_y
                
                info_lines.extend([
                    f"",
                    f"Box {i}: {bbox.get('class', class_name)}",
                    f"  Original coords ({orig_w}x{orig_h}):",
                    f"    xmin={xmin_orig}, ymin={ymin_orig}",
                    f"    xmax={xmax_orig}, ymax={ymax_orig}",
                    f"    size={xmax_orig-xmin_orig}x{ymax_orig-ymin_orig}",
                    f"  Scaled coords (224x224):",
                    f"    xmin={xmin_scaled:.1f}, ymin={ymin_scaled:.1f}",
                    f"    xmax={xmax_scaled:.1f}, ymax={ymax_scaled:.1f}",
                    f"    size={xmax_scaled-xmin_scaled:.1f}x{ymax_scaled-ymin_scaled:.1f}",
                ])
            
            info_text = '\n'.join(info_lines)
            axes[1].text(0.05, 0.95, info_text, transform=axes[1].transAxes,
                        fontsize=9, verticalalignment='top', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(output_dir / f'debug_sample{idx}_{image_id}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Sample {idx} ({image_id}) debug visualization saved")
        
        print(f"\nAll debug visualizations saved to: {output_dir}")
    
    def generate_comprehensive_comparison(self, images, class_names, bboxes_batch, image_ids, layers, output_dir):
        """
        生成全面对比: 4种模式 x 12层热图
        
        Args:
            images: [B, 3, H, W]
            class_names: list of str
            bboxes_batch: list of bbox dicts
            image_ids: list of image IDs
            layers: list of layer indices
            output_dir: Path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        B = images.shape[0]
        num_layers = len(layers)
        num_modes = len(self.mode_configs)
        
        for b in range(min(B, 5)):  # 最多5个样本
            # 创建网格: num_modes行 x (1+12)列
            fig, axes = plt.subplots(num_modes, num_layers + 1, 
                                    figsize=(2.5 * (num_layers + 1), 2.5 * num_modes))
            
            # 准备原图
            img = images[b].cpu().permute(1, 2, 0).numpy()
            mean = np.array([0.48145466, 0.4578275, 0.40821073])
            std = np.array([0.26862954, 0.26130258, 0.27577711])
            img = img * std + mean
            img = np.clip(img, 0, 1)
            
            # 对每个模式
            for row, mode_name in enumerate(self.mode_configs.keys()):
                # 第0列: 原图 + 模式标签 + 图像ID
                axes[row, 0].imshow(img)
                prompt_text = f"an aerial photo of {class_names[b]}"
                title_text = f'{mode_name}\n{prompt_text}\nID: {image_ids[b]}'
                axes[row, 0].set_title(title_text, fontsize=6.5)
                axes[row, 0].axis('off')
                
                # 绘制GT边界框（仅匹配类别）
                if b < len(bboxes_batch):
                    bbox_info = bboxes_batch[b]
                    target_class = class_names[b]
                    
                    if 'boxes' in bbox_info and len(bbox_info['boxes']) > 0:
                        original_h, original_w = bbox_info.get('original_size', (224, 224))
                        scale_x = 224.0 / original_w
                        scale_y = 224.0 / original_h
                        
                        for bbox in bbox_info['boxes']:
                            # 仅绘制匹配文本查询的bbox
                            bbox_class = bbox.get('class', target_class)
                            if bbox_class != target_class:
                                continue
                            
                            # 缩放bbox到224x224
                            xmin = bbox['xmin'] * scale_x
                            ymin = bbox['ymin'] * scale_y
                            xmax = bbox['xmax'] * scale_x
                            ymax = bbox['ymax'] * scale_y
                            w, h = xmax - xmin, ymax - ymin
                            
                            rect = patches.Rectangle((xmin, ymin), w, h, 
                                                    linewidth=1.5, edgecolor='lime', facecolor='none')
                            axes[row, 0].add_patch(rect)
                
                # 后续列: 层热图
                # 为这个模式生成热图
                image_tensor = images[b:b+1].to(self.device)
                heatmaps_per_mode = self.generate_multi_mode_heatmaps(
                    image_tensor, class_names[b], layers
                )
                
                for col, layer_idx in enumerate(layers):
                    heatmap = heatmaps_per_mode[mode_name][layer_idx][0, 0].detach().cpu().numpy()
                    
                    # 叠加热图
                    axes[row, col + 1].imshow(img)
                    axes[row, col + 1].imshow(heatmap, cmap='jet', alpha=0.5)
                    
                    # 仅在第一行显示层标签
                    if row == 0:
                        axes[row, col + 1].set_title(f'L{layer_idx}', fontsize=8)
                    axes[row, col + 1].axis('off')
                    
                    # 在热图上绘制GT框（仅匹配类别）
                    if b < len(bboxes_batch):
                        bbox_info = bboxes_batch[b]
                        target_class = class_names[b]
                        
                        if 'boxes' in bbox_info and len(bbox_info['boxes']) > 0:
                            original_h, original_w = bbox_info.get('original_size', (224, 224))
                            scale_x = 224.0 / original_w
                            scale_y = 224.0 / original_h
                            
                            for bbox in bbox_info['boxes']:
                                # 仅绘制匹配的bbox
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
    
    def process_multi_class_images(self, dataset, max_samples=5, layers=None):
        """
        处理多类别图像，为每个类别生成4模式对比
        
        Args:
            dataset: SeenUnseenDataset
            max_samples: 最大样本数
            layers: 要分析的层，默认L1-L12
        """
        if layers is None:
            layers = list(range(1, 13))
        
        output_dir = Path(__file__).parent / 'multi_class_results'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processed = 0
        for idx in range(len(dataset)):
            if processed >= max_samples:
                break
            
            sample = dataset[idx]
            classes_in_image = sample.get('classes', [sample['class_name']])
            
            # 获取唯一类别
            unique_classes = list(set(classes_in_image))
            
            # 只处理多类别图像
            if len(unique_classes) < 2:
                continue
            
            print(f"\n{'='*70}")
            print(f"Sample {idx}: {sample['image_id']}")
            print(f"Unique classes: {unique_classes} (total {len(classes_in_image)} objects)")
            print(f"{'='*70}")
            
            # 准备数据
            image_tensor = sample['image'].unsqueeze(0).to(self.device)
            image_data = {
                'image_tensor': image_tensor[0],
                'image_id': sample['image_id'],
                'original_size': sample['original_size']
            }
            
            # 为每个唯一类别生成4模式热图
            print(f"Generating 4-mode heatmaps for {len(unique_classes)} classes...")
            
            for query_class in unique_classes:
                print(f"  Query: {query_class}")
                
                # 生成4模式热图
                heatmaps_per_mode = self.generate_multi_mode_heatmaps(
                    image_tensor, query_class, layers
                )
                
                # 可视化4模式对比
                output_path = output_dir / f'{sample["image_id"]}_{query_class}.png'
                self.visualize_4mode_comparison(
                    image_data, query_class, heatmaps_per_mode,
                    sample['bboxes'], layers, output_path
                )
                
                print(f"    Saved: {output_path.name}")
            
            processed += 1
        
        print(f"\n{'='*70}")
        print(f"Total processed: {processed} multi-class images")
        print(f"Output directory: {output_dir}")
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='统一热图生成器 - 整合所有功能')
    parser.add_argument('--dataset', type=str, default='datasets/mini_dataset',
                        help='数据集路径')
    parser.add_argument('--mode', type=str, 
                        choices=['multi_class', 'debug_gt', 'comprehensive', 'all'],
                        default='all',
                        help='运行模式: multi_class(多类别), debug_gt(GT调试), comprehensive(全面对比), all(全部)')
    parser.add_argument('--max-samples', type=int, default=5,
                        help='最大样本数')
    parser.add_argument('--layers', type=int, nargs='+', default=list(range(1, 13)),
                        help='要分析的层')
    parser.add_argument('--debug-samples', type=int, default=3,
                        help='GT调试样本数')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("统一热图生成器 - 整合所有功能")
    print("=" * 80)
    print(f"数据集: {args.dataset}")
    print(f"运行模式: {args.mode}")
    print(f"最大样本数: {args.max_samples}")
    print(f"分析层: {args.layers}")
    print(f"GT调试样本数: {args.debug_samples}")
    
    # 配置
    config = Config()
    config.dataset_root = args.dataset
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建生成器
    generator = UnifiedHeatmapGenerator(config)
    
    # 加载数据集
    print(f"\n加载数据集...")
    unseen_classes = ['airplane', 'bridge', 'storagetank', 'vehicle', 'windmill']
    dataset = SeenUnseenDataset(
        config.dataset_root,
        split='all',
        mode='val',
        unseen_classes=unseen_classes
    )
    print(f"✓ 数据集加载完成: {len(dataset)}个样本")
    
    # 根据模式运行
    if args.mode in ['multi_class', 'all']:
        print(f"\n{'='*50}")
        print("多类别4模式对比")
        print(f"{'='*50}")
        generator.process_multi_class_images(dataset, args.max_samples, args.layers)
    
    if args.mode in ['debug_gt', 'all']:
        print(f"\n{'='*50}")
        print("GT边界框调试可视化")
        print(f"{'='*50}")
        generator.debug_gt_boxes(dataset, args.debug_samples)
    
    if args.mode in ['comprehensive', 'all']:
        print(f"\n{'='*50}")
        print("全面对比实验")
        print(f"{'='*50}")
        
        # 收集样本
        all_images = []
        all_class_names = []
        all_bboxes = []
        all_image_ids = []
        
        for idx in range(min(args.max_samples, len(dataset))):
            sample = dataset[idx]
            all_images.append(sample['image'])
            all_class_names.append(sample['class_name'])
            all_image_ids.append(sample.get('image_id', f'sample_{idx}'))
            
            # 重新格式化bbox信息
            bbox_info = {
                'boxes': sample['bboxes'],
                'original_size': sample.get('original_size', (224, 224))
            }
            all_bboxes.append(bbox_info)
        
        images = torch.stack(all_images).to(config.device)
        print(f"收集了 {len(all_class_names)} 个样本")
        
        # 生成全面对比
        output_dir = Path(__file__).parent / 'comprehensive_comparison_results'
        generator.generate_comprehensive_comparison(
            images, all_class_names, all_bboxes, all_image_ids, args.layers, output_dir
        )
    
    print(f"\n{'='*80}")
    print("所有任务完成！")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
