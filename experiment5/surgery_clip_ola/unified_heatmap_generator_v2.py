# -*- coding: utf-8 -*-
"""
Experiment 5: 3种模式对比 - OLA去接缝验证

3行对比:
- Row 1: Baseline (RemoteCLIP + 余弦)
- Row 2: Complete Surgery (Surgery + VV)  
- Row 3: Baseline + OLA (RemoteCLIP + 余弦 + OLA去接缝)

目的: 验证OLA对去接缝条纹的效果
"""

import sys
from pathlib import Path
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from PIL import Image

# 添加项目根目录
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from experiment4.core.config import Config
from experiment4.core.models.clip_surgery import CLIPSurgeryWrapper, get_similarity_map
from experiment4.experiments.surgery_clip.utils.seen_unseen_split import SeenUnseenDataset


# ========== OLA Functions ==========

def create_blending_weight(h: int, w: int, device='cuda'):
    """生成余弦权重窗口（中心高、边缘低）"""
    device = torch.device(device) if isinstance(device, str) else device
    y = torch.linspace(-np.pi, np.pi, h, device=device)
    x = torch.linspace(-np.pi, np.pi, w, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    wy = (torch.cos(yy) + 1) * 0.5
    wx = (torch.cos(xx) + 1) * 0.5
    weight = wy * wx
    weight = (weight / (weight.max() + 1e-8)).clamp_min(1e-6)
    return weight  # [h, w]


def extract_sliding_windows(H: int, W: int, win_h: int, win_w: int, stride: int):
    """生成滑窗坐标"""
    ys = list(range(0, max(1, H - win_h + 1), stride))
    xs = list(range(0, max(1, W - win_w + 1), stride))
    if len(ys) > 0 and ys[-1] != H - win_h and H > win_h:
        ys.append(max(0, H - win_h))
    if len(xs) > 0 and xs[-1] != W - win_w and W > win_w:
        xs.append(max(0, W - win_w))
    return [(y, x) for y in ys for x in xs]


@torch.no_grad()
def stitch_ola(tiles, coords, out_h, out_w, device, pmin=5.0, pmax=95.0):
    """OLA拼接 + 分位归一化"""
    th, tw = tiles[0].shape[-2], tiles[0].shape[-1]
    weight = create_blending_weight(th, tw, device).view(1, 1, th, tw)
    
    acc = torch.zeros(1, 1, out_h, out_w, device=device)
    acc_w = torch.zeros(1, 1, out_h, out_w, device=device)
    
    for t, (ty, tx) in zip(tiles, coords):
        acc[:, :, ty:ty+th, tx:tx+tw] += t * weight
        acc_w[:, :, ty:ty+th, tx:tx+tw] += weight
    
    heat = acc / (acc_w + 1e-8)
    
    # 分位归一化
    vals = heat.flatten()
    lo = torch.quantile(vals, float(pmin / 100.0))
    hi = torch.quantile(vals, float(pmax / 100.0))
    heat = (heat - lo) / (hi - lo + 1e-8)
    heat = heat.clamp(0, 1)
    
    return heat, acc_w


# ========== Main Generator ==========

class ThreeModeComparisonGenerator:
    """3种模式对比生成器"""
    
    def __init__(self, config, tile_size=224, tile_stride=112, pmin=5.0, pmax=95.0):
        self.config = config
        self.device = config.device
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.pmin = pmin
        self.pmax = pmax
        
        # DIOR类别
        self.dior_classes_raw = [
            'airplane', 'airport', 'baseballfield', 'basketballcourt', 
            'bridge', 'chimney', 'dam', 'Expressway-Service-area',
            'Expressway-toll-station', 'golffield', 'groundtrackfield',
            'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
            'tenniscourt', 'trainstation', 'vehicle', 'windmill'
        ]
        self.dior_prompts = [f"an aerial photo of {cls}" for cls in self.dior_classes_raw]
        
        # 3种模式配置
        self.mode_configs = {
            '1.Baseline': {'use_surgery': False, 'use_vv': False, 'use_ola': False},
            '2.Complete Surgery': {'use_surgery': True, 'use_vv': True, 'use_ola': False},
            '3.Baseline+OLA': {'use_surgery': False, 'use_vv': False, 'use_ola': True},
        }
        
        # 加载模型（只需2个：baseline和complete）
        self.models = self._load_models()
    
    def _load_models(self):
        """加载baseline和complete surgery两个模型"""
        print("Loading models for 3-mode comparison...")
        models = {}
        
        # Baseline model
        cfg_baseline = Config()
        cfg_baseline.dataset_root = self.config.dataset_root
        cfg_baseline.device = self.device
        cfg_baseline.use_surgery = False
        cfg_baseline.use_vv_mechanism = False
        models['baseline'] = CLIPSurgeryWrapper(cfg_baseline)
        print("  Baseline: loaded (RemoteCLIP + cosine)")
        
        # Complete Surgery model
        cfg_complete = Config()
        cfg_complete.dataset_root = self.config.dataset_root
        cfg_complete.device = self.device
        cfg_complete.use_surgery = True
        cfg_complete.use_vv_mechanism = True
        models['complete'] = CLIPSurgeryWrapper(cfg_complete)
        print("  Complete Surgery: loaded (Surgery + VV)")
        
        return models
    
    def generate_3mode_heatmaps(self, image, query_class, layers):
        """生成3种模式的热图"""
        class_idx = self.dior_classes_raw.index(query_class)
        heatmaps_per_mode = {}
        out_h, out_w = self.config.image_size, self.config.image_size
        
        for mode_name, mode_cfg in self.mode_configs.items():
            heatmaps_per_mode[mode_name] = {}
            
            # 选择模型
            if mode_name == '2.Complete Surgery':
                model = self.models['complete']
            else:
                model = self.models['baseline']
            
            # 编码文本
            all_text_features = model.encode_text(self.dior_prompts)
            all_text_features = F.normalize(all_text_features, dim=-1)
            
            # 提取多层特征
            layer_features_dict = model.get_layer_features(image, layer_indices=layers)
            
            for layer_idx in layers:
                image_feature = layer_features_dict[layer_idx]
                
                # 计算相似度
                similarity = model.compute_similarity(image_feature, all_text_features)
                target_similarity = similarity[:, :, class_idx:class_idx+1]
                
                # 生成热图
                if mode_cfg['use_ola']:
                    # OLA路径
                    full_heat = get_similarity_map(target_similarity, (out_h, out_w))
                    coords = extract_sliding_windows(out_h, out_w, self.tile_size, self.tile_size, self.tile_stride)
                    tiles = [full_heat[:, :, y:y+self.tile_size, x:x+self.tile_size] for (y, x) in coords]
                    stitched, acc_w = stitch_ola(tiles, coords, out_h, out_w, self.device, self.pmin, self.pmax)
                    heatmaps_per_mode[mode_name][layer_idx] = stitched
                    if layer_idx == layers[-1]:
                        heatmaps_per_mode[mode_name]['acc_w'] = acc_w
                else:
                    # 标准路径
                    heatmap = get_similarity_map(target_similarity, (out_h, out_w))
                    heatmaps_per_mode[mode_name][layer_idx] = heatmap
        
        return heatmaps_per_mode
    
    def visualize_3mode_comparison(self, image_data, query_class, heatmaps_per_mode, bboxes, layers, output_path):
        """可视化3模式对比"""
        num_modes = 3
        num_layers = len(layers)
        
        # 反归一化图像
        img = image_data['image_tensor'].cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        # 获取缩放信息（参考exp3）
        original_h, original_w = image_data['original_size']
        scale_x = 224.0 / original_w
        scale_y = 224.0 / original_h
        
        # 创建图形: 3 modes x (1 + num_layers) columns
        fig, axes = plt.subplots(num_modes, num_layers + 1, 
                                figsize=(2.5 * (num_layers + 1), 2.5 * num_modes))
        
        # 对每个模式
        for row, mode_name in enumerate(self.mode_configs.keys()):
            # 第0列: 原图 + GT框
            axes[row, 0].imshow(img)
            prompt = f"an aerial photo of {query_class}"
            title = f'{mode_name}\n{prompt}\nID: {image_data["image_id"]}'
            axes[row, 0].set_title(title, fontsize=7)
            axes[row, 0].axis('off')
            
            # 绘制查询类别的GT框（参考exp3）
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
    
    def visualize_ola_diagnosis(self, image, heatmap, acc_w, output_path):
        """诊断OLA Row 3"""
        img = image[0].detach().cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        img = np.clip(img * std + mean, 0, 1)
        
        hm = heatmap[0, 0].detach().cpu().numpy()
        w = acc_w[0, 0].detach().cpu().numpy()
        w_norm = (w - w.min()) / (w.max() - w.min() + 1e-8)
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(img); axes[0].set_title('Original', fontsize=10); axes[0].axis('off')
        axes[1].imshow(img); axes[1].imshow(hm, cmap='jet', alpha=0.7)
        axes[1].set_title('OLA Heatmap', fontsize=10); axes[1].axis('off')
        axes[2].imshow(hm, cmap='jet'); axes[2].set_title('Pure Heatmap', fontsize=10); axes[2].axis('off')
        
        im = axes[3].imshow(w_norm, cmap='viridis')
        axes[3].set_title('Coverage (Uniform=No Seams)', fontsize=10); axes[3].axis('off')
        plt.colorbar(im, ax=axes[3], fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Coverage: min={w.min():.2f}, max={w.max():.2f}, std={w.std():.4f}")
    
    def process_multi_class_images(self, dataset, max_samples=10, layers=None, save_diagnosis=True):
        """处理多类别图像"""
        if layers is None:
            layers = list(range(1, 13))
        
        output_dir = Path(__file__).parent / 'results' / '3mode_comparison'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processed = 0
        for idx in range(len(dataset)):
            if processed >= max_samples:
                break
            
            sample = dataset[idx]
            classes_in_image = sample.get('classes', [sample['class_name']])
            unique_classes = list(set(classes_in_image))
            
            # 优先多类别
            if len(unique_classes) < 2 and processed >= max_samples // 2:
                pass
            elif len(unique_classes) < 2:
                continue
            
            print(f"\n{'='*70}")
            print(f"Sample {idx}: {sample['image_id']}")
            print(f"Unique classes: {unique_classes} (total {len(classes_in_image)} objects)")
            print(f"{'='*70}")
            
            image_tensor = sample['image'].unsqueeze(0).to(self.device)
            image_data = {
                'image_tensor': image_tensor[0],
                'image_id': sample['image_id'],
                'original_size': sample.get('original_size', (224, 224))
            }
            
            print(f"Generating 3-mode heatmaps for {len(unique_classes)} classes...")
            
            for query_class in unique_classes:
                print(f"  Query: {query_class}")
                
                # 生成3模式热图
                heatmaps_per_mode = self.generate_3mode_heatmaps(image_tensor, query_class, layers)
                
                # 可视化
                output_path = output_dir / f'{sample["image_id"]}_{query_class}.png'
                self.visualize_3mode_comparison(image_data, query_class, heatmaps_per_mode,
                                               sample['bboxes'], layers, output_path)
                print(f"    Saved: {output_path.name}")
                
                # OLA诊断（可选）
                if save_diagnosis and 'acc_w' in heatmaps_per_mode.get('3.Baseline+OLA', {}):
                    diag_path = output_dir / f'{sample["image_id"]}_{query_class}_ola_diag.png'
                    acc_w = heatmaps_per_mode['3.Baseline+OLA']['acc_w']
                    last_heat = heatmaps_per_mode['3.Baseline+OLA'][layers[-1]]
                    self.visualize_ola_diagnosis(image_tensor, last_heat, acc_w, diag_path)
                    print(f"    Diagnosis: {diag_path.name}")
            
            processed += 1
        
        print(f"\n{'='*70}")
        print(f"Total: {processed} images processed")
        print(f"Output: {output_dir}")
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Exp5: 3种模式对比 (Baseline / Complete / OLA)')
    parser.add_argument('--dataset', type=str, default='datasets/mini_dataset')
    parser.add_argument('--max-samples', type=int, default=10)
    parser.add_argument('--layers', type=int, nargs='+', default=list(range(1, 13)))
    parser.add_argument('--tile-size', type=int, default=224)
    parser.add_argument('--tile-stride', type=int, default=112)
    parser.add_argument('--pmin', type=float, default=5.0)
    parser.add_argument('--pmax', type=float, default=95.0)
    parser.add_argument('--no-diagnosis', action='store_true', help='不保存OLA诊断图')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Experiment 5: 3-Mode Comparison (Baseline / Complete Surgery / OLA)")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Max samples: {args.max_samples}")
    print(f"Layers: {args.layers}")
    print(f"OLA config: tile_size={args.tile_size}, stride={args.tile_stride}")
    print(f"Percentile: pmin={args.pmin}, pmax={args.pmax}")
    
    # 配置
    config = Config()
    config.dataset_root = args.dataset
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建生成器
    generator = ThreeModeComparisonGenerator(config, args.tile_size, args.tile_stride, 
                                            args.pmin, args.pmax)
    
    # 加载数据集
    print(f"\n加载数据集...")
    unseen_classes = ['airplane', 'bridge', 'storagetank', 'vehicle', 'windmill']
    dataset = SeenUnseenDataset(config.dataset_root, split='all', mode='val', 
                               unseen_classes=unseen_classes)
    print(f"✓ 数据集: {len(dataset)}个样本")
    
    # 处理
    generator.process_multi_class_images(dataset, args.max_samples, args.layers, 
                                        save_diagnosis=not args.no_diagnosis)
    
    print(f"\n{'='*80}")
    print("完成！")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

