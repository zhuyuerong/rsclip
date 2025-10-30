# -*- coding: utf-8 -*-
"""
统一热图生成器 - OLA去接缝版本 (Experiment 5)

整合功能:
1. 文本引导VV^T热图生成 (text_guided_vvt.py)
2. 多类别4模式对比 (multi_class_heatmap.py) 
3. GT边界框调试可视化 (debug_gt_boxes.py)
4. OLA加权拼接（消除滑窗接缝条纹）

支持模式:
- 4种模式对比: With Surgery, Without Surgery, With VV, Complete Surgery
- 12层热图分析: L1-L12
- 多类别图像处理: 每个类别独立查询
- GT边界框可视化: 精确坐标缩放和显示
- OLA加权拼接: 消除接缝条纹，提升小目标可见性
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
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

# 添加exp4目录
exp4_dir = root_dir / 'experiment4'
sys.path.append(str(exp4_dir))

from experiment4.core.config import Config
from experiment4.core.models.clip_surgery import CLIPSurgeryWrapper, clip_feature_surgery, get_similarity_map
from experiment4.experiments.surgery_clip.utils.seen_unseen_split import SeenUnseenDataset


# ========== OLA (Overlap-Add) Functions for Seamless Stitching ==========

def create_blending_weight(h: int, w: int, blend_type='cosine', device='cuda'):
    """
    生成平滑过渡权重窗口（中心高、边缘低）
    
    Args:
        h, w: 窗口高宽
        blend_type: 'cosine'(推荐) | 'gaussian' | 'linear'
        device: torch device
    
    Returns:
        weight: [h, w] 归一化权重，中心≈1，边缘→0
    """
    device = torch.device(device) if isinstance(device, str) else device
    
    if blend_type == 'cosine':
        # 余弦窗口（Hann）：最平滑
        y = torch.linspace(-np.pi, np.pi, h, device=device)
        x = torch.linspace(-np.pi, np.pi, w, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        wy = (torch.cos(yy) + 1) * 0.5
        wx = (torch.cos(xx) + 1) * 0.5
        weight = wy * wx
    elif blend_type == 'gaussian':
        # 高斯窗口
        y = torch.linspace(-3, 3, h, device=device)
        x = torch.linspace(-3, 3, w, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        weight = torch.exp(-(xx**2 + yy**2) * 0.5)
    else:  # linear
        # 线性渐变
        y = torch.linspace(0, 1, h, device=device)
        x = torch.linspace(0, 1, w, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        wy = torch.minimum(yy, 1 - yy) * 2
        wx = torch.minimum(xx, 1 - xx) * 2
        weight = wy * wx
    
    weight = (weight / (weight.max() + 1e-8)).clamp_min(1e-6)
    return weight  # [h, w]


def extract_sliding_windows(H: int, W: int, win_h: int, win_w: int, stride_h: int, stride_w: int):
    """
    生成滑窗坐标（保证覆盖右下角）
    
    Returns:
        coords: list of (y, x) 左上角坐标
    """
    ys = list(range(0, max(1, H - win_h + 1), stride_h))
    xs = list(range(0, max(1, W - win_w + 1), stride_w))
    
    # 确保覆盖边界
    if len(ys) > 0 and ys[-1] != H - win_h and H > win_h:
        ys.append(max(0, H - win_h))
    if len(xs) > 0 and xs[-1] != W - win_w and W > win_w:
        xs.append(max(0, W - win_w))
    
    coords = [(y, x) for y in ys for x in xs]
    return coords


@torch.no_grad()
def stitch_ola(tiles, coords, out_h, out_w, device,
               global_normalize=True, use_percentile=True, pmin=5.0, pmax=95.0):
    """
    重叠-加权-平均 (Overlap-Add) 拼接 + 全图统一归一化
    
    Args:
        tiles: list of [1,1,th,tw]（未单窗归一化的热图值）
        coords: list of (y, x) 左上角坐标
        out_h, out_w: 输出大图尺寸
        device: torch device
        global_normalize: 是否拼接后统一归一化
        use_percentile: 使用分位归一化（抗异常值）
        pmin, pmax: 分位阈值百分比
    
    Returns:
        heat: [1,1,H,W] 归一化后热图
        acc_w: [1,1,H,W] 权重和（用于诊断覆盖均匀性）
    """
    assert len(tiles) == len(coords) and len(tiles) > 0, "tiles and coords must not be empty"
    
    th, tw = tiles[0].shape[-2], tiles[0].shape[-1]
    weight = create_blending_weight(th, tw, 'cosine', device).view(1, 1, th, tw)
    
    acc = torch.zeros(1, 1, out_h, out_w, device=device)
    acc_w = torch.zeros(1, 1, out_h, out_w, device=device)
    
    for t, (ty, tx) in zip(tiles, coords):
        acc[:, :, ty:ty+th, tx:tx+tw] += t * weight
        acc_w[:, :, ty:ty+th, tx:tx+tw] += weight
    
    heat = acc / (acc_w + 1e-8)
    
    if global_normalize:
        if use_percentile:
            # 分位归一化，抗极端值
            vals = heat.flatten()
            lo = torch.quantile(vals, float(pmin / 100.0))
            hi = torch.quantile(vals, float(pmax / 100.0))
            heat = (heat - lo) / (hi - lo + 1e-8)
        else:
            # 标准min-max
            mn = heat.amin(dim=(-2, -1), keepdim=True)
            mx = heat.amax(dim=(-2, -1), keepdim=True)
            heat = (heat - mn) / (mx - mn + 1e-8)
        
        heat = heat.clamp(0, 1)
    
    return heat, acc_w


def visualize_ola_diagnosis(image, heatmap, acc_w, output_path):
    """
    诊断OLA拼接质量（覆盖均匀性）
    
    Args:
        image: [1, 3, H, W] 原图
        heatmap: [1, 1, H, W] OLA热图
        acc_w: [1, 1, H, W] 权重和
        output_path: 保存路径
    """
    # 反归一化图像
    img = image[0].detach().cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    img = np.clip(img * std + mean, 0, 1)
    
    hm = heatmap[0, 0].detach().cpu().numpy()
    w = acc_w[0, 0].detach().cpu().numpy()
    w_norm = (w - w.min()) / (w.max() - w.min() + 1e-8)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. 原图
    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=10)
    axes[0].axis('off')
    
    # 2. 叠加热图
    axes[1].imshow(img)
    axes[1].imshow(hm, cmap='jet', alpha=0.7)
    axes[1].set_title('Heatmap (OLA)', fontsize=10)
    axes[1].axis('off')
    
    # 3. 纯热图
    axes[2].imshow(hm, cmap='jet')
    axes[2].set_title('Pure Heatmap', fontsize=10)
    axes[2].axis('off')
    
    # 4. 权重和图（诊断接缝）
    im = axes[3].imshow(w_norm, cmap='viridis')
    axes[3].set_title('Weight Sum\n(Uniform=No Seams)', fontsize=10)
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 打印覆盖统计
    print(f"  Coverage stats: min={w.min():.2f}, max={w.max():.2f}, std={w.std():.4f}")
    if w.std() / (w.mean() + 1e-8) < 0.1:
        print("  ✓ Coverage uniform, no visible seams")
    else:
        print("  ⚠ Coverage uneven, consider smaller stride")


# ========== End of OLA Functions ==========


class UnifiedHeatmapGenerator:
    """统一热图生成器（支持OLA去接缝）"""
    
    def __init__(self, config, enable_ola=False, tile_size=224, tile_stride=112, 
                 use_percentile=True, pmin=5.0, pmax=95.0):
        self.config = config
        self.device = config.device
        
        # OLA配置
        self.enable_ola = enable_ola
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.use_percentile = use_percentile
        self.pmin = pmin
        self.pmax = pmax
        
        # DIOR数据集20个类别
        self.dior_classes_raw = [
            'airplane', 'airport', 'baseballfield', 'basketballcourt', 
            'bridge', 'chimney', 'dam', 'Expressway-Service-area',
            'Expressway-toll-station', 'golffield', 'groundtrackfield',
            'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
            'tenniscourt', 'trainstation', 'vehicle', 'windmill'
        ]
        self.dior_prompts = [f"an aerial photo of {cls}" for cls in self.dior_classes_raw]
        
        # 3种模式配置：Baseline / Complete Surgery / Baseline+OLA
        self.mode_configs = {
            '1.Baseline': {'use_surgery': False, 'use_vv': False, 'use_ola': False},
            '2.Complete Surgery': {'use_surgery': True, 'use_vv': True, 'use_ola': False},
            '3.Baseline+OLA': {'use_surgery': False, 'use_vv': False, 'use_ola': True},
        }
        
        # 加载所有模式模型
        self.models = self._load_all_models()
    
    def _load_all_models(self):
        """加载baseline和complete surgery两个模型（Row3复用baseline）"""
        print("Loading models for 3-mode comparison...")
        models = {}
        
        # Baseline model (用于Row1和Row3)
        cfg_baseline = Config()
        cfg_baseline.dataset_root = self.config.dataset_root
        cfg_baseline.device = self.device
        cfg_baseline.use_surgery = False
        cfg_baseline.use_vv_mechanism = False
        models['baseline'] = CLIPSurgeryWrapper(cfg_baseline)
        print("  Baseline: loaded (RemoteCLIP + cosine)")
        
        # Complete Surgery model (用于Row2)
        cfg_complete = Config()
        cfg_complete.dataset_root = self.config.dataset_root
        cfg_complete.device = self.device
        cfg_complete.use_surgery = True
        cfg_complete.use_vv_mechanism = True
        models['complete'] = CLIPSurgeryWrapper(cfg_complete)
        print("  Complete Surgery: loaded (Surgery + VV)")
        
        return models
    
    def generate_multi_mode_heatmaps(self, image, query_class, layers):
        """
        为一个查询类别生成3种模式的热图（支持OLA去接缝）
        
        Args:
            image: [1, 3, H, W] single image
            query_class: str (query class name)
            layers: list of layer indices
        
        Returns:
            heatmaps_per_mode: {mode_name: {layer_idx: [1, 1, H, W]}}
                               {mode_name: {'acc_w': [1, 1, H, W]}}  # 如果enable_ola
        """
        class_idx = self.dior_classes_raw.index(query_class)
        heatmaps_per_mode = {}
        
        out_h, out_w = self.config.image_size, self.config.image_size
        
        for mode_name, mode_cfg in self.mode_configs.items():
            # 选择模型（Row1和Row3用baseline，Row2用complete）
            if mode_name == '2.Complete Surgery':
                model = self.models['complete']
            else:
                model = self.models['baseline']
            heatmaps_per_mode[mode_name] = {}
            
            # 编码所有类别文本
            all_text_features = model.encode_text(self.dior_prompts)
            all_text_features = F.normalize(all_text_features, dim=-1)
            
            # Row3启用OLA，Row1和Row2不启用
            use_ola_for_this_mode = mode_cfg['use_ola']
            
            if not use_ola_for_this_mode:
                # ========== 原始路径：整图一次性计算 ==========
                layer_features_dict = model.get_layer_features(image, layer_indices=layers)
                
                for layer_idx in layers:
                    image_feature = layer_features_dict[layer_idx]  # [1, N+1, C]
                    
                    # 统一由模型包装器决定（Surgery / VV / 二者组合）
                    similarity = model.compute_similarity(image_feature, all_text_features)
                    
                    # 提取目标类别的相似度
                    target_similarity = similarity[:, :, class_idx:class_idx+1]  # [1, N_patches, 1]
                    
                    # 生成热图
                    heatmap = get_similarity_map(target_similarity, (out_h, out_w))
                    heatmaps_per_mode[mode_name][layer_idx] = heatmap
            
            else:
                # ========== OLA路径：滑窗+加权平均拼接（消除接缝条纹） ==========
                layer_features_dict = model.get_layer_features(image, layer_indices=layers)
                
                for layer_idx in layers:
                    image_feature = layer_features_dict[layer_idx]  # [1, N+1, C]
                    
                    # 计算全图相似度
                    similarity = model.compute_similarity(image_feature, all_text_features)
                    target_similarity = similarity[:, :, class_idx:class_idx+1]
                    
                    # 先生成整图热图（未归一化）
                    full_heat = get_similarity_map(target_similarity, (out_h, out_w))
                    
                    # 生成滑窗坐标
                    coords = extract_sliding_windows(out_h, out_w, 
                                                    self.tile_size, self.tile_size,
                                                    self.tile_stride, self.tile_stride)
                    
                    # 裁切tiles
                    tiles = []
                    for (y, x) in coords:
                        tiles.append(full_heat[:, :, y:y+self.tile_size, x:x+self.tile_size])
                    
                    # OLA拼接
                    stitched, acc_w = stitch_ola(tiles, coords, out_h, out_w, self.device,
                                                global_normalize=True, 
                                                use_percentile=self.use_percentile,
                                                pmin=self.pmin, pmax=self.pmax)
                    
                    heatmaps_per_mode[mode_name][layer_idx] = stitched
                    
                    # 保存最后一层的acc_w用于诊断
                    if layer_idx == layers[-1]:
                        heatmaps_per_mode[mode_name]['acc_w'] = acc_w
        
        return heatmaps_per_mode
    
    def visualize_3mode_comparison(self, image_data, query_class, heatmaps_per_mode, bboxes, layers, output_path):
        """
        可视化3模式对比
        
        Layout: 3 modes x (1 original + N layers) columns
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
    
    # comprehensive_comparison 已由 multi_class 产出的4模式网格全面替代（按需删除保留逻辑）
    
    def process_multi_class_images(self, dataset, max_samples=10, layers=None):
        """
        处理多类别图像，为每个类别生成4模式对比
        
        Args:
            dataset: SeenUnseenDataset
            max_samples: 最大样本数
            layers: 要分析的层，默认L1-L12
        """
        if layers is None:
            layers = list(range(1, 13))
        
        # 统一输出目录至 results/3mode_comparison
        output_dir = Path(__file__).parent / 'results' / '3mode_comparison'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processed = 0
        for idx in range(len(dataset)):
            if processed >= max_samples:
                break
            
            sample = dataset[idx]
            classes_in_image = sample.get('classes', [sample['class_name']])
            
            # 获取唯一类别
            unique_classes = list(set(classes_in_image))
            
            # 优先多类别；若不足则也允许单类别，确保样图数量
            if len(unique_classes) < 2 and processed >= max_samples // 2:
                # 前半配额保证多类别，后半允许单类别以补足到max_samples
                pass
            elif len(unique_classes) < 2:
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
                'original_size': sample.get('original_size', (224, 224))
            }
            
            # 为每个唯一类别生成3模式热图
            print(f"Generating 3-mode heatmaps for {len(unique_classes)} classes...")
            
            for query_class in unique_classes:
                print(f"  Query: {query_class}")
                
                # 生成3模式热图
                heatmaps_per_mode = self.generate_multi_mode_heatmaps(
                    image_tensor, query_class, layers
                )
                
                # 可视化3模式对比
                output_path = output_dir / f'{sample["image_id"]}_{query_class}.png'
                self.visualize_3mode_comparison(
                    image_data, query_class, heatmaps_per_mode,
                    sample['bboxes'], layers, output_path
                )
                
                print(f"    Saved: {output_path.name}")
                
                # 保存Row3的OLA诊断图
                if 'acc_w' in heatmaps_per_mode.get('3.Baseline+OLA', {}):
                    diagnosis_path = output_dir / f'{sample["image_id"]}_{query_class}_ola_diag.png'
                    acc_w = heatmaps_per_mode['3.Baseline+OLA']['acc_w']
                    last_layer = layers[-1]
                    last_heatmap = heatmaps_per_mode['3.Baseline+OLA'][last_layer]
                    
                    visualize_ola_diagnosis(image_tensor, last_heatmap, acc_w, diagnosis_path)
                    print(f"    Diagnosis: {diagnosis_path.name}")
            
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
                        choices=['multi_class', 'debug_gt', 'all'],
                        default='multi_class',
                        help='运行模式: multi_class(多类别), debug_gt(GT调试), all(全部)')
    parser.add_argument('--max-samples', type=int, default=10,
                        help='最大样本数')
    parser.add_argument('--layers', type=int, nargs='+', default=list(range(1, 13)),
                        help='要分析的层')
    parser.add_argument('--debug-samples', type=int, default=3,
                        help='GT调试样本数')
    # OLA参数（Row3始终启用）
    parser.add_argument('--tile-size', type=int, default=224,
                        help='OLA滑窗大小')
    parser.add_argument('--tile-stride', type=int, default=112,
                        help='OLA滑窗步长（推荐tile-size//2）')
    parser.add_argument('--percentile', action='store_true',
                        help='使用分位归一化（抗异常值）')
    parser.add_argument('--pmin', type=float, default=5.0,
                        help='分位归一化下限百分比')
    parser.add_argument('--pmax', type=float, default=95.0,
                        help='分位归一化上限百分比')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Experiment 5: 3-Mode Comparison (Baseline / Complete / OLA)")
    print("=" * 80)
    print(f"数据集: {args.dataset}")
    print(f"运行模式: {args.mode}")
    print(f"最大样本数: {args.max_samples}")
    print(f"分析层: {args.layers}")
    print(f"GT调试样本数: {args.debug_samples}")
    print(f"OLA (Row3): Always enabled for Baseline+OLA")
    print(f"  Tile size: {args.tile_size}")
    print(f"  Tile stride: {args.tile_stride}")
    print(f"  Percentile: pmin={args.pmin}, pmax={args.pmax}")
    
    # 配置
    config = Config()
    config.dataset_root = args.dataset
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建生成器（OLA参数用于Row3）
    generator = UnifiedHeatmapGenerator(
        config, 
        enable_ola=True,  # Row3始终启用OLA
        tile_size=args.tile_size,
        tile_stride=args.tile_stride,
        use_percentile=args.percentile if hasattr(args, 'percentile') else True,
        pmin=args.pmin,
        pmax=args.pmax
    )
    
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
    
    # comprehensive_comparison 输出已废弃，multi_class_4mode 已覆盖所有需求
    
    print(f"\n{'='*80}")
    print("所有任务完成！")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
