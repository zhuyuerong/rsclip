# -*- coding: utf-8 -*-
"""
Experiment 5: 层级注意力搜索（Hierarchical Attention Search）

递归细化检测小目标：
1. 从粗粒度热图开始（7×7 patch）
2. 找高响应区域 → 超分2x → 重新检测
3. 递归细化，直到收敛或达到深度限制
4. 对比4种分区策略：Grid / Threshold / Peaks / Hybrid
"""

import sys
from pathlib import Path
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# 添加项目根目录
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from experiment4.core.config import Config
from experiment4.core.models.clip_surgery import CLIPSurgeryWrapper, get_similarity_map
from experiment4.experiments.surgery_clip.utils.seen_unseen_split import SeenUnseenDataset


# ========== SearchNode数据结构 ==========

@dataclass
class SearchNode:
    """搜索树节点"""
    bbox: Tuple[int, int, int, int]  # (y, x, h, w) 在原图的位置
    margin: float                     # prediction margin (top1-top2)
    confidence: float                 # top-1概率
    class_idx: int                    # 预测类别
    depth: int                        # 搜索深度
    parent: Optional['SearchNode'] = None
    children: List['SearchNode'] = field(default_factory=list)
    heatmap: Optional[np.ndarray] = None  # 该区域的热图


# ========== 层级注意力搜索 ==========

class HierarchicalAttentionSearch:
    """层级注意力搜索（递归细化检测）"""
    
    def __init__(self, 
                 model,
                 text_features,
                 query_class_idx: int,
                 max_depth: int = 3,
                 scale_factor: float = 2.0,
                 top_k_regions: int = 3,
                 margin_threshold: float = 0.2,
                 min_region_size: int = 32,
                 partition_method: str = 'hybrid'):
        self.model = model
        self.text_features = text_features
        self.query_class_idx = query_class_idx
        self.max_depth = max_depth
        self.scale_factor = scale_factor
        self.top_k_regions = top_k_regions
        self.margin_threshold = margin_threshold
        self.min_region_size = min_region_size
        self.partition_method = partition_method
        
        self.total_nodes = 0
        self.total_inferences = 0
    
    def search(self, image: torch.Tensor, verbose: bool = False):
        """
        执行层级搜索
        
        Returns:
            best_node: 最高margin的叶子节点
            root_node: 搜索树根节点
        """
        H, W = image.shape[2:]
        
        # 初始化根节点
        root_margin, root_conf, root_class = self._compute_margin(image)
        root_heatmap = self._generate_heatmap(image)
        
        root_node = SearchNode(
            bbox=(0, 0, H, W),
            margin=root_margin,
            confidence=root_conf,
            class_idx=root_class,
            depth=0,
            heatmap=root_heatmap
        )
        
        self.total_nodes = 1
        self.total_inferences = 1
        
        if verbose:
            print(f"  Root: margin={root_margin:.4f}, conf={root_conf:.4f}")
        
        # 递归搜索
        self._recursive_search(image, root_node, verbose=verbose)
        
        # 找最佳叶子节点
        best_node = self._find_best_leaf(root_node)
        
        if verbose:
            print(f"  Total nodes: {self.total_nodes}, Best margin: {best_node.margin:.4f}")
        
        return best_node, root_node
    
    def _recursive_search(self, image: torch.Tensor, node: SearchNode, verbose: bool):
        """递归搜索（DFS）"""
        if self._should_stop(node):
            return
        
        # 分区域
        candidate_regions = self._partition_region(node)
        
        if len(candidate_regions) == 0:
            return
        
        # 对每个候选区域
        for i, region_info in enumerate(candidate_regions[:self.top_k_regions]):
            y, x, h, w = region_info['bbox']
            
            # 边界检查
            if h <= 0 or w <= 0 or y < 0 or x < 0:
                continue
            
            # 裁剪区域
            _, _, H_img, W_img = image.shape
            y_end = min(y + h, H_img)
            x_end = min(x + w, W_img)
            h_actual = y_end - y
            w_actual = x_end - x
            
            if h_actual <= 0 or w_actual <= 0:
                continue
            
            crop = image[:, :, y:y_end, x:x_end]
            
            # 超分
            crop_sr = self._super_resolve(crop)
            
            # 重新检测
            margin, conf, class_idx = self._compute_margin(crop_sr)
            heatmap = self._generate_heatmap(crop_sr)
            
            # 创建子节点
            child_node = SearchNode(
                bbox=(y, x, h, w),
                margin=margin,
                confidence=conf,
                class_idx=class_idx,
                depth=node.depth + 1,
                parent=node,
                heatmap=heatmap
            )
            
            node.children.append(child_node)
            self.total_nodes += 1
            self.total_inferences += 1
            
            if verbose:
                indent = "    " * (node.depth + 1)
                print(f"{indent}├─ Region{i+1}: margin={margin:.3f}")
            
            # 递归
            self._recursive_search(crop_sr, child_node, verbose=verbose)
    
    def _should_stop(self, node: SearchNode) -> bool:
        """判断是否停止"""
        if node.depth >= self.max_depth:
            return True
        if node.margin < self.margin_threshold:
            return True
        y, x, h, w = node.bbox
        if h < self.min_region_size or w < self.min_region_size:
            return True
        return False
    
    def _partition_region(self, node: SearchNode) -> List[Dict]:
        """分区域（根据策略）"""
        if self.partition_method == 'grid':
            return self._partition_grid(node)
        elif self.partition_method == 'threshold':
            return self._partition_threshold(node)
        elif self.partition_method == 'peaks':
            return self._partition_peaks(node)
        elif self.partition_method == 'hybrid':
            return self._partition_hybrid(node)
        else:
            return self._partition_grid(node)
    
    def _partition_grid(self, node: SearchNode) -> List[Dict]:
        """网格分区（2×2）"""
        y, x, h, w = node.bbox
        regions = []
        
        step_h = h // 2
        step_w = w // 2
        
        for i in range(2):
            for j in range(2):
                yi = y + i * step_h
                xi = x + j * step_w
                hi = step_h if i == 0 else (y + h - yi)
                wi = step_w if j == 0 else (x + w - xi)
                
                # 计算该网格的得分
                if node.heatmap is not None:
                    H_hm, W_hm = node.heatmap.shape
                    yi_hm = int((yi - y) / h * H_hm)
                    xi_hm = int((xi - x) / w * W_hm)
                    hi_hm = max(1, int(hi / h * H_hm))
                    wi_hm = max(1, int(wi / w * W_hm))
                    
                    score = node.heatmap[yi_hm:yi_hm+hi_hm, xi_hm:xi_hm+wi_hm].mean()
                else:
                    score = 0.5
                
                regions.append({'bbox': (yi, xi, hi, wi), 'score': float(score)})
        
        regions.sort(key=lambda x: x['score'], reverse=True)
        return regions
    
    def _partition_threshold(self, node: SearchNode) -> List[Dict]:
        """阈值分区（连通域）"""
        if node.heatmap is None:
            return []
        
        heatmap = node.heatmap
        y_base, x_base, h, w = node.bbox
        H_hm, W_hm = heatmap.shape
        
        # 二值化
        threshold = heatmap.mean() + heatmap.std() * 0.5
        binary = (heatmap > threshold).astype(np.uint8)
        
        # 连通域
        from scipy import ndimage
        labeled, num_features = ndimage.label(binary)
        
        regions = []
        for i in range(1, min(num_features + 1, 10)):  # 最多10个
            ys, xs = np.where(labeled == i)
            if len(xs) < 5:
                continue
            
            # 热图坐标
            y_min_hm, y_max_hm = ys.min(), ys.max()
            x_min_hm, x_max_hm = xs.min(), xs.max()
            
            # 映射回原图（相对于node.bbox）
            y_min = y_base + int(y_min_hm / H_hm * h)
            x_min = x_base + int(x_min_hm / W_hm * w)
            h_region = int((y_max_hm - y_min_hm + 1) / H_hm * h)
            w_region = int((x_max_hm - x_min_hm + 1) / W_hm * w)
            
            # 扩展10%
            expand = 0.1
            y_min = max(y_base, int(y_min - h_region * expand))
            x_min = max(x_base, int(x_min - w_region * expand))
            h_region = int(h_region * (1 + 2 * expand))
            w_region = int(w_region * (1 + 2 * expand))
            
            score = heatmap[labeled == i].mean()
            regions.append({'bbox': (y_min, x_min, h_region, w_region), 'score': float(score)})
        
        regions.sort(key=lambda x: x['score'], reverse=True)
        return regions
    
    def _partition_peaks(self, node: SearchNode) -> List[Dict]:
        """峰值分区"""
        if node.heatmap is None:
            return []
        
        try:
            from skimage.feature import peak_local_max
        except ImportError:
            # fallback到grid
            return self._partition_grid(node)
        
        heatmap = node.heatmap
        y_base, x_base, h, w = node.bbox
        H_hm, W_hm = heatmap.shape
        
        # 找峰值
        coordinates = peak_local_max(
            heatmap,
            min_distance=max(3, min(H_hm, W_hm) // 10),
            threshold_abs=heatmap.mean()
        )
        
        regions = []
        for y_peak, x_peak in coordinates[:10]:  # 最多10个
            # 以峰值为中心
            window_size = min(H_hm, W_hm) // 3
            
            y_min_hm = max(0, y_peak - window_size // 2)
            x_min_hm = max(0, x_peak - window_size // 2)
            y_max_hm = min(H_hm, y_min_hm + window_size)
            x_max_hm = min(W_hm, x_min_hm + window_size)
            
            # 映射回原图
            y_min = y_base + int(y_min_hm / H_hm * h)
            x_min = x_base + int(x_min_hm / W_hm * w)
            h_region = int((y_max_hm - y_min_hm) / H_hm * h)
            w_region = int((x_max_hm - x_min_hm) / W_hm * w)
            
            score = float(heatmap[y_peak, x_peak])
            regions.append({'bbox': (y_min, x_min, h_region, w_region), 'score': score})
        
        regions.sort(key=lambda x: x['score'], reverse=True)
        return regions
    
    def _partition_hybrid(self, node: SearchNode) -> List[Dict]:
        """混合策略（Grid + Threshold + Peaks + NMS）"""
        all_regions = []
        
        # 收集3种策略的结果
        all_regions.extend(self._partition_grid(node))
        all_regions.extend(self._partition_threshold(node))
        all_regions.extend(self._partition_peaks(node))
        
        # NMS去重
        kept_regions = []
        for region in all_regions:
            is_duplicate = False
            for kept in kept_regions:
                if self._compute_iou(region['bbox'], kept['bbox']) > 0.7:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept_regions.append(region)
        
        kept_regions.sort(key=lambda x: x['score'], reverse=True)
        return kept_regions
    
    def _compute_iou(self, box1, box2):
        """计算IoU"""
        y1, x1, h1, w1 = box1
        y2, x2, h2, w2 = box2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1+w1, x2+w2)
        yi2 = min(y1+h1, y2+h2)
        
        inter = max(0, xi2-xi1) * max(0, yi2-yi1)
        union = w1*h1 + w2*h2 - inter
        
        return inter / (union + 1e-8)
    
    def _super_resolve(self, crop: torch.Tensor) -> torch.Tensor:
        """超分区域"""
        _, _, h, w = crop.shape
        h_sr = int(h * self.scale_factor)
        w_sr = int(w * self.scale_factor)
        
        crop_sr = F.interpolate(
            crop, size=(h_sr, w_sr),
            mode='bicubic', align_corners=False, antialias=True
        )
        
        return crop_sr
    
    def _compute_margin(self, image: torch.Tensor) -> Tuple[float, float, int]:
        """
        计算prediction margin
        
        Returns:
            margin: top1 - top2
            confidence: top1概率
            class_idx: top1类别
        """
        # Resize到224
        if image.shape[2] != 224 or image.shape[3] != 224:
            image_resized = F.interpolate(image, size=(224, 224), mode='bilinear')
        else:
            image_resized = image
        
        # 编码（使用CLS token）
        features = self.model.encode_image(image_resized)
        cls_feature = features[:, 0, :]
        cls_feature = F.normalize(cls_feature, dim=-1)
        
        # 相似度
        similarity = cls_feature @ self.text_features.t()
        probs = F.softmax(similarity * 100, dim=-1)[0]
        
        # Top-2
        top2_probs, top2_indices = torch.topk(probs, k=2)
        
        margin = (top2_probs[0] - top2_probs[1]).item()
        confidence = top2_probs[0].item()
        class_idx = top2_indices[0].item()
        
        return margin, confidence, class_idx
    
    def _generate_heatmap(self, image: torch.Tensor) -> np.ndarray:
        """生成热图"""
        if image.shape[2] != 224 or image.shape[3] != 224:
            image_resized = F.interpolate(image, size=(224, 224), mode='bilinear')
        else:
            image_resized = image
        
        features = self.model.encode_image(image_resized)
        similarity = self.model.compute_similarity(features, self.text_features)
        target_sim = similarity[:, :, self.query_class_idx:self.query_class_idx+1]
        
        heatmap = get_similarity_map(target_sim, (224, 224))
        return heatmap[0, 0].cpu().numpy()
    
    def _find_best_leaf(self, root: SearchNode) -> SearchNode:
        """找最高margin的叶子节点"""
        best = root
        
        def dfs(node):
            nonlocal best
            if len(node.children) == 0:
                if node.margin > best.margin:
                    best = node
            else:
                for child in node.children:
                    dfs(child)
        
        dfs(root)
        return best


# ========== 可视化函数 ==========

def visualize_search_tree(root_node: SearchNode, image: torch.Tensor, 
                          query_class: str, output_path: str):
    """可视化搜索树"""
    # 找最优路径
    best_leaf = _find_best_leaf_helper(root_node)
    best_path_nodes = _get_path_to_root(best_leaf)
    
    # 收集所有节点
    all_nodes = []
    def collect(node):
        all_nodes.append(node)
        for child in node.children:
            collect(child)
    collect(root_node)
    
    # 反归一化图像
    img = image[0].cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    img = np.clip(img * std + mean, 0, 1)
    
    # 创建2子图
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：所有探索区域
    axes[0].imshow(img)
    axes[0].set_title(f'All Explored Regions\n{query_class}', fontsize=12)
    
    colors = ['red', 'yellow', 'cyan', 'magenta']
    for node in all_nodes:
        y, x, h, w = node.bbox
        color = colors[node.depth % len(colors)]
        linewidth = 1 + node.margin * 2
        linestyle = '-' if node in best_path_nodes else '--'
        
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=linewidth,
            edgecolor=color,
            facecolor='none',
            linestyle=linestyle
        )
        axes[0].add_patch(rect)
    
    axes[0].axis('off')
    
    # 右图：最优路径
    axes[1].imshow(img)
    axes[1].set_title(f'Best Path (margin={best_leaf.margin:.4f})', fontsize=12)
    
    for node in best_path_nodes:
        y, x, h, w = node.bbox
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=3,
            edgecolor='lime',
            facecolor='none'
        )
        axes[1].add_patch(rect)
        
        axes[1].text(x+5, y+15, f'L{node.depth}: {node.margin:.3f}',
                    color='lime', fontsize=10, weight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _find_best_leaf_helper(root):
    best = root
    def dfs(node):
        nonlocal best
        if len(node.children) == 0 and node.margin > best.margin:
            best = node
        for child in node.children:
            dfs(child)
    dfs(root)
    return best


def _get_path_to_root(node):
    path = []
    current = node
    while current is not None:
        path.append(current)
        current = current.parent
    return list(reversed(path))


# ========== 4策略对比生成器 ==========

class FourStrategyComparison:
    """4种分区策略对比"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # DIOR类别
        self.dior_classes_raw = [
            'airplane', 'airport', 'baseballfield', 'basketballcourt', 
            'bridge', 'chimney', 'dam', 'Expressway-Service-area',
            'Expressway-toll-station', 'golffield', 'groundtrackfield',
            'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
            'tenniscourt', 'trainstation', 'vehicle', 'windmill'
        ]
        self.dior_prompts = [f"an aerial photo of {cls}" for cls in self.dior_classes_raw]
        
        # 加载模型（使用Complete Surgery）
        cfg = Config()
        cfg.dataset_root = config.dataset_root
        cfg.device = self.device
        cfg.use_surgery = True
        cfg.use_vv_mechanism = True
        self.model = CLIPSurgeryWrapper(cfg)
        print("Model loaded: Complete Surgery (Surgery + VV)")
    
    def process_multi_class_images(self, dataset, max_samples=10, layers=None):
        """处理多类别图像"""
        if layers is None:
            layers = [1, 3, 6, 9, 12]
        
        output_dir = Path(__file__).parent / 'results' / '4strategy_comparison'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tree_dir = Path(__file__).parent / 'results' / 'search_trees'
        tree_dir.mkdir(parents=True, exist_ok=True)
        
        processed = 0
        for idx in range(len(dataset)):
            if processed >= max_samples:
                break
            
            sample = dataset[idx]
            classes_in_image = sample.get('classes', [sample['class_name']])
            unique_classes = list(set(classes_in_image))
            
            if len(unique_classes) < 2 and processed >= max_samples // 2:
                pass
            elif len(unique_classes) < 2:
                continue
            
            print(f"\n{'='*70}")
            print(f"Sample {idx}: {sample['image_id']}")
            print(f"Classes: {unique_classes}")
            print(f"{'='*70}")
            
            image_tensor = sample['image'].unsqueeze(0).to(self.device)
            
            for query_class in unique_classes:
                print(f"\nQuery: {query_class}")
                
                # 4种策略搜索
                strategies = ['grid', 'threshold', 'peaks', 'hybrid']
                search_results = {}
                
                # 编码文本
                all_text_features = self.model.encode_text(self.dior_prompts)
                all_text_features = F.normalize(all_text_features, dim=-1)
                class_idx = self.dior_classes_raw.index(query_class)
                
                for strategy in strategies:
                    print(f"  Strategy: {strategy}")
                    searcher = HierarchicalAttentionSearch(
                        self.model, all_text_features, class_idx,
                        max_depth=2, scale_factor=2.0, top_k_regions=2,
                        margin_threshold=0.15, partition_method=strategy
                    )
                    
                    best_node, root_node = searcher.search(image_tensor, verbose=False)
                    search_results[strategy] = (best_node, root_node)
                    print(f"    Nodes: {searcher.total_nodes}, Best margin: {best_node.margin:.4f}")
                
                # 可视化搜索树（hybrid策略）
                tree_path = tree_dir / f'{sample["image_id"]}_{query_class}_tree.png'
                _, root_hybrid = search_results['hybrid']
                visualize_search_tree(root_hybrid, image_tensor, query_class, tree_path)
                print(f"  Tree saved: {tree_path.name}")
            
            processed += 1
        
        print(f"\n{'='*70}")
        print(f"Total: {processed} images")
        print(f"Output: {output_dir}")
        print(f"{'='*70}")


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser(description='Exp5: Hierarchical Attention Search')
    parser.add_argument('--dataset', type=str, default='datasets/mini_dataset')
    parser.add_argument('--max-samples', type=int, default=10)
    parser.add_argument('--layers', type=int, nargs='+', default=[1, 3, 6, 9, 12])
    parser.add_argument('--max-depth', type=int, default=2)
    parser.add_argument('--top-k', type=int, default=2)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Experiment 5: Hierarchical Attention Search")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Max samples: {args.max_samples}")
    print(f"Layers: {args.layers}")
    print(f"Max depth: {args.max_depth}")
    print(f"Top-k regions: {args.top_k}")
    
    # 配置
    config = Config()
    config.dataset_root = args.dataset
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建生成器
    generator = FourStrategyComparison(config)
    
    # 加载数据集
    print(f"\n加载数据集...")
    unseen_classes = ['airplane', 'bridge', 'storagetank', 'vehicle', 'windmill']
    dataset = SeenUnseenDataset(config.dataset_root, split='all', mode='val', 
                               unseen_classes=unseen_classes)
    print(f"✓ Dataset: {len(dataset)} samples")
    
    # 处理
    generator.process_multi_class_images(dataset, args.max_samples, args.layers)
    
    print(f"\n{'='*80}")
    print("完成！")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

