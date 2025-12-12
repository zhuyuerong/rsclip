# -*- coding: utf-8 -*-
"""
CLIP热图可视化工具（共享模块）

生成热图对比、边界框对比等可视化

来源: src/legacy_experiments/experiment6/exp/exp1/utils/visualization.py
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import List, Dict


def create_side_by_side_heatmap(image: np.ndarray, 
                                  heatmap_with: np.ndarray,
                                  heatmap_without: np.ndarray,
                                  class_name: str,
                                  image_id: str,
                                  output_path: str):
    """
    创建并排热图对比
    
    Args:
        image: 原始图像 [H, W, 3] RGB
        heatmap_with: 有冗余去除的热图 [H, W]
        heatmap_without: 无冗余去除的热图 [H, W]
        class_name: 类别名称
        image_id: 图像ID
        output_path: 输出路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title(f'Original Image\nID: {image_id}', fontsize=12)
    axes[0].axis('off')
    
    # With redundancy removal
    heatmap_vis_with = cv2.applyColorMap((heatmap_with * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_vis_with = cv2.cvtColor(heatmap_vis_with, cv2.COLOR_BGR2RGB)
    overlay_with = (image * 0.4 + heatmap_vis_with * 0.6).astype(np.uint8)
    
    axes[1].imshow(overlay_with)
    axes[1].set_title(f'With Redundancy Removal\nRange: [{heatmap_with.min():.3f}, {heatmap_with.max():.3f}]', 
                      fontsize=12)
    axes[1].axis('off')
    
    # Without redundancy removal
    heatmap_vis_without = cv2.applyColorMap((heatmap_without * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_vis_without = cv2.cvtColor(heatmap_vis_without, cv2.COLOR_BGR2RGB)
    overlay_without = (image * 0.4 + heatmap_vis_without * 0.6).astype(np.uint8)
    
    axes[2].imshow(overlay_without)
    axes[2].set_title(f'Without Redundancy Removal\nRange: [{heatmap_without.min():.3f}, {heatmap_without.max():.3f}]', 
                      fontsize=12)
    axes[2].axis('off')
    
    plt.suptitle(f'Redundancy Removal Ablation - {class_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_bbox_comparison(image: np.ndarray,
                            bboxes_with: List[Dict],
                            bboxes_without: List[Dict],
                            gt_bboxes: List[Dict],
                            class_name: str,
                            image_id: str,
                            threshold: float,
                            output_path: str):
    """
    创建bbox对比可视化
    
    Args:
        image: 原始图像 [H, W, 3] RGB
        bboxes_with: 有冗余去除的bbox
        bboxes_without: 无冗余去除的bbox
        gt_bboxes: GT bbox
        class_name: 类别名称
        image_id: 图像ID
        threshold: 使用的阈值
        output_path: 输出路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # GT boxes
    axes[0].imshow(image)
    axes[0].set_title(f'Ground Truth\n{len(gt_bboxes)} boxes', fontsize=12)
    for gt in gt_bboxes:
        x1, y1, x2, y2 = gt['box']
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                         linewidth=2, edgecolor='green', facecolor='none')
        axes[0].add_patch(rect)
    axes[0].axis('off')
    
    # With redundancy removal
    axes[1].imshow(image)
    axes[1].set_title(f'With Redundancy\n{len(bboxes_with)} boxes (thresh={threshold})', fontsize=12)
    for bbox in bboxes_with[:10]:  # Max 10 boxes
        x1, y1, x2, y2 = bbox['box']
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                         linewidth=2, edgecolor='red', facecolor='none', alpha=0.7)
        axes[1].add_patch(rect)
    axes[1].axis('off')
    
    # Without redundancy removal
    axes[2].imshow(image)
    axes[2].set_title(f'Without Redundancy\n{len(bboxes_without)} boxes (thresh={threshold})', fontsize=12)
    for bbox in bboxes_without[:10]:
        x1, y1, x2, y2 = bbox['box']
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                         linewidth=2, edgecolor='blue', facecolor='none', alpha=0.7)
        axes[2].add_patch(rect)
    axes[2].axis('off')
    
    plt.suptitle(f'Bbox Comparison - {class_name} (ID: {image_id})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_similarity_distribution(stats_list: List[Dict], output_path: str):
    """
    绘制similarity分布对比
    
    Args:
        stats_list: 统计信息列表
        output_path: 输出路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    versions = [s['version'] for s in stats_list]
    colors = ['red', 'blue']
    
    # Mean对比
    means = [s['mean'] for s in stats_list]
    axes[0, 0].bar(versions, means, color=colors, alpha=0.7)
    axes[0, 0].set_title('Mean Similarity', fontsize=12)
    axes[0, 0].set_ylabel('Mean')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Std对比
    stds = [s['std'] for s in stats_list]
    axes[0, 1].bar(versions, stds, color=colors, alpha=0.7)
    axes[0, 1].set_title('Standard Deviation', fontsize=12)
    axes[0, 1].set_ylabel('Std')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Min/Max对比
    mins = [s['min'] for s in stats_list]
    maxs = [s['max'] for s in stats_list]
    x = np.arange(len(versions))
    width = 0.35
    axes[1, 0].bar(x - width/2, mins, width, label='Min', color='lightblue', alpha=0.7)
    axes[1, 0].bar(x + width/2, maxs, width, label='Max', color='orange', alpha=0.7)
    axes[1, 0].set_title('Min/Max Range', fontsize=12)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(versions)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 箱线图
    data_to_plot = []
    for s in stats_list:
        data_to_plot.append([s['min'], s['q25'], s['median'], s['q75'], s['max']])
    
    bp = axes[1, 1].boxplot(data_to_plot, labels=versions, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 1].set_title('Distribution (Box Plot)', fontsize=12)
    axes[1, 1].set_ylabel('Similarity')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Similarity Statistics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_iou_comparison(results_with: Dict, results_without: Dict, output_path: str, class_name: str = ""):
    """
    绘制IoU对比图
    
    Args:
        results_with: 有冗余版本的评估结果
        results_without: 无冗余版本的评估结果
        output_path: 输出路径
        class_name: 类别名称
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    thresholds = sorted(results_with.keys())
    
    # Precision对比
    precision_with = [results_with[t]['metrics']['precision'] for t in thresholds]
    precision_without = [results_without[t]['metrics']['precision'] for t in thresholds]
    
    x = np.arange(len(thresholds))
    width = 0.35
    axes[0, 0].bar(x - width/2, precision_with, width, label='With Redundancy', color='red', alpha=0.7)
    axes[0, 0].bar(x + width/2, precision_without, width, label='Without Redundancy', color='blue', alpha=0.7)
    axes[0, 0].set_title('Precision Comparison', fontsize=12)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([f"{t.split('_')[1]}" for t in thresholds])
    axes[0, 0].set_xlabel('Heatmap Threshold')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Recall对比
    recall_with = [results_with[t]['metrics']['recall'] for t in thresholds]
    recall_without = [results_without[t]['metrics']['recall'] for t in thresholds]
    
    axes[0, 1].bar(x - width/2, recall_with, width, label='With Redundancy', color='red', alpha=0.7)
    axes[0, 1].bar(x + width/2, recall_without, width, label='Without Redundancy', color='blue', alpha=0.7)
    axes[0, 1].set_title('Recall Comparison', fontsize=12)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([f"{t.split('_')[1]}" for t in thresholds])
    axes[0, 1].set_xlabel('Heatmap Threshold')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Avg IoU对比
    avg_iou_with = [results_with[t]['metrics']['avg_iou'] for t in thresholds]
    avg_iou_without = [results_without[t]['metrics']['avg_iou'] for t in thresholds]
    
    axes[1, 0].bar(x - width/2, avg_iou_with, width, label='With Redundancy', color='red', alpha=0.7)
    axes[1, 0].bar(x + width/2, avg_iou_without, width, label='Without Redundancy', color='blue', alpha=0.7)
    axes[1, 0].set_title('Average IoU Comparison', fontsize=12)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([f"{t.split('_')[1]}" for t in thresholds])
    axes[1, 0].set_xlabel('Heatmap Threshold')
    axes[1, 0].set_ylabel('Avg IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 预测框数量对比
    num_pred_with = [results_with[t]['metrics']['num_pred'] for t in thresholds]
    num_pred_without = [results_without[t]['metrics']['num_pred'] for t in thresholds]
    
    axes[1, 1].bar(x - width/2, num_pred_with, width, label='With Redundancy', color='red', alpha=0.7)
    axes[1, 1].bar(x + width/2, num_pred_without, width, label='Without Redundancy', color='blue', alpha=0.7)
    axes[1, 1].set_title('Predicted Box Count Comparison', fontsize=12)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([f"{t.split('_')[1]}" for t in thresholds])
    axes[1, 1].set_xlabel('Heatmap Threshold')
    axes[1, 1].set_ylabel('Num Boxes')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'IoU Evaluation Comparison - {class_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

