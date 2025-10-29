# -*- coding: utf-8 -*-
"""
实验4.1：打印patch相似度grid
验证GT区域的相似度是否真的低于背景
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# 添加路径
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.append(str(root_dir))

from experiment4.config import Config
from experiment4.models.clip_surgery import CLIPSurgeryWrapper
from experiment4.data.dataset import get_dataloaders


def identify_gt_patches(gt_bbox, grid_size=7, image_size=224):
    """
    确定GT bbox覆盖的patch索引
    
    Args:
        gt_bbox: [x_min, y_min, x_max, y_max] 归一化坐标 (0-1)
        grid_size: patch grid大小 (7 for ViT-B/32)
        image_size: 图像尺寸 (224)
    
    Returns:
        gt_patches: set of (row, col) tuples
    """
    # 转换为像素坐标
    x_min = int(gt_bbox[0] * image_size)
    y_min = int(gt_bbox[1] * image_size)
    x_max = int(gt_bbox[2] * image_size)
    y_max = int(gt_bbox[3] * image_size)
    
    # patch尺寸
    patch_size = image_size / grid_size  # 32 for 224/7
    
    # 确定覆盖的patch
    gt_patches = set()
    for row in range(grid_size):
        for col in range(grid_size):
            # patch的中心点
            patch_x = (col + 0.5) * patch_size
            patch_y = (row + 0.5) * patch_size
            
            # 判断是否在GT框内
            if x_min <= patch_x <= x_max and y_min <= patch_y <= y_max:
                gt_patches.add((row, col))
    
    return gt_patches


def diagnose_single_sample(model, image, class_name, gt_bbox, device):
    """
    诊断单个样本的patch相似度分布
    """
    print("\n" + "="*70)
    print(f"样本诊断：{class_name}")
    print("="*70)
    
    # 提取特征
    image = image.unsqueeze(0).to(device)
    
    # 获取完整特征
    with torch.no_grad():
        image_features = model.get_all_features(image)  # [1, 50, 512]
        text_features = model.encode_text([class_name])  # [1, 512]
    
    # 提取patch特征（去掉CLS）
    patch_features = image_features[:, 1:, :]  # [1, 49, 512]
    
    # L2归一化
    patch_norm = F.normalize(patch_features, dim=-1, p=2)
    text_norm = F.normalize(text_features, dim=-1, p=2)
    
    # 计算相似度
    similarity = (patch_norm @ text_norm.T).squeeze()  # [49]
    similarity_np = similarity.cpu().numpy()
    
    # Reshape到7x7 grid
    similarity_grid = similarity_np.reshape(7, 7)
    
    # 确定GT区域的patches
    gt_patches = identify_gt_patches(gt_bbox.cpu().numpy() if isinstance(gt_bbox, torch.Tensor) else gt_bbox, 
                                     grid_size=7)
    
    # 打印grid
    print(f"\n相似度Grid (7x7):")
    print(f"GT bbox: [{gt_bbox[0]:.3f}, {gt_bbox[1]:.3f}, {gt_bbox[2]:.3f}, {gt_bbox[3]:.3f}]")
    print(f"GT覆盖的patches: {sorted(gt_patches)}")
    print()
    
    for i in range(7):
        for j in range(7):
            sim_val = similarity_grid[i, j]
            marker = "[GT]" if (i, j) in gt_patches else "    "
            
            # 颜色编码（用于文本输出）
            if sim_val > 0.20:
                color = "🔴"  # 高相似度
            elif sim_val > 0.18:
                color = "🟠"  # 中高
            elif sim_val > 0.16:
                color = "🟡"  # 中等
            else:
                color = "🔵"  # 低
            
            print(f"{color}{sim_val:.4f}{marker}", end="  ")
        print()
    
    # 统计分析
    gt_similarities = [similarity_grid[i, j] for i, j in gt_patches]
    bg_similarities = [similarity_grid[i, j] for i in range(7) for j in range(7) 
                       if (i, j) not in gt_patches]
    
    print(f"\n" + "-"*70)
    print(f"统计分析:")
    print(f"-"*70)
    print(f"GT区域相似度: {np.mean(gt_similarities):.4f} ± {np.std(gt_similarities):.4f}")
    print(f"  最小: {np.min(gt_similarities):.4f}")
    print(f"  最大: {np.max(gt_similarities):.4f}")
    print(f"  中位: {np.median(gt_similarities):.4f}")
    
    print(f"\n背景区域相似度: {np.mean(bg_similarities):.4f} ± {np.std(bg_similarities):.4f}")
    print(f"  最小: {np.min(bg_similarities):.4f}")
    print(f"  最大: {np.max(bg_similarities):.4f}")
    print(f"  中位: {np.median(bg_similarities):.4f}")
    
    print(f"\n全图相似度: {similarity_np.mean():.4f} ± {similarity_np.std():.4f}")
    print(f"  范围: [{similarity_np.min():.4f}, {similarity_np.max():.4f}]")
    
    # 关键指标
    gt_avg = np.mean(gt_similarities)
    bg_avg = np.mean(bg_similarities)
    diff = gt_avg - bg_avg
    
    print(f"\n" + "="*70)
    print(f"关键发现:")
    print(f"="*70)
    print(f"GT区域 vs 背景: {gt_avg:.4f} vs {bg_avg:.4f}")
    print(f"差异: {diff:+.4f} ({diff/bg_avg*100:+.2f}%)")
    
    if gt_avg > bg_avg:
        print(f"✅ GT区域相似度更高（正常）")
    else:
        print(f"❌ GT区域相似度更低（异常！）")
        print(f"   → 可能原因：Surgery去冗余抑制了目标特征")
    
    # 百分位分析
    print(f"\n相似度分布（百分位）:")
    for p in [10, 25, 50, 75, 90, 95]:
        percentile_val = np.percentile(similarity_np, p)
        print(f"  {p:3d}%tile: {percentile_val:.4f}", end="")
        
        # 检查GT区域在哪个百分位
        if p == 75:
            gt_below_75 = sum(1 for s in gt_similarities if s < percentile_val)
            print(f"  (GT中有{gt_below_75}/{len(gt_similarities)}低于此值)", end="")
        print()
    
    return {
        'class_name': class_name,
        'gt_avg': float(gt_avg),
        'bg_avg': float(bg_avg),
        'diff': float(diff),
        'gt_similarities': [float(s) for s in gt_similarities],
        'bg_similarities': [float(s) for s in bg_similarities],
        'all_similarities': similarity_np.tolist()
    }


def main():
    """主函数"""
    print("="*70)
    print("实验4.1：Patch相似度Grid诊断")
    print("="*70)
    
    config = Config()
    device = config.device
    
    # 加载数据
    print("\n加载数据...")
    train_loader, val_loader, test_loader, dataset = get_dataloaders(config)
    
    # 加载模型
    print("\n加载CLIP Surgery模型...")
    model = CLIPSurgeryWrapper(config)
    
    # 诊断前3个样本
    print("\n开始诊断...")
    
    all_results = []
    sample_count = 0
    max_samples = 5
    
    for batch in val_loader:
        images = batch['image']
        class_names = batch['class_name']
        bboxes = batch['bbox']
        has_bbox = batch['has_bbox']
        
        for i in range(len(images)):
            if not has_bbox[i]:
                continue
            
            result = diagnose_single_sample(
                model,
                images[i],
                class_names[i],
                bboxes[i],
                device
            )
            all_results.append(result)
            
            sample_count += 1
            if sample_count >= max_samples:
                break
        
        if sample_count >= max_samples:
            break
    
    # 汇总分析
    print("\n" + "="*70)
    print("汇总分析")
    print("="*70)
    
    gt_better_count = sum(1 for r in all_results if r['gt_avg'] > r['bg_avg'])
    bg_better_count = sum(1 for r in all_results if r['bg_avg'] > r['gt_avg'])
    
    print(f"\n样本统计（共{len(all_results)}个）:")
    print(f"  GT区域更高: {gt_better_count}个 ({gt_better_count/len(all_results)*100:.1f}%)")
    print(f"  背景更高: {bg_better_count}个 ({bg_better_count/len(all_results)*100:.1f}%)")
    
    avg_diff = np.mean([r['diff'] for r in all_results])
    print(f"\n平均差异（GT - 背景）: {avg_diff:+.4f}")
    
    if avg_diff < 0:
        print(f"\n❌ 严重问题：GT区域平均相似度低于背景{abs(avg_diff):.4f}")
        print(f"   → 这解释了为什么热图颜色反转")
        print(f"   → Surgery去冗余可能抑制了目标的公共特征")
    else:
        print(f"\n✅ GT区域相似度更高（正常）")
    
    # 保存结果
    output_dir = Path("experiment4/experiments/exp4_diagnosis/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(output_dir / "patch_grid_analysis.json", 'w', encoding='utf-8') as f:
        json.dump({
            'results': all_results,
            'summary': {
                'gt_better_count': gt_better_count,
                'bg_better_count': bg_better_count,
                'avg_diff': float(avg_diff),
                'conclusion': 'GT低于背景' if avg_diff < 0 else 'GT高于背景'
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存: {output_dir}/patch_grid_analysis.json")


if __name__ == "__main__":
    main()

