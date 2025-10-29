# -*- coding: utf-8 -*-
"""
实验4.2：Surgery去冗余前后的相似度对比
量化Surgery操作对GT区域的影响
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import clip

# 添加路径
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.append(str(root_dir))

from experiment4.core.config import Config
from experiment4.core.data.dataset import get_dataloaders


def extract_raw_clip_features(clip_model, images, device):
    """
    提取原始CLIP特征（不经过Surgery去冗余）
    """
    with torch.no_grad():
        # 确保输入类型匹配
        if images.dtype != clip_model.visual.conv1.weight.dtype:
            images = images.to(clip_model.visual.conv1.weight.dtype)
        
        # 获取ViT的patch embeddings
        x = clip_model.visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        
        # 添加CLS token
        x = torch.cat([
            clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            ),
            x
        ], dim=1)
        
        x = x + clip_model.visual.positional_embedding.to(x.dtype)
        x = clip_model.visual.ln_pre(x)
        
        # 通过transformer
        x = x.permute(1, 0, 2)
        x = clip_model.visual.transformer(x)
        x = x.permute(1, 0, 2)
        
        # Layer norm
        x = clip_model.visual.ln_post(x)
        
        # 投影到512维
        if hasattr(clip_model.visual, 'proj') and clip_model.visual.proj is not None:
            B, N, D = x.shape
            x_reshaped = x.reshape(B * N, D)
            x_proj = x_reshaped @ clip_model.visual.proj
            features = x_proj.reshape(B, N, -1)
        else:
            features = x
    
    return features


def extract_surgery_features(features):
    """
    应用Surgery去冗余: F_surgery = F - mean(F, dim=1)
    
    Args:
        features: [B, N+1, 512] 包含CLS的特征
    
    Returns:
        features_surgery: [B, N+1, 512] 去冗余后的特征
    """
    # 注意：Surgery应该只在patch上做去冗余，不包括CLS
    patch_features = features[:, 1:, :]  # [B, N, 512]
    
    # Surgery去冗余
    redundant = patch_features.mean(dim=1, keepdim=True)  # [B, 1, 512]
    patch_surgery = patch_features - redundant  # [B, N, 512]
    
    # 重新组合
    features_surgery = torch.cat([features[:, 0:1, :], patch_surgery], dim=1)
    
    return features_surgery


def compare_surgery_impact(clip_model, image, class_name, gt_bbox, device):
    """
    对比Surgery前后的patch相似度
    """
    print("\n" + "="*70)
    print(f"Surgery影响分析：{class_name}")
    print("="*70)
    
    # 提取原始特征
    image_tensor = image.unsqueeze(0).to(device)
    features_raw = extract_raw_clip_features(clip_model, image_tensor, device)  # [1, 50, 512]
    
    # 应用Surgery去冗余
    features_surgery = extract_surgery_features(features_raw)  # [1, 50, 512]
    
    # 提取patch特征
    patch_raw = features_raw[:, 1:, :]  # [1, 49, 512]
    patch_surgery = features_surgery[:, 1:, :]  # [1, 49, 512]
    
    # 获取文本特征
    text_tokens = clip.tokenize([class_name]).to(device)
    text_features = clip_model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # 计算相似度（L2归一化后）
    patch_raw_norm = F.normalize(patch_raw, dim=-1, p=2)
    patch_surgery_norm = F.normalize(patch_surgery, dim=-1, p=2)
    text_norm = F.normalize(text_features, dim=-1, p=2)
    
    sim_raw = (patch_raw_norm @ text_norm.T).squeeze().detach().cpu().numpy()  # [49]
    sim_surgery = (patch_surgery_norm @ text_norm.T).squeeze().detach().cpu().numpy()  # [49]
    
    # Reshape到7x7
    sim_raw_grid = sim_raw.reshape(7, 7)
    sim_surgery_grid = sim_surgery.reshape(7, 7)
    delta_grid = sim_surgery_grid - sim_raw_grid
    
    # 确定GT区域
    from experiment4.experiments.exp4_diagnosis.print_patch_grid import identify_gt_patches
    gt_patches = identify_gt_patches(gt_bbox.cpu().numpy() if isinstance(gt_bbox, torch.Tensor) else gt_bbox)
    
    # 打印对比
    print(f"\n原始CLIP相似度 (Surgery前):")
    for i in range(7):
        for j in range(7):
            marker = "[GT]" if (i, j) in gt_patches else "    "
            print(f"{sim_raw_grid[i,j]:.4f}{marker}", end="  ")
        print()
    
    print(f"\nSurgery后相似度:")
    for i in range(7):
        for j in range(7):
            marker = "[GT]" if (i, j) in gt_patches else "    "
            print(f"{sim_surgery_grid[i,j]:.4f}{marker}", end="  ")
        print()
    
    print(f"\n变化量 (Surgery - 原始):")
    for i in range(7):
        for j in range(7):
            marker = "[GT]" if (i, j) in gt_patches else "    "
            delta_val = delta_grid[i, j]
            sign = "+" if delta_val >= 0 else ""
            print(f"{sign}{delta_val:.4f}{marker}", end="  ")
        print()
    
    # 统计
    gt_raw = [sim_raw_grid[i, j] for i, j in gt_patches]
    gt_surgery = [sim_surgery_grid[i, j] for i, j in gt_patches]
    gt_delta = [delta_grid[i, j] for i, j in gt_patches]
    
    bg_raw = [sim_raw_grid[i, j] for i in range(7) for j in range(7) if (i, j) not in gt_patches]
    bg_surgery = [sim_surgery_grid[i, j] for i in range(7) for j in range(7) if (i, j) not in gt_patches]
    bg_delta = [delta_grid[i, j] for i in range(7) for j in range(7) if (i, j) not in gt_patches]
    
    print(f"\n" + "-"*70)
    print(f"统计对比:")
    print(f"-"*70)
    
    print(f"\nGT区域:")
    print(f"  原始: {np.mean(gt_raw):.4f} ± {np.std(gt_raw):.4f}")
    print(f"  Surgery: {np.mean(gt_surgery):.4f} ± {np.std(gt_surgery):.4f}")
    print(f"  变化: {np.mean(gt_delta):+.4f} ({np.mean(gt_delta)/np.mean(gt_raw)*100:+.1f}%)")
    
    print(f"\n背景区域:")
    print(f"  原始: {np.mean(bg_raw):.4f} ± {np.std(bg_raw):.4f}")
    print(f"  Surgery: {np.mean(bg_surgery):.4f} ± {np.std(bg_surgery):.4f}")
    print(f"  变化: {np.mean(bg_delta):+.4f} ({np.mean(bg_delta)/np.mean(bg_raw)*100:+.1f}%)")
    
    # 关键指标
    gap_before = np.mean(gt_raw) - np.mean(bg_raw)
    gap_after = np.mean(gt_surgery) - np.mean(bg_surgery)
    
    print(f"\n" + "="*70)
    print(f"Surgery的影响:")
    print(f"="*70)
    print(f"Surgery前 (GT - 背景): {gap_before:+.4f}")
    print(f"Surgery后 (GT - 背景): {gap_after:+.4f}")
    print(f"Surgery导致gap变化: {gap_after - gap_before:+.4f}")
    
    if gap_before > 0 and gap_after < 0:
        print(f"\n❌ Surgery反转了GT-背景关系！")
        print(f"   Surgery前：GT > 背景 ({gap_before:+.4f})")
        print(f"   Surgery后：GT < 背景 ({gap_after:+.4f})")
        print(f"   → Surgery去冗余破坏了目标特征")
    elif abs(gap_after) < abs(gap_before):
        print(f"\n⚠️ Surgery减小了GT-背景差异")
        print(f"   → 降低了目标的判别性")
    else:
        print(f"\n✅ Surgery增强了GT-背景差异（正常）")
    
    return {
        'class_name': class_name,
        'gt_raw_avg': float(np.mean(gt_raw)),
        'gt_surgery_avg': float(np.mean(gt_surgery)),
        'bg_raw_avg': float(np.mean(bg_raw)),
        'bg_surgery_avg': float(np.mean(bg_surgery)),
        'gap_before': float(gap_before),
        'gap_after': float(gap_after),
        'gap_change': float(gap_after - gap_before)
    }


def main():
    """主函数"""
    print("="*70)
    print("实验4.2：Surgery去冗余影响分析")
    print("="*70)
    
    config = Config()
    device = config.device
    
    # 加载数据
    print("\n加载数据...")
    train_loader, val_loader, test_loader, dataset = get_dataloaders(config)
    
    # 加载原始CLIP模型
    print("\n加载原始CLIP模型...")
    clip_model, _ = clip.load("ViT-B/32", device=device)
    
    # 加载RemoteCLIP权重
    remoteclip_path = Path("checkpoints/RemoteCLIP-ViT-B-32.pt")
    if remoteclip_path.exists():
        print(f"加载RemoteCLIP权重: {remoteclip_path}")
        checkpoint = torch.load(remoteclip_path, map_location=device)
        if 'state_dict' in checkpoint:
            clip_model.load_state_dict(checkpoint['state_dict'])
        else:
            clip_model.load_state_dict(checkpoint)
    
    clip_model.eval()
    
    # 分析样本
    print("\n开始对比分析...")
    
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
            
            result = compare_surgery_impact(
                clip_model,
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
    
    # 汇总
    print("\n" + "="*70)
    print("汇总分析")
    print("="*70)
    
    avg_gap_before = np.mean([r['gap_before'] for r in all_results])
    avg_gap_after = np.mean([r['gap_after'] for r in all_results])
    avg_gap_change = np.mean([r['gap_change'] for r in all_results])
    
    reversed_count = sum(1 for r in all_results if r['gap_before'] > 0 and r['gap_after'] < 0)
    
    print(f"\n平均GT-背景gap:")
    print(f"  Surgery前: {avg_gap_before:+.4f}")
    print(f"  Surgery后: {avg_gap_after:+.4f}")
    print(f"  平均变化: {avg_gap_change:+.4f}")
    
    print(f"\nSurgery反转GT-背景关系的样本数: {reversed_count}/{len(all_results)}")
    
    if reversed_count > len(all_results) / 2:
        print(f"\n❌ Surgery在多数样本上反转了GT-背景关系")
        print(f"   → Surgery去冗余确实是问题根源")
    
    # 保存
    output_dir = Path("experiment4/experiments/exp4_diagnosis/outputs")
    with open(output_dir / "surgery_impact_analysis.json", 'w', encoding='utf-8') as f:
        json.dump({
            'results': all_results,
            'summary': {
                'avg_gap_before': float(avg_gap_before),
                'avg_gap_after': float(avg_gap_after),
                'avg_gap_change': float(avg_gap_change),
                'reversed_count': reversed_count,
                'total_samples': len(all_results)
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存: {output_dir}/surgery_impact_analysis.json")


if __name__ == "__main__":
    main()

