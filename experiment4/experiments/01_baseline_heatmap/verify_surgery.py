# -*- coding: utf-8 -*-
"""
实验A：验证Surgery去冗余是否在热图生成中执行

关键问题：
- CLIPSurgeryWrapper.get_patch_features() 是否应用了 F - mean(F)?
- 热图生成用的是Surgery特征还是RemoteCLIP原始特征?
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import clip
from tqdm import tqdm
from collections import defaultdict

# 添加路径
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.append(str(root_dir))

from experiment4.core.config import Config
from experiment4.core.models.clip_surgery import CLIPSurgeryWrapper
from experiment4.core.data.dataset import get_dataloaders
from experiment4.core.utils.heatmap_generator_v2 import generate_bboxes_from_heatmap_v2, compute_bbox_score
from experiment4.core.utils.map_calculator import calculate_map


def extract_raw_remoteclip_features(clip_model, images):
    """
    提取原始RemoteCLIP特征（完全不经过Surgery去冗余）
    """
    with torch.no_grad():
        if images.dtype != clip_model.visual.conv1.weight.dtype:
            images = images.to(clip_model.visual.conv1.weight.dtype)
        
        x = clip_model.visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        
        x = torch.cat([
            clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            ),
            x
        ], dim=1)
        
        x = x + clip_model.visual.positional_embedding.to(x.dtype)
        x = clip_model.visual.ln_pre(x)
        
        x = x.permute(1, 0, 2)
        x = clip_model.visual.transformer(x)
        x = x.permute(1, 0, 2)
        
        x = clip_model.visual.ln_post(x)
        
        if hasattr(clip_model.visual, 'proj') and clip_model.visual.proj is not None:
            B, N, D = x.shape
            x_reshaped = x.reshape(B * N, D)
            x_proj = x_reshaped @ clip_model.visual.proj
            features = x_proj.reshape(B, N, -1)
        else:
            features = x
    
    return features


def apply_surgery_manually(features):
    """
    手动应用Surgery去冗余: F_surgery = F - mean(F, dim=1)
    
    Args:
        features: [B, N+1, 512] 包含CLS的特征
    
    Returns:
        features_surgery: [B, N+1, 512] 去冗余后的特征
    """
    patch_features = features[:, 1:, :]  # [B, N, 512]
    
    # Surgery去冗余
    redundant = patch_features.mean(dim=1, keepdim=True)  # [B, 1, 512]
    patch_surgery = patch_features - redundant  # [B, N, 512]
    
    # 重新组合（CLS保持不变）
    features_surgery = torch.cat([features[:, 0:1, :], patch_surgery], dim=1)
    
    return features_surgery


def verify_surgery_execution_single_image(wrapper, clip_model, image, device):
    """
    验证单张图像的Surgery执行情况
    """
    print("\n" + "="*70)
    print("单图像Surgery执行验证")
    print("="*70)
    
    image = image.unsqueeze(0).to(device)
    
    # 1. 通过wrapper提取
    features_wrapper = wrapper.get_all_features(image)  # [1, 50, 512]
    patch_wrapper = features_wrapper[:, 1:, :]  # [1, 49, 512]
    
    # 2. 直接从CLIP提取原始特征
    features_raw = extract_raw_remoteclip_features(clip_model, image)  # [1, 50, 512]
    patch_raw = features_raw[:, 1:, :]  # [1, 49, 512]
    
    # 3. 手动应用Surgery
    features_manual_surgery = apply_surgery_manually(features_raw)
    patch_manual_surgery = features_manual_surgery[:, 1:, :]
    
    # 统计对比
    print(f"\n特征统计对比:")
    print(f"-"*70)
    
    print(f"原始RemoteCLIP特征 (patch):")
    print(f"  Mean: {patch_raw.mean().item():.6f}")
    print(f"  Std:  {patch_raw.std().item():.6f}")
    print(f"  Min:  {patch_raw.min().item():.6f}")
    print(f"  Max:  {patch_raw.max().item():.6f}")
    
    print(f"\n手动Surgery特征 (F - mean(F)):")
    print(f"  Mean: {patch_manual_surgery.mean().item():.6f}")
    print(f"  Std:  {patch_manual_surgery.std().item():.6f}")
    print(f"  Min:  {patch_manual_surgery.min().item():.6f}")
    print(f"  Max:  {patch_manual_surgery.max().item():.6f}")
    
    print(f"\nWrapper提取的特征:")
    print(f"  Mean: {patch_wrapper.mean().item():.6f}")
    print(f"  Std:  {patch_wrapper.std().item():.6f}")
    print(f"  Min:  {patch_wrapper.min().item():.6f}")
    print(f"  Max:  {patch_wrapper.max().item():.6f}")
    
    # 差异检查
    diff_raw_vs_manual = (patch_raw - patch_manual_surgery).abs().mean().item()
    diff_raw_vs_wrapper = (patch_raw - patch_wrapper).abs().mean().item()
    diff_manual_vs_wrapper = (patch_manual_surgery - patch_wrapper).abs().mean().item()
    
    print(f"\n差异检查:")
    print(f"-"*70)
    print(f"原始 vs 手动Surgery:  {diff_raw_vs_manual:.6f}")
    print(f"原始 vs Wrapper:      {diff_raw_vs_wrapper:.6f}")
    print(f"手动Surgery vs Wrapper: {diff_manual_vs_wrapper:.6f}")
    
    # 判断
    print(f"\n" + "="*70)
    print(f"关键结论:")
    print(f"="*70)
    
    if diff_raw_vs_wrapper < 1e-5:
        print(f"❌ Wrapper输出 = 原始RemoteCLIP（差异{diff_raw_vs_wrapper:.8f}）")
        print(f"   → Surgery去冗余未在热图生成中执行！")
        surgery_applied = False
    elif diff_manual_vs_wrapper < 1e-5:
        print(f"✅ Wrapper输出 = 手动Surgery（差异{diff_manual_vs_wrapper:.8f}）")
        print(f"   → Surgery去冗余正常执行")
        surgery_applied = True
    else:
        print(f"⚠️ Wrapper输出与两者都不同")
        print(f"   → Surgery实现可能有其他逻辑")
        surgery_applied = "unknown"
    
    return surgery_applied, {
        'patch_raw_mean': float(patch_raw.mean().item()),
        'patch_raw_std': float(patch_raw.std().item()),
        'patch_manual_surgery_mean': float(patch_manual_surgery.mean().item()),
        'patch_manual_surgery_std': float(patch_manual_surgery.std().item()),
        'patch_wrapper_mean': float(patch_wrapper.mean().item()),
        'patch_wrapper_std': float(patch_wrapper.std().item()),
        'diff_raw_vs_wrapper': diff_raw_vs_wrapper,
        'diff_manual_vs_wrapper': diff_manual_vs_wrapper
    }


def generate_heatmap_with_surgery_control(features, text_features, apply_surgery=False):
    """
    生成热图，可选是否应用Surgery去冗余
    
    Args:
        features: [B, N+1, 512] 完整特征
        text_features: [K, 512] 文本特征
        apply_surgery: 是否应用Surgery去冗余
    
    Returns:
        similarity_map: [B, H, W, K]
    """
    # 提取patch
    patch_features = features[:, 1:, :]  # [B, N, 512]
    
    # 可选：应用Surgery
    if apply_surgery:
        redundant = patch_features.mean(dim=1, keepdim=True)
        patch_features = patch_features - redundant
    
    # 归一化并计算相似度
    patch_norm = F.normalize(patch_features, dim=-1, p=2)
    text_norm = F.normalize(text_features, dim=-1, p=2)
    
    similarity = torch.einsum('bnd,kd->bnk', patch_norm, text_norm)
    
    B, N, K = similarity.shape
    H = W = int(N ** 0.5)
    similarity_map = similarity.reshape(B, H, W, K)
    
    return similarity_map


def evaluate_with_surgery_option(wrapper, clip_model, val_loader, apply_surgery, use_inverted, threshold_pct, device):
    """
    评估特定Surgery配置
    
    Args:
        apply_surgery: 是否应用Surgery去冗余
        use_inverted: 是否反转相似度
        threshold_pct: 阈值百分位
    """
    all_predictions = defaultdict(list)
    all_ground_truths = defaultdict(list)
    
    method_name = f"{'Surgery+' if apply_surgery else '原始'}{'反转+' if use_inverted else ''}阈值{threshold_pct}%"
    
    for batch in tqdm(val_loader, desc=method_name, leave=False):
        images = batch['image'].to(device)
        class_names = batch['class_name']
        gt_bboxes = batch['bbox']
        has_bbox = batch['has_bbox']
        
        # 提取原始特征
        with torch.no_grad():
            features_raw = extract_raw_remoteclip_features(clip_model, images)
            
            # 文本特征
            unique_classes = list(set(class_names))
            text_features = wrapper.encode_text(unique_classes)
        
        # 生成热图（可选Surgery）
        similarity_map = generate_heatmap_with_surgery_control(
            features_raw, text_features, apply_surgery=apply_surgery
        )
        
        # 可选反转
        if use_inverted:
            similarity_map = -similarity_map
        
        # 处理每个样本
        for i in range(len(images)):
            if not has_bbox[i]:
                continue
            
            class_name = class_names[i]
            class_idx = unique_classes.index(class_name)
            heatmap = similarity_map[i, :, :, class_idx].cpu().numpy()
            
            # 生成bbox
            pred_bboxes = generate_bboxes_from_heatmap_v2(heatmap, threshold_percentile=threshold_pct)
            
            for bbox in pred_bboxes:
                score = compute_bbox_score(heatmap, bbox)
                all_predictions[class_name].append({'bbox': bbox, 'score': score})
            
            # GT
            gt_bbox_norm = gt_bboxes[i].cpu().numpy()
            gt_bbox_pixel = [
                int(gt_bbox_norm[0] * 224),
                int(gt_bbox_norm[1] * 224),
                int(gt_bbox_norm[2] * 224),
                int(gt_bbox_norm[3] * 224)
            ]
            all_ground_truths[class_name].append(gt_bbox_pixel)
    
    # 计算mAP
    mAP_005, _ = calculate_map(all_predictions, all_ground_truths, iou_threshold=0.05)
    mAP_050, per_class = calculate_map(all_predictions, all_ground_truths, iou_threshold=0.50)
    
    return {
        'method': method_name,
        'apply_surgery': apply_surgery,
        'use_inverted': use_inverted,
        'threshold': threshold_pct,
        'mAP_005': mAP_005,
        'mAP_050': mAP_050,
        'per_class': per_class
    }


def main():
    """主函数"""
    print("="*70)
    print("实验A：Surgery去冗余验证")
    print("="*70)
    
    config = Config()
    device = config.device
    
    # 加载数据
    print("\n加载数据...")
    train_loader, val_loader, test_loader, dataset = get_dataloaders(config)
    
    # 加载模型
    print("\n加载模型...")
    wrapper = CLIPSurgeryWrapper(config)
    
    # 加载原始CLIP模型（用于提取原始特征）
    clip_model, _ = clip.load("ViT-B/32", device=device)
    remoteclip_path = Path("checkpoints/RemoteCLIP-ViT-B-32.pt")
    if remoteclip_path.exists():
        checkpoint = torch.load(remoteclip_path, map_location=device)
        if 'state_dict' in checkpoint:
            clip_model.load_state_dict(checkpoint['state_dict'])
        else:
            clip_model.load_state_dict(checkpoint)
    clip_model.eval()
    
    # ===== 步骤1：单图像验证Surgery是否执行 =====
    print("\n" + "="*70)
    print("步骤1：单图像Surgery执行验证")
    print("="*70)
    
    # 获取一张图
    sample_batch = next(iter(val_loader))
    sample_image = sample_batch['image'][0]
    
    surgery_applied, stats = verify_surgery_execution_single_image(
        wrapper, clip_model, sample_image, device
    )
    
    # ===== 步骤2：评估不同Surgery配置的mAP =====
    print("\n" + "="*70)
    print("步骤2：Surgery配置对mAP的影响")
    print("="*70)
    
    # 测试配置矩阵
    test_configs = [
        # (apply_surgery, use_inverted, threshold)
        (False, False, 30),  # 当前版本（RemoteCLIP原始）
        (True, False, 30),   # 添加Surgery去冗余
        (True, True, 30),    # Surgery + 反转
        (False, True, 30),   # 原始 + 反转（对照）
    ]
    
    all_results = []
    
    for apply_surgery, use_inverted, threshold in test_configs:
        print(f"\n测试配置: Surgery={apply_surgery}, 反转={use_inverted}, 阈值={threshold}%")
        
        result = evaluate_with_surgery_option(
            wrapper, clip_model, val_loader, 
            apply_surgery, use_inverted, threshold, device
        )
        all_results.append(result)
        
        print(f"  mAP@0.05: {result['mAP_005']:.4f}")
        print(f"  mAP@0.50: {result['mAP_050']:.4f}")
    
    # ===== 步骤3：对比分析 =====
    print("\n" + "="*70)
    print("步骤3：对比分析")
    print("="*70)
    
    print(f"\n完整对比表:")
    print(f"{'配置':<25} {'mAP@0.05':>10} {'mAP@0.50':>10}")
    print(f"-"*50)
    
    for r in all_results:
        print(f"{r['method']:<25} {r['mAP_005']:>10.4f} {r['mAP_050']:>10.4f}")
    
    # 找出Surgery的实际影响
    baseline = all_results[0]  # 原始RemoteCLIP
    with_surgery = all_results[1]  # +Surgery
    surgery_inverted = all_results[2]  # Surgery+反转
    
    surgery_impact = with_surgery['mAP_005'] - baseline['mAP_005']
    inverted_impact = surgery_inverted['mAP_005'] - with_surgery['mAP_005']
    
    print(f"\nSurgery影响分析:")
    print(f"-"*70)
    print(f"添加Surgery去冗余的影响: {surgery_impact:+.4f} ({surgery_impact/baseline['mAP_005']*100:+.1f}%)")
    print(f"反转修正的影响: {inverted_impact:+.4f} ({inverted_impact/with_surgery['mAP_005']*100:+.1f}%)")
    
    # ===== 结论 =====
    print(f"\n" + "="*70)
    print(f"实验结论:")
    print(f"="*70)
    
    if surgery_applied == False:
        print(f"\n❌ 关键发现：CLIPSurgeryWrapper.get_patch_features() 未应用Surgery去冗余")
        print(f"   - 当前热图使用的是RemoteCLIP原始特征")
        print(f"   - SimplifiedDenoiser中的Surgery去冗余只在训练时使用")
        print(f"   - 热图评估和训练使用了不同的特征！")
    else:
        print(f"\n✅ Surgery去冗余正常执行")
    
    if abs(surgery_impact) > 0.05:
        print(f"\n✅ Surgery去冗余显著影响mAP ({surgery_impact:+.4f})")
        if surgery_impact < 0:
            print(f"   → Surgery降低了mAP（抑制GT特征）")
            if inverted_impact > abs(surgery_impact):
                print(f"   → 但反转修正后整体提升")
        else:
            print(f"   → Surgery提升了mAP")
    else:
        print(f"\n⚠️ Surgery去冗余对mAP影响很小 ({surgery_impact:+.4f})")
    
    # 保存结果
    output_dir = Path("experiment4/experiments/01_baseline_heatmap/results")
    
    results_to_save = {
        'surgery_execution_check': {
            'surgery_applied_in_wrapper': surgery_applied,
            'statistics': stats
        },
        'mAP_comparison': all_results
    }
    
    with open(output_dir / "surgery_verification.json", 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存: {output_dir}/surgery_verification.json")


if __name__ == "__main__":
    main()

