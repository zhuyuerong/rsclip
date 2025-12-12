#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
调试归一化前后的相似度分布
"""
import sys
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.competitors.clip_methods.surgeryclip.model_wrapper import SurgeryCLIPWrapper

print("="*70)
print("🔍 调试归一化前后的相似度分布")
print("="*70)

# 测试图像
image_path = "datasets/mini-DIOR/images/00679.jpg"
image = Image.open(image_path).convert('RGB')

# ============ 测试1: 单类别Surgery ============
print("\n" + "="*70)
print("测试1: 单类别Surgery")
print("="*70)

model1 = SurgeryCLIPWrapper(
    model_name='surgeryclip',
    checkpoint_path='checkpoints/ViT-B-32.pt',
    use_surgery_single_class=True,
    use_surgery_multi_class=False,
    device='cpu'
)
model1.load_model()

with torch.no_grad():
    image_tensor = model1.preprocess(image).unsqueeze(0).to(model1.device)
    image_features_all = model1.model.encode_image(image_tensor)
    image_features_all = image_features_all / image_features_all.norm(dim=-1, keepdim=True)
    
    from src.competitors.clip_methods.surgeryclip.clip import encode_text_with_prompt_ensemble, clip_feature_surgery
    text_features = encode_text_with_prompt_ensemble(model1.model, ['vehicle'], model1.device)
    
    redundant_features = encode_text_with_prompt_ensemble(model1.model, [""], model1.device)
    similarity_maps = clip_feature_surgery(image_features_all, text_features, redundant_features)
    
    # 排除class token
    similarity_maps_patches = similarity_maps[:, 1:, :]  # [1, 49, 1]
    
    print(f"\n归一化前统计:")
    similarity_np = similarity_maps_patches.detach().cpu().numpy().flatten()
    print(f"  min={similarity_np.min():.6f}, max={similarity_np.max():.6f}")
    print(f"  mean={similarity_np.mean():.6f}, std={similarity_np.std():.6f}")
    print(f"  负值%: {(similarity_np < 0).sum() / len(similarity_np) * 100:.2f}%")
    
    # 归一化
    from src.competitors.clip_methods.surgeryclip.clip import get_similarity_map
    target_h, target_w = image.size[1], image.size[0]
    heatmap_tensor = get_similarity_map(similarity_maps_patches, (target_h, target_w))
    heatmap = heatmap_tensor[0, :, :, 0].detach().cpu().numpy()
    
    print(f"\n归一化后统计:")
    print(f"  min={heatmap.min():.6f}, max={heatmap.max():.6f}")
    print(f"  mean={heatmap.mean():.6f}, std={heatmap.std():.6f}")
    
    # 分位数
    print(f"\n归一化后分位数:")
    for p in [0, 10, 25, 50, 75, 90, 100]:
        val = np.percentile(heatmap, p)
        print(f"  {p:3d}%: {val:.6f}")

# ============ 测试2: 单类别余弦 ============
print("\n" + "="*70)
print("测试2: 单类别余弦")
print("="*70)

model2 = SurgeryCLIPWrapper(
    model_name='surgeryclip',
    checkpoint_path='checkpoints/ViT-B-32.pt',
    use_surgery_single_class=False,
    use_surgery_multi_class=False,
    device='cpu'
)
model2.load_model()

with torch.no_grad():
    image_tensor = model2.preprocess(image).unsqueeze(0).to(model2.device)
    image_features_all = model2.model.encode_image(image_tensor)
    image_features_all = image_features_all / image_features_all.norm(dim=-1, keepdim=True)
    
    from src.competitors.clip_methods.surgeryclip.clip import encode_text_with_prompt_ensemble
    text_features = encode_text_with_prompt_ensemble(model2.model, ['vehicle'], model2.device)
    
    similarity_maps = image_features_all @ text_features.t()
    
    # 排除class token
    similarity_maps_patches = similarity_maps[:, 1:, :]  # [1, 49, 1]
    
    print(f"\n归一化前统计:")
    similarity_np = similarity_maps_patches.detach().cpu().numpy().flatten()
    print(f"  min={similarity_np.min():.6f}, max={similarity_np.max():.6f}")
    print(f"  mean={similarity_np.mean():.6f}, std={similarity_np.std():.6f}")
    print(f"  负值%: {(similarity_np < 0).sum() / len(similarity_np) * 100:.2f}%")
    
    # 归一化
    from src.competitors.clip_methods.surgeryclip.clip import get_similarity_map
    target_h, target_w = image.size[1], image.size[0]
    heatmap_tensor = get_similarity_map(similarity_maps_patches, (target_h, target_w))
    heatmap = heatmap_tensor[0, :, :, 0].detach().cpu().numpy()
    
    print(f"\n归一化后统计:")
    print(f"  min={heatmap.min():.6f}, max={heatmap.max():.6f}")
    print(f"  mean={heatmap.mean():.6f}, std={heatmap.std():.6f}")
    
    # 分位数
    print(f"\n归一化后分位数:")
    for p in [0, 10, 25, 50, 75, 90, 100]:
        val = np.percentile(heatmap, p)
        print(f"  {p:3d}%: {val:.6f}")

# ============ 对比分析 ============
print("\n" + "="*70)
print("对比分析")
print("="*70)

print("\n关键发现:")
print("1. 如果归一化后的std（标准差）很接近 → 视觉上会很像")
print("2. 如果归一化后的分位数分布很接近 → 热图看起来一样")
print("3. 需要检查归一化前的相对分布是否相似")

print("\n" + "="*70)






