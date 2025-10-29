#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试边界框微调功能
创建一个模拟场景来演示框微调的效果
"""

import torch
import open_clip
from PIL import Image
import numpy as np
import cv2
from bbox_refinement import BBoxRefinement, compute_saliency_map


def test_bbox_refinement():
    """测试框微调功能"""
    print("=" * 70)
    print("边界框微调功能测试")
    print("=" * 70)
    
    # 1. 加载模型
    print("\n🔄 加载RemoteCLIP模型...")
    model_name = 'RN50'
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    
    checkpoint_path = f"checkpoints/RemoteCLIP-{model_name}.pt"
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"✅ 模型已加载到 {device}")
    
    # 2. 加载测试图像
    image_path = "assets/airport.jpg"
    pil_image = Image.open(image_path)
    cv_image = np.array(pil_image)
    print(f"✅ 图像已加载: {cv_image.shape}")
    
    # 3. 计算显著性图
    print("\n🔄 计算显著性图...")
    saliency_map = compute_saliency_map(cv_image)
    print(f"✅ 显著性图: {saliency_map.shape}, 范围: {saliency_map.min()}-{saliency_map.max()}")
    
    # 4. 创建模拟的正负样本原型
    print("\n🔄 创建模拟原型...")
    
    # 正样本：航站楼区域
    positive_crops = [
        cv_image[50:100, 50:100],
        cv_image[100:150, 100:150],
    ]
    
    # 负样本：跑道区域
    negative_crops = [
        cv_image[0:50, 0:50],
        cv_image[150:200, 150:200],
    ]
    
    positive_features = []
    for crop in positive_crops:
        crop_pil = Image.fromarray(crop)
        crop_tensor = preprocess(crop_pil).unsqueeze(0)
        with torch.no_grad():
            feat = model.encode_image(crop_tensor.to(device))
            feat /= feat.norm(dim=-1, keepdim=True)
            positive_features.append(feat)
    
    positive_prototype = torch.cat(positive_features).mean(dim=0, keepdim=True)
    
    negative_features = []
    for crop in negative_crops:
        crop_pil = Image.fromarray(crop)
        crop_tensor = preprocess(crop_pil).unsqueeze(0)
        with torch.no_grad():
            feat = model.encode_image(crop_tensor.to(device))
            feat /= feat.norm(dim=-1, keepdim=True)
            negative_features.append(feat)
    
    negative_prototype = torch.cat(negative_features).mean(dim=0, keepdim=True)
    print("✅ 原型已创建")
    
    # 5. 创建微调器
    bbox_refiner = BBoxRefinement(model, preprocess, device)
    
    # 6. 测试不同的微调方法
    test_bbox = (80, 80, 150, 150)
    print(f"\n初始边界框: {test_bbox}")
    
    methods = ['position', 'scale', 'both', 'boundary']
    
    for method in methods:
        print(f"\n{'='*70}")
        print(f"测试方法: {method}")
        print(f"{'='*70}")
        
        result = bbox_refiner.refine_bbox_hybrid(
            image=cv_image,
            bbox=test_bbox,
            saliency_map=saliency_map,
            positive_prototype=positive_prototype,
            negative_prototype=negative_prototype,
            method=method
        )
        
        print(f"结果:")
        print(f"  初始bbox: {test_bbox}")
        print(f"  优化bbox: {result['bbox']}")
        print(f"  是否优化: {result.get('refined', False)}")
        print(f"  对比分数: {result.get('contrast_score', 0.0):.4f}")
        print(f"  显著性分数: {result.get('saliency_score', 0.0):.4f}")
        print(f"  综合分数: {result.get('composite_score', 0.0):.4f}")
        
        if 'scale' in result:
            print(f"  尺度因子: {result['scale']}")
        
        if 'iterations' in result:
            print(f"  迭代次数: {result['iterations']}")
    
    print("\n" + "="*70)
    print("✅ 框微调功能测试完成！")
    print("="*70)
    
    print("\n💡 框微调说明:")
    print("  - position: 基于显著性峰值优化位置")
    print("  - scale: 多尺度搜索最佳尺寸")
    print("  - both: 先优化位置，再优化尺寸（推荐）")
    print("  - boundary: 逐步调整四条边界")
    print("\n综合得分 = 0.7 × 对比分数 + 0.3 × 显著性分数")


if __name__ == "__main__":
    test_bbox_refinement()

