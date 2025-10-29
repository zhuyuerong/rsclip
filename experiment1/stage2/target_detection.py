#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目标检测脚本（遥感通用）
输入目标类别（如ship），输出得分最高的那一个框
"""

import torch
import open_clip
from PIL import Image
import numpy as np
import cv2
import argparse
from typing import List, Dict

from sampling import sample_regions
from wordnet_vocabulary import get_expansion_words, get_synonyms, WORDNET_REMOTE_SENSING_CLASSES
from output_manager import get_output_manager


def detect_target_with_contrastive_learning(
    image_path: str,
    target_class: str,
    model_name: str = 'RN50',
    strategy: str = 'multi_threshold_saliency',
    max_regions: int = 50,
    output_path: str = None
):
    """
    使用对比学习检测目标类别
    
    流程:
    1. 区域采样 → 候选框
    2. 定义类别:
       - 目标类 + 近义词 → 正样本参考
       - 其他100类 → 负样本参考
    3. 批量推理 → 相似度矩阵
    4. 对比学习 → 找出与目标类最相似的框
    5. 输出得分最高的1个框
    
    参数:
        image_path: 输入图像
        target_class: 目标类别（如 "ship", "airplane"等）
        model_name: 模型名称
        strategy: 采样策略
        max_regions: 最大区域数
        output_path: 输出图像路径
    """
    print("=" * 70)
    print(f"🎯 目标检测: {target_class}")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 加载模型
    print(f"\n🔄 加载RemoteCLIP模型: {model_name}")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    
    checkpoint_path = f"checkpoints/RemoteCLIP-{model_name}.pt"
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt)
    model = model.to(device).eval()
    print(f"✅ 模型已加载")
    
    # 2. 加载图像
    pil_image = Image.open(image_path)
    cv_image = np.array(pil_image)
    print(f"✅ 图像已加载: {cv_image.shape}")
    
    # 3. 定义类别体系
    # 正样本参考：目标类 + 近义词
    target_synonyms = get_synonyms(target_class)
    if target_synonyms:
        positive_classes = [target_class] + target_synonyms[:4]  # 目标类+最多4个近义词
    else:
        positive_classes = [target_class]
    
    # 负样本参考：100个基础类别（排除目标类及其近义词）
    negative_classes = [c for c in WORDNET_REMOTE_SENSING_CLASSES 
                        if c not in positive_classes]
    
    # 所有类别
    all_classes = positive_classes + negative_classes
    n_positive = len(positive_classes)
    
    print(f"\n📋 类别设置:")
    print(f"   目标类: {target_class}")
    print(f"   正样本参考 ({n_positive}个): {positive_classes}")
    print(f"   负样本参考: {len(negative_classes)}个基础类别")
    print(f"   总类别数: {len(all_classes)}")
    
    # 4. 预编码文本特征
    print(f"\n🔄 编码文本特征...")
    text = tokenizer(all_classes)
    with torch.no_grad():
        text_features = model.encode_text(text.to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # 分离正负样本特征
    positive_text_features = text_features[:n_positive]  # 前n个是正样本
    negative_text_features = text_features[n_positive:]  # 后面是负样本
    
    print(f"✅ 文本特征已编码")
    
    # 5. 区域采样
    print(f"\n🔍 Step 1: 区域采样")
    regions = sample_regions(cv_image, strategy=strategy, max_regions=max_regions)
    print(f"✅ 提取到 {len(regions)} 个候选区域")
    
    # 6. 对每个区域计算对比得分
    print(f"\n🔄 Step 2: 对比学习评分...")
    results = []
    
    for idx, region in enumerate(regions):
        x1, y1, x2, y2 = region['bbox']
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(cv_image.shape[1], x2), min(cv_image.shape[0], y2)
        
        # 裁剪
        crop = cv_image[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            continue
        
        crop_pil = Image.fromarray(crop)
        crop_tensor = preprocess(crop_pil).unsqueeze(0)
        
        # 编码
        with torch.no_grad():
            image_features = model.encode_image(crop_tensor.to(device))
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # 计算与所有类别的相似度
            similarities = (image_features @ text_features.T).squeeze(0).cpu().numpy()
            
            # 正样本得分（与目标类+近义词的相似度）
            positive_scores = similarities[:n_positive]
            positive_avg = positive_scores.mean()
            positive_max = positive_scores.max()
            best_positive_idx = positive_scores.argmax()
            
            # 负样本得分（与其他类别的相似度）
            negative_scores = similarities[n_positive:]
            negative_avg = negative_scores.mean()
            negative_max = negative_scores.max()
            
            # 改进的对比得分计算
            # 1. 基础对比得分
            base_contrast = positive_avg - negative_avg
            
            # 2. 加强背景降分：负样本最高分也要惩罚
            background_penalty = negative_max * 0.5  # 负样本最高分的一半作为惩罚
            
            # 3. 最终对比得分 = 基础对比 - 背景惩罚
            contrast_score = base_contrast - background_penalty
            
            # 4. 更严格的判断条件
            is_target = (positive_avg > negative_avg) and (positive_max > negative_max)
        
        if is_target:  # 只保留目标类别的框
            results.append({
                'bbox': (x1, y1, x2, y2),
                'region_idx': idx,
                'target_class': positive_classes[best_positive_idx],
                'target_score': float(positive_max),
                'positive_avg': float(positive_avg),
                'negative_avg': float(negative_avg),
                'contrast_score': float(contrast_score),
                'saliency': region.get('saliency', 0),
                'priority': region.get('priority', 'N/A')
            })
    
    # 7. 按对比得分排序，只保留最佳的1个
    results.sort(key=lambda x: x['contrast_score'], reverse=True)
    
    print(f"✅ 找到 {len(results)} 个{target_class}候选框")
    
    if len(results) == 0:
        print(f"❌ 未找到{target_class}类别的框")
        return None
    
    # 8. 输出最佳结果
    best_result = results[0]
    
    print(f"\n{'='*70}")
    print(f"🏆 最佳{target_class}框选结果")
    print(f"{'='*70}")
    print(f"  📍 位置: {best_result['bbox']}")
    print(f"  🏷️  匹配类别: {best_result['target_class']}")
    print(f"  📊 目标类得分: {best_result['target_score']:.3f}")
    print(f"  📊 正样本平均: {best_result['positive_avg']:.3f}")
    print(f"  📊 负样本平均: {best_result['negative_avg']:.3f}")
    print(f"  💯 对比得分: {best_result['contrast_score']:.3f}")
    print(f"  ⭐ 显著性: {best_result['saliency']:.1f}")
    
    # 9. 可视化（只画1个框）
    if output_path:
        vis_image = cv2.imread(image_path)
        x1, y1, x2, y2 = best_result['bbox']
        
        # 确保坐标是整数
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        print(f"\n🔍 绘制框选: ({x1}, {y1}) -> ({x2}, {y2})")
        
        # 绘制更明显的框（红色，很粗的线）
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 5)  # 红色，5像素粗
        
        # 绘制内框（白色，细线）
        cv2.rectangle(vis_image, (x1+2, y1+2), (x2-2, y2-2), (255, 255, 255), 2)
        
        # 标签
        label1 = f"{best_result['target_class']}: {best_result['target_score']:.1%}"
        label2 = f"Contrast: {best_result['contrast_score']:.3f}"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 标签1（红色背景）
        (tw, th), _ = cv2.getTextSize(label1, font, 0.8, 2)
        cv2.rectangle(vis_image, (x1, y1-th-15), (x1+tw+15, y1), (0, 0, 255), -1)
        cv2.putText(vis_image, label1, (x1+8, y1-8), font, 0.8, (255, 255, 255), 2)
        
        # 标签2（红色背景）
        (tw2, th2), _ = cv2.getTextSize(label2, font, 0.6, 2)
        cv2.rectangle(vis_image, (x1, y2+5), (x1+tw2+15, y2+th2+15), (0, 0, 255), -1)
        cv2.putText(vis_image, label2, (x1+8, y2+th2+12), font, 0.6, (255, 255, 255), 2)
        
        # 标题
        title = f"Target: {target_class} (Best Detection)"
        cv2.putText(vis_image, title, (10, 40), font, 1.0, (255, 255, 255), 4)
        cv2.putText(vis_image, title, (10, 40), font, 1.0, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, vis_image)
        print(f"\n✅ 可视化已保存: {output_path}")
    
    return best_result


def main():
    parser = argparse.ArgumentParser(description='遥感目标检测（通用）')
    parser.add_argument('--image', type=str, required=True,
                        help='输入图像路径')
    parser.add_argument('--target', type=str, required=True,
                        help='目标类别（如: ship, airplane, building等）')
    parser.add_argument('--model', type=str, default='RN50',
                        choices=['RN50', 'ViT-B-32', 'ViT-L-14'],
                        help='模型选择')
    parser.add_argument('--strategy', type=str, default='multi_threshold_saliency',
                        choices=['layered', 'pyramid', 'multi_threshold_saliency'],
                        help='采样策略')
    parser.add_argument('--max-regions', type=int, default=50,
                        help='最大区域数')
    parser.add_argument('--output', type=str, default=None,
                        help='输出图像路径')
    
    args = parser.parse_args()
    
    # 自动生成输出路径
    if args.output is None:
        om = get_output_manager()
        args.output = om.get_detection_result_path(args.target, args.model)
    
    # 运行检测
    result = detect_target_with_contrastive_learning(
        image_path=args.image,
        target_class=args.target,
        model_name=args.model,
        strategy=args.strategy,
        max_regions=args.max_regions,
        output_path=args.output
    )
    
    print(f"\n{'='*70}")
    print("✅ 检测完成！")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

