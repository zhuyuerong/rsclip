#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RemoteCLIP 区域可视化工具
将提取的区域和识别结果绘制在图像上
"""

import torch
import open_clip
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import cv2
import argparse
from sampling import sample_regions
from output_manager import get_output_manager

# 颜色方案（BGR格式）
COLORS = {
    'critical': (0, 0, 255),      # 红色
    'high': (0, 165, 255),        # 橙色
    'medium': (0, 255, 255),      # 黄色
    'low': (0, 255, 0),           # 绿色
    'fallback': (128, 128, 128),  # 灰色
    'default': (255, 0, 0),       # 蓝色
}

def draw_regions_on_image(image_path, regions, output_path, region_labels=None, strategy_name=""):
    """
    在图像上绘制区域框和标签
    
    参数:
        image_path: 输入图像路径
        regions: 区域列表
        output_path: 输出图像路径
        region_labels: 每个区域的标签字典 {region_idx: (label, confidence)}
        strategy_name: 采样策略名称
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return
    
    # 创建副本用于绘制
    vis_image = image.copy()
    
    # 绘制每个区域
    for idx, region in enumerate(regions):
        x1, y1, x2, y2 = region['bbox']
        
        # 确保坐标在有效范围内
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(image.shape[1], int(x2)), min(image.shape[0], int(y2))
        
        # 选择颜色
        priority = region.get('priority', 'default')
        threshold = region.get('threshold', None)
        
        if priority in COLORS:
            color = COLORS[priority]
        elif threshold is not None:
            # 根据阈值选择颜色
            if threshold >= 0.7:
                color = COLORS['critical']
            elif threshold >= 0.5:
                color = COLORS['high']
            elif threshold >= 0.3:
                color = COLORS['medium']
            else:
                color = COLORS['low']
        else:
            color = COLORS['default']
        
        # 绘制矩形框（线条细一倍）
        thickness = 1
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
        
        # 准备标签文本
        if region_labels and idx in region_labels:
            label, confidence = region_labels[idx]
            text = f"{idx+1}: {label} {confidence:.1f}%"
        else:
            text = f"{idx+1}"
        
        # 添加优先级信息
        if priority != 'default':
            text += f" [{priority}]"
        elif threshold is not None:
            text += f" [t:{threshold}]"
        
        # 绘制标签背景（字号小一倍）
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.25
        font_thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # 标签位置（在框的上方）
        label_y = y1 - 5
        if label_y - text_height - 5 < 0:
            label_y = y1 + text_height + 5
        
        # 绘制标签背景
        cv2.rectangle(vis_image, 
                     (x1, label_y - text_height - 5), 
                     (x1 + text_width + 5, label_y + baseline),
                     color, -1)
        
        # 绘制标签文字
        cv2.putText(vis_image, text, (x1 + 2, label_y - 2),
                   font, font_scale, (255, 255, 255), font_thickness)
    
    # 添加标题（字号小一倍）
    title = f"Strategy: {strategy_name} | Regions: {len(regions)}"
    cv2.putText(vis_image, title, (10, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(vis_image, title, (10, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # 保存图像
    cv2.imwrite(output_path, vis_image)
    print(f"✅ 可视化结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='RemoteCLIP 区域可视化')
    parser.add_argument('--strategy', type=str, default='multi_threshold_saliency',
                        choices=['layered', 'pyramid', 'multi_threshold_saliency', 'all'],
                        help='采样策略 (all表示运行所有策略)')
    parser.add_argument('--model', type=str, default='RN50',
                        choices=['RN50', 'ViT-B-32', 'ViT-L-14'],
                        help='模型选择')
    parser.add_argument('--image', type=str, default='assets/airport.jpg',
                        help='输入图像路径')
    parser.add_argument('--output-dir', type=str, default='extensions/outputs/visualizations',
                        help='输出目录')
    parser.add_argument('--max-regions', type=int, default=50,
                        help='最大区域数')
    parser.add_argument('--top-k', type=int, default=10,
                        help='分析前K个重要区域')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("RemoteCLIP 区域可视化工具")
    print("=" * 70)
    
    # 创建输出目录
    om = get_output_manager()
    if args.output_dir == 'extensions/outputs/visualizations':
        # 使用默认输出管理器路径
        args.output_dir = om.dirs['visualizations']
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 加载模型
    model_name = args.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n🔄 正在加载 {model_name} 模型...")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    
    checkpoint_path = f"checkpoints/RemoteCLIP-{model_name}.pt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ 找不到模型文件: {checkpoint_path}")
        return
    
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt)
    model = model.to(device).eval()
    print(f"✅ 模型已加载到 {device}")
    
    # 2. 加载图像
    if not os.path.exists(args.image):
        print(f"❌ 找不到图像: {args.image}")
        return
    
    pil_image = Image.open(args.image)
    cv_image = np.array(pil_image)
    print(f"✅ 图像已加载: {cv_image.shape}")
    
    # 3. 定义查询
    region_queries = [
        "airport", "runway", "airplane", "aircraft", 
        "building", "terminal", "parking lot", "road",
        "vegetation", "water"
    ]
    
    # 预编码文本特征
    region_text = tokenizer(region_queries)
    with torch.no_grad():
        region_text_features = model.encode_text(region_text.to(device))
        region_text_features /= region_text_features.norm(dim=-1, keepdim=True)
    
    # 4. 处理策略
    strategies = ['multi_threshold_saliency', 'layered', 'pyramid'] if args.strategy == 'all' else [args.strategy]
    
    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"📊 处理策略: {strategy}")
        print(f"{'='*70}")
        
        # 区域采样
        regions = sample_regions(
            cv_image,
            strategy=strategy,
            max_regions=args.max_regions
        )
        
        if len(regions) == 0:
            print("❌ 未找到区域")
            continue
        
        # 分析每个区域
        region_labels = {}
        print(f"\n分析前 {min(args.top_k, len(regions))} 个区域...")
        
        for idx, region in enumerate(regions[:args.top_k]):
            x1, y1, x2, y2 = region['bbox']
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(cv_image.shape[1], x2), min(cv_image.shape[0], y2)
            
            # 裁剪区域
            cropped = cv_image[y1:y2, x1:x2]
            if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
                continue
            
            cropped_pil = Image.fromarray(cropped)
            cropped_tensor = preprocess(cropped_pil).unsqueeze(0)
            
            # 推理
            with torch.no_grad():
                crop_features = model.encode_image(cropped_tensor.to(device))
                crop_features /= crop_features.norm(dim=-1, keepdim=True)
                probs = (100.0 * crop_features @ region_text_features.T).softmax(dim=-1).cpu().numpy()[0]
            
            # 保存最佳匹配
            best_idx = probs.argmax()
            best_label = region_queries[best_idx]
            best_confidence = probs[best_idx] * 100
            
            region_labels[idx] = (best_label, best_confidence)
            
            print(f"  区域 {idx+1}: {best_label} ({best_confidence:.1f}%)")
        
        # 5. 可视化并保存
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        output_path = os.path.join(args.output_dir, f"{base_name}_{strategy}_visualization.jpg")
        
        draw_regions_on_image(
            args.image,
            regions[:args.top_k],
            output_path,
            region_labels,
            strategy_name=strategy
        )
    
    print(f"\n{'='*70}")
    print(f"✅ 所有可视化完成！")
    print(f"   输出目录: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

