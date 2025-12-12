#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成热图脚本 - 使用恢复后的原始SurgeryCLIP实现
"""
import sys
import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.competitors.clip_methods.surgeryclip.model_wrapper import SurgeryCLIPWrapper


def generate_heatmap_overlay(image_path: str, class_name: str, model: SurgeryCLIPWrapper, 
                            output_path: str):
    """生成单个类别的热图叠加图"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)
    h, w = image_array.shape[:2]
    
    # 生成热图
    print(f"  生成 '{class_name}' 的热图...")
    heatmap = model.generate_heatmap(image, [class_name])
    
    if heatmap is None:
        print(f"  ⚠️  热图生成失败")
        return False
    
    # 检查尺寸
    if heatmap.shape[0] != h or heatmap.shape[1] != w:
        print(f"  ⚠️  尺寸不匹配: 热图 {heatmap.shape} vs 图像 ({h}, {w})")
        from scipy.ndimage import zoom
        heatmap = zoom(heatmap, (h / heatmap.shape[0], w / heatmap.shape[1]), order=1)
        heatmap = np.clip(heatmap, 0.0, 1.0)
    
    # 转换为uint8
    heatmap_uint8 = (heatmap * 255).astype('uint8')
    
    # 应用JET colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # 与原图混合（原图0.4权重 + 热图0.6权重）
    overlay = (image_array.astype(np.float32) * 0.4 + 
              heatmap_colored_rgb.astype(np.float32) * 0.6).astype('uint8')
    
    # 保存叠加图
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.title(class_name, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"  ✅ 已保存: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='生成SurgeryCLIP热图')
    parser.add_argument('--image', type=str, required=True,
                       help='图像路径')
    parser.add_argument('--classes', type=str, nargs='+', required=True,
                       help='类别名称列表')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/ViT-B-32.pt',
                       help='模型权重路径')
    parser.add_argument('--output-dir', type=str, default='outputs/heatmaps',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建模型（使用原始配置）
    print("="*60)
    print("创建SurgeryCLIP模型（原始配置）")
    print("="*60)
    model = SurgeryCLIPWrapper(
        model_name="surgeryclip",
        checkpoint_path=args.checkpoint,
        use_surgery_single_class=True,
        use_surgery_multi_class=True,
        single_class_redundant_mode="empty",
        similarity_processing="none",  # 关键：不做后处理
        device=args.device
    )
    
    # 加载模型
    model.load_model()
    
    # 获取图像ID
    image_id = Path(args.image).stem
    
    # 为每个类别生成热图
    print(f"\n处理图像: {args.image}")
    print(f"类别: {args.classes}")
    print(f"输出目录: {output_dir}")
    
    for class_name in args.classes:
        safe_class_name = class_name.replace(' ', '_').replace('/', '_')
        output_path = output_dir / f"{image_id}_{safe_class_name}_overlay.png"
        
        generate_heatmap_overlay(args.image, class_name, model, str(output_path))
    
    print(f"\n✅ 所有热图已保存到: {output_dir}")


if __name__ == "__main__":
    main()







