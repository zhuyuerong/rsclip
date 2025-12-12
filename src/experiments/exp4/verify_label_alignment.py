# -*- coding: utf-8 -*-
"""
验证标签和图像对齐
随机选择3-5张图像，读取GT框和类别，画在图像上，人眼检查
"""

import torch
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import yaml
import sys
from typing import List, Dict

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))
from datasets.dior_detection import get_detection_dataloader


# DIOR类别列表
DIOR_CLASSES = [
    "airplane", "airport", "baseball field", "basketball court",
    "bridge", "chimney", "dam", "expressway service area",
    "expressway toll station", "golf course", "ground track field",
    "harbor", "overpass", "ship", "stadium", "storage tank",
    "tennis court", "train station", "vehicle", "wind mill"
]


def denormalize_box(box: torch.Tensor, img_width: int, img_height: int) -> List[int]:
    """
    将归一化的框坐标转换为像素坐标
    
    Args:
        box: [xmin, ymin, xmax, ymax] 归一化坐标 (0-1)
        img_width: 图像宽度
        img_height: 图像高度
    
    Returns:
        [xmin, ymin, xmax, ymax] 像素坐标
    """
    xmin, ymin, xmax, ymax = box.tolist()
    return [
        int(xmin * img_width),
        int(ymin * img_height),
        int(xmax * img_width),
        int(ymax * img_height)
    ]


def draw_boxes_on_image(image: Image.Image, boxes: torch.Tensor, 
                       labels: torch.Tensor, original_size: tuple) -> Image.Image:
    """
    在图像上绘制边界框和类别标签
    
    Args:
        image: PIL Image对象
        boxes: [N, 4] 归一化的框坐标 [xmin, ymin, xmax, ymax]
        labels: [N] 类别索引
        original_size: (width, height) 原始图像尺寸
    
    Returns:
        绘制了框的图像
    """
    # 创建副本
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    # 获取图像尺寸
    img_width, img_height = original_size
    
    # 颜色列表（为不同类别分配不同颜色）
    colors = [
        '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF',
        '#00FFFF', '#FFA500', '#800080', '#FFC0CB', '#A52A2A',
        '#808080', '#000080', '#008000', '#800000', '#008080',
        '#FFD700', '#4B0082', '#FF1493', '#00CED1', '#32CD32'
    ]
    
    # 绘制每个框
    for i, (box, label_idx) in enumerate(zip(boxes, labels)):
        label_idx = label_idx.item()
        
        # 转换坐标
        xmin, ymin, xmax, ymax = denormalize_box(box, img_width, img_height)
        
        # 验证坐标有效性
        if xmax <= xmin or ymax <= ymin:
            print(f"⚠️  警告: 框 {i} 坐标无效: [{xmin}, {ymin}, {xmax}, {ymax}]")
            continue
        
        if xmin < 0 or ymin < 0 or xmax > img_width or ymax > img_height:
            print(f"⚠️  警告: 框 {i} 超出图像范围: [{xmin}, {ymin}, {xmax}, {ymax}], 图像尺寸: {img_width}x{img_height}")
        
        # 选择颜色
        color = colors[label_idx % len(colors)]
        
        # 绘制矩形框
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
        
        # 准备标签文本
        class_name = DIOR_CLASSES[label_idx] if label_idx < len(DIOR_CLASSES) else f"Unknown({label_idx})"
        label_text = f"{class_name} ({label_idx})"
        
        # 计算文本位置（框的上方）
        try:
            # 尝试使用默认字体
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            try:
                # 尝试其他字体
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 16)
            except:
                # 使用默认字体
                font = ImageFont.load_default()
        
        # 获取文本尺寸
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 绘制文本背景
        text_x = xmin
        text_y = max(0, ymin - text_height - 4)
        draw.rectangle(
            [text_x, text_y, text_x + text_width + 4, text_y + text_height + 4],
            fill=color,
            outline=color
        )
        
        # 绘制文本
        draw.text((text_x + 2, text_y + 2), label_text, fill='white', font=font)
    
    return img


def verify_label_alignment(config_path: str = None, num_samples: int = 5, 
                          split: str = 'trainval', output_dir: str = 'outputs/label_verification'):
    """
    验证标签和图像对齐
    
    Args:
        config_path: 配置文件路径
        num_samples: 要验证的图像数量
        split: 数据集划分 ('trainval' 或 'test')
        output_dir: 输出目录
    """
    print("=" * 80)
    print("🔍 开始验证标签和图像对齐")
    print("=" * 80)
    
    # 加载配置
    if config_path is None:
        config_path = Path(__file__).parent / 'configs' / 'improved_detector_config.yaml'
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    dataset_root = config.get('dataset_root')
    if dataset_root is None:
        raise ValueError("配置文件中未找到 dataset_root")
    
    print(f"📁 数据集根目录: {dataset_root}")
    print(f"📊 数据集划分: {split}")
    print(f"🖼️  验证样本数: {num_samples}")
    
    # 创建输出目录
    output_path = Path(__file__).parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"💾 输出目录: {output_path}")
    
    # 加载数据集（不使用transform，保持原始图像）
    from datasets.dior_detection import DIORDetectionDataset
    
    # 从配置读取train_only_seen设置
    train_only_seen = config.get('train_only_seen', True)
    
    print(f"📊 训练配置: train_only_seen = {train_only_seen}")
    if train_only_seen:
        print("   ⚠️  训练时只使用seen类别，unseen类别的框会被过滤掉")
    
    # 加载完整数据集（用于对比）
    dataset_full = DIORDetectionDataset(
        root=dataset_root,
        split=split,
        transform=None,  # 不使用transform，保持原始图像
        anno_type='horizontal',
        train_only_seen=False  # 显示所有类别
    )
    
    # 加载训练时使用的数据集（如果train_only_seen=True，会过滤unseen类别）
    dataset_train = DIORDetectionDataset(
        root=dataset_root,
        split=split,
        transform=None,  # 不使用transform，保持原始图像
        anno_type='horizontal',
        train_only_seen=train_only_seen  # 使用训练配置
    )
    
    # 使用训练数据集进行采样
    dataset = dataset_train
    
    print(f"✅ 数据集加载成功，共 {len(dataset)} 张图像")
    
    # 随机选择样本（优先选择有GT框的图像）
    # 先尝试找到有框的图像
    indices_with_boxes = []
    indices_without_boxes = []
    
    # 快速采样检查哪些图像有框
    check_indices = random.sample(range(len(dataset)), min(100, len(dataset)))
    for idx in check_indices:
        try:
            sample = dataset[idx]
            if len(sample['boxes']) > 0:
                indices_with_boxes.append(idx)
            else:
                indices_without_boxes.append(idx)
        except:
            pass
    
    # 优先选择有框的图像
    if len(indices_with_boxes) >= num_samples:
        sample_indices = random.sample(indices_with_boxes, num_samples)
        print(f"🎲 随机选择的样本索引（有GT框）: {sample_indices}")
    else:
        # 如果不够，补充一些无框的图像用于对比
        sample_indices = indices_with_boxes.copy()
        remaining = num_samples - len(sample_indices)
        if remaining > 0 and len(indices_without_boxes) > 0:
            additional = random.sample(indices_without_boxes, min(remaining, len(indices_without_boxes)))
            sample_indices.extend(additional)
        print(f"🎲 随机选择的样本索引: {sample_indices}")
        print(f"   其中 {len([i for i in sample_indices if i in indices_with_boxes])} 个有GT框")
    
    # 验证每个样本
    for idx, sample_idx in enumerate(sample_indices):
        print(f"\n{'='*80}")
        print(f"📸 样本 {idx+1}/{len(sample_indices)}: 索引 {sample_idx}")
        print(f"{'='*80}")
        
        try:
            # 加载训练时使用的样本
            sample_train = dataset[sample_idx]
            
            # 加载完整样本（用于对比）
            sample_full = dataset_full[sample_idx]
            
            image = sample_train['image']  # PIL Image
            boxes_train = sample_train['boxes']  # 训练时使用的框（可能被过滤）
            labels_train = sample_train['labels']  # 训练时使用的标签
            boxes_full = sample_full['boxes']  # 完整的框（所有类别）
            labels_full = sample_full['labels']  # 完整的标签
            image_id = sample_train['image_id']
            original_size = sample_train['original_size']  # (width, height)
            
            print(f"🆔 图像ID: {image_id}")
            print(f"📏 原始图像尺寸: {original_size[0]}x{original_size[1]}")
            print(f"📦 完整GT框数量: {len(boxes_full)}")
            print(f"📦 训练时使用的GT框数量: {len(boxes_train)}")
            
            if len(boxes_full) != len(boxes_train):
                filtered_count = len(boxes_full) - len(boxes_train)
                print(f"   ⚠️  有 {filtered_count} 个框被过滤（unseen类别）")
            
            # 使用训练时的框进行可视化（这是实际训练时使用的数据）
            boxes = boxes_train
            labels = labels_train
            
            # 打印训练时使用的框信息
            if len(boxes) > 0:
                print("\n📋 训练时使用的GT框详细信息:")
                for i, (box, label) in enumerate(zip(boxes, labels)):
                    label_idx = label.item()
                    class_name = DIOR_CLASSES[label_idx] if label_idx < len(DIOR_CLASSES) else f"Unknown({label_idx})"
                    xmin, ymin, xmax, ymax = box.tolist()
                    
                    # 转换为像素坐标
                    px_xmin = int(xmin * original_size[0])
                    px_ymin = int(ymin * original_size[1])
                    px_xmax = int(xmax * original_size[0])
                    px_ymax = int(ymax * original_size[1])
                    
                    print(f"  框 {i+1}:")
                    print(f"    类别: {class_name} (索引: {label_idx})")
                    print(f"    归一化坐标: [{xmin:.4f}, {ymin:.4f}, {xmax:.4f}, {ymax:.4f}]")
                    print(f"    像素坐标: [{px_xmin}, {px_ymin}, {px_xmax}, {px_ymax}]")
                    print(f"    框尺寸: {px_xmax - px_xmin} x {px_ymax - px_ymin}")
                    
                    # 验证坐标
                    if xmax <= xmin or ymax <= ymin:
                        print(f"    ⚠️  警告: 归一化坐标无效!")
                    if px_xmax <= px_xmin or px_ymax <= px_ymin:
                        print(f"    ⚠️  警告: 像素坐标无效!")
                    if px_xmin < 0 or px_ymin < 0 or px_xmax > original_size[0] or px_ymax > original_size[1]:
                        print(f"    ⚠️  警告: 框超出图像范围!")
            else:
                print("⚠️  该图像训练时没有GT框（空图像或被全部过滤）")
            
            # 如果有被过滤的框，显示它们
            if len(boxes_full) > len(boxes_train):
                print(f"\n📋 被过滤的框（unseen类别，训练时不使用）:")
                seen_indices = set()
                for box, label in zip(boxes_train, labels_train):
                    seen_indices.add(label.item())
                
                filtered_idx = 1
                for i, (box, label) in enumerate(zip(boxes_full, labels_full)):
                    label_idx = label.item()
                    if label_idx not in seen_indices:
                        class_name = DIOR_CLASSES[label_idx] if label_idx < len(DIOR_CLASSES) else f"Unknown({label_idx})"
                        xmin, ymin, xmax, ymax = box.tolist()
                        px_xmin = int(xmin * original_size[0])
                        px_ymin = int(ymin * original_size[1])
                        px_xmax = int(xmax * original_size[0])
                        px_ymax = int(ymax * original_size[1])
                        print(f"  被过滤框 {filtered_idx}: {class_name} (索引: {label_idx}) - [{px_xmin}, {px_ymin}, {px_xmax}, {px_ymax}]")
                        filtered_idx += 1
            
            # 绘制训练时使用的框
            img_with_boxes = draw_boxes_on_image(image, boxes, labels, original_size)
            
            # 保存图像（训练时使用的数据）
            output_file = output_path / f"sample_{idx+1}_{image_id}_train.jpg"
            img_with_boxes.save(output_file, quality=95)
            print(f"💾 已保存训练数据可视化: {output_file}")
            
            # 如果有被过滤的框，也保存完整版本用于对比
            if len(boxes_full) > len(boxes_train):
                img_with_all_boxes = draw_boxes_on_image(image, boxes_full, labels_full, original_size)
                output_file_full = output_path / f"sample_{idx+1}_{image_id}_full.jpg"
                img_with_all_boxes.save(output_file_full, quality=95)
                print(f"💾 已保存完整数据可视化（包含unseen类别）: {output_file_full}")
            
        except Exception as e:
            print(f"❌ 处理样本 {sample_idx} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("✅ 验证完成！")
    print(f"📁 所有可视化结果保存在: {output_path}")
    print("\n📝 检查要点:")
    print("  1. 框是否在正确的物体上？")
    print("  2. 类别标签是否正确？")
    print("  3. 框的坐标是否合理（不超出图像边界）？")
    print("  4. 框的格式是否正确（xyxy格式）？")
    print("  5. 图像和标注文件是否匹配？")
    print("\n⚠️  重要提示:")
    print(f"  - 训练配置: train_only_seen = {config.get('train_only_seen', True)}")
    if config.get('train_only_seen', True):
        print("  - 训练时只使用seen类别（10个类别）")
        print("  - unseen类别的框会被过滤，不会参与训练")
        print("  - 文件名带 '_train.jpg' 的是训练时使用的数据")
        print("  - 文件名带 '_full.jpg' 的是完整数据（包含unseen类别）")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='验证标签和图像对齐')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--num_samples', type=int, default=5, help='验证样本数量')
    parser.add_argument('--split', type=str, default='trainval', choices=['trainval', 'test'], help='数据集划分')
    parser.add_argument('--output_dir', type=str, default='outputs/label_verification', help='输出目录')
    
    args = parser.parse_args()
    
    verify_label_alignment(
        config_path=args.config,
        num_samples=args.num_samples,
        split=args.split,
        output_dir=args.output_dir
    )

