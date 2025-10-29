#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment1 标准评估脚本

功能：
1. 在数据集上运行检测
2. 计算mAP和其他指标
3. 生成评估报告
"""

import sys
import argparse
from pathlib import Path
import json
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

sys.path.append('..')

from inference.model_loader import ModelLoader
from utils.evaluation import evaluate_detections


# DIOR类别
DIOR_CLASSES = [
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
    'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(DIOR_CLASSES)}


def load_dataset(data_dir, split='val'):
    """加载数据集"""
    
    data_dir = Path(data_dir)
    
    # 检查是否是mini_dataset
    if (data_dir / 'splits').exists():
        # mini_dataset结构
        images_dir = data_dir / 'images'
        annos_dir = data_dir / 'annotations'
        split_file = data_dir / 'splits' / f'{split}.txt'
    else:
        # DIOR结构
        images_dir = data_dir / 'images' / 'trainval' if split != 'test' else data_dir / 'images' / 'test'
        annos_dir = data_dir / 'annotations' / 'horizontal'
        split_file = data_dir / 'splits' / f'{split}.txt'
    
    # 读取image_ids
    if split_file.exists():
        with open(split_file) as f:
            image_ids = [line.strip() for line in f if line.strip()]
    else:
        image_ids = [img.stem for img in sorted(images_dir.glob('*.jpg'))]
    
    samples = []
    for img_id in image_ids:
        # 尝试不同的文件名格式
        img_file = images_dir / f'{img_id}.jpg'
        if not img_file.exists():
            img_file = images_dir / f'DIOR_{img_id}.jpg'
        
        xml_file = annos_dir / f'{img_id}.xml'
        if not xml_file.exists():
            xml_file = annos_dir / f'DIOR_{img_id}.xml'
        
        if img_file.exists() and xml_file.exists():
            samples.append({
                'image_path': img_file,
                'xml_path': xml_file,
                'image_id': img_id
            })
    
    print(f"✅ 加载{len(samples)}个样本")
    return samples


def parse_xml(xml_path):
    """解析XML标注"""
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    boxes = []
    labels = []
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in CLASS_TO_IDX:
            continue
        
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(CLASS_TO_IDX[name])
    
    return np.array(boxes), np.array(labels)


def detect_image_simple(image_path, model_loader, score_threshold=0.15):
    """
    简化的图像检测（基于全局图像-文本相似度）
    
    参数:
        image_path: 图像路径
        model_loader: 模型加载器
        score_threshold: 分数阈值
    
    返回:
        检测结果
    """
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    orig_w, orig_h = image.size
    
    # 编码图像
    image_features = model_loader.encode_image(image)
    
    # 编码文本
    text_features = model_loader.encode_text_batch(DIOR_CLASSES)
    
    # 计算相似度
    similarities = (image_features @ text_features.T).cpu().numpy()[0]
    
    # 检测高相似度的类别
    detected = np.where(similarities > score_threshold)[0]
    
    # 生成检测框（简化版：使用全图）
    boxes = []
    scores = []
    labels = []
    
    for cls_id in detected:
        boxes.append([0, 0, orig_w, orig_h])  # 简化为全图
        scores.append(similarities[cls_id])
        labels.append(cls_id)
    
    return {
        'boxes': np.array(boxes) if len(boxes) > 0 else np.zeros((0, 4)),
        'scores': np.array(scores) if len(scores) > 0 else np.array([]),
        'labels': np.array(labels) if len(labels) > 0 else np.array([])
    }


def evaluate(args):
    """主评估函数"""
    
    print("=" * 70)
    print("Experiment1 评估")
    print("=" * 70)
    
    # 加载模型
    print(f"\n加载 RemoteCLIP {args.model}...")
    model_loader = ModelLoader(model_name=args.model, device=args.device)
    
    # 加载数据
    print(f"\n加载数据集: {args.data_dir}")
    samples = load_dataset(args.data_dir, args.split)
    
    # 运行检测
    print(f"\n运行检测...")
    all_predictions = []
    all_targets = []
    
    for i, sample in enumerate(samples):
        # 预测
        pred = detect_image_simple(sample['image_path'], model_loader, args.score_threshold)
        all_predictions.append(pred)
        
        # 目标
        boxes, labels = parse_xml(sample['xml_path'])
        all_targets.append({
            'boxes': boxes,
            'labels': labels
        })
        
        if (i + 1) % 10 == 0:
            print(f"  已处理 {i+1}/{len(samples)} 张图片")
    
    # 计算指标
    print(f"\n计算评估指标...")
    metrics = evaluate_detections(
        all_predictions,
        all_targets,
        num_classes=len(DIOR_CLASSES),
        iou_threshold=args.iou_threshold
    )
    
    # 打印结果
    print("\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)
    print(f"\nmAP@{args.iou_threshold}: {metrics['mAP']:.4f}")
    print(f"评估类别数: {metrics['num_classes_evaluated']}/{len(DIOR_CLASSES)}")
    
    if len(metrics['AP_per_class']) > 0:
        print("\n各类别AP:")
        for cls_id, ap in sorted(metrics['AP_per_class'].items(), key=lambda x: x[1], reverse=True):
            cls_name = DIOR_CLASSES[cls_id]
            print(f"  {cls_name:30s}: {ap:.4f}")
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'mAP': float(metrics['mAP']),
            'iou_threshold': args.iou_threshold,
            'score_threshold': args.score_threshold,
            'num_classes_evaluated': metrics['num_classes_evaluated'],
            'AP_per_class': {DIOR_CLASSES[k]: float(v) for k, v in metrics['AP_per_class'].items()},
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall'])
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ 结果保存到: {output_path}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment1 评估')
    
    parser.add_argument('--data_dir', type=str, default='datasets/mini_dataset',
                       help='数据集目录')
    parser.add_argument('--split', type=str, default='val',
                       help='数据分割 (train/val/test)')
    parser.add_argument('--model', type=str, default='RN50',
                       help='RemoteCLIP模型 (RN50/ViT-B-32/ViT-L-14)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='设备 (cuda/cpu)')
    parser.add_argument('--score_threshold', type=float, default=0.15,
                       help='分数阈值')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                       help='IoU阈值')
    parser.add_argument('--output', type=str, default='experiment1/evaluation_results.json',
                       help='输出文件路径')
    
    args = parser.parse_args()
    evaluate(args)


