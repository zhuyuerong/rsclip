# -*- coding: utf-8 -*-
"""
检查特定样本的标签
"""

import torch
import sys
from pathlib import Path
import yaml
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from datasets.dior_detection import DIORDetectionDataset

# DIOR类别列表
DIOR_CLASSES = [
    "airplane", "airport", "baseball field", "basketball court",
    "bridge", "chimney", "dam", "expressway service area",
    "expressway toll station", "golf course", "ground track field",
    "harbor", "overpass", "ship", "stadium", "storage tank",
    "tennis court", "train station", "vehicle", "wind mill"
]

# Seen类别索引
SEEN_CLASS_INDICES = {0, 1, 4, 9, 11, 13, 14, 15, 18, 19}

def check_sample(image_id, config_path=None):
    """检查特定样本"""
    if config_path is None:
        config_path = Path(__file__).parent / 'configs' / 'improved_detector_config.yaml'
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    dataset_root = config.get('dataset_root')
    train_only_seen = config.get('train_only_seen', True)
    
    print("=" * 80)
    print(f"检查样本: {image_id}")
    print("=" * 80)
    
    # 加载完整数据集
    dataset_full = DIORDetectionDataset(
        root=dataset_root,
        split='trainval',
        transform=None,
        anno_type='horizontal',
        train_only_seen=False
    )
    
    # 加载训练数据集
    dataset_train = DIORDetectionDataset(
        root=dataset_root,
        split='trainval',
        transform=None,
        anno_type='horizontal',
        train_only_seen=train_only_seen
    )
    
    # 找到样本索引
    try:
        idx = dataset_full.image_ids.index(image_id)
    except ValueError:
        print(f"❌ 未找到图像ID: {image_id}")
        return
    
    # 加载样本
    sample_full = dataset_full[idx]
    sample_train = dataset_train[idx]
    
    print(f"\n📊 数据统计:")
    print(f"  完整GT框数量: {len(sample_full['boxes'])}")
    print(f"  训练时使用的GT框数量: {len(sample_train['boxes'])}")
    
    # 显示完整标注
    if len(sample_full['boxes']) > 0:
        print(f"\n📋 完整标注（所有类别）:")
        for i, (box, label) in enumerate(zip(sample_full['boxes'], sample_full['labels'])):
            label_idx = label.item()
            class_name = DIOR_CLASSES[label_idx]
            is_seen = label_idx in SEEN_CLASS_INDICES
            status = "✅ Seen (训练时使用)" if is_seen else "❌ Unseen (训练时过滤)"
            xmin, ymin, xmax, ymax = box.tolist()
            print(f"  框 {i+1}: {class_name} (索引: {label_idx}) - [{xmin:.4f}, {ymin:.4f}, {xmax:.4f}, {ymax:.4f}] {status}")
    else:
        print(f"\n⚠️  该图像没有GT框（空图像）")
    
    # 显示训练时使用的标注
    if len(sample_train['boxes']) > 0:
        print(f"\n📋 训练时使用的标注（只包含seen类别）:")
        for i, (box, label) in enumerate(zip(sample_train['boxes'], sample_train['labels'])):
            label_idx = label.item()
            class_name = DIOR_CLASSES[label_idx]
            xmin, ymin, xmax, ymax = box.tolist()
            print(f"  框 {i+1}: {class_name} (索引: {label_idx}) - [{xmin:.4f}, {ymin:.4f}, {xmax:.4f}, {ymax:.4f}]")
    else:
        print(f"\n⚠️  训练时该图像没有GT框（空图像或被全部过滤）")
    
    # 检查是否有unseen类别被过滤
    if len(sample_full['boxes']) > len(sample_train['boxes']):
        print(f"\n📋 被过滤的框（unseen类别，训练时不使用）:")
        seen_labels = set([l.item() for l in sample_train['labels']])
        filtered_count = 0
        for i, (box, label) in enumerate(zip(sample_full['boxes'], sample_full['labels'])):
            label_idx = label.item()
            if label_idx not in seen_labels:
                class_name = DIOR_CLASSES[label_idx]
                xmin, ymin, xmax, ymax = box.tolist()
                print(f"  被过滤框 {filtered_count+1}: {class_name} (索引: {label_idx}) - [{xmin:.4f}, {ymin:.4f}, {xmax:.4f}, {ymax:.4f}]")
                filtered_count += 1
    
    print("\n" + "=" * 80)
    print("📝 标签括号中数字的含义:")
    print("=" * 80)
    print("括号中的数字是类别索引（Class Index），范围是 0-19")
    print("\n类别索引对应关系:")
    for idx, cls_name in enumerate(DIOR_CLASSES):
        is_seen = idx in SEEN_CLASS_INDICES
        status = "✅ Seen" if is_seen else "❌ Unseen"
        print(f"  {idx:2d}: {cls_name:30s} {status}")
    print("\n说明:")
    print("  - 类别索引是固定的，不会改变")
    print("  - Seen类别（✅）: 训练时使用，共10个")
    print("  - Unseen类别（❌）: 训练时过滤，共10个")
    print("  - 如果 train_only_seen=True，只有seen类别的框会参与训练")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='检查特定样本的标签')
    parser.add_argument('--image_id', type=str, required=True, help='图像ID（不含扩展名）')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    
    args = parser.parse_args()
    
    check_sample(args.image_id, args.config)


