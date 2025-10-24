#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 mini_dataset 创建 train/val/test 分割文件
"""

from pathlib import Path
import json
import random

# 设置随机种子
random.seed(42)

# 目录
mini_dataset_dir = Path(__file__).parent
images_dir = mini_dataset_dir / 'images'
splits_dir = mini_dataset_dir / 'splits'
splits_dir.mkdir(exist_ok=True)

# 获取所有图片ID
image_files = sorted(images_dir.glob('DIOR_*.jpg'))
image_ids = [img.stem for img in image_files]

print(f"找到 {len(image_ids)} 个图片")

# 分割比例: 70% train, 15% val, 15% test
total = len(image_ids)
train_size = int(total * 0.7)
val_size = int(total * 0.15)
test_size = total - train_size - val_size

# 随机打乱
random.shuffle(image_ids)

# 分割
train_ids = image_ids[:train_size]
val_ids = image_ids[train_size:train_size+val_size]
test_ids = image_ids[train_size+val_size:]

print(f"\n分割结果:")
print(f"  训练集: {len(train_ids)} ({len(train_ids)/total*100:.1f}%)")
print(f"  验证集: {len(val_ids)} ({len(val_ids)/total*100:.1f}%)")
print(f"  测试集: {len(test_ids)} ({len(test_ids)/total*100:.1f}%)")

# 保存
for split_name, split_ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
    split_file = splits_dir / f'{split_name}.txt'
    with open(split_file, 'w') as f:
        for img_id in split_ids:
            f.write(f"{img_id}\n")
    print(f"✅ 保存 {split_file.name}")

print("\n✅ 分割文件创建完成！")

