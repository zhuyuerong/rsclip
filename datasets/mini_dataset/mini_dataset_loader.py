#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mini Dataset 专用数据加载器

适配 mini_dataset 的特殊结构
"""

import torch
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Dict


# DIOR类别
DIOR_CLASSES = [
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
    'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(DIOR_CLASSES)}


class MiniDataset(Dataset):
    """Mini Dataset 加载器"""
    
    def __init__(
        self,
        root_dir: str = 'datasets/mini_dataset',
        split: str = 'train',
        transforms=None
    ):
        """
        参数:
            root_dir: mini_dataset 根目录
            split: 'train', 'val', 或 'test'
            transforms: 数据转换
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transforms = transforms
        
        # 图片和标注目录 (mini_dataset 的结构)
        self.images_dir = self.root_dir / 'images'
        self.annos_dir = self.root_dir / 'annotations'
        
        # 读取split文件
        split_file = self.root_dir / 'splits' / f'{split}.txt'
        if split_file.exists():
            with open(split_file, 'r') as f:
                self.image_ids = [line.strip() for line in f if line.strip()]
        else:
            # 如果没有split文件，使用所有图片
            self.image_ids = [img.stem for img in sorted(self.images_dir.glob('DIOR_*.jpg'))]
        
        print(f"✅ 加载Mini Dataset ({split}): {len(self.image_ids)}张图片")
    
    def __len__(self):
        return len(self.image_ids)
    
    def parse_xml(self, xml_path: Path):
        """解析XML标注"""
        
        boxes = []
        labels = []
        
        if not xml_path.exists():
            return boxes, labels
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size = root.find('size')
        img_width = int(size.find('width').text) if size is not None else 800
        img_height = int(size.find('height').text) if size is not None else 800
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            
            if name not in CLASS_TO_IDX:
                continue
            
            bndbox = obj.find('bndbox')
            if bndbox is None:
                continue
            
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            # 边界检查
            xmin = max(0, min(xmin, img_width))
            ymin = max(0, min(ymin, img_height))
            xmax = max(0, min(xmax, img_width))
            ymax = max(0, min(ymax, img_height))
            
            if xmax <= xmin or ymax <= ymin:
                continue
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(name)
        
        return boxes, labels
    
    def __getitem__(self, idx):
        """获取数据样本"""
        
        image_id = self.image_ids[idx]
        
        # 加载图片
        img_path = self.images_dir / f'{image_id}.jpg'
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        
        # 解析标注
        anno_path = self.annos_dir / f'{image_id}.xml'
        boxes_xyxy, labels_str = self.parse_xml(anno_path)
        
        # 转换为tensor
        if len(boxes_xyxy) > 0:
            boxes_xyxy = torch.as_tensor(boxes_xyxy, dtype=torch.float32)
            
            # xyxy -> cxcywh
            cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
            cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2
            w = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
            h = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
            
            # 归一化
            cx = cx / orig_w
            cy = cy / orig_h
            w = w / orig_w
            h = h / orig_h
            
            boxes = torch.stack([cx, cy, w, h], dim=1)
            
            # 转换标签
            labels = torch.tensor([CLASS_TO_IDX[name] for name in labels_str], dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        
        # 构建目标
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'orig_size': torch.tensor([orig_h, orig_w])
        }
        
        # 应用转换
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target


def collate_fn(batch):
    """批次整理函数"""
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)


def create_mini_dataloader(split='train', batch_size=4, transforms=None):
    """创建mini dataset数据加载器"""
    
    dataset = MiniDataset(
        root_dir='datasets/mini_dataset',
        split=split,
        transforms=transforms
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        collate_fn=collate_fn,
        num_workers=0
    )
    
    return dataloader


if __name__ == '__main__':
    print("=" * 70)
    print("测试 Mini Dataset 加载器")
    print("=" * 70)
    
    dataset = MiniDataset(split='train')
    
    print(f"\n数据集大小: {len(dataset)}")
    print(f"类别数: {len(DIOR_CLASSES)}")
    
    # 测试加载
    image, target = dataset[0]
    print(f"\n样本0:")
    print(f"  图像: {image.size if hasattr(image, 'size') else image.shape}")
    print(f"  目标数: {len(target['labels'])}")
    print(f"  边界框: {target['boxes'].shape}")
    print(f"  标签: {target['labels']}")
    
    # 测试数据加载器
    loader = create_mini_dataloader(split='train', batch_size=2)
    
    images, targets = next(iter(loader))
    print(f"\n批次:")
    print(f"  图像shape: {images.shape if isinstance(images, torch.Tensor) else 'PIL Images'}")
    print(f"  目标数: {[len(t['labels']) for t in targets]}")
    
    print("\n✅ Mini Dataset 加载器测试完成！")


