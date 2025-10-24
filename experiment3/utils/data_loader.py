#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIOR数据集加载器

功能：
1. 加载DIOR数据集
2. 解析VOC XML标注
3. 支持训练/验证/测试划分
"""

import torch
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple


# DIOR数据集类别
DIOR_CLASSES = [
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
    'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'
]

# 类别到索引的映射
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(DIOR_CLASSES)}


class DiorDataset(Dataset):
    """
    DIOR数据集加载器
    """
    
    def __init__(
        self,
        root_dir: str = '/home/ubuntu22/Projects/RemoteCLIP-main/datasets/DIOR',
        split: str = 'train',
        anno_type: str = 'horizontal',
        transforms=None
    ):
        """
        参数:
            root_dir: DIOR数据集根目录
            split: 'train', 'val', 或 'test'
            anno_type: 'horizontal' 或 'oriented'
            transforms: 数据转换
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.anno_type = anno_type
        self.transforms = transforms
        
        # 图像和标注目录
        if split == 'train' or split == 'val':
            self.images_dir = self.root_dir / 'images' / 'trainval'
        else:
            self.images_dir = self.root_dir / 'images' / 'test'
        
        self.annos_dir = self.root_dir / 'annotations' / anno_type
        
        # 读取split文件
        split_file = self.root_dir / 'splits' / f'{split}.txt'
        if split_file.exists():
            with open(split_file, 'r') as f:
                self.image_ids = [line.strip() for line in f if line.strip()]
        else:
            # 从images目录读取
            self.image_ids = [img.stem for img in sorted(self.images_dir.glob('*.jpg'))]
        
        print(f"✅ 加载DIOR数据集 ({split}): {len(self.image_ids)}张图片")
    
    def __len__(self):
        return len(self.image_ids)
    
    def parse_xml(self, xml_path: Path) -> Tuple[List, List]:
        """
        解析VOC XML标注
        
        返回:
            boxes: List of [xmin, ymin, xmax, ymax]
            labels: List of class names
        """
        boxes = []
        labels = []
        
        if not xml_path.exists():
            return boxes, labels
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 获取图像尺寸
        size = root.find('size')
        img_width = int(size.find('width').text) if size is not None else 800
        img_height = int(size.find('height').text) if size is not None else 800
        
        # 解析目标
        for obj in root.findall('object'):
            name = obj.find('name').text
            
            # 过滤未知类别
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
            
            # 过滤无效框
            if xmax <= xmin or ymax <= ymin:
                continue
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(name)
        
        return boxes, labels
    
    def __getitem__(self, idx):
        """
        获取数据样本
        
        返回:
            image: PIL Image
            target: {
                'boxes': (N, 4) [cx, cy, w, h] 归一化,
                'labels': (N,) class indices,
                'image_id': str,
                'orig_size': (H, W)
            }
        """
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
    """
    批次整理函数
    
    处理不同数量的目标
    """
    images, targets = zip(*batch)
    
    # 堆叠图像
    images = torch.stack(images, dim=0)
    
    # 目标保持为列表
    return images, list(targets)


def create_data_loader(
    root_dir: str,
    split: str = 'train',
    batch_size: int = 8,
    num_workers: int = 4,
    transforms=None
) -> DataLoader:
    """
    创建数据加载器
    
    参数:
        root_dir: 数据集根目录
        split: 'train', 'val', 或 'test'
        batch_size: 批次大小
        num_workers: 工作进程数
        transforms: 数据转换
    
    返回:
        data_loader: DataLoader
    """
    dataset = DiorDataset(
        root_dir=root_dir,
        split=split,
        anno_type='horizontal',
        transforms=transforms
    )
    
    shuffle = (split == 'train')
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False
    )
    
    return data_loader


if __name__ == "__main__":
    import sys
    sys.path.append('/home/ubuntu22/Projects/RemoteCLIP-main/experiment3')
    from utils.transforms import get_transforms
    
    print("=" * 70)
    print("测试DIOR数据加载器")
    print("=" * 70)
    
    # 创建数据集
    transforms = get_transforms(mode='train', image_size=(800, 800))
    dataset = DiorDataset(
        root_dir='/home/ubuntu22/Projects/RemoteCLIP-main/datasets/DIOR',
        split='train',
        transforms=transforms
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    print(f"类别数量: {len(DIOR_CLASSES)}")
    print(f"类别: {DIOR_CLASSES[:5]}...")
    
    # 测试加载
    image, target = dataset[0]
    
    print(f"\n样本0:")
    print(f"  图像: {image.shape}")
    print(f"  目标数量: {len(target['labels'])}")
    print(f"  边界框: {target['boxes'].shape}")
    print(f"  标签: {target['labels']}")
    print(f"  图像ID: {target['image_id']}")
    
    # 创建数据加载器
    data_loader = create_data_loader(
        root_dir='/home/ubuntu22/Projects/RemoteCLIP-main/datasets/DIOR',
        split='train',
        batch_size=2,
        num_workers=0,
        transforms=transforms
    )
    
    print(f"\n数据加载器:")
    print(f"  批次数: {len(data_loader)}")
    
    # 测试一个批次
    images, targets = next(iter(data_loader))
    print(f"\n批次:")
    print(f"  图像: {images.shape}")
    print(f"  目标数量: {[len(t['labels']) for t in targets]}")
    
    print("\n✅ DIOR数据加载器测试完成！")

