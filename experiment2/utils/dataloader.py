#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment2 数据加载器

支持 DIOR 和 mini_dataset
"""

import torch
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import torchvision.transforms as T


# DIOR类别
DIOR_CLASSES = [
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
    'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(DIOR_CLASSES)}


class DIORDataset(Dataset):
    """DIOR 数据集加载器"""
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        image_size: tuple = (800, 800),
        augment: bool = False
    ):
        """
        参数:
            root_dir: 数据集根目录
            split: 'train', 'val', 或 'test'
            image_size: 目标图像大小
            augment: 是否使用数据增强
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        
        # 确定目录结构
        if (self.root_dir / 'splits').exists() and not (self.root_dir / 'images' / 'trainval').exists():
            # mini_dataset 结构
            self.images_dir = self.root_dir / 'images'
            self.annos_dir = self.root_dir / 'annotations'
            self.is_mini = True
        else:
            # DIOR 结构
            if split == 'test':
                self.images_dir = self.root_dir / 'images' / 'test'
            else:
                self.images_dir = self.root_dir / 'images' / 'trainval'
            self.annos_dir = self.root_dir / 'annotations' / 'horizontal'
            self.is_mini = False
        
        # 读取split文件
        split_file = self.root_dir / 'splits' / f'{split}.txt'
        if split_file.exists():
            with open(split_file) as f:
                self.image_ids = [line.strip() for line in f if line.strip()]
        else:
            self.image_ids = [img.stem for img in sorted(self.images_dir.glob('*.jpg'))]
        
        # 转换
        self.transforms = self._get_transforms()
        
        print(f"✅ 加载DIOR数据集 ({split}): {len(self.image_ids)}张图片")
    
    def _get_transforms(self):
        """获取数据转换"""
        transforms = []
        
        if self.augment:
            transforms.extend([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.RandomHorizontalFlip(p=0.5)
            ])
        
        transforms.extend([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return T.Compose(transforms)
    
    def parse_xml(self, xml_path):
        """解析XML标注"""
        boxes = []
        labels = []
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size = root.find('size')
        img_w = int(size.find('width').text) if size is not None else self.image_size[1]
        img_h = int(size.find('height').text) if size is not None else self.image_size[0]
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in CLASS_TO_IDX:
                continue
            
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            # 归一化
            xmin /= img_w
            ymin /= img_h
            xmax /= img_w
            ymax /= img_h
            
            # 转换为cxcywh
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            w = xmax - xmin
            h = ymax - ymin
            
            boxes.append([cx, cy, w, h])
            labels.append(CLASS_TO_IDX[name])
        
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """获取样本"""
        image_id = self.image_ids[idx]
        
        # 加载图片
        if self.is_mini:
            img_path = self.images_dir / f'{image_id}.jpg'
            xml_path = self.annos_dir / f'{image_id}.xml'
        else:
            img_path = self.images_dir / f'{image_id}.jpg'
            xml_path = self.annos_dir / f'{image_id}.xml'
        
        image = Image.open(img_path).convert('RGB')
        orig_size = image.size  # (W, H)
        
        # 解析标注
        boxes, labels = self.parse_xml(xml_path)
        
        # 转换图像
        image = self.transforms(image)
        
        target = {
            'boxes': boxes,  # (N, 4) [cx, cy, w, h] 归一化
            'labels': labels,  # (N,)
            'orig_size': torch.tensor(orig_size),  # (W, H)
            'image_id': image_id
        }
        
        return image, target


def collate_fn(batch):
    """批次整理函数"""
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)


def create_dataloader(
    root_dir: str,
    split: str = 'train',
    batch_size: int = 16,
    image_size: tuple = (800, 800),
    augment: bool = False,
    num_workers: int = 4
) -> DataLoader:
    """创建数据加载器"""
    
    dataset = DIORDataset(
        root_dir=root_dir,
        split=split,
        image_size=image_size,
        augment=augment
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    print("=" * 70)
    print("测试数据加载器")
    print("=" * 70)
    
    # 测试mini_dataset
    dataset = DIORDataset(
        root_dir='../datasets/mini_dataset',
        split='train',
        image_size=(800, 800)
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    
    # 测试加载
    image, target = dataset[0]
    print(f"\n样本0:")
    print(f"  图像: {image.shape}")
    print(f"  边界框: {target['boxes'].shape}")
    print(f"  标签: {target['labels']}")
    
    # 测试dataloader
    loader = create_dataloader(
        root_dir='../datasets/mini_dataset',
        split='train',
        batch_size=2,
        num_workers=0
    )
    
    images, targets = next(iter(loader))
    print(f"\n批次:")
    print(f"  图像: {images.shape}")
    print(f"  目标数: {[len(t['labels']) for t in targets]}")
    
    print("\n✅ 数据加载器测试完成！")

