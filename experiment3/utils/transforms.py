#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增强和转换

功能：
1. 图像预处理
2. 边界框转换
3. 数据增强
"""

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
import random
from typing import Tuple
from PIL import Image


class Compose:
    """组合多个转换"""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize:
    """调整图像大小"""
    
    def __init__(self, size: Tuple[int, int]):
        self.size = size
    
    def __call__(self, image, target):
        # 原始尺寸
        orig_w, orig_h = image.size
        
        # 调整图像
        image = F.resize(image, self.size)
        
        # 调整边界框
        if target is not None and 'boxes' in target:
            boxes = target['boxes']
            
            # 转换为相对坐标
            scale_x = self.size[1] / orig_w
            scale_y = self.size[0] / orig_h
            
            if len(boxes) > 0:
                boxes[:, 0] *= scale_x  # cx
                boxes[:, 1] *= scale_y  # cy
                boxes[:, 2] *= scale_x  # w
                boxes[:, 3] *= scale_y  # h
            
            target['boxes'] = boxes
        
        return image, target


class ToTensor:
    """转换为Tensor"""
    
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize:
    """归一化"""
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, image, target):
        image = F.normalize(image, self.mean, self.std)
        return image, target


class RandomHorizontalFlip:
    """随机水平翻转"""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, target):
        if random.random() < self.p:
            image = F.hflip(image)
            
            if target is not None and 'boxes' in target:
                boxes = target['boxes']
                if len(boxes) > 0:
                    # 翻转边界框
                    boxes[:, 0] = 1.0 - boxes[:, 0]  # cx
                target['boxes'] = boxes
        
        return image, target


class RandomCrop:
    """随机裁剪"""
    
    def __init__(self, size: Tuple[int, int]):
        self.size = size
    
    def __call__(self, image, target):
        # TODO: 实现随机裁剪和边界框调整
        return image, target


class ColorJitter:
    """颜色抖动"""
    
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.color_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
    
    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target


def get_transforms(mode='train', image_size=(800, 800)):
    """
    获取转换
    
    参数:
        mode: 'train' 或 'val'
        image_size: 目标图像大小
    
    返回:
        transforms: 转换组合
    """
    normalize = Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if mode == 'train':
        return Compose([
            Resize(image_size),
            ColorJitter(),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize
        ])
    else:
        return Compose([
            Resize(image_size),
            ToTensor(),
            normalize
        ])


if __name__ == "__main__":
    from PIL import Image
    import torch
    
    print("=" * 70)
    print("测试数据转换")
    print("=" * 70)
    
    # 创建测试图像
    image = Image.new('RGB', (1024, 768), color='red')
    target = {
        'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.3, 0.4, 0.1, 0.2]]),
        'labels': torch.tensor([1, 2])
    }
    
    # 训练转换
    train_transforms = get_transforms(mode='train', image_size=(800, 800))
    image_t, target_t = train_transforms(image, target)
    
    print(f"\n训练转换:")
    print(f"  原始图像: {image.size}")
    print(f"  转换后: {image_t.shape}")
    print(f"  原始边界框: {target['boxes']}")
    print(f"  转换后: {target_t['boxes']}")
    
    # 验证转换
    val_transforms = get_transforms(mode='val', image_size=(800, 800))
    image_v, target_v = val_transforms(image, target)
    
    print(f"\n验证转换:")
    print(f"  图像: {image_v.shape}")
    print(f"  边界框: {target_v['boxes']}")
    
    print("\n✅ 数据转换测试完成！")

