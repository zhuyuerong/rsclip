#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage1: 数据加载模块
负责加载和预处理遥感图像数据
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader


class RemoteSensingDataLoader:
    """遥感数据加载器"""
    
    def __init__(self, data_dir: str, image_size: Tuple[int, int] = (512, 512)):
        """
        初始化数据加载器
        
        参数:
            data_dir: 数据目录路径
            image_size: 图像尺寸 (width, height)
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.image_paths = []
        self._load_image_paths()
    
    def _load_image_paths(self):
        """加载图像路径"""
        if not os.path.exists(self.data_dir):
            raise ValueError(f"数据目录不存在: {self.data_dir}")
        
        # 支持的图像格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    self.image_paths.append(os.path.join(root, file))
        
        print(f"✅ 找到 {len(self.image_paths)} 张图像")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        加载单张图像
        
        参数:
            image_path: 图像路径
        
        返回:
            图像数组 (H, W, C)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        # 使用OpenCV加载图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        # 转换BGR到RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整图像尺寸
        image = cv2.resize(image, self.image_size)
        
        return image
    
    def load_batch(self, batch_size: int = 1, shuffle: bool = False) -> List[np.ndarray]:
        """
        批量加载图像
        
        参数:
            batch_size: 批量大小
            shuffle: 是否打乱顺序
        
        返回:
            图像批次列表
        """
        indices = list(range(len(self.image_paths)))
        if shuffle:
            np.random.shuffle(indices)
        
        batch = []
        for i in range(min(batch_size, len(self.image_paths))):
            idx = indices[i]
            image_path = self.image_paths[idx]
            image = self.load_image(image_path)
            batch.append(image)
        
        return batch
    
    def get_image_info(self, image_path: str) -> Dict:
        """
        获取图像信息
        
        参数:
            image_path: 图像路径
        
        返回:
            图像信息字典
        """
        image = cv2.imread(image_path)
        if image is None:
            return {}
        
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        
        return {
            'path': image_path,
            'width': width,
            'height': height,
            'channels': channels,
            'size_mb': os.path.getsize(image_path) / (1024 * 1024)
        }


class RemoteSensingDataset(Dataset):
    """遥感图像数据集类"""
    
    def __init__(self, data_dir: str, image_size: Tuple[int, int] = (512, 512)):
        """
        初始化数据集
        
        参数:
            data_dir: 数据目录
            image_size: 图像尺寸
        """
        self.data_loader = RemoteSensingDataLoader(data_dir, image_size)
        self.image_paths = self.data_loader.image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = self.data_loader.load_image(image_path)
        
        return {
            'image': image,
            'path': image_path,
            'index': idx
        }


def create_data_loader(data_dir: str, batch_size: int = 1, 
                      image_size: Tuple[int, int] = (512, 512),
                      shuffle: bool = False) -> DataLoader:
    """
    创建数据加载器
    
    参数:
        data_dir: 数据目录
        batch_size: 批量大小
        image_size: 图像尺寸
        shuffle: 是否打乱顺序
    
    返回:
        PyTorch DataLoader
    """
    dataset = RemoteSensingDataset(data_dir, image_size)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # 避免多进程问题
        collate_fn=lambda batch: batch
    )


def main():
    """测试数据加载器"""
    print("=" * 70)
    print("测试遥感数据加载器")
    print("=" * 70)
    
    # 测试数据目录
    test_dir = "assets"
    if not os.path.exists(test_dir):
        print(f"❌ 测试目录不存在: {test_dir}")
        return
    
    # 创建数据加载器
    data_loader = RemoteSensingDataLoader(test_dir)
    
    # 加载单张图像
    if data_loader.image_paths:
        image_path = data_loader.image_paths[0]
        print(f"\n📷 加载图像: {image_path}")
        
        image = data_loader.load_image(image_path)
        print(f"✅ 图像尺寸: {image.shape}")
        
        # 获取图像信息
        info = data_loader.get_image_info(image_path)
        print(f"📊 图像信息: {info}")
        
        # 批量加载
        print(f"\n📦 批量加载测试:")
        batch = data_loader.load_batch(batch_size=2)
        print(f"✅ 批量大小: {len(batch)}")
        for i, img in enumerate(batch):
            print(f"  图像 {i+1}: {img.shape}")
    
    print("\n✅ 数据加载器测试完成!")


if __name__ == "__main__":
    main()
