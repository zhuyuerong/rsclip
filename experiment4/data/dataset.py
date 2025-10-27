# -*- coding: utf-8 -*-
"""
数据集加载器
支持mini_dataset的seen/unseen分割
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
from torchvision import transforms


class MiniDataset(Dataset):
    """
    Mini数据集（100个样本）
    
    支持seen/unseen类别分割
    """
    
    def __init__(self, root_dir, split='train', seen_ratio=0.75, config=None, transform=None):
        """
        Args:
            root_dir: 数据集根目录
            split: 'train', 'val', 'test'
            seen_ratio: seen类的比例（用于seen/unseen分割）
            config: 配置对象
            transform: 图像变换
        """
        self.root_dir = root_dir
        self.split = split
        self.config = config
        
        # 设置transform
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
        
        # 加载数据
        self.samples = self._load_samples()
        
        # 获取所有类别
        all_classes = sorted(list(set([s['class_name'] for s in self.samples])))
        
        # 分割seen/unseen类别
        num_seen = int(len(all_classes) * seen_ratio)
        self.seen_classes = all_classes[:num_seen]
        self.unseen_classes = all_classes[num_seen:]
        
        # 类别到索引的映射
        self.class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        print(f"数据集 [{split}] 加载完成:")
        print(f"  总样本数: {len(self.samples)}")
        print(f"  总类别数: {len(all_classes)}")
        print(f"  Seen类: {len(self.seen_classes)}")
        print(f"  Unseen类: {len(self.unseen_classes)}")
    
    def _load_samples(self):
        """加载样本列表"""
        samples = []
        
        # 读取split文件
        split_file = os.path.join(self.root_dir, 'splits', f'{self.split}.txt')
        
        if not os.path.exists(split_file):
            print(f"警告: split文件不存在 {split_file}，加载所有样本")
            # 加载所有图像
            images_dir = os.path.join(self.root_dir, 'images')
            image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
        else:
            with open(split_file, 'r') as f:
                image_files = [line.strip() for line in f if line.strip()]
        
        # 加载每个样本
        for img_file in image_files:
            # 确保有扩展名（split文件可能只有ID）
            if not img_file.endswith(('.jpg', '.png')):
                # 尝试添加.jpg扩展名
                img_file_with_ext = img_file + '.jpg'
                img_path = os.path.join(self.root_dir, 'images', img_file_with_ext)
                # 如果.jpg不存在，尝试.png
                if not os.path.exists(img_path):
                    img_file_with_ext = img_file + '.png'
                    img_path = os.path.join(self.root_dir, 'images', img_file_with_ext)
            else:
                img_file_with_ext = img_file
                img_path = os.path.join(self.root_dir, 'images', img_file_with_ext)
            
            # 标注路径
            ann_file = img_file_with_ext.replace('.jpg', '.xml').replace('.png', '.xml')
            ann_path = os.path.join(self.root_dir, 'annotations', ann_file)
            
            if not os.path.exists(img_path):
                continue
            
            # 解析标注
            if os.path.exists(ann_path):
                annotation = self._parse_annotation(ann_path)
            else:
                # 从文件名推断类别
                class_name = img_file.split('_')[0]
                annotation = {
                    'class_name': class_name,
                    'bbox': None
                }
            
            samples.append({
                'image_path': img_path,
                'class_name': annotation['class_name'],
                'bbox': annotation['bbox']
            })
        
        return samples
    
    def _parse_annotation(self, ann_path):
        """解析XML标注"""
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        # 获取类别名
        class_name = root.find('.//name').text if root.find('.//name') is not None else 'unknown'
        
        # 获取bbox（如果有）
        bbox = None
        bbox_elem = root.find('.//bndbox')
        if bbox_elem is not None:
            xmin = int(bbox_elem.find('xmin').text)
            ymin = int(bbox_elem.find('ymin').text)
            xmax = int(bbox_elem.find('xmax').text)
            ymax = int(bbox_elem.find('ymax').text)
            
            # 获取图像尺寸
            size_elem = root.find('.//size')
            if size_elem is not None:
                width = int(size_elem.find('width').text)
                height = int(size_elem.find('height').text)
                
                # 归一化bbox
                bbox = [
                    xmin / width,
                    ymin / height,
                    xmax / width,
                    ymax / height
                ]
        
        return {
            'class_name': class_name,
            'bbox': bbox
        }
    
    def _get_default_transform(self):
        """获取默认变换"""
        if self.config is not None:
            normalize = transforms.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            )
            img_size = self.config.image_size
        else:
            normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
            img_size = 224
        
        if self.split == 'train':
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                normalize
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                normalize
            ])
        
        return transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像
        image = Image.open(sample['image_path']).convert('RGB')
        
        # 变换
        if self.transform:
            image = self.transform(image)
        
        # 获取标签
        class_name = sample['class_name']
        label = self.class_to_idx[class_name]
        
        # 获取bbox
        bbox = sample['bbox']
        if bbox is not None:
            bbox = torch.tensor(bbox, dtype=torch.float32)
        else:
            bbox = torch.zeros(4, dtype=torch.float32)
        
        return {
            'image': image,
            'label': label,
            'class_name': class_name,
            'bbox': bbox,
            'has_bbox': sample['bbox'] is not None
        }
    
    def get_seen_samples(self):
        """获取seen类的样本"""
        seen_indices = [i for i, s in enumerate(self.samples) 
                       if s['class_name'] in self.seen_classes]
        return torch.utils.data.Subset(self, seen_indices)
    
    def get_unseen_samples(self):
        """获取unseen类的样本"""
        unseen_indices = [i for i, s in enumerate(self.samples) 
                         if s['class_name'] in self.unseen_classes]
        return torch.utils.data.Subset(self, unseen_indices)


class SeenDataset(Dataset):
    """Seen类数据集（用于训练）"""
    
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.seen_classes = base_dataset.seen_classes
        
        # 筛选seen样本
        self.indices = [i for i, s in enumerate(base_dataset.samples) 
                       if s['class_name'] in self.seen_classes]
        
        print(f"Seen数据集: {len(self.indices)} 个样本，{len(self.seen_classes)} 个类别")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.base_dataset[real_idx]


class UnseenDataset(Dataset):
    """Unseen类数据集（用于zero-shot评估）"""
    
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.unseen_classes = base_dataset.unseen_classes
        
        # 筛选unseen样本
        self.indices = [i for i, s in enumerate(base_dataset.samples) 
                       if s['class_name'] in self.unseen_classes]
        
        print(f"Unseen数据集: {len(self.indices)} 个样本，{len(self.unseen_classes)} 个类别")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.base_dataset[real_idx]


def get_dataloaders(config):
    """
    获取数据加载器
    
    Returns:
        seen_train_loader, seen_val_loader, unseen_test_loader
    """
    # 创建数据集
    train_dataset = MiniDataset(
        root_dir=config.dataset_root,
        split='train',
        seen_ratio=0.75,
        config=config
    )
    
    val_dataset = MiniDataset(
        root_dir=config.dataset_root,
        split='val',
        seen_ratio=0.75,
        config=config
    )
    
    test_dataset = MiniDataset(
        root_dir=config.dataset_root,
        split='test',
        seen_ratio=0.75,
        config=config
    )
    
    # 创建seen/unseen子集
    seen_train = SeenDataset(train_dataset)
    seen_val = SeenDataset(val_dataset)
    unseen_test = UnseenDataset(test_dataset)
    
    # 创建数据加载器
    seen_train_loader = torch.utils.data.DataLoader(
        seen_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    seen_val_loader = torch.utils.data.DataLoader(
        seen_val,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    unseen_test_loader = torch.utils.data.DataLoader(
        unseen_test,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    return seen_train_loader, seen_val_loader, unseen_test_loader, train_dataset


def test_dataset():
    """测试数据集"""
    print("测试数据集...")
    
    # 模拟配置
    class DummyConfig:
        dataset_root = "datasets/mini_dataset"
        batch_size = 4
        eval_batch_size = 8
        num_workers = 0
        pin_memory = False
        image_size = 224
        normalize_mean = [0.48145466, 0.4578275, 0.40821073]
        normalize_std = [0.26862954, 0.26130258, 0.27577711]
    
    config = DummyConfig()
    
    # 创建数据集
    dataset = MiniDataset(
        root_dir=config.dataset_root,
        split='train',
        config=config
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"Seen类: {len(dataset.seen_classes)}")
    print(f"Unseen类: {len(dataset.unseen_classes)}")
    
    # 测试加载
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"样本键: {sample.keys()}")
        print(f"图像形状: {sample['image'].shape}")
        print(f"类别: {sample['class_name']}")
        print(f"标签: {sample['label']}")


if __name__ == "__main__":
    test_dataset()

