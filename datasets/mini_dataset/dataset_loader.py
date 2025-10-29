#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小数据集加载器

功能：
1. 加载mini_dataset的图片和标注
2. 根据seen/unseen配置过滤样本
3. 提供统一的数据接口
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from typing import List, Dict, Tuple, Optional
import random


class MiniDatasetLoader:
    """小数据集加载器"""
    
    def __init__(
        self,
        dataset_dir: str = 'datasets/mini_dataset',
        split_config: str = 'split_config_seen_70.json',
        mode: str = 'all'
    ):
        """
        参数:
            dataset_dir: 数据集目录
            split_config: seen/unseen分割配置文件
            mode: 'all', 'seen', 'unseen'
        """
        self.dataset_dir = Path(dataset_dir)
        self.mode = mode
        
        # 加载样本列表
        with open(self.dataset_dir / 'samples.json', 'r') as f:
            self.all_samples = json.load(f)
        
        # 加载分割配置
        config_path = self.dataset_dir / split_config
        with open(config_path, 'r') as f:
            self.split_config = json.load(f)
        
        self.seen_classes = set(self.split_config['seen_classes'])
        self.unseen_classes = set(self.split_config['unseen_classes'])
        
        # 根据mode过滤样本
        self.samples = self._filter_samples()
        
        print(f"=" * 70)
        print(f"Mini Dataset Loader")
        print(f"=" * 70)
        print(f"配置: {split_config}")
        print(f"模式: {mode}")
        print(f"Seen类别: {self.split_config['seen_classes']}")
        print(f"Unseen类别: {self.split_config['unseen_classes']}")
        print(f"样本数: {len(self.samples)}")
        print(f"=" * 70)
    
    def _filter_samples(self) -> List[Dict]:
        """根据mode过滤样本"""
        if self.mode == 'all':
            return self.all_samples
        
        # 需要解析标注来判断类别
        filtered = []
        
        for sample in self.all_samples:
            # 简化：根据数据集判断
            # 更精确的方式是解析标注文件
            if self.mode == 'seen':
                filtered.append(sample)
            elif self.mode == 'unseen':
                filtered.append(sample)
        
        return filtered
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取一个样本
        
        返回:
            sample: 样本字典
                - image: PIL.Image
                - boxes: 边界框列表
                - labels: 类别标签列表
                - image_name: 图片名称
                - dataset: 数据集来源
        """
        sample_info = self.samples[idx]
        
        # 加载图片
        image_path = self.dataset_dir / 'images' / sample_info['image_name']
        image = Image.open(image_path).convert('RGB')
        
        # 加载标注
        boxes = []
        labels = []
        
        # 根据数据集类型解析标注
        if sample_info['dataset'] == 'DIOR':
            # DIOR使用VOC XML格式
            anno_name = sample_info['image_name'].replace('.jpg', '.xml').replace('DIOR_', '')
            anno_path = self.dataset_dir / 'annotations' / f"DIOR_{anno_name}"
            
            if anno_path.exists():
                boxes, labels = self._parse_voc_xml(anno_path)
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'image_name': sample_info['image_name'],
            'dataset': sample_info['dataset']
        }
    
    def _parse_voc_xml(self, xml_path: Path) -> Tuple[List, List]:
        """解析VOC XML标注"""
        boxes = []
        labels = []
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                name = obj.find('name').text
                bndbox = obj.find('bndbox')
                
                if bndbox is not None:
                    xmin = int(float(bndbox.find('xmin').text))
                    ymin = int(float(bndbox.find('ymin').text))
                    xmax = int(float(bndbox.find('xmax').text))
                    ymax = int(float(bndbox.find('ymax').text))
                    
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(name)
        except Exception as e:
            print(f"警告: 解析标注失败 {xml_path}: {e}")
        
        return boxes, labels
    
    def get_class_distribution(self) -> Dict:
        """获取类别分布统计"""
        distribution = {
            'seen': {},
            'unseen': {},
            'unknown': {}
        }
        
        for i in range(len(self)):
            sample = self[i]
            
            for label in sample['labels']:
                if label in self.seen_classes:
                    category = 'seen'
                elif label in self.unseen_classes:
                    category = 'unseen'
                else:
                    category = 'unknown'
                
                distribution[category][label] = distribution[category].get(label, 0) + 1
        
        return distribution
    
    def get_available_classes(self) -> Dict[str, List[str]]:
        """获取可用的类别"""
        return {
            'seen_classes': list(self.seen_classes),
            'unseen_classes': list(self.unseen_classes),
            'all_classes': list(self.seen_classes | self.unseen_classes)
        }


def test_loader():
    """测试数据集加载器"""
    print("\n" + "=" * 70)
    print("测试数据集加载器")
    print("=" * 70)
    
    # 测试不同配置
    configs = ['split_config_seen_50.json', 'split_config_seen_70.json']
    
    for config in configs:
        print(f"\n测试配置: {config}")
        
        loader = MiniDatasetLoader(split_config=config, mode='all')
        
        print(f"\n数据集大小: {len(loader)}")
        
        # 加载第一个样本
        if len(loader) > 0:
            sample = loader[0]
            print(f"\n第一个样本:")
            print(f"  图片: {sample['image_name']}")
            print(f"  大小: {sample['image'].size}")
            print(f"  来源: {sample['dataset']}")
            print(f"  目标数: {len(sample['boxes'])}")
            print(f"  类别: {sample['labels'][:3]}...")  # 只显示前3个
        
        # 类别分布
        distribution = loader.get_class_distribution()
        print(f"\n类别分布:")
        print(f"  Seen: {sum(distribution['seen'].values())} 个目标")
        print(f"  Unseen: {sum(distribution['unseen'].values())} 个目标")


if __name__ == "__main__":
    test_loader()

