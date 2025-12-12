# -*- coding: utf-8 -*-
"""
DIOR Detection Dataset Loader
支持从VOC XML格式加载bounding boxes和类别标签
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import xml.etree.ElementTree as ET
from pathlib import Path
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional


class DIORDetectionDataset(Dataset):
    """
    DIOR Remote Sensing Detection Dataset
    
    支持目标检测任务，从VOC XML格式加载bounding boxes和类别标签
    支持seen/unseen类别划分，训练时只使用seen类别
    """
    
    def __init__(self, root, split='trainval', transform=None, 
                 target_transform=None, anno_type='horizontal',
                 train_only_seen=False):
        """
        Args:
            root: Root directory of DIOR dataset
            split: 'trainval' or 'test'
            transform: Image transform pipeline
            target_transform: Target transform (for boxes/labels)
            anno_type: 'horizontal' or 'oriented'
            train_only_seen: 如果为True，训练时只保留seen类别的标注
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.anno_type = anno_type
        self.train_only_seen = train_only_seen
        
        # 导入seen/unseen类别划分
        try:
            import sys
            import importlib.util
            # 添加utils路径
            current_file_path = Path(__file__)
            utils_path = current_file_path.parent.parent / 'utils' / 'class_split.py'
            if utils_path.exists():
                spec = importlib.util.spec_from_file_location("class_split", utils_path)
                class_split_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(class_split_module)
                self.seen_class_indices = set(class_split_module.get_seen_class_indices())
            else:
                raise FileNotFoundError(f"class_split.py not found at {utils_path}")
        except Exception as e:
            # 如果导入失败，使用默认值（seen类别索引）
            # Seen: airplane(0), ship(13), vehicle(18), bridge(4), harbor(11),
            #       stadium(14), storage tank(15), airport(1), golf course(9), wind mill(19)
            self.seen_class_indices = {0, 1, 4, 9, 11, 13, 14, 15, 18, 19}
            if train_only_seen:
                print(f"⚠️  Warning: 无法导入class_split模块，使用默认seen类别索引: {self.seen_class_indices}")
                print(f"   错误信息: {e}")
        
        # DIOR class list (20 classes)
        self.classes = [
            "airplane", "airport", "baseball field", "basketball court",
            "bridge", "chimney", "dam", "expressway service area",
            "expressway toll station", "golf course", "ground track field",
            "harbor", "overpass", "ship", "stadium", "storage tank",
            "tennis court", "train station", "vehicle", "wind mill"
        ]
        
        # Normalize class names (lowercase, handle variations)
        self.class_to_idx = {cls.lower(): idx for idx, cls in enumerate(self.classes)}
        
        # Handle class name variations
        self.name_mapping = {
            'golffield': 'golf course',
            'golf course': 'golf course',
            'groundtrackfield': 'ground track field',
            'ground track field': 'ground track field',
            'expressway service area': 'expressway service area',
            'expressway toll station': 'expressway toll station',
        }
        
        # Load image IDs from split file or directory
        split_file = self.root / 'splits' / f'{split}.txt'
        if split_file.exists():
            with open(split_file, 'r') as f:
                self.image_ids = [line.strip() for line in f if line.strip()]
        else:
            # Fallback: read from images directory
            images_dir = self.root / 'images' / split
            if images_dir.exists():
                self.image_ids = [f.stem for f in images_dir.glob('*.jpg')]
            else:
                raise FileNotFoundError(
                    f"Neither split file nor images directory found: "
                    f"{split_file} or {images_dir}"
                )
        
        # Images and annotations directories
        # Handle split mapping: val -> trainval (val images are in trainval folder)
        if split == 'val':
            images_split = 'trainval'
        elif split == 'test':
            images_split = 'test'
        else:
            images_split = split
        
        self.images_dir = self.root / 'images' / images_split
        self.annos_dir = self.root / 'annotations' / anno_type
    
    def __len__(self):
        return len(self.image_ids)
    
    def _normalize_class_name(self, name: str) -> Optional[str]:
        """Normalize class name to match class list"""
        name_lower = name.lower().strip()
        
        # Direct match
        if name_lower in self.class_to_idx:
            return name_lower
        
        # Try mapping
        if name_lower in self.name_mapping:
            mapped = self.name_mapping[name_lower].lower()
            if mapped in self.class_to_idx:
                return mapped
        
        # Try partial match
        for cls_key in self.class_to_idx.keys():
            if name_lower in cls_key or cls_key in name_lower:
                return cls_key
        
        return None
    
    def _parse_xml(self, xml_path: Path) -> Tuple[List[List[float]], List[int], Tuple[int, int]]:
        """
        Parse VOC XML annotation file
        
        Returns:
            boxes: List of [xmin, ymin, xmax, ymax] in normalized coordinates
            labels: List of class indices
            image_size: (width, height)
        """
        boxes = []
        labels = []
        
        if not xml_path.exists():
            return boxes, labels, (800, 800)  # Default size
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image size
        size_elem = root.find('size')
        if size_elem is not None:
            width = int(size_elem.find('width').text)
            height = int(size_elem.find('height').text)
        else:
            width, height = 800, 800  # Default
        
        # Parse objects
        for obj in root.findall('object'):
            name_elem = obj.find('name')
            if name_elem is None:
                continue
            
            class_name = self._normalize_class_name(name_elem.text)
            if class_name is None:
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            bndbox = obj.find('bndbox')
            if bndbox is None:
                continue
            
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            # Normalize coordinates to [0, 1]
            xmin_norm = xmin / width
            ymin_norm = ymin / height
            xmax_norm = xmax / width
            ymax_norm = ymax / height
            
            # Validate box
            if xmax_norm > xmin_norm and ymax_norm > ymin_norm:
                boxes.append([xmin_norm, ymin_norm, xmax_norm, ymax_norm])
                labels.append(class_idx)
        
        return boxes, labels, (width, height)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image
        image_path = self.images_dir / f'{image_id}.jpg'
        if not image_path.exists():
            # Try without extension
            image_path = self.images_dir / image_id
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (width, height)
        
        # Load annotations
        anno_path = self.annos_dir / f'{image_id}.xml'
        boxes, labels, xml_size = self._parse_xml(anno_path)
        
        # 训练时只保留seen类别的标注
        if self.train_only_seen and self.split == 'trainval':
            filtered_boxes = []
            filtered_labels = []
            for box, label in zip(boxes, labels):
                if label in self.seen_class_indices:
                    filtered_boxes.append(box)
                    filtered_labels.append(label)
            boxes = filtered_boxes
            labels = filtered_labels
        
        # Convert to tensors
        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            # Empty image
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            boxes_tensor, labels_tensor = self.target_transform(boxes_tensor, labels_tensor)
        
        return {
            'image': image,
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': image_id,
            'original_size': original_size,
            'text_queries': self.classes  # All classes for text queries
        }


class RandomHorizontalFlip:
    """Random horizontal flip for image and boxes"""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, boxes):
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            # Flip boxes: xmin -> 1-xmax, xmax -> 1-xmin
            boxes = boxes.clone()
            xmin, ymin, xmax, ymax = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            boxes[:, 0] = 1 - xmax
            boxes[:, 2] = 1 - xmin
        
        return image, boxes


def get_detection_dataloader(root=None, split='trainval', batch_size=8,
                            num_workers=4, shuffle=None, 
                            image_size=224, augment=False,
                            train_only_seen=False):
    """
    Get DataLoader for DIOR detection dataset
    
    Args:
        root: Root directory of DIOR dataset
        split: 'trainval' or 'test'
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle
        image_size: Target image size
        augment: Whether to apply data augmentation
        train_only_seen: 训练时只使用seen类别（默认False）
    
    Returns:
        DataLoader instance
    """
    # Default dataset root
    if root is None:
        project_root = Path(__file__).parent.parent.parent.parent.parent
        possible_paths = [
            project_root / 'datasets' / 'DIOR',
            project_root.parent / 'datasets' / 'DIOR',
            Path('/media/ubuntu22/新加卷1/Projects/RemoteCLIP-main/datasets/DIOR')
        ]
        
        for path in possible_paths:
            if path.exists():
                root = str(path)
                break
        
        if root is None:
            raise FileNotFoundError(
                f"DIOR dataset not found. Please specify root path. "
                f"Tried: {possible_paths}"
            )
    
    # Image preprocessing (CLIP standard)
    if augment and split == 'trainval':
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
    
    # Create dataset
    dataset = DIORDetectionDataset(
        root=root,
        split=split,
        transform=transform,
        anno_type='horizontal',
        train_only_seen=train_only_seen
    )
    
    # Determine shuffle
    if shuffle is None:
        shuffle = (split == 'trainval')
    
    # Custom collate function for variable number of boxes
    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        boxes = [item['boxes'] for item in batch]
        labels = [item['labels'] for item in batch]
        image_ids = [item['image_id'] for item in batch]
        original_sizes = [item['original_size'] for item in batch]
        text_queries = batch[0]['text_queries']  # Same for all samples
        
        return {
            'images': images,
            'boxes': boxes,
            'labels': labels,
            'image_ids': image_ids,
            'original_sizes': original_sizes,
            'text_queries': text_queries
        }
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=(split == 'trainval' and augment)
    )
    
    return dataloader

