# -*- coding: utf-8 -*-
"""
Data loading utilities for DIOR dataset
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from torchvision import transforms


class DIORDataset(Dataset):
    """
    DIOR Remote Sensing Dataset
    
    Supports multi-label classification with image-level labels extracted from XML annotations.
    """
    
    def __init__(self, root, split='trainval', transform=None):
        """
        Args:
            root: Root directory of DIOR dataset (should contain images/, annotations/, splits/)
            split: 'trainval' or 'test'
            transform: Image transform pipeline
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        
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
                raise FileNotFoundError(f"Neither split file nor images directory found: {split_file} or {images_dir}")
        
        # Images and annotations directories
        self.images_dir = self.root / 'images' / split
        self.annos_dir = self.root / 'annotations' / 'horizontal'
        
    def __len__(self):
        return len(self.image_ids)
    
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
        
        # Load labels from XML annotation
        labels = torch.zeros(len(self.classes), dtype=torch.float32)
        anno_path = self.annos_dir / f'{image_id}.xml'
        
        if anno_path.exists():
            tree = ET.parse(anno_path)
            root = tree.getroot()
            
            # Extract unique class names from annotations
            found_classes = set()
            for obj in root.findall('object'):
                name_elem = obj.find('name')
                if name_elem is not None:
                    class_name = name_elem.text.strip().lower()
                    # Handle class name variations
                    if class_name in self.class_to_idx:
                        found_classes.add(class_name)
                    else:
                        # Try to match with variations
                        for cls_key, cls_idx in self.class_to_idx.items():
                            if class_name in cls_key or cls_key in class_name:
                                found_classes.add(cls_key)
                                break
            
            # Set labels
            for class_name in found_classes:
                if class_name in self.class_to_idx:
                    labels[self.class_to_idx[class_name]] = 1.0
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            'images': image,
            'labels': labels,
            'text_queries': self.classes,  # All classes for text queries
            'image_id': image_id
        }


def get_dataloader(dataset_name='DIOR', root=None, split='trainval', batch_size=8, 
                   num_workers=4, shuffle=None):
    """
    Get DataLoader for specified dataset
    
    Args:
        dataset_name: Name of dataset ('DIOR')
        root: Root directory of dataset (if None, uses default path)
        split: 'trainval' or 'test'
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle (if None, shuffles for 'trainval', not for 'test')
    
    Returns:
        DataLoader instance
    """
    # Default dataset root
    if root is None:
        # Try to find DIOR dataset in common locations
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
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    # Create dataset
    if dataset_name == 'DIOR':
        dataset = DIORDataset(
            root=root,
            split=split,
            transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Determine shuffle
    if shuffle is None:
        shuffle = (split == 'trainval')
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'trainval')  # Drop last batch for training
    )
    
    return dataloader





