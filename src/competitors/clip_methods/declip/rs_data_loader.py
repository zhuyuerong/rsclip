# -*- coding: utf-8 -*-
"""
DeCLIP遥感数据加载器

替换COCO/LVIS数据加载，适配DIOR/DOTA/LAE-1M
"""
import os
import json
import logging
import random
from typing import List, Optional, Tuple
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from .open_clip.transform import det_image_transform
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None


class RSProposalDistillDataset(Dataset):
    """
    遥感数据蒸馏数据集（替换COCO ProposalDistillDataset）
    
    支持DIOR/DOTA/LAE-1M格式
    """
    def __init__(self, data_root: str, split: str = "train",
                 transforms=None, crop_size=224, tokenizer=None, args=None):
        """
        Args:
            data_root: 数据集根目录
            split: 数据集划分（train/val/test）
            transforms: 数据变换
            crop_size: 裁剪尺寸
            tokenizer: 分词器
            args: 其他参数
        """
        self.data_root = data_root
        self.split = split
        self.transforms = transforms
        self.tokenize = tokenizer
        self.crop_size = crop_size if isinstance(crop_size, (tuple, list)) else [crop_size, crop_size]
        self.args = args
        self.min_size = getattr(args, 'min_size', 10) if args else 10
        self.max_size = getattr(args, 'max_size', 1000) if args else 1000
        
        # 加载数据集
        self.samples = self._load_rs_dataset()
        logger.info(f"加载了 {len(self.samples)} 个样本")
        
        # 设置proxy transform（用于VFM）
        if args and hasattr(args, 'use_vfm') and args.use_vfm:
            L = getattr(args, 'det_image_size', 224) // getattr(args, 'downsample_factor', 1)
            if args.use_vfm == "dino-B-8":
                proxy_resolution = L * 8
            elif args.use_vfm in ["dinov2-L", "dinov2-B"]:
                proxy_resolution = L * 14
            elif args.use_vfm in ["sam-B", "sam-L", "dino-B-16"]:
                proxy_resolution = L * 16
            else:
                proxy_resolution = L * 16
            self.proxy_transform = det_image_transform(
                proxy_resolution,
                is_train=False,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        else:
            self.proxy_transform = None
    
    def _load_rs_dataset(self) -> List[dict]:
        """加载遥感数据集"""
        samples = []
        
        # 尝试加载DIOR格式
        annotation_dir = os.path.join(self.data_root, "annotations", "horizontal")
        if not os.path.exists(annotation_dir):
            annotation_dir = os.path.join(self.data_root, "annotations")
        
        image_dir = os.path.join(self.data_root, "images", self.split)
        if not os.path.exists(image_dir):
            image_dir = os.path.join(self.data_root, "images")
        
        split_file = os.path.join(self.data_root, "splits", f"{self.split}.txt")
        if os.path.exists(split_file):
            # DIOR格式
            with open(split_file, 'r') as f:
                image_ids = [line.strip() for line in f if line.strip()]
            
            for image_id in image_ids:
                xml_path = os.path.join(annotation_dir, f"{image_id}.xml")
                img_path = self._find_image_path(image_dir, image_id)
                
                if not os.path.exists(xml_path) or not img_path:
                    continue
                
                # 解析XML标注
                import xml.etree.ElementTree as ET
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                boxes = []
                for obj in root.findall("object"):
                    bbox = obj.find("bndbox")
                    if bbox is not None:
                        xmin = int(bbox.find("xmin").text)
                        ymin = int(bbox.find("ymin").text)
                        xmax = int(bbox.find("xmax").text)
                        ymax = int(bbox.find("ymax").text)
                        boxes.append([xmin, ymin, xmax, ymax])
                
                if boxes:
                    samples.append({
                        'image_path': img_path,
                        'image_id': image_id,
                        'boxes': boxes
                    })
        
        return samples
    
    def _find_image_path(self, image_dir: str, image_id: str) -> Optional[str]:
        """查找图像路径"""
        extensions = ['.jpg', '.png', '.jpeg', '.tif', '.bmp']
        for ext in extensions:
            path = os.path.join(image_dir, f"{image_id}{ext}")
            if os.path.exists(path):
                return path
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample['image_path']
        boxes = sample['boxes']
        
        # 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"无法加载图像 {image_path}: {e}")
            # 返回下一个样本
            return self.__getitem__((idx + 1) % len(self.samples))
        
        img_w, img_h = image.size
        
        # 应用变换
        if self.transforms:
            new_image = self.transforms[0](image)
        else:
            new_image = image
        
        # 处理边界框
        max_anns = getattr(self.args, 'max_anns', 20) if self.args else 20
        boxes_template = torch.zeros(max_anns, 5)  # xyxy + score
        image_crops = torch.zeros(max_anns, 3, *self.crop_size)
        
        num_valid_boxes = 0
        indices = list(range(len(boxes)))
        random.shuffle(indices)
        
        for i, box_idx in enumerate(indices[:max_anns]):
            xmin, ymin, xmax, ymax = boxes[box_idx]
            w, h = xmax - xmin, ymax - ymin
            
            # 过滤太小或太大的框
            if w * h < (self.min_size ** 2) or w * h > (self.max_size ** 2):
                continue
            
            num_valid_boxes += 1
            
            # 扩展框（用于裁剪）
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
            expand_w, expand_h = w * 1.5, h * 1.5
            x0 = max(0, cx - expand_w / 2)
            y0 = max(0, cy - expand_h / 2)
            x1 = min(img_w, cx + expand_w / 2)
            y1 = min(img_h, cy + expand_h / 2)
            
            # 裁剪图像
            if self.transforms and len(self.transforms) > 1:
                crop = self.transforms[1](image.crop((x0, y0, x1, y1)))
            else:
                crop = image.crop((x0, y0, x1, y1))
                crop = crop.resize(self.crop_size)
            
            image_crops[i] = crop if isinstance(crop, torch.Tensor) else torch.tensor(crop)
            boxes_template[i] = torch.tensor([xmin, ymin, xmax, ymax, 1.0])
        
        # 如果没有有效框，创建一个默认框
        if num_valid_boxes == 0:
            boxes_template[0] = torch.tensor([0, 0, img_w / 4, img_h / 4, 1.0])
            if self.transforms and len(self.transforms) > 1:
                image_crops[0] = self.transforms[1](image.crop((0, 0, img_w // 4, img_h // 4)))
            else:
                crop = image.crop((0, 0, img_w // 4, img_h // 4))
                image_crops[0] = torch.tensor(crop.resize(self.crop_size))
        
        # 归一化边界框
        if isinstance(new_image, torch.Tensor):
            _, h, w = new_image.shape
        else:
            h, w = new_image.size[1], new_image.size[0]
        
        boxes_template[:, [0, 2]] /= w
        boxes_template[:, [1, 3]] /= h
        
        # Proxy image（用于VFM）
        if self.proxy_transform:
            proxy_image = self.proxy_transform(image)
        else:
            proxy_image = new_image
        
        return new_image, boxes_template, image_crops, proxy_image


def get_rs_proposal_distill_dataset(args, preprocess_fn, is_train=True, epoch=0, tokenizer=None):
    """获取遥感数据蒸馏数据集"""
    dataset = RSProposalDistillDataset(
        data_root=args.train_data if is_train else args.val_data,
        split="train" if is_train else "val",
        transforms=preprocess_fn,
        crop_size=getattr(args, 'crop_size', 224),
        tokenizer=tokenizer,
        args=args
    )
    
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if getattr(args, 'distributed', False) and is_train else None
    shuffle = is_train and sampler is None
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=getattr(args, 'workers', 4),
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    
    return DataInfo(dataloader, sampler)








