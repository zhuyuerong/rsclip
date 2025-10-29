# -*- coding: utf-8 -*-
"""
Seen/Unseen数据集划分工具
"""

import os
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# 添加项目根目录到路径
root_dir = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(root_dir))


# DIOR数据集20个类别
ALL_DIOR_CLASSES = [
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 
    'bridge', 'chimney', 'dam', 'Expressway-Service-area',
    'Expressway-toll-station', 'golffield', 'groundtrackfield',
    'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
    'tenniscourt', 'trainstation', 'vehicle', 'windmill'
]


def create_seen_unseen_splits(unseen_classes):
    """
    创建seen/unseen数据集划分
    
    Args:
        unseen_classes: unseen类别列表
    
    Returns:
        dict: {'seen': [...], 'unseen': [...], 'all': [...]}
    """
    seen_classes = [c for c in ALL_DIOR_CLASSES if c not in unseen_classes]
    
    return {
        'seen': seen_classes,
        'unseen': unseen_classes,
        'all': ALL_DIOR_CLASSES
    }


class SeenUnseenDataset(Dataset):
    """
    支持seen/unseen划分的DIOR数据集
    
    Args:
        root_dir: 数据集根目录
        split: 'seen', 'unseen', 或 'all'
        mode: 'train', 'val', 或 'test'
        unseen_classes: unseen类别列表
        transform: 图像变换
    """
    
    def __init__(self, root_dir, split='seen', mode='val', unseen_classes=None, transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.mode = mode
        
        # 默认unseen类别
        if unseen_classes is None:
            unseen_classes = ['airplane', 'bridge', 'storagetank', 'vehicle', 'windmill']
        
        # 获取seen/unseen划分
        split_info = create_seen_unseen_splits(unseen_classes)
        
        if split == 'seen':
            self.target_classes = split_info['seen']
        elif split == 'unseen':
            self.target_classes = split_info['unseen']
        else:
            self.target_classes = split_info['all']
        
        # 加载样本
        self.samples = self._load_samples()
        
        # 图像变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
        else:
            self.transform = transform
    
    def _load_samples(self):
        """加载符合条件的样本"""
        samples = []
        
        # 检测是否为DIOR数据集或mini_dataset
        if (self.root_dir / "images" / "train").exists():
            # 完整DIOR数据集
            image_dir = self.root_dir / "images" / "trainval" if self.mode != "test" else self.root_dir / "images" / "test"
            anno_dir = self.root_dir / "annotations" / "horizontal"
        elif (self.root_dir / "images").exists() and (self.root_dir / "annotations").exists():
            # mini_dataset
            image_dir = self.root_dir / "images"
            anno_dir = self.root_dir / "annotations"
        else:
            raise ValueError(f"无法识别的数据集结构: {self.root_dir}")
        
        # 遍历标注文件
        if not anno_dir.exists():
            print(f"警告: 标注目录不存在: {anno_dir}")
            return samples
        
        for anno_file in anno_dir.glob("*.xml"):
            try:
                tree = ET.parse(anno_file)
                root = tree.getroot()
                
                # 提取所有对象类别
                objects = root.findall('.//object')
                if not objects:
                    continue
                
                # 检查是否有目标类别
                has_target_class = False
                sample_classes = []
                bboxes = []
                
                for obj in objects:
                    class_name = obj.find('name').text
                    
                    # 归一化类别名称
                    class_name = class_name.replace('-', '')
                    if class_name.lower() == 'expresswayservicearea':
                        class_name = 'Expressway-Service-area'
                    elif class_name.lower() == 'expresswaytollstation':
                        class_name = 'Expressway-toll-station'
                    
                    if class_name in self.target_classes:
                        has_target_class = True
                        sample_classes.append(class_name)
                        
                        # 提取bbox
                        bndbox = obj.find('bndbox')
                        if bndbox is not None:
                            bbox = {
                                'xmin': int(bndbox.find('xmin').text),
                                'ymin': int(bndbox.find('ymin').text),
                                'xmax': int(bndbox.find('xmax').text),
                                'ymax': int(bndbox.find('ymax').text),
                                'class': class_name
                            }
                            bboxes.append(bbox)
                
                if not has_target_class:
                    continue
                
                # 查找对应图像
                image_name = anno_file.stem + '.jpg'
                image_path = image_dir / image_name
                
                if not image_path.exists():
                    continue
                
                samples.append({
                    'image_path': str(image_path),
                    'image_id': anno_file.stem,
                    'classes': sample_classes,
                    'bboxes': bboxes
                })
                
            except Exception as e:
                print(f"警告: 解析{anno_file}失败: {e}")
                continue
        
        print(f"加载{self.split} {self.mode}数据集: {len(samples)}个样本")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像
        image = Image.open(sample['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'image_id': sample['image_id'],
            'class_name': sample['classes'][0] if sample['classes'] else 'unknown',  # 取第一个类别
            'classes': sample['classes'],
            'bboxes': sample['bboxes']
        }


def test_seen_unseen_split():
    """测试seen/unseen划分"""
    print("="*60)
    print("测试Seen/Unseen数据集划分")
    print("="*60)
    
    # 测试划分
    unseen_classes = ['airplane', 'bridge', 'storagetank', 'vehicle', 'windmill']
    splits = create_seen_unseen_splits(unseen_classes)
    
    print(f"\nUnseen类别 ({len(splits['unseen'])}个):")
    print(splits['unseen'])
    
    print(f"\nSeen类别 ({len(splits['seen'])}个):")
    print(splits['seen'])
    
    # 测试数据集加载
    dataset_root = "/media/ubuntu22/新加卷1/Projects/RemoteCLIP-main/datasets/mini_dataset"
    
    if Path(dataset_root).exists():
        print(f"\n测试数据集加载: {dataset_root}")
        
        seen_dataset = SeenUnseenDataset(dataset_root, split='seen', mode='val', unseen_classes=unseen_classes)
        print(f"Seen验证集: {len(seen_dataset)}个样本")
        
        unseen_dataset = SeenUnseenDataset(dataset_root, split='unseen', mode='val', unseen_classes=unseen_classes)
        print(f"Unseen验证集: {len(unseen_dataset)}个样本")
        
        # 测试加载一个样本
        if len(seen_dataset) > 0:
            sample = seen_dataset[0]
            print(f"\n样本示例:")
            print(f"  图像形状: {sample['image'].shape}")
            print(f"  类别: {sample['class_name']}")
            print(f"  所有类别: {sample['classes']}")
    else:
        print(f"警告: 数据集不存在: {dataset_root}")
    
    print("\n✓ 测试完成！")


if __name__ == "__main__":
    test_seen_unseen_split()

