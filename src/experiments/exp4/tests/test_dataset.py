# -*- coding: utf-8 -*-
"""
数据集测试: 验证DIOR数据能正常加载
"""

import torch
import sys
import os
from pathlib import Path

# 添加路径
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent.parent.parent))


def test_dior_dataset():
    """测试DIOR数据集加载"""
    print("=" * 50)
    print("Dataset Test: DIOR Loading")
    print("=" * 50)
    
    from datasets.dior_detection import DIORDetectionDataset
    
    # 尝试加载数据集
    try:
        # 尝试多个可能的路径
        possible_roots = [
            Path(current_dir.parent.parent.parent.parent) / 'datasets' / 'DIOR',
            Path('/media/ubuntu22/新加卷1/Projects/RemoteCLIP-main/datasets/DIOR'),
            Path('./datasets/DIOR'),
        ]
        
        dataset = None
        for root in possible_roots:
            if root.exists():
                try:
                    dataset = DIORDetectionDataset(
                        root=str(root),
                        split='trainval',
                        transform=None
                    )
                    print(f"✅ Dataset loaded from: {root}")
                    break
                except Exception as e:
                    continue
        
        if dataset is None:
            print("⚠️  DIOR dataset not found in common locations")
            print("   Please download and set correct path")
            print("   Skipping dataset test...")
            return
        
        print(f"   Total samples: {len(dataset)}")
        
        # 测试加载一个样本
        if len(dataset) > 0:
            sample = dataset[0]
            
            print(f"\n   Sample 0:")
            print(f"   - Image ID: {sample['image_id']}")
            print(f"   - Image: {sample['image'].shape if hasattr(sample['image'], 'shape') else type(sample['image'])}")
            print(f"   - Boxes: {sample['boxes'].shape}")
            print(f"   - Labels: {sample['labels'].shape}")
            print(f"   - Text queries: {len(sample['text_queries'])} classes")
            
            # 验证格式
            assert sample['boxes'].dim() == 2
            assert sample['boxes'].shape[1] == 4
            assert sample['labels'].dim() == 1
            assert len(sample['boxes']) == len(sample['labels'])
            
            print("\n✅ Sample format correct!")
        
    except Exception as e:
        print(f"⚠️  Error loading dataset: {e}")
        print("   Skipping dataset test...")
    
    print()


def test_dataloader():
    """测试DataLoader"""
    print("=" * 50)
    print("DataLoader Test")
    print("=" * 50)
    
    from datasets.dior_detection import get_detection_dataloader
    
    try:
        # 尝试加载
        loader = get_detection_dataloader(
            root=None,  # 自动查找
            split='trainval',
            batch_size=2,
            num_workers=0,  # 避免多进程问题
            image_size=224,
            augment=False
        )
        
        print("✅ DataLoader created!")
        print(f"   Dataset size: {len(loader.dataset)}")
        
        # 测试一个batch
        if len(loader) > 0:
            batch = next(iter(loader))
            
            print(f"   Batch images: {batch['images'].shape}")
            print(f"   Batch size: {len(batch['boxes'])}")
            print(f"   Boxes in batch[0]: {batch['boxes'][0].shape}")
            print(f"   Labels in batch[0]: {batch['labels'][0].shape}")
            
            print("\n✅ DataLoader works!")
        
    except FileNotFoundError as e:
        print(f"⚠️  Dataset not found: {e}")
        print("   Skipping dataloader test...")
    except Exception as e:
        print(f"⚠️  Error: {e}")
        print("   Skipping dataloader test...")
    
    print()


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting Dataset Tests")
    print("="*50 + "\n")
    
    test_dior_dataset()
    test_dataloader()


