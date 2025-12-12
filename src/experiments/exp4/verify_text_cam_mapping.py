#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证文本编码顺序和CAM类别映射是否正确
"""

import torch
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.simple_surgery_cam_detector import create_simple_surgery_cam_detector
from datasets.dior_detection import get_detection_dataloader
import yaml

# DIOR类别列表（按顺序）
DIOR_CLASSES = [
    "airplane", "airport", "baseball field", "basketball court",
    "bridge", "chimney", "dam", "expressway service area",
    "expressway toll station", "golf course", "ground track field",
    "harbor", "overpass", "ship", "stadium", "storage tank",
    "tennis court", "train station", "vehicle", "wind mill"
]

def test_text_encoding_order():
    """测试文本编码顺序"""
    print("=" * 80)
    print("测试1: 文本编码顺序")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载配置
    with open('configs/surgery_cam_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建模型
    surgery_checkpoint = config.get('surgery_clip_checkpoint', 
                                   'checkpoints/RemoteCLIP-ViT-B-32.pt')
    if not Path(surgery_checkpoint).is_absolute():
        project_root = Path(__file__).parent.parent.parent.parent
        surgery_checkpoint = project_root / surgery_checkpoint
        surgery_checkpoint = str(surgery_checkpoint)
    
    model = create_simple_surgery_cam_detector(
        surgery_clip_checkpoint=surgery_checkpoint,
        num_classes=20,
        cam_resolution=7,
        device=device,
        unfreeze_cam_last_layer=True
    )
    
    # 测试文本编码
    test_queries = ["airplane", "ship", "vehicle", "bridge"]
    print(f"\n输入text_queries: {test_queries}")
    
    # 获取CLIP模型
    clip_model = model.simple_surgery_cam.clip
    
    # 编码文本
    from src.competitors.clip_methods.surgeryclip.clip import tokenize
    text_tokens = tokenize(test_queries).to(device)
    text_features = clip_model.encode_text(text_tokens)
    
    print(f"text_features shape: {text_features.shape}")
    print(f"text_features[0] 应该对应 'airplane'")
    print(f"text_features[1] 应该对应 'ship'")
    print(f"text_features[2] 应该对应 'vehicle'")
    print(f"text_features[3] 应该对应 'bridge'")
    
    # 验证：计算每个text_feature与对应类名的相似度
    print("\n验证：计算text_features与对应类名的相似度")
    for i, query in enumerate(test_queries):
        # 单独编码这个类名
        single_tokens = tokenize([query]).to(device)
        single_features = clip_model.encode_text(single_tokens)
        
        # 计算相似度
        similarity = torch.cosine_similarity(
            text_features[i:i+1], 
            single_features,
            dim=1
        ).item()
        
        print(f"  text_features[{i}] vs '{query}': {similarity:.4f} (应该接近1.0)")
    
    return model

def test_cam_class_mapping(model):
    """测试CAM类别映射"""
    print("\n" + "=" * 80)
    print("测试2: CAM类别映射")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载一张测试图像
    loader = get_detection_dataloader(
        root=None,
        split='test',
        batch_size=1,
        num_workers=0,
        image_size=224,
        shuffle=False
    )
    
    batch = next(iter(loader))
    images = batch['images'].to(device)
    gt_boxes = batch['boxes'][0]
    gt_labels = batch['labels'][0]
    
    print(f"\n图像GT标签: {gt_labels.tolist()}")
    print(f"GT类别名称: {[DIOR_CLASSES[l.item()] for l in gt_labels]}")
    
    # 使用所有20个类别进行推理
    text_queries = DIOR_CLASSES.copy()
    print(f"\n使用text_queries顺序: {text_queries[:5]}... (共{len(text_queries)}个)")
    
    # 生成CAM
    model.eval()
    with torch.no_grad():
        outputs = model(images, text_queries)
        cam = outputs['cam'][0]  # [C, H, W]
    
    print(f"\nCAM shape: {cam.shape}")
    print(f"CAM维度: [C={cam.shape[0]}, H={cam.shape[1]}, W={cam.shape[2]}]")
    
    # 检查每个GT类别对应的CAM响应
    print("\n检查GT类别对应的CAM响应:")
    for i, label in enumerate(gt_labels):
        label_idx = label.item()
        class_name = DIOR_CLASSES[label_idx]
        cam_class = cam[label_idx]  # 对应类别的CAM
        
        # 计算CAM在GT框内的平均响应
        box = gt_boxes[i]
        x1, y1, x2, y2 = box
        
        # 将归一化坐标转换为CAM空间坐标
        H_cam, W_cam = cam.shape[1], cam.shape[2]
        x1_cam = int(x1 * W_cam)
        y1_cam = int(y1 * H_cam)
        x2_cam = int(x2 * W_cam)
        y2_cam = int(y2 * H_cam)
        
        # 确保坐标在范围内
        x1_cam = max(0, min(x1_cam, W_cam-1))
        y1_cam = max(0, min(y1_cam, H_cam-1))
        x2_cam = max(0, min(x2_cam, W_cam-1))
        y2_cam = max(0, min(y2_cam, H_cam-1))
        
        cam_inside = cam_class[y1_cam:y2_cam+1, x1_cam:x2_cam+1]
        cam_outside = cam_class.clone()
        cam_outside[y1_cam:y2_cam+1, x1_cam:x2_cam+1] = -float('inf')
        cam_outside = cam_outside[cam_outside > -float('inf')]
        
        inside_mean = cam_inside.mean().item() if cam_inside.numel() > 0 else 0
        outside_mean = cam_outside.mean().item() if cam_outside.numel() > 0 else 0
        
        # 检查其他类别的CAM响应（应该更低）
        other_cams = []
        for c in range(cam.shape[0]):
            if c != label_idx:
                other_cam = cam[c]
                other_inside = other_cam[y1_cam:y2_cam+1, x1_cam:x2_cam+1]
                other_mean = other_inside.mean().item() if other_inside.numel() > 0 else 0
                other_cams.append((c, DIOR_CLASSES[c], other_mean))
        
        # 找到响应最高的其他类别
        other_cams.sort(key=lambda x: x[2], reverse=True)
        top_other = other_cams[0] if other_cams else (None, None, 0)
        
        print(f"\n  GT类别 [{label_idx}] {class_name}:")
        print(f"    正确CAM响应 (cam[{label_idx}]): {inside_mean:.4f}")
        print(f"    框外响应: {outside_mean:.4f}")
        print(f"    对比度: {inside_mean / (outside_mean + 1e-6):.2f}")
        print(f"    最高其他类别CAM: cam[{top_other[0]}] '{top_other[1]}' = {top_other[2]:.4f}")
        
        if top_other[2] > inside_mean:
            print(f"    ⚠️  警告: 其他类别 '{top_other[1]}' 的CAM响应更高！")
            print(f"       这可能表示类别映射有问题")

def main():
    model = test_text_encoding_order()
    test_cam_class_mapping(model)

if __name__ == '__main__':
    main()


