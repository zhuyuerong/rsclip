#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
调试评估：检查为什么mAP为0
"""

import torch
import yaml
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.improved_direct_detection_detector import create_improved_direct_detection_detector
from datasets.dior_detection import get_detection_dataloader

def debug_model_outputs(model, dataloader, device, num_samples=5):
    """调试模型输出"""
    model.eval()
    
    print("=" * 80)
    print("调试模型输出")
    print("=" * 80)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_samples:
                break
            
            images = batch['images'].to(device)
            text_queries = batch['text_queries']
            gt_boxes = batch['boxes']
            gt_labels = batch['labels']
            
            print(f"\n样本 {batch_idx + 1}:")
            print(f"  图像数量: {len(images)}")
            print(f"  GT框数量: {[len(b) for b in gt_boxes]}")
            
            # Forward
            outputs = model(images, text_queries)
            boxes = outputs['pred_boxes']  # [B, C, H, W, 4]
            confidences = outputs['confidences']  # [B, C, H, W]
            
            print(f"  预测框形状: {boxes.shape}")
            print(f"  置信度形状: {confidences.shape}")
            print(f"  置信度范围: [{confidences.min().item():.4f}, {confidences.max().item():.4f}]")
            print(f"  置信度均值: {confidences.mean().item():.4f}")
            
            # 检查不同阈值下的检测数量
            for threshold in [0.1, 0.2, 0.3, 0.5]:
                num_detections = (confidences > threshold).sum().item()
                print(f"  阈值 {threshold}: {num_detections} 个检测")
            
            # 推理
            detections = model.inference(
                images, text_queries,
                conf_threshold=0.1,  # 降低阈值
                nms_threshold=0.5
            )
            
            print(f"  检测结果数量: {[len(d) for d in detections]}")
            
            # 显示第一个图像的检测详情
            if len(detections) > 0 and len(detections[0]) > 0:
                print(f"\n  第一个图像的检测详情:")
                for i, det in enumerate(detections[0][:5]):  # 只显示前5个
                    print(f"    检测 {i+1}: 类别={det['class']}, 置信度={det['confidence']:.4f}, "
                          f"框={det['box'].tolist() if isinstance(det['box'], torch.Tensor) else det['box']}")

if __name__ == '__main__':
    config_path = Path(__file__).parent / 'configs/improved_detector_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = config.get('device', 'cuda')
    
    # 创建模型
    surgery_checkpoint = config.get('surgery_clip_checkpoint', 
                                   'checkpoints/RemoteCLIP-ViT-B-32.pt')
    if not Path(surgery_checkpoint).is_absolute():
        project_root = Path(__file__).parent.parent.parent.parent
        surgery_checkpoint = project_root / surgery_checkpoint
        surgery_checkpoint = str(surgery_checkpoint)
    
    print("创建模型...")
    model = create_improved_direct_detection_detector(
        surgery_clip_checkpoint=surgery_checkpoint,
        num_classes=config.get('num_classes', 20),
        cam_resolution=config.get('cam_resolution', 7),
        device=device,
        unfreeze_cam_last_layer=config.get('unfreeze_cam_last_layer', True)
    )
    model.to(device)
    
    # 加载checkpoint
    checkpoint_path = Path(__file__).parent / 'checkpoints/improved_detector/best_improved_detector_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if 'attn.in_proj_weight' in key or 'attn.in_proj_bias' in key:
            qkv_key = key.replace('in_proj_weight', 'qkv.weight').replace('in_proj_bias', 'qkv.bias')
            if qkv_key not in state_dict:
                continue
        elif 'attn.qkv.weight' in key or 'attn.qkv.bias' in key:
            if key in model_state_dict:
                filtered_state_dict[key] = value
        elif key in model_state_dict:
            if model_state_dict[key].shape == value.shape:
                filtered_state_dict[key] = value
    
    model.load_state_dict(filtered_state_dict, strict=False)
    print("✅ 模型加载成功")
    
    # 加载数据
    val_loader = get_detection_dataloader(
        root=config.get('dataset_root'),
        split='val',
        batch_size=2,
        num_workers=0,
        image_size=config.get('image_size', 224),
        augment=False,
        train_only_seen=False
    )
    
    debug_model_outputs(model, val_loader, device, num_samples=3)


