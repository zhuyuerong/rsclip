#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
调试mAP计算：检查为什么mAP为0
"""

import torch
import yaml
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.improved_direct_detection_detector import create_improved_direct_detection_detector
from datasets.dior_detection import get_detection_dataloader
from utils.metrics import DetectionMetrics, box_iou

def debug_map_calculation(model, dataloader, device, num_samples=3):
    """调试mAP计算"""
    model.eval()
    metrics = DetectionMetrics(num_classes=20)
    
    print("=" * 80)
    print("调试mAP计算")
    print("=" * 80)
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if sample_count >= num_samples:
                break
            
            images = batch['images'].to(device)
            text_queries = batch['text_queries']
            gt_boxes = batch['boxes']
            gt_labels = batch['labels']
            
            print(f"\n样本 {batch_idx + 1}:")
            print(f"  GT框数量: {[len(b) for b in gt_boxes]}")
            print(f"  GT标签: {[l.tolist() for l in gt_labels]}")
            
            # 推理
            detections = model.inference(
                images, text_queries,
                conf_threshold=0.1,
                nms_threshold=0.5
            )
            
            print(f"  检测结果数量: {[len(d) for d in detections]}")
            
            # 手动检查第一个图像的IoU
            if len(gt_boxes) > 0 and len(gt_boxes[0]) > 0 and len(detections[0]) > 0:
                gt_boxes_0 = gt_boxes[0].to(device)  # [N, 4]
                gt_labels_0 = gt_labels[0].to(device)  # [N]
                
                print(f"\n  第一个图像:")
                print(f"    GT框: {gt_boxes_0.tolist()}")
                print(f"    GT标签: {gt_labels_0.tolist()}")
                
                # 提取预测框
                pred_boxes_list = []
                pred_labels_list = []
                pred_scores_list = []
                
                for det in detections[0]:
                    box = det['box']
                    if isinstance(box, torch.Tensor):
                        box_tensor = box.to(device)
                        if box_tensor.dim() == 1:
                            box_tensor = box_tensor.unsqueeze(0)
                    elif isinstance(box, (list, tuple)):
                        box_tensor = torch.tensor(box, device=device).unsqueeze(0)
                    else:
                        box_tensor = torch.tensor([box['xmin'], box['ymin'], box['xmax'], box['ymax']], device=device).unsqueeze(0)
                    
                    pred_boxes_list.append(box_tensor)
                    pred_labels_list.append(det['class'])
                    pred_scores_list.append(det.get('confidence', 0.5))
                
                if pred_boxes_list:
                    pred_boxes_0 = torch.cat(pred_boxes_list, dim=0).to(device)  # [M, 4]
                    pred_labels_0 = torch.tensor(pred_labels_list, dtype=torch.long, device=device)
                    pred_scores_0 = torch.tensor(pred_scores_list, device=device)
                    
                    print(f"    预测框数量: {len(pred_boxes_0)}")
                    print(f"    预测框: {pred_boxes_0.tolist()[:3]}...")  # 只显示前3个
                    print(f"    预测标签: {pred_labels_0.tolist()[:3]}...")
                    print(f"    预测分数: {pred_scores_0.tolist()[:3]}...")
                    
                    # 计算IoU
                    ious = box_iou(pred_boxes_0, gt_boxes_0)  # [M, N]
                    print(f"\n    IoU矩阵形状: {ious.shape}")
                    print(f"    IoU最大值: {ious.max().item():.4f}")
                    print(f"    IoU均值: {ious.mean().item():.4f}")
                    
                    # 检查每个GT框的最佳匹配
                    for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes_0, gt_labels_0)):
                        # 只检查相同类别的预测
                        same_class_mask = (pred_labels_0 == gt_label)
                        if same_class_mask.sum() > 0:
                            same_class_ious = ious[same_class_mask, gt_idx]
                            max_iou = same_class_ious.max().item()
                            max_idx = same_class_ious.argmax().item()
                            pred_idx = torch.where(same_class_mask)[0][max_idx].item()
                            
                            print(f"\n    GT框 {gt_idx} (类别 {gt_label.item()}):")
                            print(f"      框: {gt_box.tolist()}")
                            print(f"      最佳匹配预测框 {pred_idx}: {pred_boxes_0[pred_idx].tolist()}")
                            print(f"      IoU: {max_iou:.4f}")
                            if max_iou > 0.5:
                                print(f"      ✅ 匹配成功 (IoU > 0.5)")
                            else:
                                print(f"      ❌ 匹配失败 (IoU < 0.5)")
                        else:
                            print(f"\n    GT框 {gt_idx} (类别 {gt_label.item()}):")
                            print(f"      框: {gt_box.tolist()}")
                            print(f"      ❌ 没有相同类别的预测")
            
            # 转换为metrics格式
            pred_boxes = []
            pred_labels = []
            pred_scores = []
            
            for img_detections in detections:
                if len(img_detections) > 0:
                    boxes_list = []
                    labels_list = []
                    scores_list = []
                    
                    for det in img_detections:
                        box = det['box']
                        if isinstance(box, torch.Tensor):
                            box_tensor = box.to(device)
                            if box_tensor.dim() == 1:
                                box_tensor = box_tensor.unsqueeze(0)
                        elif isinstance(box, (list, tuple)):
                            box_tensor = torch.tensor(box, device=device).unsqueeze(0)
                        else:
                            box_tensor = torch.tensor([box['xmin'], box['ymin'], box['xmax'], box['ymax']], device=device).unsqueeze(0)
                        
                        boxes_list.append(box_tensor)
                        labels_list.append(det['class'])
                        scores_list.append(det.get('confidence', det.get('score', 0.5)))
                    
                    pred_boxes.append(torch.cat(boxes_list, dim=0).to(device))
                    pred_labels.append(torch.tensor(labels_list, dtype=torch.long, device=device))
                    pred_scores.append(torch.tensor(scores_list, device=device))
                else:
                    pred_boxes.append(torch.zeros((0, 4), device=device))
                    pred_labels.append(torch.zeros((0,), dtype=torch.long, device=device))
                    pred_scores.append(torch.zeros((0,), device=device))
            
            # 更新metrics
            metrics.update(
                pred_boxes=pred_boxes,
                pred_labels=pred_labels,
                pred_scores=pred_scores,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                image_ids=batch.get('image_ids', None)
            )
            
            sample_count += 1
    
    # 计算mAP
    print("\n" + "=" * 80)
    print("计算mAP...")
    print("=" * 80)
    
    map_results = metrics.compute_map()
    
    print(f"\nmAP@0.5: {map_results.get('map_50', map_results.get('mAP_50', 0.0)):.4f}")
    print(f"mAP@0.5:0.95: {map_results.get('map_50_95', map_results.get('mAP_50_95', 0.0)):.4f}")
    
    # 检查每个类别的统计
    ap_per_class = map_results.get('ap_per_class', {})
    print(f"\n每类AP@0.5:")
    for class_name, ap in sorted(ap_per_class.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {class_name}: {ap:.4f}")

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
    
    debug_map_calculation(model, val_loader, device, num_samples=3)


