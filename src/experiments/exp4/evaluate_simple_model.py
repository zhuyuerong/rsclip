#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估简化SurgeryCAM模型
"""

import torch
import sys
import json
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.simple_surgery_cam_detector import create_simple_surgery_cam_detector
from datasets.dior_detection import get_detection_dataloader
from utils.metrics import DetectionMetrics
from utils.class_split import SEEN_CLASSES, UNSEEN_CLASSES, ALL_CLASSES


def evaluate_model(model, dataloader, device, conf_threshold=0.1, nms_threshold=0.5):
    """评估模型"""
    model.eval()
    metrics = DetectionMetrics(num_classes=20)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中"):
            images = batch['images'].to(device)
            text_queries = batch['text_queries']
            
            # 推理
            detections = model.inference(
                images, text_queries,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold
            )
            
            # detections是一个列表，每个元素是一个列表的字典（每个图像一个列表）
            # 转换为metrics格式
            pred_boxes = []
            pred_labels = []
            pred_scores = []
            
            for img_detections in detections:
                # img_detections是一个列表，包含该图像的所有检测结果
                if len(img_detections) > 0:
                    # 收集所有检测框
                    boxes_list = []
                    labels_list = []
                    scores_list = []
                    
                    for det in img_detections:
                        boxes_list.append(det['box'].unsqueeze(0))
                        labels_list.append(det['class'])
                        scores_list.append(det['score'])
                    
                    pred_boxes.append(torch.cat(boxes_list, dim=0).to(device))
                    pred_labels.append(torch.tensor(labels_list, dtype=torch.long, device=device))
                    pred_scores.append(torch.tensor(scores_list, device=device))
                else:
                    # 如果没有检测结果，创建空的tensor
                    pred_boxes.append(torch.zeros((0, 4), device=device))
                    pred_labels.append(torch.zeros((0,), dtype=torch.long, device=device))
                    pred_scores.append(torch.zeros((0,), device=device))
            
            # 更新metrics
            metrics.update(
                pred_boxes=pred_boxes,
                pred_labels=pred_labels,
                pred_scores=pred_scores,
                gt_boxes=batch['boxes'],
                gt_labels=batch['labels'],
                image_ids=batch.get('image_ids', None)
            )
    
    # 计算mAP
    map_results = metrics.compute_map()
    
    return map_results


def main():
    parser = argparse.ArgumentParser(description='评估简化SurgeryCAM模型')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_simple_model.pth',
                       help='模型checkpoint路径')
    parser.add_argument('--config', type=str, default='configs/surgery_cam_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--output', type=str, default='outputs/simple_model_evaluation.json',
                       help='输出结果文件')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='评估样本数量（None表示全部）')
    
    args = parser.parse_args()
    
    # 加载配置
    config_path = Path(__file__).parent / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = config.get('device', 'cuda')
    print(f"使用设备: {device}")
    
    # 创建模型
    surgery_checkpoint = config.get('surgery_clip_checkpoint', 
                                   'checkpoints/RemoteCLIP-ViT-B-32.pt')
    if not Path(surgery_checkpoint).is_absolute():
        project_root = Path(__file__).parent.parent.parent.parent
        surgery_checkpoint = project_root / surgery_checkpoint
        surgery_checkpoint = str(surgery_checkpoint)
    
    print("创建简化SurgeryCAM模型...")
    model = create_simple_surgery_cam_detector(
        surgery_clip_checkpoint=surgery_checkpoint,
        num_classes=config.get('num_classes', 20),
        cam_resolution=config.get('cam_resolution', 7),
        upsample_cam=config.get('upsample_cam', False),
        device=device,
        unfreeze_cam_last_layer=True
    )
    
    # 加载checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 只加载可训练部分的权重（BoxHead和CAMGenerator的learnable_proj）
    model_state = checkpoint['model_state_dict']
    model_dict = model.state_dict()
    
    # 过滤出可训练的权重
    trainable_state = {}
    for key, value in model_state.items():
        if key in model_dict and ('box_head' in key or 'cam_generator.learnable_proj' in key):
            trainable_state[key] = value
    
    model_dict.update(trainable_state)
    model.load_state_dict(model_dict, strict=False)
    
    print(f"✅ 已加载checkpoint: {args.checkpoint}")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    print(f"   已加载 {len(trainable_state)} 个可训练参数")
    
    # 加载验证集
    val_loader = get_detection_dataloader(
        root=config.get('dataset_root'),
        split='test',
        batch_size=config.get('batch_size', 8),
        num_workers=config.get('num_workers', 4),
        image_size=config.get('image_size', 224),
        augment=False,
        train_only_seen=False  # 评估所有类别
    )
    
    # 如果指定了样本数量，创建一个限制版本的迭代器
    if args.num_samples:
        original_dataset_size = len(val_loader.dataset)
        limited_size = min(args.num_samples, original_dataset_size)
        print(f"限制评估样本数量: {limited_size}/{original_dataset_size}")
        
        # 创建一个包装器来限制迭代次数
        class LimitedDataLoader:
            def __init__(self, dataloader, max_samples):
                self.dataloader = dataloader
                self.max_samples = max_samples
                self.count = 0
            
            def __iter__(self):
                self.count = 0
                for batch in self.dataloader:
                    if self.count >= self.max_samples:
                        break
                    self.count += len(batch['images'])
                    yield batch
            
            def __len__(self):
                return min(len(self.dataloader), (self.max_samples + self.dataloader.batch_size - 1) // self.dataloader.batch_size)
        
        val_loader = LimitedDataLoader(val_loader, args.num_samples)
    
    # 评估
    print("\n开始评估...")
    map_results = evaluate_model(
        model, val_loader, device,
        conf_threshold=config.get('conf_threshold', 0.1),
        nms_threshold=config.get('nms_threshold', 0.5)
    )
    
    # 提取mAP结果
    map_50 = map_results.get('mAP@0.5', map_results.get('map_50', 0.0))
    map_50_95 = map_results.get('mAP@0.5:0.95', map_results.get('map_50_95', 0.0))
    per_class_ap = map_results.get('per_class_AP@0.5', map_results.get('per_class_ap', [0.0] * 20))
    
    # 计算seen和unseen类别的mAP
    seen_indices = [ALL_CLASSES.index(c) for c in SEEN_CLASSES if c in ALL_CLASSES]
    unseen_indices = [ALL_CLASSES.index(c) for c in UNSEEN_CLASSES if c in ALL_CLASSES]
    
    seen_map = [per_class_ap[i] for i in seen_indices if i < len(per_class_ap)]
    unseen_map = [per_class_ap[i] for i in unseen_indices if i < len(per_class_ap)]
    
    results = {
        'checkpoint': args.checkpoint,
        'mAP@0.5': map_50,
        'mAP@0.5:0.95': map_50_95,
        'seen_mAP@0.5': sum(seen_map) / len(seen_map) if seen_map else 0.0,
        'unseen_mAP@0.5': sum(unseen_map) / len(unseen_map) if unseen_map else 0.0,
        'per_class_ap': {
            ALL_CLASSES[i]: per_class_ap[i] 
            for i in range(min(len(ALL_CLASSES), len(per_class_ap)))
        }
    }
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印结果
    print("\n" + "=" * 80)
    print("评估结果")
    print("=" * 80)
    print(f"mAP@0.5: {results['mAP@0.5']:.4f}")
    print(f"mAP@0.5:0.95: {results['mAP@0.5:0.95']:.4f}")
    print(f"Seen mAP@0.5: {results['seen_mAP@0.5']:.4f}")
    print(f"Unseen mAP@0.5: {results['unseen_mAP@0.5']:.4f}")
    print(f"\n结果已保存到: {output_path}")


if __name__ == '__main__':
    main()

