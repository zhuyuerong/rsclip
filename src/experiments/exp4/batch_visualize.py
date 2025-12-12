#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量生成可视化：为10张不同的图像生成可视化
确保数据全面（不同类别、不同场景）
"""

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免GUI问题

import torch
import sys
import json
from pathlib import Path
import argparse
import yaml
from PIL import Image
import numpy as np
from torchvision import transforms
import random

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets.dior_detection import get_detection_dataloader, DIORDetectionDataset
from models.surgery_cam_detector import create_surgery_cam_detector
from visualize_with_real_results import (
    load_image, visualize_comprehensive_results,
    create_metrics_comparison_table, create_seen_unseen_comparison,
    DIOR_CLASSES
)


def find_diverse_images(dataset, num_images=10):
    """
    找到多样化的图像（不同类别、不同场景）
    
    Args:
        dataset: DIORDetectionDataset实例
        num_images: 需要的图像数量
    
    Returns:
        List of (image_id, classes_in_image) tuples
    """
    # 统计每个类别的图像
    class_to_images = {i: [] for i in range(len(DIOR_CLASSES))}
    
    print("分析数据集，寻找多样化图像...")
    for idx in range(min(len(dataset), 1000)):  # 只检查前1000张以加快速度
        try:
            sample = dataset[idx]
            image_id = dataset.image_ids[idx]
            labels = sample['labels'].numpy()
            
            # 记录包含的类别
            unique_classes = np.unique(labels)
            for cls_id in unique_classes:
                class_to_images[int(cls_id)].append((image_id, unique_classes))
        except Exception as e:
            continue
    
    # 选择策略：优先选择包含不同类别的图像
    selected = []
    used_image_ids = set()
    
    # 1. 优先选择包含多个类别的图像
    print("选择包含多个类别的图像...")
    for image_id, classes in sorted(class_to_images[0], key=lambda x: len(x[1]), reverse=True):
        if len(selected) >= num_images:
            break
        if image_id not in used_image_ids:
            selected.append((image_id, classes))
            used_image_ids.add(image_id)
    
    # 2. 如果还不够，随机选择不同类别的图像
    if len(selected) < num_images:
        print("补充随机图像...")
        all_images = []
        for cls_id, images in class_to_images.items():
            for image_id, classes in images:
                if image_id not in used_image_ids:
                    all_images.append((image_id, classes))
        
        random.shuffle(all_images)
        for image_id, classes in all_images:
            if len(selected) >= num_images:
                break
            if image_id not in used_image_ids:
                selected.append((image_id, classes))
                used_image_ids.add(image_id)
    
    # 3. 如果还不够，从所有图像中随机选择
    if len(selected) < num_images:
        print("从所有图像中随机选择...")
        remaining = num_images - len(selected)
        all_ids = [img_id for img_id in dataset.image_ids if img_id not in used_image_ids]
        random.shuffle(all_ids)
        for image_id in all_ids[:remaining]:
            selected.append((image_id, []))
    
    return selected[:num_images]


def get_image_path(image_id, dataset_root, split='test'):
    """获取图像完整路径"""
    if dataset_root is None:
        return None
    
    dataset_root = Path(dataset_root)
    image_path = dataset_root / 'images' / split / f'{image_id}.jpg'
    
    # 如果不存在，尝试其他可能的路径
    if not image_path.exists():
        # 尝试trainval
        image_path = dataset_root / 'images' / 'trainval' / f'{image_id}.jpg'
    
    # 如果还是不存在，尝试从dataset对象获取
    if not image_path.exists() and hasattr(dataset, 'images_dir'):
        image_path = dataset.images_dir / f'{image_id}.jpg'
    
    return image_path if image_path.exists() else None


def main():
    parser = argparse.ArgumentParser(description='批量生成可视化（10张图像）')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='SurgeryCAM checkpoint路径')
    parser.add_argument('--config', type=str, default='configs/surgery_cam_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--results', type=str, default='outputs/evaluation_results.json',
                       help='评估结果JSON文件路径')
    parser.add_argument('--output-dir', type=str, default='outputs/visualizations',
                       help='输出目录')
    parser.add_argument('--num-images', type=int, default=10,
                       help='要生成的图像数量')
    parser.add_argument('--dataset-root', type=str, default=None,
                       help='数据集根目录')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== 1. 加载配置 =====
    config_path = Path(__file__).parent / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # ===== 2. 加载数据集并找到多样化图像 =====
    print("\n" + "=" * 80)
    print("加载数据集并寻找多样化图像...")
    print("=" * 80)
    
    # 使用DataLoader自动查找数据集路径
    try:
        loader = get_detection_dataloader(
            root=args.dataset_root or config.get('dataset_root'),
            split='test',
            batch_size=1,
            num_workers=0,
            shuffle=False
        )
        dataset = loader.dataset
        # 从dataset获取实际的数据集根目录
        dataset_root = dataset.root if hasattr(dataset, 'root') else None
        if dataset_root:
            dataset_root = str(dataset_root)
        print(f"✅ 通过DataLoader加载成功，数据集根目录: {dataset_root}")
    except Exception as e:
        print(f"❌ DataLoader失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 找到多样化图像
    selected_images = find_diverse_images(dataset, args.num_images)
    print(f"\n✅ 选择了 {len(selected_images)} 张图像")
    
    # ===== 3. 加载模型 =====
    print("\n" + "=" * 80)
    print("加载模型...")
    print("=" * 80)
    
    surgery_checkpoint = config.get('surgery_clip_checkpoint',
                                   'checkpoints/RemoteCLIP-ViT-B-32.pt')
    if not Path(surgery_checkpoint).is_absolute():
        project_root = Path(__file__).parent.parent.parent.parent
        surgery_checkpoint = project_root / surgery_checkpoint
        surgery_checkpoint = str(surgery_checkpoint)
    
    model = create_surgery_cam_detector(
        surgery_clip_checkpoint=surgery_checkpoint,
        num_classes=config.get('num_classes', 20),
        cam_resolution=config.get('cam_resolution', 7),
        upsample_cam=config.get('upsample_cam', False),
        device=device
    )
    
    checkpoint_path = Path(__file__).parent / args.checkpoint
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.box_head.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 已加载checkpoint: {args.checkpoint}")
    else:
        print(f"⚠️  Checkpoint不存在: {checkpoint_path}")
        return
    
    model.eval()
    
    # ===== 4. 加载评估结果 =====
    results_path = Path(args.results)
    eval_results = {}
    if results_path.exists():
        with open(results_path, 'r') as f:
            eval_results = json.load(f)
        print(f"✅ 加载评估结果: {results_path}")
    else:
        print(f"⚠️  评估结果文件不存在: {results_path}")
    
    # ===== 5. 为每张图像生成可视化 =====
    print("\n" + "=" * 80)
    print(f"开始生成 {len(selected_images)} 张图像的可视化...")
    print("=" * 80)
    
    text_queries = DIOR_CLASSES.copy()
    success_count = 0
    
    for idx, (image_id, classes_in_image) in enumerate(selected_images):
        print(f"\n[{idx+1}/{len(selected_images)}] 处理图像: {image_id}")
        
        # 获取图像路径
        image_path = get_image_path(image_id, dataset_root, 'test')
        if image_path is None:
            image_path = get_image_path(image_id, dataset_root, 'trainval')
        
        # 如果还是None，尝试直接从dataset获取
        if image_path is None and hasattr(dataset, 'images_dir'):
            image_path = dataset.images_dir / f'{image_id}.jpg'
        
        if image_path is None:
            print(f"  ⚠️  图像不存在，跳过: {image_id}")
            continue
        
        try:
            # 加载图像
            image_tensor, image_pil, original_size = load_image(
                str(image_path),
                image_size=config.get('image_size', 224)
            )
            image_tensor = image_tensor.to(device)
            
            # 确定要可视化的类别（优先选择图像中存在的类别）
            class_idx = None
            if len(classes_in_image) > 0:
                # 选择第一个存在的类别
                class_idx = int(classes_in_image[0])
                class_name = DIOR_CLASSES[class_idx] if class_idx < len(DIOR_CLASSES) else None
                print(f"  类别: {class_name} (索引: {class_idx})")
            else:
                # 如果没有类别信息，使用ship作为默认
                if 'ship' in DIOR_CLASSES:
                    class_idx = DIOR_CLASSES.index('ship')
                    print(f"  使用默认类别: ship")
            
            # 推理
            with torch.no_grad():
                outputs = model(image_tensor, text_queries)
                cam = outputs['cam'][0]  # [C, H, W]
                
                detections = model.inference(
                    image_tensor, text_queries,
                    conf_threshold=config.get('conf_threshold', 0.3),
                    nms_threshold=config.get('nms_threshold', 0.5)
                )
                detections = detections[0]  # 单张图像的检测结果
            
            # 转换检测结果格式
            detections_list = []
            if isinstance(detections, dict):
                # 如果detections是字典格式
                boxes = detections.get('boxes', torch.zeros((0, 4)))
                labels = detections.get('labels', torch.zeros((0,), dtype=torch.int64))
                scores = detections.get('scores', torch.zeros((0,)))
                class_names_list = detections.get('class_names', [])
                
                for i in range(len(boxes)):
                    detections_list.append({
                        'box': boxes[i],
                        'label': labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i],
                        'score': scores[i].item() if isinstance(scores[i], torch.Tensor) else scores[i],
                        'class_name': class_names_list[i] if i < len(class_names_list) else text_queries[labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i]]
                    })
            elif isinstance(detections, list):
                # 如果detections是列表格式
                detections_list = detections
            
            print(f"  检测到 {len(detections_list)} 个目标")
            
            # 生成可视化
            output_path = output_dir / f"{image_id}_comprehensive.jpg"
            visualize_comprehensive_results(
                image_pil, cam, detections_list, text_queries,
                class_idx=class_idx,
                output_path=str(output_path)
            )
            
            success_count += 1
            print(f"  ✅ 已保存: {output_path}")
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ===== 6. 生成指标对比图（只生成一次） =====
    if eval_results:
        print("\n生成指标对比图...")
        metrics_comparison = {}
        
        if 'SurgeryCAM' in eval_results:
            metrics_comparison['SurgeryCAM'] = {
                'AP@0.5': eval_results['SurgeryCAM']['AP@0.5'],
                'AP@0.5:0.95': eval_results['SurgeryCAM']['AP@0.5:0.95']
            }
        else:
            metrics_comparison['SurgeryCAM'] = {'AP@0.5': 0.0, 'AP@0.5:0.95': 0.0}
        
        if 'OWL-ViT-style' in eval_results:
            metrics_comparison['OWL-ViT-style'] = {
                'AP@0.5': eval_results['OWL-ViT-style']['AP@0.5'],
                'AP@0.5:0.95': eval_results['OWL-ViT-style']['AP@0.5:0.95']
            }
        else:
            metrics_comparison['OWL-ViT-style'] = {'AP@0.5': 0.0, 'AP@0.5:0.95': 0.0}
        
        metrics_comparison['CAM-only'] = {
            'AP@0.5': eval_results.get('CAM-only', {}).get('AP@0.5', 0.0),
            'AP@0.5:0.95': eval_results.get('CAM-only', {}).get('AP@0.5:0.95', 0.0)
        }
        
        metrics_path = output_dir / "metrics_comparison.jpg"
        create_metrics_comparison_table(metrics_comparison, str(metrics_path))
        
        # Seen/Unseen对比
        if 'SurgeryCAM_seen' in eval_results and 'SurgeryCAM_unseen' in eval_results:
            seen_metrics = {
                'AP@0.5': eval_results['SurgeryCAM_seen']['AP@0.5'],
                'AP@0.5:0.95': eval_results['SurgeryCAM'].get('AP@0.5:0.95', 0.0)
            }
            unseen_metrics = {
                'AP@0.5': eval_results['SurgeryCAM_unseen']['AP@0.5'],
                'AP@0.5:0.95': eval_results['SurgeryCAM'].get('AP@0.5:0.95', 0.0)
            }
            
            seen_unseen_path = output_dir / "seen_unseen_comparison.jpg"
            create_seen_unseen_comparison(seen_metrics, unseen_metrics, str(seen_unseen_path))
    
    print("\n" + "=" * 80)
    print("✅ 批量可视化完成！")
    print("=" * 80)
    print(f"成功处理: {success_count}/{len(selected_images)} 张图像")
    print(f"输出目录: {output_dir}")
    print(f"  - {success_count} 张综合可视化图像")
    if eval_results:
        print(f"  - metrics_comparison.jpg")
        print(f"  - seen_unseen_comparison.jpg")


if __name__ == '__main__':
    main()

