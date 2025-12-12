#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用实际评估结果进行可视化
从evaluation_results.json读取真实指标
"""

import torch
import sys
import json
from pathlib import Path
import argparse
import yaml
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from torchvision import transforms
import cv2

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.surgery_cam_detector import create_surgery_cam_detector
from visualize_results import (
    load_image, visualize_comprehensive_results,
    create_metrics_comparison_table, create_seen_unseen_comparison,
    DIOR_CLASSES
)


def load_evaluation_results(results_path):
    """加载评估结果"""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results


def main():
    parser = argparse.ArgumentParser(description='使用实际评估结果进行可视化')
    parser.add_argument('--image', type=str, required=True,
                       help='输入图像路径')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='SurgeryCAM checkpoint路径')
    parser.add_argument('--config', type=str, default='configs/surgery_cam_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--results', type=str, default='outputs/evaluation_results.json',
                       help='评估结果JSON文件路径')
    parser.add_argument('--class-name', type=str, default=None,
                       help='要可视化的类别名称')
    parser.add_argument('--conf-threshold', type=float, default=0.3,
                       help='置信度阈值')
    parser.add_argument('--nms-threshold', type=float, default=0.5,
                       help='NMS IoU阈值')
    parser.add_argument('--output-dir', type=str, default='outputs/visualizations',
                       help='输出目录')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== 1. 加载评估结果 =====
    results_path = Path(args.results)
    if results_path.exists():
        print(f"✅ 加载评估结果: {results_path}")
        eval_results = load_evaluation_results(results_path)
    else:
        print(f"⚠️  评估结果文件不存在: {results_path}")
        print("   将使用默认值（示例数据）")
        eval_results = {}
    
    # ===== 2. 加载配置 =====
    config_path = Path(__file__).parent / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # ===== 3. 加载模型 =====
    print("\n" + "=" * 80)
    print("加载 SurgeryCAM 模型...")
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
    
    model.eval()
    
    # ===== 4. 加载图像 =====
    print("\n加载图像...")
    image_tensor, image_pil, original_size = load_image(
        args.image,
        image_size=config.get('image_size', 224)
    )
    image_tensor = image_tensor.to(device)
    
    # ===== 5. 准备文本查询 =====
    text_queries = DIOR_CLASSES.copy()
    
    # 确定要可视化的类别
    class_idx = None
    if args.class_name:
        if args.class_name in text_queries:
            class_idx = text_queries.index(args.class_name)
            print(f"可视化类别: {args.class_name} (索引: {class_idx})")
    
    # ===== 6. 推理 =====
    print("\n运行推理...")
    with torch.no_grad():
        outputs = model(image_tensor, text_queries)
        cam = outputs['cam'][0]  # [C, H, W]
        
        detections = model.inference(
            image_tensor, text_queries,
            conf_threshold=args.conf_threshold,
            nms_threshold=args.nms_threshold
        )
        detections = detections[0]
    
    print(f"检测到 {len(detections)} 个目标")
    
    # ===== 7. 生成综合可视化 =====
    print("\n生成可视化...")
    image_stem = Path(args.image).stem
    output_path = output_dir / f"{image_stem}_comprehensive.jpg"
    
    visualize_comprehensive_results(
        image_pil, cam, detections, text_queries,
        class_idx=class_idx,
        output_path=str(output_path)
    )
    
    # ===== 8. 生成指标对比（使用实际结果） =====
    print("\n生成指标对比...")
    
    # 从评估结果构建指标字典
    metrics_comparison = {}
    
    if 'SurgeryCAM' in eval_results:
        metrics_comparison['SurgeryCAM'] = {
            'AP@0.5': eval_results['SurgeryCAM']['AP@0.5'],
            'AP@0.5:0.95': eval_results['SurgeryCAM']['AP@0.5:0.95']
        }
    else:
        print("⚠️  SurgeryCAM结果不存在，使用默认值")
        metrics_comparison['SurgeryCAM'] = {
            'AP@0.5': 0.0,
            'AP@0.5:0.95': 0.0
        }
    
    if 'OWL-ViT-style' in eval_results:
        metrics_comparison['OWL-ViT-style'] = {
            'AP@0.5': eval_results['OWL-ViT-style']['AP@0.5'],
            'AP@0.5:0.95': eval_results['OWL-ViT-style']['AP@0.5:0.95']
        }
    else:
        print("⚠️  OWL-ViT结果不存在，使用默认值")
        metrics_comparison['OWL-ViT-style'] = {
            'AP@0.5': 0.0,
            'AP@0.5:0.95': 0.0
        }
    
    # CAM-only（如果评估结果中没有，使用默认值）
    if 'CAM-only' in eval_results and eval_results['CAM-only']['AP@0.5'] > 0:
        metrics_comparison['CAM-only'] = {
            'AP@0.5': eval_results['CAM-only']['AP@0.5'],
            'AP@0.5:0.95': eval_results['CAM-only']['AP@0.5:0.95']
        }
    else:
        print("⚠️  CAM-only结果不存在，使用默认值")
        metrics_comparison['CAM-only'] = {
            'AP@0.5': 0.0,
            'AP@0.5:0.95': 0.0
        }
    
    metrics_path = output_dir / f"{image_stem}_metrics_comparison.jpg"
    create_metrics_comparison_table(metrics_comparison, str(metrics_path))
    
    # ===== 9. Seen/Unseen对比 =====
    if 'SurgeryCAM_seen' in eval_results and 'SurgeryCAM_unseen' in eval_results:
        seen_metrics = {
            'AP@0.5': eval_results['SurgeryCAM_seen']['AP@0.5'],
            'AP@0.5:0.95': eval_results['SurgeryCAM'].get('AP@0.5:0.95', 0.0)  # 近似值
        }
        unseen_metrics = {
            'AP@0.5': eval_results['SurgeryCAM_unseen']['AP@0.5'],
            'AP@0.5:0.95': eval_results['SurgeryCAM'].get('AP@0.5:0.95', 0.0)  # 近似值
        }
        
        seen_unseen_path = output_dir / f"{image_stem}_seen_unseen.jpg"
        create_seen_unseen_comparison(seen_metrics, unseen_metrics, str(seen_unseen_path))
    else:
        print("⚠️  Seen/Unseen结果不存在，跳过")
    
    print("\n" + "=" * 80)
    print("✅ 可视化完成！")
    print("=" * 80)
    print(f"输出目录: {output_dir}")
    print(f"  - {image_stem}_comprehensive.jpg")
    print(f"  - {image_stem}_metrics_comparison.jpg")
    if 'SurgeryCAM_seen' in eval_results:
        print(f"  - {image_stem}_seen_unseen.jpg")


if __name__ == '__main__':
    main()


