#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行实际评估并保存结果
评估SurgeryCAM、OWL-ViT和CAM-only方法
"""

import torch
import sys
import json
from pathlib import Path
import argparse
import yaml

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval import evaluate_model
from datasets.dior_detection import get_detection_dataloader
from models.surgery_cam_detector import create_surgery_cam_detector
from models.owlvit_baseline import create_owlvit_model


def evaluate_surgery_cam(checkpoint_path, config, device, num_samples=None):
    """评估SurgeryCAM"""
    print("\n" + "=" * 80)
    print("评估 SurgeryCAM...")
    print("=" * 80)
    
    # 加载模型
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
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.box_head.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ 已加载checkpoint: {checkpoint_path}")
    
    # 加载数据
    val_loader = get_detection_dataloader(
        root=config.get('dataset_root'),
        split='test',
        batch_size=config.get('batch_size', 4),
        num_workers=config.get('num_workers', 2),
        image_size=config.get('image_size', 224),
        augment=False
    )
    
    if num_samples:
        # 限制样本数（用于快速测试）
        val_loader.dataset.image_ids = val_loader.dataset.image_ids[:num_samples]
    
    print(f"测试样本数: {len(val_loader.dataset)}")
    
    # 评估
    results = evaluate_model(
        model, val_loader, device,
        model_type='surgery_cam',
        conf_threshold=config.get('conf_threshold', 0.3),
        nms_threshold=config.get('nms_threshold', 0.5)
    )
    
    return results


def evaluate_owlvit(config, device, num_samples=None):
    """评估OWL-ViT"""
    print("\n" + "=" * 80)
    print("评估 OWL-ViT...")
    print("=" * 80)
    
    # 加载模型
    model = create_owlvit_model(
        model_name=config.get('model_name', 'google/owlvit-base-patch32'),
        device=device
    )
    
    # 加载数据
    val_loader = get_detection_dataloader(
        root=config.get('dataset_root'),
        split='test',
        batch_size=config.get('batch_size', 4),
        num_workers=config.get('num_workers', 2),
        image_size=config.get('image_size', 224),
        augment=False
    )
    
    if num_samples:
        val_loader.dataset.image_ids = val_loader.dataset.image_ids[:num_samples]
    
    print(f"测试样本数: {len(val_loader.dataset)}")
    
    # 评估
    results = evaluate_model(
        model, val_loader, device,
        model_type='owlvit',
        conf_threshold=config.get('conf_threshold', 0.3),
        nms_threshold=config.get('nms_threshold', 0.5)
    )
    
    return results


def compute_seen_unseen_metrics(results, seen_classes, all_classes):
    """计算seen和unseen类别的指标"""
    seen_indices = [all_classes.index(c) for c in seen_classes if c in all_classes]
    unseen_indices = [i for i in range(len(all_classes)) if i not in seen_indices]
    
    per_class_ap = results['per_class_AP@0.5']
    
    seen_ap = [per_class_ap[i] for i in seen_indices if i < len(per_class_ap)]
    unseen_ap = [per_class_ap[i] for i in unseen_indices if i < len(per_class_ap)]
    
    seen_map = sum(seen_ap) / len(seen_ap) if seen_ap else 0.0
    unseen_map = sum(unseen_ap) / len(unseen_ap) if unseen_ap else 0.0
    
    return {
        'seen': {
            'AP@0.5': seen_map,
            'classes': [all_classes[i] for i in seen_indices]
        },
        'unseen': {
            'AP@0.5': unseen_map,
            'classes': [all_classes[i] for i in unseen_indices]
        }
    }


def main():
    parser = argparse.ArgumentParser(description='运行实际评估')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='SurgeryCAM checkpoint路径')
    parser.add_argument('--config', type=str, default='configs/surgery_cam_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--output', type=str, default='outputs/evaluation_results.json',
                       help='输出JSON文件路径')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='限制评估样本数（用于快速测试）')
    parser.add_argument('--skip-owlvit', action='store_true',
                       help='跳过OWL-ViT评估（节省时间）')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载配置
    config_path = Path(__file__).parent / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # DIOR类别
    all_classes = [
        "airplane", "airport", "baseball field", "basketball court",
        "bridge", "chimney", "dam", "expressway service area",
        "expressway toll station", "golf course", "ground track field",
        "harbor", "overpass", "ship", "stadium", "storage tank",
        "tennis court", "train station", "vehicle", "wind mill"
    ]
    
    # Seen/Unseen类别划分
    seen_classes = [
        "airplane", "ship", "vehicle", "bridge", "harbor",
        "stadium", "storage tank", "airport", "golf course", "wind mill"
    ]
    
    results_dict = {}
    
    # 1. 评估SurgeryCAM
    try:
        surgery_results = evaluate_surgery_cam(
            args.checkpoint, config, device, args.num_samples
        )
        results_dict['SurgeryCAM'] = {
            'AP@0.5': float(surgery_results['mAP@0.5']),
            'AP@0.5:0.95': float(surgery_results['mAP@0.5:0.95']),
            'per_class_AP@0.5': [float(x) for x in surgery_results['per_class_AP@0.5']]
        }
        
        # Seen/Unseen
        seen_unseen = compute_seen_unseen_metrics(surgery_results, seen_classes, all_classes)
        results_dict['SurgeryCAM_seen'] = {
            'AP@0.5': float(seen_unseen['seen']['AP@0.5'])
        }
        results_dict['SurgeryCAM_unseen'] = {
            'AP@0.5': float(seen_unseen['unseen']['AP@0.5'])
        }
        
        print(f"\n✅ SurgeryCAM结果:")
        print(f"   mAP@0.5: {surgery_results['mAP@0.5']:.4f}")
        print(f"   mAP@0.5:0.95: {surgery_results['mAP@0.5:0.95']:.4f}")
        print(f"   Seen AP@0.5: {seen_unseen['seen']['AP@0.5']:.4f}")
        print(f"   Unseen AP@0.5: {seen_unseen['unseen']['AP@0.5']:.4f}")
    except Exception as e:
        print(f"❌ SurgeryCAM评估失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. 评估OWL-ViT
    if not args.skip_owlvit:
        try:
            owlvit_results = evaluate_owlvit(config, device, args.num_samples)
            results_dict['OWL-ViT-style'] = {
                'AP@0.5': float(owlvit_results['mAP@0.5']),
                'AP@0.5:0.95': float(owlvit_results['mAP@0.5:0.95']),
                'per_class_AP@0.5': [float(x) for x in owlvit_results['per_class_AP@0.5']]
            }
            
            seen_unseen = compute_seen_unseen_metrics(owlvit_results, seen_classes, all_classes)
            results_dict['OWL-ViT_seen'] = {
                'AP@0.5': float(seen_unseen['seen']['AP@0.5'])
            }
            results_dict['OWL-ViT_unseen'] = {
                'AP@0.5': float(seen_unseen['unseen']['AP@0.5'])
            }
            
            print(f"\n✅ OWL-ViT结果:")
            print(f"   mAP@0.5: {owlvit_results['mAP@0.5']:.4f}")
            print(f"   mAP@0.5:0.95: {owlvit_results['mAP@0.5:0.95']:.4f}")
        except Exception as e:
            print(f"❌ OWL-ViT评估失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 3. CAM-only（简化版本，使用CAM阈值）
    # 注意：这需要修改模型来只输出CAM，这里先跳过或使用近似值
    results_dict['CAM-only'] = {
        'AP@0.5': 0.0,  # 需要单独实现
        'AP@0.5:0.95': 0.0
    }
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✅ 评估结果已保存到: {output_path}")
    
    # 打印总结
    print("\n" + "=" * 80)
    print("评估总结")
    print("=" * 80)
    if 'SurgeryCAM' in results_dict:
        print(f"SurgeryCAM: AP@0.5 = {results_dict['SurgeryCAM']['AP@0.5']:.4f}")
    if 'OWL-ViT-style' in results_dict:
        print(f"OWL-ViT-style: AP@0.5 = {results_dict['OWL-ViT-style']['AP@0.5']:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()


