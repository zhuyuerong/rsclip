#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估修复后的CAM生成器
检查CAM质量和类别映射正确性
"""

import torch
import numpy as np
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets.dior_detection import get_detection_dataloader
from models.surgery_cam_detector import create_surgery_cam_detector


def evaluate_cam_quality(model, dataloader, device, num_samples=50):
    """评估CAM质量和类别映射"""
    print("\n" + "="*80)
    print("评估修复后的CAM生成器")
    print("="*80)
    
    model.eval()
    
    all_cam_values = []
    all_inside_responses = []
    all_outside_responses = []
    all_contrasts = []
    
    correct_mappings = 0
    total_mappings = 0
    
    # 每个类别的统计
    class_stats = {}
    
    with torch.no_grad():
        count = 0
        for batch in dataloader:
            if count >= num_samples:
                break
            
            images = batch['images'].to(device)
            text_queries = batch['text_queries']
            gt_boxes = batch['boxes']
            gt_labels = batch['labels']
            
            # 获取CAM
            cam, aux = model.surgery_aaf(images, text_queries)
            
            B, C, H, W = cam.shape
            
            for b in range(B):
                boxes = gt_boxes[b]
                labels = gt_labels[b]
                
                if len(boxes) == 0:
                    continue
                
                for k, (box, label) in enumerate(zip(boxes, labels)):
                    label = label.item() if isinstance(label, torch.Tensor) else int(label)
                    class_name = text_queries[label] if label < len(text_queries) else f"class_{label}"
                    
                    # 初始化类别统计
                    if class_name not in class_stats:
                        class_stats[class_name] = {
                            'total': 0,
                            'correct': 0,
                            'contrasts': [],
                            'inside_responses': [],
                            'outside_responses': []
                        }
                    
                    # 框内响应
                    xmin, ymin, xmax, ymax = box
                    i_min = max(0, int(ymin * H))
                    i_max = min(H - 1, int(ymax * H))
                    j_min = max(0, int(xmin * W))
                    j_max = min(W - 1, int(xmax * W))
                    
                    if i_max >= i_min and j_max >= j_min:
                        # 该GT框对应的类别CAM
                        target_cam = cam[b, label, i_min:i_max+1, j_min:j_max+1]
                        target_response = target_cam.mean().item()
                        
                        # 所有类别的CAM在该框内的响应
                        all_responses = []
                        for c in range(C):
                            cam_c = cam[b, c, i_min:i_max+1, j_min:j_max+1]
                            response = cam_c.mean().item()
                            all_responses.append((c, response))
                        
                        # 排序
                        all_responses.sort(key=lambda x: x[1], reverse=True)
                        
                        # 检查是否正确
                        total_mappings += 1
                        class_stats[class_name]['total'] += 1
                        
                        is_correct = (all_responses[0][0] == label)
                        if is_correct:
                            correct_mappings += 1
                            class_stats[class_name]['correct'] += 1
                        
                        # 统计框内外响应
                        cam_class = cam[b, label].cpu().numpy()
                        mask_out = np.ones((H, W), dtype=bool)
                        mask_out[i_min:i_max+1, j_min:j_max+1] = False
                        outside_cam = cam_class[mask_out]
                        outside_mean = outside_cam.mean()
                        
                        contrast = target_response / outside_mean if outside_mean > 0 else 0
                        
                        all_inside_responses.append(target_response)
                        all_outside_responses.append(outside_mean)
                        all_contrasts.append(contrast)
                        
                        class_stats[class_name]['contrasts'].append(contrast)
                        class_stats[class_name]['inside_responses'].append(target_response)
                        class_stats[class_name]['outside_responses'].append(outside_mean)
                
                # 统计CAM值
                all_cam_values.extend(cam[b].cpu().flatten().tolist())
            
            count += 1
    
    # 整体统计
    print("\n" + "="*80)
    print("整体CAM质量统计")
    print("="*80)
    print(f"  样本数: {count}")
    print(f"  GT框总数: {total_mappings}")
    print(f"  CAM值范围: [{np.min(all_cam_values):.6f}, {np.max(all_cam_values):.6f}]")
    print(f"  CAM值均值: {np.mean(all_cam_values):.6f}")
    print(f"  CAM值中位数: {np.median(all_cam_values):.6f}")
    print(f"  CAM值标准差: {np.std(all_cam_values):.6f}")
    
    if all_inside_responses:
        print(f"\n  框内外响应统计:")
        print(f"    框内平均响应: {np.mean(all_inside_responses):.6f} ± {np.std(all_inside_responses):.6f}")
        print(f"    框外平均响应: {np.mean(all_outside_responses):.6f} ± {np.std(all_outside_responses):.6f}")
        print(f"    平均对比度: {np.mean(all_contrasts):.6f} ± {np.std(all_contrasts):.6f}")
        print(f"    对比度 > 1.5 的比例: {(np.array(all_contrasts) > 1.5).sum() / len(all_contrasts) * 100:.1f}%")
        print(f"    对比度 > 2.0 的比例: {(np.array(all_contrasts) > 2.0).sum() / len(all_contrasts) * 100:.1f}%")
        print(f"    对比度 > 3.0 的比例: {(np.array(all_contrasts) > 3.0).sum() / len(all_contrasts) * 100:.1f}%")
    
    print(f"\n  类别映射正确率: {correct_mappings}/{total_mappings} ({correct_mappings/total_mappings*100:.1f}%)")
    
    # 每个类别的统计
    print("\n" + "="*80)
    print("每个类别的CAM质量统计")
    print("="*80)
    print(f"{'类别':<25} {'正确率':<10} {'平均对比度':<15} {'框内响应':<15} {'框外响应':<15}")
    print("-" * 80)
    
    for class_name in sorted(class_stats.keys()):
        stats = class_stats[class_name]
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total'] * 100
            avg_contrast = np.mean(stats['contrasts']) if stats['contrasts'] else 0
            avg_inside = np.mean(stats['inside_responses']) if stats['inside_responses'] else 0
            avg_outside = np.mean(stats['outside_responses']) if stats['outside_responses'] else 0
            print(f"{class_name:<25} {accuracy:>6.1f}%    {avg_contrast:>8.3f}        {avg_inside:>8.4f}        {avg_outside:>8.4f}")
    
    return {
        'total_mappings': total_mappings,
        'correct_mappings': correct_mappings,
        'accuracy': correct_mappings / total_mappings if total_mappings > 0 else 0,
        'avg_contrast': np.mean(all_contrasts) if all_contrasts else 0,
        'contrast_above_1.5': (np.array(all_contrasts) > 1.5).sum() / len(all_contrasts) * 100 if all_contrasts else 0,
        'contrast_above_2.0': (np.array(all_contrasts) > 2.0).sum() / len(all_contrasts) * 100 if all_contrasts else 0,
        'class_stats': class_stats
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='评估修复后的CAM生成器')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='模型checkpoint路径')
    parser.add_argument('--config', type=str, default='configs/surgery_cam_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--num-samples', type=int, default=50,
                       help='评估的样本数量')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 加载模型
    surgery_checkpoint = config.get('surgery_clip_checkpoint', 
                                   'checkpoints/RemoteCLIP-ViT-B-32.pt')
    if not Path(surgery_checkpoint).is_absolute():
        project_root = Path(__file__).parent.parent.parent.parent
        surgery_checkpoint = project_root / surgery_checkpoint
        surgery_checkpoint = str(surgery_checkpoint)
    
    print(f"加载SurgeryCLIP权重: {surgery_checkpoint}")
    model = create_surgery_cam_detector(
        surgery_clip_checkpoint=surgery_checkpoint,
        num_classes=config.get('num_classes', 20),
        cam_resolution=config.get('cam_resolution', 7),
        upsample_cam=config.get('upsample_cam', False),
        device=device
    )
    
    # 加载checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.box_head.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ 已加载checkpoint: {args.checkpoint}")
    
    # 检查CAMGenerator是否使用了CLIP投影
    if hasattr(model.surgery_aaf.cam_generator, 'use_clip_projection'):
        print(f"✅ CAMGenerator使用CLIP投影: {model.surgery_aaf.cam_generator.use_clip_projection}")
    else:
        print(f"⚠️  CAMGenerator可能未使用CLIP投影")
    
    # 加载数据
    dataloader = get_detection_dataloader(
        root=config.get('dataset_root'),
        split='test',
        batch_size=1,
        num_workers=2,
        image_size=config.get('image_size', 224),
        augment=False
    )
    
    # 评估CAM质量
    results = evaluate_cam_quality(model, dataloader, device, num_samples=args.num_samples)
    
    # 保存结果
    output_dir = Path(config.get('output_dir', 'outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / 'fixed_cam_evaluation.txt'
    
    with open(results_file, 'w') as f:
        f.write("修复后的CAM评估结果\n")
        f.write("="*80 + "\n\n")
        f.write(f"类别映射正确率: {results['accuracy']*100:.1f}%\n")
        f.write(f"平均对比度: {results['avg_contrast']:.4f}\n")
        f.write(f"对比度>1.5的比例: {results['contrast_above_1.5']:.1f}%\n")
        f.write(f"对比度>2.0的比例: {results['contrast_above_2.0']:.1f}%\n")
        f.write("\n每个类别的统计:\n")
        for class_name in sorted(results['class_stats'].keys()):
            stats = results['class_stats'][class_name]
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total'] * 100
                avg_contrast = np.mean(stats['contrasts']) if stats['contrasts'] else 0
                f.write(f"  {class_name}: 正确率={accuracy:.1f}%, 平均对比度={avg_contrast:.3f}\n")
    
    print(f"\n✅ 评估结果已保存到: {results_file}")


if __name__ == '__main__':
    main()


