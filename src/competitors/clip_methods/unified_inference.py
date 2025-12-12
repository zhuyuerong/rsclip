# -*- coding: utf-8 -*-
"""
统一推理脚本

支持所有CLIP方法（surgeryclip, declip, diffclip）的统一推理接口
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from PIL import Image

from .base_interface import BaseCLIPMethod
from .surgeryclip.model_wrapper import SurgeryCLIPWrapper
from .declip.model_wrapper import DeCLIPWrapper
from .diffclip.model_wrapper import DiffCLIPWrapper
from .eval import heatmap_to_bboxes, evaluate_bboxes_with_gt


def create_model(method: str, model_name: str = "ViT-B/32", 
                device: str = "cuda", **kwargs) -> BaseCLIPMethod:
    """
    创建CLIP模型（统一接口）
    
    Args:
        method: 方法名称 ('surgeryclip', 'declip', 'diffclip')
        model_name: 模型名称
        device: 设备
        **kwargs: 其他参数
    
    Returns:
        model: CLIP模型实例
    """
    if method == 'surgeryclip':
        return SurgeryCLIPWrapper(
            model_name=model_name,
            device=device,
            mode=kwargs.get('mode', 'full'),
            checkpoint_path=kwargs.get('checkpoint_path')
        )
    elif method == 'declip':
        return DeCLIPWrapper(
            model_name=model_name,
            device=device,
            checkpoint_path=kwargs.get('checkpoint_path'),
            mode=kwargs.get('mode', 'qq_vfm_distill')
        )
    elif method == 'diffclip':
        return DiffCLIPWrapper(
            model_name=model_name,
            device=device,
            checkpoint_path=kwargs.get('checkpoint_path')
        )
    else:
        raise ValueError(f"未知方法: {method}")


def inference_single_image(model: BaseCLIPMethod, image_path: str, 
                          text_queries: List[str], threshold: float = 0.5,
                          output_dir: Optional[str] = None) -> Dict:
    """
    单张图像推理
    
    Args:
        model: CLIP模型
        image_path: 图像路径
        text_queries: 文本查询列表
        threshold: 相似度阈值
        output_dir: 输出目录（可选）
    
    Returns:
        results: 推理结果
    """
    # 加载模型
    if model.model is None:
        model.load_model()
    
    # 推理
    results = model.inference(image_path, text_queries, threshold=threshold)
    
    # 保存结果
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存热图
        import matplotlib.pyplot as plt
        heatmap_path = output_dir / f"{Path(image_path).stem}_heatmap.png"
        plt.imsave(str(heatmap_path), results['heatmap'], cmap='jet')
        
        # 保存结果JSON
        results_json = {
            'image_path': str(image_path),
            'text_queries': text_queries,
            'bboxes': results['bboxes'],
            'similarities': results['similarities'].tolist(),
            'num_detections': len(results['bboxes'])
        }
        json_path = output_dir / f"{Path(image_path).stem}_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    return results


def inference_dataset(model: BaseCLIPMethod, dataset, text_queries: List[str],
                     threshold: float = 0.5, output_dir: Optional[str] = None) -> Dict:
    """
    数据集推理
    
    Args:
        model: CLIP模型
        dataset: 数据集
        text_queries: 文本查询列表
        threshold: 相似度阈值
        output_dir: 输出目录（可选）
    
    Returns:
        all_results: 所有结果
    """
    all_results = []
    all_predictions = []
    all_ground_truths = []
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        image = sample.get('image') or sample.get('image_path')
        
        # 推理
        results = model.inference(image, text_queries, threshold=threshold)
        all_results.append(results)
        
        # 收集预测和GT
        all_predictions.append(results['bboxes'])
        all_ground_truths.append(sample.get('gt_boxes', []))
    
    # 评估
    if len(all_ground_truths) > 0 and any(len(gt) > 0 for gt in all_ground_truths):
        eval_results = evaluate_bboxes_with_gt(
            all_predictions, all_ground_truths, iou_threshold=0.5
        )
    else:
        eval_results = {}
    
    # 保存结果
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary = {
            'num_images': len(all_results),
            'text_queries': text_queries,
            'threshold': threshold,
            'evaluation': eval_results,
            'per_image_results': [
                {
                    'image_idx': i,
                    'num_detections': len(r['bboxes']),
                    'similarities': r['similarities'].tolist()
                }
                for i, r in enumerate(all_results)
            ]
        }
        
        summary_path = output_dir / 'inference_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return {
        'all_results': all_results,
        'evaluation': eval_results
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='统一CLIP方法推理')
    parser.add_argument('--method', type=str, required=True,
                       choices=['surgeryclip', 'declip', 'diffclip'],
                       help='CLIP方法')
    parser.add_argument('--model-name', type=str, default='ViT-B/32',
                       help='模型名称')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='检查点路径')
    parser.add_argument('--image', type=str, default=None,
                       help='单张图像路径')
    parser.add_argument('--dataset', type=str, default=None,
                       help='数据集路径（待实现）')
    parser.add_argument('--text-queries', type=str, nargs='+', required=True,
                       help='文本查询列表')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='相似度阈值')
    parser.add_argument('--output-dir', type=str, default='outputs/inference',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备')
    parser.add_argument('--mode', type=str, default='full',
                       help='模式（surgeryclip: full/no_surgery, declip: qq_vfm_distill等）')
    
    args = parser.parse_args()
    
    # 创建模型
    model = create_model(
        method=args.method,
        model_name=args.model_name,
        device=args.device,
        checkpoint_path=args.checkpoint,
        mode=args.mode
    )
    
    # 推理
    if args.image:
        results = inference_single_image(
            model, args.image, args.text_queries,
            threshold=args.threshold, output_dir=args.output_dir
        )
        print(f"✅ 推理完成: {len(results['bboxes'])} 个检测框")
    elif args.dataset:
        # TODO: 实现数据集推理
        raise NotImplementedError("数据集推理待实现")
    else:
        raise ValueError("必须指定 --image 或 --dataset")


if __name__ == '__main__':
    main()

