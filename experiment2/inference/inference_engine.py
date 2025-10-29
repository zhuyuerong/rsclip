#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理引擎

支持：
1. 单张图像 + 单个类别
2. 单张图像 + 多个类别
3. 批量图像 + 开放词汇
"""

import torch
import argparse
import cv2
import numpy as np
from PIL import Image
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ContextGuidedDetector
from .post_processor import PostProcessor


class InferenceEngine:
    """推理引擎 - 基于RemoteCLIP"""
    
    def __init__(
        self,
        model_name: str = 'RN50',
        pretrained_path: str = 'checkpoints/RemoteCLIP-RN50.pt',
        score_threshold: float = 0.5,
        nms_threshold: float = 0.7,
        device: str = 'cuda'
    ):
        """
        参数:
            model_name: RemoteCLIP模型名称
            pretrained_path: RemoteCLIP预训练权重路径
            score_threshold: 分数阈值
            nms_threshold: NMS阈值
            device: 计算设备
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        print(f"🔄 初始化推理引擎（RemoteCLIP-{model_name}）...")
        
        # 加载模型
        self.model = ContextGuidedDetector(
            model_name=model_name,
            pretrained_path=pretrained_path
        )
        self.model = self.model.to(self.device).eval()
        
        print(f"✅ 推理引擎初始化完成")
        
        # 后处理器
        self.post_processor = PostProcessor(
            score_threshold=score_threshold,
            nms_threshold=nms_threshold
        )
        
        # 预处理
        self.preprocess = self.model.image_encoder.preprocess
    
    @torch.no_grad()
    def infer_single(
        self,
        image_path: str,
        text_query: str
    ) -> dict:
        """
        单张图像 + 单个类别推理
        
        参数:
            image_path: 图像路径
            text_query: 文本查询，如 "airplane"
        
        返回:
            result: 检测结果字典
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # 推理
        outputs = self.model(image_tensor, [text_query])
        
        # 后处理
        results = self.post_processor(
            outputs['pred_boxes'],
            outputs['scores']
        )
        
        return results[0]
    
    @torch.no_grad()
    def infer_multi_class(
        self,
        image_path: str,
        text_queries: List[str]
    ) -> Dict[str, dict]:
        """
        单张图像 + 多个类别推理
        
        参数:
            image_path: 图像路径
            text_queries: 文本查询列表
        
        返回:
            results_dict: 每个类别的检测结果字典
        """
        results_dict = {}
        
        for text_query in text_queries:
            result = self.infer_single(image_path, text_query)
            results_dict[text_query] = result
        
        return results_dict
    
    def visualize_results(
        self,
        image_path: str,
        results: dict,
        text_query: str,
        output_path: str = None
    ):
        """
        可视化结果
        
        参数:
            image_path: 图像路径
            results: 检测结果
            text_query: 文本查询
            output_path: 输出路径
        """
        # 加载图像
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        
        boxes = results['boxes'].cpu().numpy()
        scores = results['scores'].cpu().numpy()
        
        # 绘制边界框
        for i, (box, score) in enumerate(zip(boxes, scores)):
            cx, cy, bw, bh = box
            
            # 转换到像素坐标
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            
            # 绘制矩形
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"{text_query}: {score:.2f}"
            cv2.putText(image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 保存或显示
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"结果已保存: {output_path}")
        
        return image


def main():
    """测试推理引擎"""
    parser = argparse.ArgumentParser(description='Experiment2 推理引擎')
    parser.add_argument('--image', type=str, default='assets/airport.jpg',
                        help='输入图像路径')
    parser.add_argument('--text', type=str, nargs='+', default=['airplane'],
                        help='文本查询')
    parser.add_argument('--model', type=str, default='RN50',
                        help='模型名称')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='分数阈值')
    parser.add_argument('--output', type=str, default=None,
                        help='输出路径')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Experiment2 推理引擎")
    print("=" * 70)
    
    # 创建推理引擎
    engine = InferenceEngine(
        model_name=args.model,
        score_threshold=args.threshold
    )
    
    # 推理
    if len(args.text) == 1:
        # 单类别推理
        result = engine.infer_single(args.image, args.text[0])
        
        print(f"\n检测结果 ({args.text[0]}):")
        print(f"  检测数: {result['num_detections']}")
        
        # 可视化
        if args.output:
            engine.visualize_results(args.image, result, args.text[0], args.output)
    
    else:
        # 多类别推理
        results_dict = engine.infer_multi_class(args.image, args.text)
        
        print("\n检测结果:")
        for text_query, result in results_dict.items():
            print(f"  {text_query}: {result['num_detections']} 个检测")
    
    print("\n✅ 推理完成！")


if __name__ == "__main__":
    main()

