#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment1 在 mini_dataset 上的评估

基于两阶段检测方法
"""

import torch
import sys
import time
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
from collections import defaultdict

sys.path.append('..')
from inference.model_loader import ModelLoader


class Experiment1Evaluator:
    """Experiment1 评估器"""
    
    def __init__(self, model_name='RN50', device='cuda'):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # 加载模型
        self.model_loader = ModelLoader(model_name=model_name, device=self.device)
        self.model, self.preprocess, self.tokenizer = self.model_loader.load_model()
        
        # DIOR类别
        self.classes = [
            'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
            'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
            'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
            'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'
        ]
        
        # 提取文本特征
        self.text_features = self.model_loader.encode_text_batch(self.classes)
    
    def load_mini_dataset(self, root_dir='datasets/mini_dataset'):
        """加载 mini_dataset"""
        
        root = Path(root_dir)
        images_dir = root / 'images'
        annos_dir = root / 'annotations'
        
        samples = []
        
        for img_file in sorted(images_dir.glob('DIOR_*.jpg')):
            img_id = img_file.stem
            xml_file = annos_dir / f'{img_id}.xml'
            
            if xml_file.exists():
                samples.append({
                    'image_path': img_file,
                    'xml_path': xml_file,
                    'image_id': img_id
                })
        
        print(f"加载了 {len(samples)} 个样本")
        return samples
    
    def parse_xml(self, xml_path):
        """解析XML标注"""
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        size = root.find('size')
        img_w = int(size.find('width').text) if size else 800
        img_h = int(size.find('height').text) if size else 800
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.classes:
                continue
            
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.classes.index(name))
        
        return np.array(boxes), np.array(labels)
    
    def detect_image(self, image_path):
        """检测单张图像"""
        
        image = Image.open(image_path).convert('RGB')
        orig_w, orig_h = image.size
        
        # 调整大小
        image_resized = image.resize((800, 800))
        
        # 编码图像
        image_features = self.model_loader.encode_image(image_resized)
        
        # 计算相似度
        similarities = (image_features @ self.text_features.T).cpu().numpy()[0]
        
        # 简单阈值检测（Experiment1是基于区域的，这里简化为全图）
        threshold = 0.15
        detected_classes = np.where(similarities > threshold)[0]
        
        # 生成边界框（简化：使用整图）
        predictions = []
        for cls_id in detected_classes:
            predictions.append({
                'box': [0, 0, orig_w, orig_h],  # 简化为全图
                'score': similarities[cls_id],
                'label': cls_id
            })
        
        return predictions
    
    def evaluate(self, samples):
        """评估数据集"""
        
        print("\n开始评估...")
        
        all_predictions = []
        all_targets = []
        
        start_time = time.time()
        
        for sample in samples:
            # 预测
            predictions = self.detect_image(sample['image_path'])
            
            # 目标
            boxes, labels = self.parse_xml(sample['xml_path'])
            
            all_predictions.append(predictions)
            all_targets.append({
                'boxes': boxes,
                'labels': labels
            })
        
        inference_time = time.time() - start_time
        
        # 计算指标（简化版）
        total_targets = sum(len(t['labels']) for t in all_targets)
        total_predictions = sum(len(p) for p in all_predictions)
        
        print(f"\n推理完成:")
        print(f"  总图片数: {len(samples)}")
        print(f"  总目标数: {total_targets}")
        print(f"  总预测数: {total_predictions}")
        print(f"  推理时间: {inference_time:.2f}秒")
        print(f"  FPS: {len(samples)/inference_time:.2f}")
        
        return {
            'inference_time': inference_time,
            'fps': len(samples) / inference_time,
            'num_targets': total_targets,
            'num_predictions': total_predictions
        }


def main():
    """主函数"""
    
    print("=" * 70)
    print("Experiment1 Mini Dataset 评估")
    print("=" * 70)
    
    # 创建评估器
    evaluator = Experiment1Evaluator(model_name='RN50')
    
    # 加载数据
    samples = evaluator.load_mini_dataset()
    
    # 评估
    results = evaluator.evaluate(samples)
    
    # 获取模型参数
    total_params = sum(p.numel() for p in evaluator.model.parameters())
    
    print(f"\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)
    print(f"模型参数: {total_params/1e6:.2f}M")
    print(f"推理时间: {results['inference_time']:.2f}秒")
    print(f"FPS: {results['fps']:.2f}")
    print(f"目标数: {results['num_targets']}")
    print(f"预测数: {results['num_predictions']}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()


