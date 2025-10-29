#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment1 在 mini_dataset 上的完整评估（含mAP）
基于RemoteCLIP的区域提议+分类方法
"""

import torch
import sys
import time
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
import json

sys.path.append('..')
from inference.model_loader import ModelLoader

# DIOR类别
DIOR_CLASSES = [
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
    'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'
]


def calculate_iou_np(box1, box2):
    """计算单个框的IoU (xyxy格式)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-6)


def parse_xml(xml_path):
    """解析XML标注"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    boxes = []
    labels = []
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in DIOR_CLASSES:
            continue
        
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(DIOR_CLASSES.index(name))
    
    return np.array(boxes), np.array(labels)


def generate_sliding_windows(img_w, img_h, window_sizes=[100, 200, 400], stride=50):
    """生成滑动窗口"""
    windows = []
    
    for size in window_sizes:
        for y in range(0, img_h - size + 1, stride):
            for x in range(0, img_w - size + 1, stride):
                windows.append([x, y, x + size, y + size])
    
    # 添加全图
    windows.append([0, 0, img_w, img_h])
    
    return np.array(windows)


def nms(boxes, scores, threshold=0.5):
    """非极大值抑制"""
    if len(boxes) == 0:
        return []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]
    
    return keep


def calculate_map(all_predictions, all_targets, iou_threshold=0.5):
    """计算mAP"""
    num_classes = 20
    aps = []
    class_results = {}
    
    for cls in range(num_classes):
        # 收集该类的预测和GT
        cls_pred_boxes = []
        cls_pred_scores = []
        cls_gt_boxes = []
        
        for preds, targets in zip(all_predictions, all_targets):
            # 预测
            for pred in preds:
                if pred['label'] == cls:
                    cls_pred_boxes.append(pred['box'])
                    cls_pred_scores.append(pred['score'])
            
            # GT
            for box, label in zip(targets['boxes'], targets['labels']):
                if label == cls:
                    cls_gt_boxes.append(box)
        
        num_gt = len(cls_gt_boxes)
        if num_gt == 0:
            class_results[DIOR_CLASSES[cls]] = {'AP': 0.0, 'GT': 0, 'pred': 0}
            continue
        
        if len(cls_pred_boxes) == 0:
            class_results[DIOR_CLASSES[cls]] = {'AP': 0.0, 'GT': num_gt, 'pred': 0, 'TP': 0}
            continue
        
        # 按分数排序
        indices = np.argsort(cls_pred_scores)[::-1]
        cls_pred_boxes = [cls_pred_boxes[i] for i in indices]
        cls_pred_scores = [cls_pred_scores[i] for i in indices]
        
        # 计算TP/FP
        tp = np.zeros(len(cls_pred_boxes))
        gt_matched = [False] * len(cls_gt_boxes)
        
        for pred_idx, pred_box in enumerate(cls_pred_boxes):
            max_iou = 0
            max_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(cls_gt_boxes):
                if gt_matched[gt_idx]:
                    continue
                
                iou = calculate_iou_np(pred_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            
            if max_iou >= iou_threshold:
                tp[pred_idx] = 1
                gt_matched[max_gt_idx] = True
        
        # 计算precision和recall
        累计tp = np.cumsum(tp)
        累计fp = np.cumsum(1 - tp)
        
        recall = 累计tp / num_gt
        precision = 累计tp / (累计tp + 累计fp)
        
        # 计算AP (11-point)
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11
        
        aps.append(ap)
        class_results[DIOR_CLASSES[cls]] = {
            'AP': ap,
            'GT': num_gt,
            'pred': len(cls_pred_boxes),
            'TP': int(累计tp[-1])
        }
    
    return np.mean(aps) if len(aps) > 0 else 0.0, class_results


def main():
    print("=" * 70)
    print("Experiment1 在 Mini Dataset 上的评估 (含mAP)")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n设备: {device}")
    
    # 加载模型
    print("\n加载RemoteCLIP模型...")
    model_loader = ModelLoader(model_name='RN50', device=device)
    model, preprocess, tokenizer = model_loader.load_model()
    
    # 提取文本特征
    text_features = model_loader.encode_text_batch(DIOR_CLASSES)
    print(f"  文本特征: {text_features.shape}")
    
    # 加载mini_dataset
    print("\n加载mini_dataset...")
    root_dir = Path('../datasets/mini_dataset')
    images_dir = root_dir / 'images'
    annos_dir = root_dir / 'annotations'
    splits_dir = root_dir / 'splits'
    
    # 读取test split
    test_file = splits_dir / 'test.txt'
    if test_file.exists():
        with open(test_file) as f:
            test_ids = [line.strip() for line in f if line.strip()]
    else:
        test_ids = [img.stem for img in sorted(images_dir.glob('DIOR_*.jpg'))]
    
    print(f"  测试集: {len(test_ids)} 张图")
    
    # 评估
    print("\n开始推理...")
    all_predictions = []
    all_targets = []
    
    for img_id in tqdm(test_ids):
        img_path = images_dir / f'{img_id}.jpg'
        xml_path = annos_dir / f'{img_id}.xml'
        
        if not img_path.exists() or not xml_path.exists():
            continue
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        
        # 编码图像
        image_features = model_loader.encode_image(image)
        
        # 计算相似度
        similarities = (image_features @ text_features.T).cpu().numpy()[0]
        
        # 生成滑动窗口
        windows = generate_sliding_windows(orig_w, orig_h, window_sizes=[100, 200, 400], stride=100)
        
        # 对每个窗口检测
        predictions = []
        for window in windows:
            x1, y1, x2, y2 = window
            
            # 裁剪区域
            cropped = image.crop((x1, y1, x2, y2))
            
            # 编码裁剪区域
            crop_features = model_loader.encode_image(cropped)
            crop_sim = (crop_features @ text_features.T).cpu().numpy()[0]
            
            # 找到最高分类别
            max_cls = crop_sim.argmax()
            max_score = crop_sim[max_cls]
            
            if max_score > 0.20:  # 阈值
                predictions.append({
                    'box': window.tolist(),
                    'score': float(max_score),
                    'label': int(max_cls)
                })
        
        # NMS
        if len(predictions) > 0:
            boxes = np.array([p['box'] for p in predictions])
            scores = np.array([p['score'] for p in predictions])
            labels = np.array([p['label'] for p in predictions])
            
            # 对每个类别分别NMS
            final_preds = []
            for cls in range(20):
                cls_mask = labels == cls
                if cls_mask.any():
                    cls_boxes = boxes[cls_mask]
                    cls_scores = scores[cls_mask]
                    
                    keep = nms(cls_boxes, cls_scores, threshold=0.5)
                    
                    for idx in keep:
                        final_preds.append({
                            'box': cls_boxes[idx].tolist(),
                            'score': float(cls_scores[idx]),
                            'label': int(cls)
                        })
            
            all_predictions.append(final_preds)
        else:
            all_predictions.append([])
        
        # 解析GT
        gt_boxes, gt_labels = parse_xml(xml_path)
        all_targets.append({
            'boxes': gt_boxes,
            'labels': gt_labels
        })
    
    # 计算mAP
    print("\n" + "=" * 70)
    print("计算mAP@50...")
    print("=" * 70)
    
    map_50, class_results = calculate_map(all_predictions, all_targets, iou_threshold=0.5)
    
    print(f"\nmAP@50: {map_50*100:.2f}%")
    print(f"\n各类别AP:")
    
    for cls_name, result in sorted(class_results.items(), key=lambda x: x[1]['AP'], reverse=True):
        if result['GT'] > 0:
            print(f"  {cls_name:20s}: AP={result['AP']*100:5.1f}% | GT={result['GT']:2d} | 预测={result['pred']:3d} | 匹配={result.get('TP', 0):2.0f}")
    
    # 保存结果
    results = {
        'model': 'Experiment1 (RemoteCLIP RN50)',
        'mAP@50': float(map_50),
        'class_results': {k: {kk: float(vv) if isinstance(vv, (int, float, np.number)) else vv 
                              for kk, vv in v.items()} 
                         for k, v in class_results.items()},
        'total_test_images': len(test_ids),
        'total_gt': sum([len(t['labels']) for t in all_targets]),
        'total_predictions': sum([len(p) for p in all_predictions])
    }
    
    Path('outputs').mkdir(exist_ok=True)
    with open('outputs/exp1_mini_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果保存到: outputs/exp1_mini_results.json")
    
    print(f"\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print(f"  mAP@50:    {map_50*100:.2f}%")
    print(f"  测试图片:  {len(test_ids)}")
    print(f"  总GT框:    {sum([len(t['labels']) for t in all_targets])}")
    print(f"  总预测框:  {sum([len(p) for p in all_predictions])}")


if __name__ == '__main__':
    main()

