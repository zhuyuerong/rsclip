#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment2 评估脚本
加载训练好的模型，计算mAP并可视化
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append('..')

from config.default_config import DefaultConfig
from stage1_encoder.clip_text_encoder import CLIPTextEncoder
from stage1_encoder.clip_image_encoder import CLIPImageEncoder
from datasets.mini_dataset.mini_dataset_loader import MiniDataset
import torchvision.transforms as T


def calculate_iou(box1, box2):
    """计算IoU (xyxy格式)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def calculate_ap(precisions, recalls):
    """计算AP（11点插值）"""
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return ap


def evaluate_classification(image_features, text_features, labels):
    """
    评估分类性能
    
    Args:
        image_features: [N, D] 图像特征
        text_features: [C, D] 文本特征  
        labels: [N] 真实标签
    
    Returns:
        accuracy, top5_accuracy
    """
    # 归一化
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # 计算相似度
    similarity = image_features @ text_features.T  # [N, C]
    
    # Top-1准确率
    pred_labels = similarity.argmax(dim=-1)
    accuracy = (pred_labels == labels).float().mean().item()
    
    # Top-5准确率
    top5_pred = similarity.topk(5, dim=-1)[1]
    top5_accuracy = sum([labels[i] in top5_pred[i] for i in range(len(labels))]) / len(labels)
    
    return accuracy, top5_accuracy


def visualize_predictions(image, predictions, ground_truth_label, class_names, save_path):
    """
    可视化预测结果（分类模型）
    
    Args:
        image: PIL Image
        predictions: list of dict with 'label', 'score'
        ground_truth_label: int, GT标签
        class_names: list of class names
        save_path: 保存路径
    """
    # 创建画布
    draw = ImageDraw.Draw(image)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    w, h = image.size
    
    # 绘制GT标签（顶部，绿色）
    gt_text = f"Ground Truth: {class_names[ground_truth_label]}"
    draw.rectangle([0, 0, w, 40], fill='green')
    draw.text((10, 10), gt_text, fill='white', font=font)
    
    # 绘制Top-3预测（底部）
    y_offset = h - 150
    draw.rectangle([0, y_offset, w, h], fill='black')
    draw.text((10, y_offset + 5), "Top-3 Predictions:", fill='white', font=font)
    
    for i, pred in enumerate(predictions[:3]):
        label = pred['label']
        score = pred['score']
        color = 'lime' if label == ground_truth_label else 'red'
        text = f"{i+1}. {class_names[label]}: {score:.2f}"
        draw.text((10, y_offset + 35 + i*30), text, fill=color, font=small_font)
    
    # 保存
    image.save(save_path)
    return save_path


def evaluate():
    print("=" * 70)
    print("Experiment2 评估")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    
    # 加载类别名称
    from utils.dataloader import DIOR_CLASSES
    
    # 加载checkpoint
    checkpoint_path = 'outputs/checkpoints/simple_epoch_10.pth'
    print(f"\n加载checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 创建模型
    print("\n创建模型...")
    text_encoder = CLIPTextEncoder('RN50', '../checkpoints/RemoteCLIP-RN50.pt').to(device)
    image_encoder = CLIPImageEncoder('RN50', '../checkpoints/RemoteCLIP-RN50.pt', freeze=False).to(device)
    
    # 加载权重
    text_encoder.load_state_dict(checkpoint['text_encoder'])
    image_encoder.load_state_dict(checkpoint['image_encoder'])
    
    text_encoder.eval()
    image_encoder.eval()
    
    print(f"✅ 模型加载成功 (Epoch {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f})")
    
    # 提取文本特征
    print("\n提取文本特征...")
    with torch.no_grad():
        text_features = text_encoder(DIOR_CLASSES).to(device)
    
    print(f"  文本特征: {text_features.shape}")
    
    # 加载测试数据
    print("\n加载测试数据...")
    test_dataset = MiniDataset('../datasets/mini_dataset', 'test', transforms=None)
    print(f"  测试集: {len(test_dataset)} 张图")
    
    # 评估分类性能
    print("\n评估分类性能...")
    
    all_image_features = []
    all_labels = []
    all_images = []
    all_targets = []
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="提取特征"):
            image, target = test_dataset[idx]
            
            # 保存原图用于可视化
            all_images.append(image.copy())
            all_targets.append(target)
            
            # 转换图像
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # 提取特征
            _, image_feat = image_encoder(image_tensor)
            
            all_image_features.append(image_feat.squeeze(0))
            
            # 获取标签（使用第一个框的标签）
            if len(target['labels']) > 0:
                all_labels.append(target['labels'][0])
    
    # 转换为tensor
    all_image_features = torch.stack(all_image_features)
    all_labels = torch.tensor(all_labels).to(device)
    
    print(f"\n收集特征:")
    print(f"  图像特征: {all_image_features.shape}")
    print(f"  标签数量: {len(all_labels)}")
    
    # 计算分类准确率
    accuracy, top5_acc = evaluate_classification(all_image_features, text_features, all_labels)
    
    print(f"\n📊 分类性能:")
    print(f"  Top-1 准确率: {accuracy*100:.2f}%")
    print(f"  Top-5 准确率: {top5_acc*100:.2f}%")
    
    # 可视化前10张图片的预测
    print(f"\n可视化预测结果...")
    vis_dir = Path('outputs/visualizations')
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # 归一化特征用于预测
    image_features_norm = all_image_features / all_image_features.norm(dim=-1, keepdim=True)
    text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    similarity = image_features_norm @ text_features_norm.T
    
    for idx in range(min(10, len(all_images))):
        image = all_images[idx]
        gt_label = int(all_labels[idx].item())
        
        # 获取预测
        scores = similarity[idx]
        top_scores, top_labels = scores.topk(3)
        
        # 创建预测列表
        predictions = []
        for label, score in zip(top_labels, top_scores):
            predictions.append({
                'label': int(label),
                'score': float(score)
            })
        
        # 可视化
        save_path = visualize_predictions(
            image.copy(),
            predictions,
            gt_label,
            DIOR_CLASSES,
            vis_dir / f'test_{idx:03d}.jpg'
        )
    
    print(f"✅ 保存了 {min(10, len(all_images))} 张可视化图片")
    
    # 计算每个类别的准确率
    print(f"\n📋 各类别准确率:")
    pred_labels = similarity.argmax(dim=-1).cpu().numpy()
    true_labels = all_labels.cpu().numpy()
    
    class_accuracy = {}
    for class_id in range(len(DIOR_CLASSES)):
        mask = true_labels == class_id
        if mask.sum() > 0:
            acc = (pred_labels[mask] == true_labels[mask]).mean()
            class_accuracy[DIOR_CLASSES[class_id]] = acc
            print(f"  {DIOR_CLASSES[class_id]:20s}: {acc*100:.1f}% ({mask.sum()} samples)")
    
    # 保存结果
    results = {
        'checkpoint': checkpoint_path,
        'epoch': checkpoint['epoch'],
        'train_loss': checkpoint['loss'],
        'test_metrics': {
            'top1_accuracy': accuracy,
            'top5_accuracy': top5_acc,
            'num_test_samples': len(all_labels)
        },
        'class_accuracy': {k: float(v) for k, v in class_accuracy.items()}
    }
    
    results_file = 'outputs/evaluation_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 评估结果保存到: {results_file}")
    print(f"✅ 可视化结果保存到: {vis_dir}/")
    
    # 绘制混淆矩阵
    print(f"\n生成混淆矩阵...")
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(true_labels, pred_labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=DIOR_CLASSES, yticklabels=DIOR_CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Accuracy: {accuracy*100:.1f}%)')
    plt.tight_layout()
    plt.savefig(vis_dir / 'confusion_matrix.png', dpi=150)
    print(f"✅ 混淆矩阵保存到: {vis_dir}/confusion_matrix.png")
    
    print("\n" + "=" * 70)
    print("评估完成！")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    evaluate()

