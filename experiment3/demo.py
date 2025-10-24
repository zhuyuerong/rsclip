#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OVA-DETR 演示脚本

展示如何使用整个系统进行训练、推理和评估
"""

import torch
import argparse
from pathlib import Path

from config.default_config import DefaultConfig
from models.ova_detr import OVADETR
from utils.data_loader import DIOR_CLASSES


def demo_model_creation():
    """演示：创建模型"""
    
    print("=" * 70)
    print("演示1: 创建OVA-DETR模型")
    print("=" * 70)
    
    # 配置
    config = DefaultConfig()
    
    # 创建模型
    model = OVADETR(config)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型信息:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  冻结参数: {total_params - trainable_params:,}")
    
    print(f"\n配置:")
    print(f"  查询数量: {config.num_queries}")
    print(f"  解码器层数: {config.num_decoder_layers}")
    print(f"  模型维度: {config.d_model}")
    
    return model, config


def demo_forward_pass(model, config):
    """演示：前向传播"""
    
    print("\n" + "=" * 70)
    print("演示2: 前向传播")
    print("=" * 70)
    
    # 准备输入
    batch_size = 2
    images = torch.randn(batch_size, 3, 800, 800)
    
    # 提取文本特征
    with torch.no_grad():
        text_features = model.backbone.forward_text(DIOR_CLASSES)
    
    print(f"\n输入:")
    print(f"  图像: {images.shape}")
    print(f"  文本特征: {text_features.shape}")
    print(f"  类别数: {len(DIOR_CLASSES)}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        outputs = model(images, text_features)
    
    print(f"\n输出:")
    print(f"  分类logits: {outputs['pred_logits'].shape}")
    print(f"  边界框: {outputs['pred_boxes'].shape}")
    print(f"  增强文本: {outputs['text_features'].shape}")
    
    # 解释输出形状
    num_layers, B, num_queries, num_classes = outputs['pred_logits'].shape
    print(f"\n输出解释:")
    print(f"  解码器层数: {num_layers}")
    print(f"  批次大小: {B}")
    print(f"  查询数量: {num_queries}")
    print(f"  类别数: {num_classes}")


def demo_inference():
    """演示：推理流程"""
    
    print("\n" + "=" * 70)
    print("演示3: 推理流程")
    print("=" * 70)
    
    print("\n推理步骤:")
    print("1. 加载训练好的模型")
    print("   engine = InferenceEngine(checkpoint_path='best.pth')")
    
    print("\n2. 对单张图像推理")
    print("   result = engine.predict_single('test.jpg')")
    
    print("\n3. 可视化结果")
    print("   vis_image = engine.visualize('test.jpg', result)")
    print("   vis_image.save('result.jpg')")
    
    print("\n结果格式:")
    print("  {")
    print("    'boxes': (N, 4),      # 边界框 [x1, y1, x2, y2]")
    print("    'scores': (N,),       # 置信度分数")
    print("    'labels': (N,),       # 类别索引")
    print("    'class_names': [...]  # 类别名称")
    print("  }")


def demo_training_workflow():
    """演示：训练流程"""
    
    print("\n" + "=" * 70)
    print("演示4: 训练流程")
    print("=" * 70)
    
    print("\n完整训练命令:")
    print("-" * 70)
    print("""
python train.py \\
  --data_dir ../datasets/DIOR \\
  --output_dir ./outputs \\
  --batch_size 8 \\
  --epochs 50 \\
  --lr 1e-4 \\
  --num_workers 8
    """)
    
    print("\n训练流程:")
    print("1. 加载DIOR数据集")
    print("2. 创建OVA-DETR模型")
    print("3. 提取文本特征（DIOR 20个类别）")
    print("4. 设置优化器（AdamW）")
    print("5. 循环训练：")
    print("   - 前向传播")
    print("   - 匈牙利匹配")
    print("   - 计算损失（变焦 + L1 + GIoU）")
    print("   - 反向传播")
    print("   - 更新参数")
    print("6. 保存检查点")
    
    print("\n输出:")
    print("  outputs/")
    print("  ├── checkpoints/")
    print("  │   ├── best.pth")
    print("  │   ├── latest.pth")
    print("  │   └── epoch_*.pth")
    print("  └── logs/")
    print("      └── tensorboard日志")


def demo_evaluation():
    """演示：评估流程"""
    
    print("\n" + "=" * 70)
    print("演示5: 评估流程")
    print("=" * 70)
    
    print("\n评估命令:")
    print("-" * 70)
    print("""
python evaluate.py \\
  --checkpoint outputs/checkpoints/best.pth \\
  --data_dir ../datasets/DIOR \\
  --output evaluation_results.json \\
  --iou_threshold 0.5
    """)
    
    print("\n评估指标:")
    print("1. mAP@0.5 - IoU阈值为0.5的平均精度")
    print("2. AP per class - 每个类别的平均精度")
    print("3. Precision / Recall - 精确率和召回率")
    
    print("\n评估结果示例:")
    print("-" * 70)
    print("""
mAP@0.5: 0.6542
评估类别数: 20/20

各类别AP:
  airplane                      : 0.7823
  ship                          : 0.7156
  harbor                        : 0.6891
  ...
    """)


def demo_dataset_info():
    """演示：数据集信息"""
    
    print("\n" + "=" * 70)
    print("演示6: DIOR数据集")
    print("=" * 70)
    
    print("\n数据集统计:")
    print(f"  类别数: {len(DIOR_CLASSES)}")
    print(f"  训练集: ~8,000张图片")
    print(f"  验证集: ~2,000张图片")
    print(f"  测试集: ~11,738张图片")
    
    print("\n类别列表:")
    for i, cls in enumerate(DIOR_CLASSES, 1):
        print(f"  {i:2d}. {cls}")
    
    print("\n数据格式:")
    print("  图像: JPG, 800×800")
    print("  标注: VOC XML, 水平边界框")
    print("  边界框: [xmin, ymin, xmax, ymax]")


def demo_architecture():
    """演示：模型架构"""
    
    print("\n" + "=" * 70)
    print("演示7: 模型架构")
    print("=" * 70)
    
    print("""
完整架构流程:

输入图像 (B, 3, 800, 800)
    ↓
[1] RemoteCLIP骨干网络 (冻结)
    ├─ 图像编码器: 提取多层级特征
    │  └─ layer2, layer3, layer4
    └─ 文本编码器: 提取文本特征
       └─ 20个类别文本
    ↓
[2] FPN特征金字塔
    ├─ 侧向连接 (1x1卷积)
    ├─ 自顶向下融合
    └─ 输出: 4层256维特征
    ↓
[3] 混合编码器
    ├─ 位置编码
    ├─ Transformer编码 (6层)
    └─ 全局特征建模
    ↓
[4] 文本-视觉融合
    ├─ 视觉增强文本 (VAT)
    │  └─ 交叉注意力: 文本 ← 视觉
    └─ 文本引导视觉
       └─ 交叉注意力: 视觉 ← 文本
    ↓
[5] Transformer解码器 (6层)
    ├─ 目标查询: 300个可学习查询
    ├─ 自注意力: 查询之间
    ├─ 交叉注意力: 查询 ← 视觉特征
    ├─ 交叉注意力: 查询 ← 文本特征
    └─ FFN: 特征变换
    ↓
[6] 检测头
    ├─ 分类头: 对比学习
    │  └─ 相似度: 查询特征 × 文本特征
    └─ 回归头: MLP
       └─ 输出: [cx, cy, w, h]
    ↓
输出:
  - 分类logits: (6, B, 300, 20)
  - 边界框: (6, B, 300, 4)
    """)


def main():
    """主函数"""
    
    print("\n" + "🎯" * 35)
    print("OVA-DETR with RemoteCLIP - 完整演示")
    print("🎯" * 35 + "\n")
    
    try:
        # 演示1: 创建模型
        model, config = demo_model_creation()
        
        # 演示2: 前向传播
        demo_forward_pass(model, config)
        
        # 演示3: 推理
        demo_inference()
        
        # 演示4: 训练
        demo_training_workflow()
        
        # 演示5: 评估
        demo_evaluation()
        
        # 演示6: 数据集
        demo_dataset_info()
        
        # 演示7: 架构
        demo_architecture()
        
        print("\n" + "=" * 70)
        print("✅ 演示完成！")
        print("=" * 70)
        
        print("\n下一步:")
        print("1. 运行快速启动脚本: bash quick_start.sh")
        print("2. 开始训练: python train.py --help")
        print("3. 查看文档: cat README.md")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

