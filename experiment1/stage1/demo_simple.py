#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RemoteCLIP 增强示例
使用机场图片进行图像-文本匹配演示
支持：全图匹配 + 多种区域采样策略
"""

import torch
import open_clip
from PIL import Image
import os
import numpy as np
import argparse

# 导入采样策略模块
from sampling import sample_regions

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='RemoteCLIP 区域采样演示')
    parser.add_argument('--strategy', type=str, default='multi_threshold_saliency',
                        choices=['layered', 'pyramid', 'multi_threshold_saliency'],
                        help='采样策略: layered(分层), pyramid(金字塔), multi_threshold_saliency(多阈值)')
    parser.add_argument('--model', type=str, default='RN50',
                        choices=['RN50', 'ViT-B-32', 'ViT-L-14'],
                        help='模型选择')
    parser.add_argument('--image', type=str, default='assets/airport.jpg',
                        help='图像路径')
    parser.add_argument('--max-regions', type=int, default=50,
                        help='最大区域数')
    parser.add_argument('--top-k', type=int, default=10,
                        help='分析前K个重要区域')
    parser.add_argument('--no-region-sampling', action='store_true',
                        help='禁用区域采样，只进行全图匹配')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("RemoteCLIP 增强示例 - 图像-文本匹配")
    print("=" * 70)
    
    # 1. 检查环境
    print(f"\n📋 环境信息:")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    print(f"   OpenCLIP版本: {open_clip.__version__}")
    print(f"\n📋 运行配置:")
    print(f"   模型: {args.model}")
    print(f"   图像: {args.image}")
    print(f"   采样策略: {args.strategy}")
    print(f"   最大区域数: {args.max_regions}")
    print(f"   分析区域数: {args.top_k}")
    
    # 2. 加载模型
    model_name = args.model
    print(f"\n🔄 正在加载 {model_name} 模型...")
    
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    
    # 加载预训练权重
    checkpoint_path = f"checkpoints/RemoteCLIP-{model_name}.pt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ 错误: 找不到模型文件 {checkpoint_path}")
        print("请确保权重文件在checkpoints目录下")
        return
    
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    message = model.load_state_dict(ckpt)
    print(f"✅ 模型加载状态: {message}")
    
    # 将模型移到GPU并设置为评估模式
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"✅ 模型已加载到 {device} 设备")
    
    # 3. 加载图片
    image_path = args.image
    print(f"\n🖼️  加载图片: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ 错误: 找不到图片文件 {image_path}")
        return
    
    pil_image = Image.open(image_path)
    cv_image = np.array(pil_image)  # 转换为numpy数组供采样使用
    print(f"✅ 图片尺寸: {pil_image.size}")
    
    # 4. 全图匹配
    print(f"\n" + "=" * 70)
    print("📊 步骤1: 全图匹配")
    print("=" * 70)
    
    text_queries = [
        "A busy airport with many airplanes.",
        "Satellite view of Hohai university.",
        "A building next to a lake.",
        "Many people in a stadium.",
        "A cute cat",
    ]
    
    print(f"\n文本查询列表:")
    for i, query in enumerate(text_queries, 1):
        print(f"   {i}. {query}")
    
    text = tokenizer(text_queries)
    image_tensor = preprocess(pil_image).unsqueeze(0)
    
    with torch.no_grad():
        # 编码图像和文本
        image_features = model.encode_image(image_tensor.to(device))
        text_features = model.encode_text(text.to(device))
        
        # L2归一化
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # 计算相似度概率
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]
    
    # 显示结果
    print(f"\n🎯 全图匹配结果:")
    for query, prob in zip(text_queries, text_probs):
        bar_length = int(prob * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"{query:<45} {prob * 100:5.1f}% {bar}")
    
    best_match_idx = text_probs.argmax()
    print(f"\n🏆 最佳匹配: {text_queries[best_match_idx]} ({text_probs[best_match_idx] * 100:.2f}%)")
    
    # 5. 区域采样匹配
    if not args.no_region_sampling:
        print(f"\n" + "=" * 70)
        print("📊 步骤2: 区域采样匹配")
        print("=" * 70)
        
        # 使用选定的策略进行区域采样
        regions = sample_regions(
            cv_image, 
            strategy=args.strategy,
            max_regions=args.max_regions
        )
        
        if len(regions) == 0:
            print("❌ 未找到任何区域")
        else:
            # 定义区域级别的查询
            region_queries = [
                "airport", "runway", "airplane", "aircraft", 
                "building", "terminal", "parking lot", "road",
                "vegetation", "water"
            ]
            
            print(f"\n分析前 {min(args.top_k, len(regions))} 个重要区域...")
            
            region_text = tokenizer(region_queries)
            with torch.no_grad():
                region_text_features = model.encode_text(region_text.to(device))
                region_text_features /= region_text_features.norm(dim=-1, keepdim=True)
            
            # 对每个区域进行匹配
            for idx, region in enumerate(regions[:args.top_k]):
                x1, y1, x2, y2 = region['bbox']
                
                # 确保坐标在有效范围内
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(cv_image.shape[1], x2), min(cv_image.shape[0], y2)
                
                # 裁剪区域
                cropped = cv_image[y1:y2, x1:x2]
                if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
                    continue
                
                cropped_pil = Image.fromarray(cropped)
                cropped_tensor = preprocess(cropped_pil).unsqueeze(0)
                
                with torch.no_grad():
                    crop_features = model.encode_image(cropped_tensor.to(device))
                    crop_features /= crop_features.norm(dim=-1, keepdim=True)
                    
                    probs = (100.0 * crop_features @ region_text_features.T).softmax(dim=-1).cpu().numpy()[0]
                
                # 获取top3结果
                top3_indices = probs.argsort()[-3:][::-1]
                
                print(f"\n区域 {idx+1} [{x1},{y1},{x2},{y2}] - 优先级: {region.get('priority', region.get('threshold', 'N/A'))}")
                print(f"  Top3匹配:")
                for rank, top_idx in enumerate(top3_indices, 1):
                    print(f"    {rank}. {region_queries[top_idx]}: {probs[top_idx]*100:.1f}%")
    
    # 6. 完成
    print(f"\n" + "=" * 70)
    print(f"✅ 演示完成！")
    print("=" * 70)
    print(f"\n💡 使用提示:")
    print(f"   python demo_simple.py --help                    # 查看所有参数")
    print(f"   python demo_simple.py --strategy pyramid        # 使用金字塔采样")
    print(f"   python demo_simple.py --strategy layered        # 使用分层采样")
    print(f"   python demo_simple.py --model ViT-B-32          # 使用ViT-B-32模型")
    print(f"   python demo_simple.py --image your_image.jpg    # 使用自定义图片")
    print(f"   python demo_simple.py --no-region-sampling      # 只进行全图匹配")

if __name__ == "__main__":
    main()

