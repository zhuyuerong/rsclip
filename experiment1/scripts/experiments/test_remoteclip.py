#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RemoteCLIP测试脚本
用于验证RemoteCLIP模型是否正常工作
"""

import torch
import open_clip
from PIL import Image
import os

def test_remoteclip():
    """测试RemoteCLIP模型加载和推理"""
    print("=" * 60)
    print("RemoteCLIP 测试脚本")
    print("=" * 60)
    
    # 检查CUDA是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 检查模型文件是否存在
    model_name = 'RN50'  # 使用最小的模型进行测试
    checkpoint_path = f"checkpoints/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-{model_name}.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 模型文件不存在: {checkpoint_path}")
        print("请先下载模型文件，参考运行指南中的说明")
        return False
    
    try:
        # 加载模型
        print(f"正在加载 {model_name} 模型...")
        model, _, preprocess = open_clip.create_model_and_transforms(model_name)
        tokenizer = open_clip.get_tokenizer(model_name)
        
        # 加载预训练权重
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        message = model.load_state_dict(ckpt)
        print(f"模型加载状态: {message}")
        
        model = model.to(device).eval()
        print("✅ 模型加载成功！")
        
        # 测试文本查询
        text_queries = [
            "A busy airport with many airplanes.", 
            "Satellite view of a university.", 
            "A building next to a lake.", 
            "Many people in a stadium.", 
            "A cute cat",
        ]
        
        print("\n正在测试文本编码...")
        text = tokenizer(text_queries)
        text_features = model.encode_text(text.to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)
        print("✅ 文本编码成功！")
        
        # 测试图像编码（如果有测试图像）
        if os.path.exists("assets/airport.jpg"):
            print("\n正在测试图像编码...")
            image = preprocess(Image.open("assets/airport.jpg")).unsqueeze(0)
            image_features = model.encode_image(image.to(device))
            image_features /= image_features.norm(dim=-1, keepdim=True)
            print("✅ 图像编码成功！")
            
            # 计算相似度
            print("\n正在计算图像-文本相似度...")
            with torch.no_grad():
                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]
            
            print("\n预测结果:")
            for query, prob in zip(text_queries, text_probs):
                print(f"{query:<40} {prob * 100:5.1f}%")
        else:
            print("⚠️  未找到测试图像 assets/airport.jpg，跳过图像测试")
        
        print("\n✅ RemoteCLIP测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    test_remoteclip()
