#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理工具：模型加载器
统一管理RemoteCLIP模型的加载和配置
"""

import torch
import open_clip
import os
from typing import Tuple, Optional
from PIL import Image


class ModelLoader:
    """模型加载器"""
    
    def __init__(self, model_name: str = 'RN50', device: str = 'cuda'):
        """
        初始化模型加载器
        
        参数:
            model_name: 模型名称
            device: 计算设备
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._loaded = False
    
    def load_model(self) -> Tuple[torch.nn.Module, callable, callable]:
        """
        加载RemoteCLIP模型
        
        返回:
            (model, preprocess, tokenizer)
        """
        if self._loaded:
            return self.model, self.preprocess, self.tokenizer
        
        print(f"🔄 加载RemoteCLIP模型: {self.model_name}")
        
        # 创建模型
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.model_name)
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        
        # 加载权重
        checkpoint_path = f"checkpoints/RemoteCLIP-{self.model_name}.pt"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"模型权重文件不存在: {checkpoint_path}")
        
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(ckpt)
        
        # 移动到设备并设置为评估模式
        self.model = self.model.to(self.device).eval()
        self._loaded = True
        
        print(f"✅ 模型已加载到 {self.device}")
        
        return self.model, self.preprocess, self.tokenizer
    
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        编码图像
        
        参数:
            image: PIL图像
        
        返回:
            图像特征张量
        """
        if not self._loaded:
            self.load_model()
        
        # 预处理图像
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # 编码
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features /= features.norm(dim=-1, keepdim=True)
        
        return features
    
    def encode_text(self, text: str) -> torch.Tensor:
        """
        编码文本
        
        参数:
            text: 文本字符串
        
        返回:
            文本特征张量
        """
        if not self._loaded:
            self.load_model()
        
        # 分词
        text_tokens = self.tokenizer([text]).to(self.device)
        
        # 编码
        with torch.no_grad():
            features = self.model.encode_text(text_tokens)
            features /= features.norm(dim=-1, keepdim=True)
        
        return features
    
    def encode_text_batch(self, texts: list) -> torch.Tensor:
        """
        批量编码文本
        
        参数:
            texts: 文本列表
        
        返回:
            文本特征张量
        """
        if not self._loaded:
            self.load_model()
        
        # 分词
        text_tokens = self.tokenizer(texts).to(self.device)
        
        # 编码
        with torch.no_grad():
            features = self.model.encode_text(text_tokens)
            features /= features.norm(dim=-1, keepdim=True)
        
        return features
    
    def compute_similarity(self, image_features: torch.Tensor, 
                          text_features: torch.Tensor) -> torch.Tensor:
        """
        计算图像-文本相似度
        
        参数:
            image_features: 图像特征
            text_features: 文本特征
        
        返回:
            相似度张量
        """
        with torch.no_grad():
            similarity = (image_features @ text_features.T).softmax(dim=-1)
        
        return similarity
    
    def get_model_info(self) -> dict:
        """
        获取模型信息
        
        返回:
            模型信息字典
        """
        return {
            'model_name': self.model_name,
            'device': self.device,
            'loaded': self._loaded,
            'parameters': sum(p.numel() for p in self.model.parameters()) if self._loaded else 0
        }


def create_model_loader(model_name: str = 'RN50', device: str = 'cuda') -> ModelLoader:
    """
    创建模型加载器
    
    参数:
        model_name: 模型名称
        device: 计算设备
    
    返回:
        模型加载器实例
    """
    return ModelLoader(model_name, device)


def main():
    """测试模型加载器"""
    print("=" * 70)
    print("测试模型加载器")
    print("=" * 70)
    
    # 创建模型加载器
    loader = create_model_loader('RN50')
    
    # 加载模型
    model, preprocess, tokenizer = loader.load_model()
    
    # 获取模型信息
    info = loader.get_model_info()
    print(f"\n📊 模型信息:")
    print(f"  模型名称: {info['model_name']}")
    print(f"  设备: {info['device']}")
    print(f"  参数数量: {info['parameters']:,}")
    
    # 测试图像编码
    test_image_path = "assets/airport.jpg"
    if os.path.exists(test_image_path):
        print(f"\n🔧 测试图像编码: {test_image_path}")
        
        image = Image.open(test_image_path)
        image_features = loader.encode_image(image)
        
        print(f"✅ 图像特征形状: {image_features.shape}")
    
    # 测试文本编码
    print(f"\n🔧 测试文本编码")
    
    test_texts = ["airplane", "building", "runway"]
    text_features = loader.encode_text_batch(test_texts)
    
    print(f"✅ 文本特征形状: {text_features.shape}")
    
    # 测试相似度计算
    if os.path.exists(test_image_path):
        print(f"\n🔧 测试相似度计算")
        
        image = Image.open(test_image_path)
        image_features = loader.encode_image(image)
        
        similarities = loader.compute_similarity(image_features, text_features)
        
        print(f"✅ 相似度形状: {similarities.shape}")
        print(f"   相似度值: {similarities.cpu().numpy()[0]}")
    
    print("\n✅ 模型加载器测试完成!")


if __name__ == "__main__":
    main()
