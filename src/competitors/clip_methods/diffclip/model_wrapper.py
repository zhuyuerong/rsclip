# -*- coding: utf-8 -*-
"""
DiffCLIP模型包装器

实现统一接口，适配遥感数据
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image
from ..base_interface import BaseCLIPMethod
from .diff_clip import DiffCLIP_VITB16
from .tokenizer import SimpleTokenizer
from .dataset import build_default_transform


class DiffCLIPWrapper(BaseCLIPMethod):
    """
    DiffCLIP包装器，实现统一接口
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda",
                 checkpoint_path: Optional[str] = None):
        """
        初始化DiffCLIP
        
        Args:
            model_name: 模型名称（目前只支持ViT-B/32）
            device: 设备
            checkpoint_path: 检查点路径
        """
        super().__init__(model_name, device)
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.preprocess = None
        self.tokenizer = SimpleTokenizer()
    
    def load_model(self, checkpoint_path: Optional[str] = None):
        """加载模型"""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path
        
        try:
            # 创建模型
            self.model = DiffCLIP_VITB16()
            
            # 如果提供了checkpoint，加载权重
            if checkpoint_path:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # 加载权重（可能需要适配）
                try:
                    self.model.load_state_dict(state_dict, strict=False)
                except Exception as e:
                    print(f"警告：加载权重时出现问题: {e}")
                    print("尝试使用load_remoteclip_weights方法...")
                    if hasattr(self.model, 'load_remoteclip_weights'):
                        self.model.load_remoteclip_weights(checkpoint_path, strict=False)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # 创建预处理
            self.preprocess = build_default_transform(image_size=224)
            
        except Exception as e:
            raise RuntimeError(f"加载DiffCLIP模型失败: {e}")
    
    def encode_image(self, image: Union[torch.Tensor, Image.Image, np.ndarray]) -> torch.Tensor:
        """编码图像"""
        if self.model is None:
            self.load_model()
        
        # 预处理
        if isinstance(image, Image.Image):
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        elif isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
            image_tensor = self.preprocess(image_pil).unsqueeze(0).to(self.device)
        else:
            image_tensor = image.to(self.device) if not image.is_cuda else image
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
        
        return image_features
    
    def encode_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        """编码文本"""
        if self.model is None:
            self.load_model()
        
        if isinstance(text, str):
            text = [text]
        
        # Tokenize
        text_tokens = []
        for t in text:
            tokens = self.tokenizer.encode(t)
            text_tokens.append(tokens)
        
        # Pad to same length
        max_len = max(len(t) for t in text_tokens)
        text_tensor = torch.zeros(len(text_tokens), max_len, dtype=torch.long)
        for i, tokens in enumerate(text_tokens):
            text_tensor[i, :len(tokens)] = torch.tensor(tokens)
        
        text_tensor = text_tensor.to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tensor)
        
        return text_features
    
    def compute_similarity(self, image: Union[torch.Tensor, Image.Image, np.ndarray],
                          text: Union[str, List[str]]) -> torch.Tensor:
        """计算相似度"""
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        # 归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 计算相似度
        logit_scale = self.model.logit_scale.exp() if hasattr(self.model, 'logit_scale') else 100.0
        similarity = (image_features @ text_features.T) * logit_scale
        
        return similarity
    
    def generate_heatmap(self, image: Union[torch.Tensor, Image.Image, np.ndarray],
                        text: Union[str, List[str]],
                        return_features: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        生成热图
        
        注意：DiffCLIP目前只支持全局特征，热图生成需要特殊处理
        """
        if self.model is None:
            self.load_model()
        
        # 预处理图像
        if isinstance(image, Image.Image):
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        elif isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
            image_tensor = self.preprocess(image_pil).unsqueeze(0).to(self.device)
        else:
            image_tensor = image.to(self.device) if not image.is_cuda else image
        
        if isinstance(text, str):
            text = [text]
        
        # 编码文本
        text_features = self.encode_text(text)
        
        # 获取图像特征
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            
            # 计算相似度
            similarity = (image_features @ text_features.T).cpu().numpy()
            
            # 由于DiffCLIP没有dense特征提取，我们创建一个简单的热图
            # 使用全局相似度作为全图激活值
            # 实际应用中可能需要使用patch-based方法
            h, w = image_tensor.shape[-2:]
            heatmap = np.ones((h, w)) * similarity[0, 0]
            
            # 可以添加一些空间变化（基于图像内容）
            # 这里简化处理，实际应该使用patch-level特征
        
        if return_features:
            return heatmap, {
                'image_features': image_features,
                'text_features': text_features,
                'similarity': similarity
            }
        
        return heatmap








