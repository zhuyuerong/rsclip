# -*- coding: utf-8 -*-
"""
DeCLIP模型包装器

实现统一接口，适配遥感数据
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image
from ..base_interface import BaseCLIPMethod


class DeCLIPWrapper(BaseCLIPMethod):
    """
    DeCLIP包装器，实现统一接口
    """
    
    def __init__(self, model_name: str = "ViT-B-16", device: str = "cuda",
                 checkpoint_path: Optional[str] = None, mode: str = "qq_vfm_distill"):
        """
        初始化DeCLIP
        
        Args:
            model_name: 模型名称
            device: 设备
            checkpoint_path: 检查点路径
            mode: 模式（用于特征提取）
        """
        super().__init__(model_name, device)
        self.checkpoint_path = checkpoint_path
        self.mode = mode
        self.model = None
        self.preprocess = None
    
    def load_model(self, checkpoint_path: Optional[str] = None):
        """加载模型"""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path
        
        try:
            from .open_clip.factory import create_model_and_transforms
            from .open_clip import get_tokenizer
            
            # 创建模型和预处理
            self.model, self.preprocess, _ = create_model_and_transforms(
                self.model_name,
                pretrained=checkpoint_path,
                device=self.device
            )
            self.model.eval()
            
        except Exception as e:
            raise RuntimeError(f"加载DeCLIP模型失败: {e}")
    
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
        
        with torch.no_grad():
            text_features = self.model.encode_text(text)
        
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
        
        DeCLIP支持dense特征提取，可以生成空间热图
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
        
        # 获取dense特征
        with torch.no_grad():
            if hasattr(self.model, 'encode_dense'):
                image_features_dense = self.model.encode_dense(
                    image_tensor, 
                    normalize=True,
                    keep_shape=True,
                    mode=self.mode
                )
            else:
                # 回退到全局特征
                image_features = self.model.encode_image(image_tensor)
                similarity = (image_features @ text_features.T).cpu().numpy()
                h, w = image_tensor.shape[-2:]
                heatmap = np.ones((h, w)) * similarity[0, 0]
                if return_features:
                    return heatmap, {'image_features': image_features, 'text_features': text_features}
                return heatmap
        
        # 计算相似度热图
        if isinstance(image_features_dense, tuple):
            # 如果返回多个特征（q, k, v等），使用第一个
            image_features_dense = image_features_dense[0]
        
        B, C, H, W = image_features_dense.shape
        image_features_flat = image_features_dense.permute(0, 2, 3, 1).reshape(B * H * W, C)
        image_features_flat = image_features_flat / image_features_flat.norm(dim=-1, keepdim=True)
        
        # 计算每个空间位置的相似度
        similarities = (image_features_flat @ text_features.T).reshape(B, H, W, -1)
        heatmap = similarities[0, :, :, 0].cpu().numpy()  # 取第一个文本
        
        # 归一化到[0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        if return_features:
            return heatmap, {
                'image_features_dense': image_features_dense,
                'text_features': text_features,
                'similarities': similarities
            }
        
        return heatmap

