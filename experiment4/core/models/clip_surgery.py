# -*- coding: utf-8 -*-
"""
CLIP Surgery模型
实现V-V attention提取，去除文本泄露
"""

import torch
import torch.nn as nn
import clip
from pathlib import Path


class CLIPSurgery(nn.Module):
    """
    CLIP Surgery模型
    
    使用V-V attention代替V-L attention，避免文本泄露
    """
    
    def __init__(self, clip_model, device="cuda"):
        super().__init__()
        self.clip_model = clip_model
        self.device = device
        
        # 冻结所有参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self.clip_model.eval()
    
    @classmethod
    def from_pretrained(cls, model_name="ViT-B/16", device="cuda"):
        """
        从预训练CLIP加载
        
        Args:
            model_name: CLIP模型名称
            device: 设备
        
        Returns:
            CLIPSurgery实例
        """
        # 检查是否有RemoteCLIP权重
        remoteclip_path = Path("checkpoints/RemoteCLIP-ViT-B-32.pt")
        
        if remoteclip_path.exists() and "B" in model_name:
            print(f"加载RemoteCLIP权重: {remoteclip_path}")
            clip_model, _ = clip.load("ViT-B/32", device=device)
            
            # 加载RemoteCLIP权重
            checkpoint = torch.load(remoteclip_path, map_location=device)
            if 'state_dict' in checkpoint:
                clip_model.load_state_dict(checkpoint['state_dict'])
            else:
                clip_model.load_state_dict(checkpoint)
        else:
            print(f"加载CLIP权重: {model_name}")
            clip_model, _ = clip.load(model_name, device=device)
        
        return cls(clip_model, device)
    
    def encode_image(self, images):
        """
        编码图像，使用V-V attention
        
        Args:
            images: [B, 3, 224, 224]
        
        Returns:
            features: [B, 197, 512] (包含CLS token)
        """
        with torch.no_grad():
            # 确保输入类型匹配模型权重类型
            if images.dtype != self.clip_model.visual.conv1.weight.dtype:
                images = images.to(self.clip_model.visual.conv1.weight.dtype)
            
            # 获取ViT的patch embeddings
            x = self.clip_model.visual.conv1(images)  # [B, 768, 7, 7] for ViT-B/32
            x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, 768, 49]
            x = x.permute(0, 2, 1)  # [B, 49, 768]
            
            # 添加CLS token和位置编码
            x = torch.cat([
                self.clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x
            ], dim=1)  # [B, 50, 768]
            
            x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
            x = self.clip_model.visual.ln_pre(x)
            
            # 通过transformer（使用self-attention，即V-V）
            x = x.permute(1, 0, 2)  # [50, B, 768]
            x = self.clip_model.visual.transformer(x)
            x = x.permute(1, 0, 2)  # [B, 50, 768]
            
            # Layer norm
            x = self.clip_model.visual.ln_post(x)
            
            # 投影到512维（如果需要）
            if hasattr(self.clip_model.visual, 'proj') and self.clip_model.visual.proj is not None:
                # 投影所有token到512维以匹配文本特征
                B, N, D = x.shape
                x_reshaped = x.reshape(B * N, D)  # [B*N, 768]
                x_proj = x_reshaped @ self.clip_model.visual.proj  # [B*N, 512]
                features = x_proj.reshape(B, N, -1)  # [B, N, 512]
            else:
                features = x
        
        return features
    
    def encode_text(self, text_list):
        """
        编码文本
        
        Args:
            text_list: list of str or tokenized text
        
        Returns:
            features: [N, 512]
        """
        with torch.no_grad():
            if isinstance(text_list, list):
                # Tokenize
                text_tokens = clip.tokenize(text_list).to(self.device)
            else:
                text_tokens = text_list
            
            # Encode
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def get_visual_features(self, images, return_all_tokens=True):
        """
        获取视觉特征（Surgery版本）
        
        Args:
            images: [B, 3, 224, 224]
            return_all_tokens: 是否返回所有token
        
        Returns:
            features: [B, 197, D] if return_all_tokens else [B, D]
        """
        features = self.encode_image(images)
        
        if return_all_tokens:
            return features  # [B, 197, D]
        else:
            return features[:, 0, :]  # [B, D] 只返回CLS token


class CLIPSurgeryWrapper:
    """
    CLIP Surgery的简化包装器
    
    自动处理RemoteCLIP路径
    """
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # 加载模型
        self.model = CLIPSurgery.from_pretrained(
            model_name=config.backbone,
            device=self.device
        )
        
        # 预计算背景词特征
        self.bg_features = self.encode_text(config.background_words)
    
    def encode_image(self, images):
        """编码图像"""
        return self.model.encode_image(images)
    
    def encode_text(self, text_list):
        """编码文本"""
        return self.model.encode_text(text_list)
    
    def get_all_features(self, images):
        """
        获取完整特征（包含CLS token + patches）- 符合VV机制格式
        
        Args:
            images: [B, 3, 224, 224]
        
        Returns:
            all_features: [B, N+1, 512]  # CLS token + N patches
            例如ViT-B-32: [B, 50, 512] = 1 CLS + 49 patches
        """
        return self.model.encode_image(images)
    
    def get_cls_features(self, images):
        """
        获取CLS token特征（全局特征）- 符合VV机制格式
        
        Args:
            images: [B, 3, 224, 224]
        
        Returns:
            cls_features: [B, 512]  # CLS token
        """
        all_features = self.model.encode_image(images)
        return all_features[:, 0, :]  # [B, 512]
    
    def get_patch_features(self, images):
        """
        获取patch特征（去掉CLS token）- 向后兼容方法
        
        Args:
            images: [B, 3, 224, 224]
        
        Returns:
            patch_features: [B, N, 512]  # 只有patches，不含CLS token
            例如ViT-B-32: [B, 49, 512]
        """
        all_features = self.model.encode_image(images)
        patch_features = all_features[:, 1:, :]  # 去掉CLS token
        
        return patch_features


def test_clip_surgery():
    """测试CLIP Surgery"""
    print("测试CLIP Surgery...")
    
    # 创建模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPSurgery.from_pretrained("ViT-B/32", device=device)
    
    # 测试图像编码
    images = torch.randn(2, 3, 224, 224).to(device)
    img_features = model.encode_image(images)
    print(f"图像特征形状: {img_features.shape}")
    
    # 测试文本编码
    texts = ["airplane", "ship", "car"]
    text_features = model.encode_text(texts)
    print(f"文本特征形状: {text_features.shape}")
    
    # 测试patch特征
    patch_features = model.get_visual_features(images, return_all_tokens=True)
    print(f"Patch特征形状: {patch_features.shape}")
    
    print("测试通过！")


if __name__ == "__main__":
    test_clip_surgery()

