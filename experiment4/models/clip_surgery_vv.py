# -*- coding: utf-8 -*-
"""
带VV机制的CLIP Surgery模型
实现双路径VV自注意力机制，使用RemoteCLIP权重
"""

import torch
import torch.nn as nn
import clip
from pathlib import Path
import sys
import os

# 添加路径以导入VVAttention
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment4.models.vv_attention import VVAttention


class CLIPSurgeryVV(nn.Module):
    """
    带VV机制的CLIP Surgery
    
    在最后几层使用VV自注意力替换标准自注意力，同时保留原始路径的融合
    使用RemoteCLIP权重
    """
    
    def __init__(self, clip_model, device="cuda", num_vv_blocks=6):
        super().__init__()
        self.clip_model = clip_model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.num_vv_blocks = num_vv_blocks
        
        # 冻结所有参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self.clip_model.eval()
        
        # 确保模型在正确设备
        self.clip_model = self.clip_model.to(self.device)
        
        # 替换最后num_vv_blocks层的注意力为VV注意力
        self._replace_with_vv_attention()
    
    def _replace_with_vv_attention(self):
        """替换最后几层的注意力为VV注意力"""
        visual = self.clip_model.visual
        transformer = visual.transformer
        
        if not hasattr(transformer, 'resblocks'):
            print("⚠️ 无法找到transformer.resblocks，跳过VV机制替换")
            return
        
        # 获取参数信息
        first_block = transformer.resblocks[0]
        original_attn = first_block.attn
        
        # 获取维度信息
        if hasattr(original_attn, 'in_proj_weight'):
            dim = original_attn.in_proj_weight.shape[1]  # 768 for ViT-B
            num_heads = original_attn.num_heads
        elif hasattr(original_attn, 'qkv'):
            dim = original_attn.qkv.weight.shape[1]
            # 尝试从其他属性推断num_heads
            if hasattr(original_attn, 'num_heads'):
                num_heads = original_attn.num_heads
            else:
                num_heads = 12  # ViT-B/32默认值
        else:
            print("⚠️ 无法识别注意力层结构，跳过VV机制替换")
            return
        
        # 替换最后num_vv_blocks层
        replaced_count = 0
        for i in range(1, min(self.num_vv_blocks + 1, len(transformer.resblocks) + 1)):
            block = transformer.resblocks[-i]
            original_attn = block.attn
            
            # 创建VV注意力模块
            vv_attn = VVAttention(dim, num_heads, scale_multiplier=1.0).to(self.device)
            
            # 转换VV注意力权重为half precision
            if hasattr(original_attn, 'in_proj_weight') and original_attn.in_proj_weight.dtype == torch.float16:
                vv_attn.half()
            
            # 从原始注意力复制权重
            with torch.no_grad():
                if hasattr(original_attn, 'in_proj_weight'):
                    # MultiheadAttention格式：in_proj_weight是[3*dim, dim]
                    in_proj_weight = original_attn.in_proj_weight.data
                    # 复制完整QKV权重
                    vv_attn.qkv.weight.data = in_proj_weight.clone()
                    
                    # 复制输出投影权重
                    if hasattr(original_attn, 'out_proj'):
                        if hasattr(original_attn.out_proj, 'weight'):
                            vv_attn.proj.weight.data = original_attn.out_proj.weight.data.clone()
                        else:
                            print(f"  ⚠️ 第{len(transformer.resblocks) - i + 1}层out_proj无weight属性")
                elif hasattr(original_attn, 'qkv'):
                    # 某些实现可能使用qkv属性
                    qkv_weight = original_attn.qkv.weight.data
                    vv_attn.qkv.weight.data = qkv_weight.clone()
                    
                    if hasattr(original_attn, 'proj'):
                        vv_attn.proj.weight.data = original_attn.proj.weight.data.clone()
                else:
                    print(f"  ⚠️ 第{len(transformer.resblocks) - i + 1}层无法复制权重，跳过")
                    continue
            
            # 替换注意力模块
            block.attn = vv_attn
            replaced_count += 1
        
        print(f"  ✓ 已替换最后{replaced_count}层为VV注意力")
    
    @classmethod
    def from_pretrained(cls, model_name="ViT-B/32", device="cuda", num_vv_blocks=6):
        """
        从预训练CLIP加载并创建VV版本
        
        Args:
            model_name: CLIP模型名称
            device: 设备
            num_vv_blocks: 应用VV机制的层数
        
        Returns:
            CLIPSurgeryVV实例
        """
        # 检查是否有RemoteCLIP权重
        remoteclip_path = Path("checkpoints/RemoteCLIP-ViT-B-32.pt")
        if not remoteclip_path.exists():
            # 尝试相对路径
            remoteclip_path = Path(__file__).parent.parent.parent / "checkpoints" / "RemoteCLIP-ViT-B-32.pt"
        
        if remoteclip_path.exists() and "B" in model_name:
            print(f"加载RemoteCLIP权重: {remoteclip_path}")
            clip_model, _ = clip.load("ViT-B/32", device=device)
            
            # 加载RemoteCLIP权重
            checkpoint = torch.load(remoteclip_path, map_location=device)
            if 'state_dict' in checkpoint:
                clip_model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                clip_model.load_state_dict(checkpoint, strict=False)
        else:
            print(f"加载CLIP权重: {model_name}")
            clip_model, _ = clip.load(model_name, device=device)
        
        return cls(clip_model, device, num_vv_blocks)
    
    def encode_image(self, images):
        """
        编码图像，使用VV注意力
        
        Args:
            images: [B, 3, 224, 224]
        
        Returns:
            features: [B, N+1, 512] (包含CLS token)
            例如ViT-B/32: [B, 50, 512] = 1 CLS + 49 patches
        """
        with torch.no_grad():
            features, _ = self._encode_image_internal(images, return_attn=False)
        return features
    
    def encode_image_with_attn(self, images):
        """
        编码图像并返回最后一层的注意力权重
        
        Args:
            images: [B, 3, 224, 224]
        
        Returns:
            features: [B, N+1, 512]
            attn_weights: dict with 'attn_qk', 'attn_vv', 'attn_mixed'
        """
        with torch.no_grad():
            features, attn_weights = self._encode_image_internal(images, return_attn=True)
        return features, attn_weights
    
    def _encode_image_internal(self, images, return_attn=False):
        """
        内部图像编码方法，支持返回注意力权重
        
        Args:
            images: [B, 3, 224, 224]
            return_attn: 是否返回最后一层的注意力权重
        
        Returns:
            features: [B, N+1, 512]
            attn_weights: dict (仅当return_attn=True时)
        """
        # 确保输入在正确的设备上
        if images.device != self.device:
            images = images.to(self.device)
        
        # 确保输入类型匹配模型权重类型
        if images.dtype != self.clip_model.visual.conv1.weight.dtype:
            images = images.to(self.clip_model.visual.conv1.weight.dtype)
        
        # 获取ViT的patch embeddings
        x = self.clip_model.visual.conv1(images)  # [B, 768, 7, 7] for ViT-B/32
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, 768, 49]
        x = x.permute(0, 2, 1)  # [B, 49, 768]
        
        # 添加CLS token
        cls_embed = self.clip_model.visual.class_embedding.to(x.dtype).to(self.device)
        x = torch.cat([
            cls_embed + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=self.device
            ),
            x
        ], dim=1)  # [B, 50, 768]
        
        # 位置编码
        pos_embed = self.clip_model.visual.positional_embedding.to(x.dtype).to(self.device)
        x = x + pos_embed
        x = self.clip_model.visual.ln_pre(x)
        
        # 通过transformer（最后几层使用VV注意力）
        x = x.permute(1, 0, 2)  # [50, B, 768] (CLIP序列优先格式)
        
        # 逐层forward，捕获最后一层的注意力
        attn_weights = None
        for i, block in enumerate(self.clip_model.visual.transformer.resblocks):
            is_last_layer = (i == len(self.clip_model.visual.transformer.resblocks) - 1)
            
            if return_attn and is_last_layer:
                # 最后一层：提取注意力权重
                if hasattr(block.attn, 'forward') and hasattr(block.attn, 'qkv'):
                    # 这是VVAttention层
                    x_before_attn = block.ln_1(x)
                    x_attn, attn_weights = block.attn(x_before_attn, need_weights=True)
                    x = x + x_attn
                    x = x + block.mlp(block.ln_2(x))
                else:
                    # 标准层
                    x = block(x)
            else:
                # 非最后一层：正常forward
                x = block(x)
        
        x = x.permute(1, 0, 2)  # [B, 50, 768]
        
        # Layer norm
        x = self.clip_model.visual.ln_post(x)
        
        # 投影到512维
        if hasattr(self.clip_model.visual, 'proj') and self.clip_model.visual.proj is not None:
            B, N, D = x.shape
            x_reshaped = x.reshape(B * N, D)
            proj_weight = self.clip_model.visual.proj.to(self.device)
            x_proj = x_reshaped @ proj_weight
            features = x_proj.reshape(B, N, -1)
        else:
            features = x
        
        if return_attn:
            return features, attn_weights
        else:
            return features, None


class CLIPSurgeryVVWrapper:
    """
    VV机制CLIP Surgery的包装器
    提供与CLIPSurgeryWrapper相同的接口
    """
    
    def __init__(self, config, num_vv_blocks=6):
        """
        Args:
            config: Config对象
            num_vv_blocks: 应用VV机制的层数
        """
        self.config = config
        self.device = config.device
        
        # 加载VV版本的CLIP Surgery
        self.model = CLIPSurgeryVV.from_pretrained(
            model_name=config.backbone,
            device=self.device,
            num_vv_blocks=num_vv_blocks
        )
        
        # 预计算背景词特征（使用原始模型的文本编码器）
        self.bg_features = self.encode_text(config.background_words)
    
    def encode_image(self, images):
        """编码图像"""
        return self.model.encode_image(images)
    
    def encode_text(self, text_list):
        """编码文本（使用原始CLIP的文本编码器）"""
        with torch.no_grad():
            if isinstance(text_list, list):
                import clip
                text_tokens = clip.tokenize(text_list).to(self.device)
            else:
                text_tokens = text_list
            
            text_features = self.model.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def get_all_features(self, images):
        """
        获取完整特征（包含CLS token + patches）
        
        Args:
            images: [B, 3, 224, 224]
        
        Returns:
            all_features: [B, N+1, 512]  # CLS token + N patches
        """
        return self.model.encode_image(images)
    
    def get_cls_features(self, images):
        """
        获取CLS token特征（全局特征）
        
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
        """
        all_features = self.model.encode_image(images)
        patch_features = all_features[:, 1:, :]  # 去掉CLS token
        return patch_features

