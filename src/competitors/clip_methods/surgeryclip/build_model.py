# -*- coding: utf-8 -*-
"""
CLIP Surgery模型构建工具
"""
from torch import nn
from .clip_model import CLIP
from .clip_surgery_model import CLIPSurgery
from typing import Optional, Tuple
import torch
import os


def build_model(model_name: str,
                checkpoint_path: str,
                device: str = "cuda") -> Tuple[nn.Module, callable]:
    """
    构建CLIP或CLIPSurgery模型
    
    Args:
        model_name: 模型架构
            - "clip": 原始CLIP架构（无VV注意力）
            - "surgeryclip": Surgery架构（有VV注意力）
        checkpoint_path: 权重文件路径（必须）
        device: 设备
    
    Returns:
        model: CLIP模型
        preprocess: 预处理函数
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"权重文件不存在: {checkpoint_path}")
    
    print(f"📥 加载权重: {checkpoint_path}")
    
    # 加载权重文件
    state_dict = _load_checkpoint(checkpoint_path)
    
    # 根据model_name构建对应架构
    if model_name == "clip":
        print("🔧 使用CLIP架构（无VV注意力）")
        model = _build_clip_model(state_dict)
    elif model_name == "surgeryclip":
        print("🔧 使用Surgery架构（有VV注意力）")
        model = _build_surgery_model(state_dict)
    else:
        raise ValueError(f"未知的model_name: {model_name}，应该是 'clip' 或 'surgeryclip'")
    
    # 移动到设备
    model = model.to(device)
    model.eval()
    
    # 创建预处理函数
    from .clip import _transform
    preprocess = _transform(model.visual.input_resolution)
    
    print("✅ 模型加载成功")
    
    return model, preprocess


def _load_checkpoint(checkpoint_path: str) -> dict:
    """加载权重文件，返回state_dict"""
    try:
        # 先尝试作为TorchScript加载（CLIP官方权重通常是这种格式）
        try:
            jit_model = torch.jit.load(checkpoint_path, map_location='cpu')
            # 如果TorchScript模型有state_dict方法，提取state_dict
            if hasattr(jit_model, 'state_dict'):
                try:
                    return jit_model.state_dict()
                except:
                    # 如果无法提取state_dict，尝试其他方法
                    pass
        except (RuntimeError, Exception):
            # 不是TorchScript格式，继续尝试其他格式
            pass
        
        # 尝试直接加载为state_dict或包含state_dict的字典
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except TypeError:
            # 旧版本PyTorch不支持weights_only
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 提取state_dict
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                return checkpoint['state_dict']
            elif 'model' in checkpoint:
                return checkpoint['model']
            else:
                # 假设checkpoint本身就是state_dict
                return checkpoint
        else:
            raise ValueError("无法从checkpoint中提取state_dict")
    
    except Exception as e:
        raise RuntimeError(f"加载权重失败: {e}")


def _build_clip_model(state_dict: dict) -> CLIP:
    """从state_dict构建CLIP模型（无VV注意力）"""
    model_config = _extract_model_config(state_dict)
    
    model = CLIP(
        embed_dim=model_config['embed_dim'],
        image_resolution=model_config['image_resolution'],
        vision_layers=model_config['vision_layers'],
        vision_width=model_config['vision_width'],
        vision_patch_size=model_config['vision_patch_size'],
        context_length=model_config['context_length'],
        vocab_size=model_config['vocab_size'],
        transformer_width=model_config['transformer_width'],
        transformer_heads=model_config['transformer_heads'],
        transformer_layers=model_config['transformer_layers']
    )
    
    # 删除不需要的键
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    
    # 加载权重
    model.load_state_dict(state_dict)
    
    return model


def _build_surgery_model(state_dict: dict) -> CLIPSurgery:
    """从state_dict构建CLIPSurgery模型（有VV注意力）"""
    model_config = _extract_model_config(state_dict)
    
    model = CLIPSurgery(
        embed_dim=model_config['embed_dim'],
        image_resolution=model_config['image_resolution'],
        vision_layers=model_config['vision_layers'],
        vision_width=model_config['vision_width'],
        vision_patch_size=model_config['vision_patch_size'],
        context_length=model_config['context_length'],
        vocab_size=model_config['vocab_size'],
        transformer_width=model_config['transformer_width'],
        transformer_heads=model_config['transformer_heads'],
        transformer_layers=model_config['transformer_layers']
    )
    
    # 删除不需要的键
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    
    # 加载权重
    model.load_state_dict(state_dict)
    
    return model


def _extract_model_config(state_dict: dict) -> dict:
    """从state_dict中提取模型配置"""
    # 判断是ViT还是ResNet
    vit = "visual.proj" in state_dict
    
    if vit:
        # ViT配置
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() 
                            if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        # ResNet配置
        counts = [len(set(k.split(".")[2] for k in state_dict 
                         if k.startswith(f"visual.layer{b}"))) 
                 for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        image_resolution = output_width * 32
    
    # 文本编码器配置
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict 
                                 if k.startswith("transformer.resblocks")))
    
    return {
        'embed_dim': embed_dim,
        'image_resolution': image_resolution,
        'vision_layers': vision_layers,
        'vision_width': vision_width,
        'vision_patch_size': vision_patch_size,
        'context_length': context_length,
        'vocab_size': vocab_size,
        'transformer_width': transformer_width,
        'transformer_heads': transformer_heads,
        'transformer_layers': transformer_layers
    }
