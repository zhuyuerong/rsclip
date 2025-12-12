"""
通用模型加载封装
支持RemoteCLIP、CLIP Surgery等模型的统一加载接口
"""
import torch
import os
from pathlib import Path
from typing import Optional, Dict, Any
import open_clip


def load_remoteclip(
    model_name: str = 'RN50',
    checkpoint_path: Optional[str] = None,
    device: str = 'cuda',
    freeze: bool = True
) -> tuple:
    """
    加载RemoteCLIP模型
    
    Args:
        model_name: 模型名称 ('RN50', 'ViT-B-32', 'ViT-L-14')
        checkpoint_path: 检查点路径，如果为None则自动查找
        device: 设备 ('cuda' or 'cpu')
        freeze: 是否冻结模型参数
    
    Returns:
        (model, preprocess, tokenizer) 元组
    """
    if checkpoint_path is None:
        project_root = Path(__file__).parent.parent.parent
        checkpoint_path = project_root / 'checkpoints' / f'RemoteCLIP-{model_name}.pt'
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # 加载模型和预处理
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
    
    return model, preprocess, tokenizer


def load_model(
    model_type: str,
    model_name: str,
    checkpoint_path: Optional[str] = None,
    device: str = 'cuda',
    **kwargs
) -> Any:
    """
    通用模型加载接口
    
    Args:
        model_type: 模型类型 ('remoteclip', 'clip_surgery', 'diffclip')
        model_name: 模型名称
        checkpoint_path: 检查点路径
        device: 设备
        **kwargs: 其他参数
    
    Returns:
        加载的模型
    """
    if model_type == 'remoteclip':
        model, preprocess, tokenizer = load_remoteclip(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            device=device,
            freeze=kwargs.get('freeze', True)
        )
        return {
            'model': model,
            'preprocess': preprocess,
            'tokenizer': tokenizer
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

