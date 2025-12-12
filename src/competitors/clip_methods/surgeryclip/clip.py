# -*- coding: utf-8 -*-
"""
CLIP Surgery工具函数
从外部CLIP_Surgery导入核心函数
"""
import sys
import os
from pathlib import Path
import torch

# 添加外部CLIP_Surgery路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent.parent
external_clip_path = project_root / "external" / "CLIP_Surgery-master" / "clip"

if not external_clip_path.exists():
    # 尝试其他可能的路径
    external_clip_path = current_file.parent.parent.parent.parent / "external" / "CLIP_Surgery-master" / "clip"

if str(external_clip_path) not in sys.path:
    sys.path.insert(0, str(external_clip_path))

# 导入外部函数
try:
    # 将external/CLIP_Surgery-master/clip添加到sys.path，然后直接导入
    if str(external_clip_path) not in sys.path:
        sys.path.insert(0, str(external_clip_path))
    
    # 直接导入（作为模块）
    import clip as external_clip_module
    
    encode_text_with_prompt_ensemble = external_clip_module.encode_text_with_prompt_ensemble
    get_similarity_map = external_clip_module.get_similarity_map
    clip_feature_surgery = external_clip_module.clip_feature_surgery
    tokenize = external_clip_module.tokenize
    
except ImportError as e:
    # 如果直接导入失败，尝试使用importlib
    try:
        import importlib.util
        clip_file = external_clip_path / "clip.py"
        if clip_file.exists():
            # 将external/CLIP_Surgery-master添加到sys.path
            external_master_path = external_clip_path.parent
            if str(external_master_path) not in sys.path:
                sys.path.insert(0, str(external_master_path))
            
            # 作为clip.clip模块导入
            spec = importlib.util.spec_from_file_location("clip.clip", clip_file)
            clip_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(clip_module)
            
            encode_text_with_prompt_ensemble = clip_module.encode_text_with_prompt_ensemble
            get_similarity_map = clip_module.get_similarity_map
            clip_feature_surgery = clip_module.clip_feature_surgery
            tokenize = clip_module.tokenize
        else:
            raise ImportError(f"无法找到CLIP_Surgery的clip.py文件: {clip_file}")
    except Exception as e2:
        raise ImportError(
            f"无法导入CLIP_Surgery函数: {e2}\n"
            f"请确保external/CLIP_Surgery-master/clip/clip.py文件存在\n"
            f"尝试的路径: {external_clip_path / 'clip.py'}"
        )

# 导入_transform函数（build_model需要）
try:
    if 'external_clip_module' in locals():
        _transform = external_clip_module._transform
    elif 'clip_module' in locals():
        _transform = clip_module._transform
    else:
        # 如果上面的导入都失败了，从外部模块直接导入
        import importlib.util
        clip_file = external_clip_path / "clip.py"
        if clip_file.exists():
            spec = importlib.util.spec_from_file_location("clip_module", clip_file)
            temp_clip_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(temp_clip_module)
            _transform = temp_clip_module._transform
        else:
            _transform = None
except:
    _transform = None

__all__ = [
    'encode_text_with_prompt_ensemble',
    'get_similarity_map',
    'clip_feature_surgery',
    'tokenize',
    '_transform'
]

