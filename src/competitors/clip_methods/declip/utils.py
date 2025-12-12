# -*- coding: utf-8 -*-
"""
DeCLIP工具函数

从external/DeCLIP-main/src/training/提取的工具函数
"""
import logging
import os
import torch
from typing import Optional

logger = logging.getLogger(__name__)


def is_main_process():
    """检查是否为主进程"""
    if not torch.distributed.is_available():
        return True
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def get_tokenizer(model_name: str = "ViT-B-32"):
    """获取tokenizer"""
    try:
        from .open_clip import tokenizer
        return tokenizer.SimpleTokenizer()
    except ImportError:
        logger.warning("无法导入tokenizer，使用简单实现")
        return None








