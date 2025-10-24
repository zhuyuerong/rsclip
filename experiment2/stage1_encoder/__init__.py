#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage1: 特征提取模块
"""

from .clip_image_encoder import CLIPImageEncoder
from .clip_text_encoder import CLIPTextEncoder
from .global_context_extractor import GlobalContextExtractor

__all__ = [
    'CLIPImageEncoder',
    'CLIPTextEncoder',
    'GlobalContextExtractor'
]

