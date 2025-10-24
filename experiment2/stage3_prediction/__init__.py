#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage3: 预测与精细化模块
"""

from .classification_head import ClassificationHead
from .regression_head import RegressionHead

__all__ = [
    'ClassificationHead',
    'RegressionHead'
]

