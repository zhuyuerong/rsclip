#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测头模块
"""

from .classification_head import ContrastiveClassificationHead
from .regression_head import BBoxRegressionHead

__all__ = ['ContrastiveClassificationHead', 'BBoxRegressionHead']

