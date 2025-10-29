#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型模块
"""

from .ova_detr import OVADETR
from .criterion import SetCriterion

__all__ = ['OVADETR', 'SetCriterion']

