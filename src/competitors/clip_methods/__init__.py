# -*- coding: utf-8 -*-
"""
CLIP方法统一接口

所有CLIP方法都应通过此接口使用
"""
from .base_interface import BaseCLIPMethod
from .surgeryclip.model_wrapper import SurgeryCLIPWrapper

# 延迟导入，避免触发timm依赖
try:
    from .declip.model_wrapper import DeCLIPWrapper
except ImportError:
    DeCLIPWrapper = None

try:
    from .diffclip.model_wrapper import DiffCLIPWrapper
except ImportError:
    DiffCLIPWrapper = None

from .eval import (
    heatmap_to_bboxes,
    compute_iou,
    evaluate_bboxes_with_gt,
    multi_threshold_evaluation
)

__all__ = [
    'BaseCLIPMethod',
    'SurgeryCLIPWrapper',
    'heatmap_to_bboxes',
    'compute_iou',
    'evaluate_bboxes_with_gt',
    'multi_threshold_evaluation',
]

# 可选导出
if DeCLIPWrapper is not None:
    __all__.append('DeCLIPWrapper')
if DiffCLIPWrapper is not None:
    __all__.append('DiffCLIPWrapper')
