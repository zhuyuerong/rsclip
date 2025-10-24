#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
边界框工具函数

坐标格式转换和处理
"""

import torch
import numpy as np


def box_cxcywh_to_xyxy(boxes):
    """
    转换: CXCYWH -> XYXY
    
    参数:
        boxes: (..., 4) [cx, cy, w, h]
    返回:
        boxes: (..., 4) [x1, y1, x2, y2]
    """
    if isinstance(boxes, torch.Tensor):
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)
    else:
        cx, cy, w, h = np.split(boxes, 4, axis=-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return np.concatenate([x1, y1, x2, y2], axis=-1)


def box_xyxy_to_cxcywh(boxes):
    """
    转换: XYXY -> CXCYWH
    
    参数:
        boxes: (..., 4) [x1, y1, x2, y2]
    返回:
        boxes: (..., 4) [cx, cy, w, h]
    """
    if isinstance(boxes, torch.Tensor):
        x1, y1, x2, y2 = boxes.unbind(-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return torch.stack([cx, cy, w, h], dim=-1)
    else:
        x1, y1, x2, y2 = np.split(boxes, 4, axis=-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return np.concatenate([cx, cy, w, h], axis=-1)


def normalize_boxes(boxes, image_size):
    """
    归一化边界框
    
    参数:
        boxes: (..., 4) 像素坐标
        image_size: (height, width)
    返回:
        boxes: (..., 4) 归一化坐标 [0, 1]
    """
    h, w = image_size
    
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.clone()
        boxes[..., [0, 2]] /= w
        boxes[..., [1, 3]] /= h
    else:
        boxes = boxes.copy()
        boxes[..., [0, 2]] /= w
        boxes[..., [1, 3]] /= h
    
    return boxes


def denormalize_boxes(boxes, image_size):
    """
    反归一化边界框
    
    参数:
        boxes: (..., 4) 归一化坐标
        image_size: (height, width)
    返回:
        boxes: (..., 4) 像素坐标
    """
    h, w = image_size
    
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.clone()
        boxes[..., [0, 2]] *= w
        boxes[..., [1, 3]] *= h
    else:
        boxes = boxes.copy()
        boxes[..., [0, 2]] *= w
        boxes[..., [1, 3]] *= h
    
    return boxes


if __name__ == "__main__":
    import torch
    
    print("=" * 70)
    print("测试边界框工具")
    print("=" * 70)
    
    # 测试转换
    boxes_cxcywh = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
    boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)
    
    print(f"\nCXCYWH: {boxes_cxcywh}")
    print(f"XYXY: {boxes_xyxy}")
    
    # 反向转换
    boxes_back = box_xyxy_to_cxcywh(boxes_xyxy)
    print(f"转换回: {boxes_back}")
    
    # 测试归一化
    boxes_pixel = torch.tensor([[100, 150, 300, 450]])
    boxes_norm = normalize_boxes(boxes_pixel, (800, 800))
    
    print(f"\n像素坐标: {boxes_pixel}")
    print(f"归一化: {boxes_norm}")
    
    # 反归一化
    boxes_denorm = denormalize_boxes(boxes_norm, (800, 800))
    print(f"反归一化: {boxes_denorm}")
    
    print("\n✅ 边界框工具测试完成！")

