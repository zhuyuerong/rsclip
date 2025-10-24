#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
后处理模块

功能：
NMS、阈值过滤等后处理操作
"""

import torch
import torch.nn as nn
from torchvision.ops import nms
from typing import List, Dict


class PostProcessor(nn.Module):
    """后处理器"""
    
    def __init__(
        self,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.7,
        max_detections: int = 100
    ):
        super().__init__()
        
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
    
    @torch.no_grad()
    def forward(
        self,
        pred_boxes: torch.Tensor,  # (B, M, 4)，格式 (cx, cy, w, h)
        scores: torch.Tensor       # (B, M)
    ) -> List[Dict]:
        """
        后处理
        
        参数:
            pred_boxes: 预测的边界框
            scores: 预测的分数
        
        返回:
            results: 列表，每个元素是一个batch的检测结果
        """
        from ..stage4_supervision.box_loss import box_cxcywh_to_xyxy
        
        batch_size = pred_boxes.size(0)
        results = []
        
        for b in range(batch_size):
            boxes_b = pred_boxes[b]  # (M, 4)
            scores_b = scores[b]      # (M,)
            
            # 阈值过滤
            keep_mask = scores_b > self.score_threshold
            boxes_filtered = boxes_b[keep_mask]
            scores_filtered = scores_b[keep_mask]
            
            if len(boxes_filtered) == 0:
                results.append({
                    'boxes': torch.tensor([]).reshape(0, 4),
                    'scores': torch.tensor([]),
                    'num_detections': 0
                })
                continue
            
            # 转换到xyxy格式用于NMS
            boxes_xyxy = box_cxcywh_to_xyxy(boxes_filtered)
            
            # NMS
            keep_indices = nms(boxes_xyxy, scores_filtered, self.nms_threshold)
            
            # 限制最大检测数
            if len(keep_indices) > self.max_detections:
                # 按分数排序，保留top-k
                scores_kept = scores_filtered[keep_indices]
                _, top_indices = scores_kept.topk(self.max_detections)
                keep_indices = keep_indices[top_indices]
            
            # 最终结果
            final_boxes = boxes_filtered[keep_indices]
            final_scores = scores_filtered[keep_indices]
            
            results.append({
                'boxes': final_boxes,
                'scores': final_scores,
                'num_detections': len(final_boxes)
            })
        
        return results


if __name__ == "__main__":
    processor = PostProcessor(score_threshold=0.5, nms_threshold=0.7)
    
    # 测试数据
    batch_size = 2
    num_queries = 100
    
    pred_boxes = torch.rand(batch_size, num_queries, 4)
    scores = torch.rand(batch_size, num_queries)
    
    results = processor(pred_boxes, scores)
    
    print("后处理结果:")
    for b, result in enumerate(results):
        print(f"  Batch {b}: {result['num_detections']} 个检测")
    
    print("✅ 后处理器测试完成！")

