# -*- coding: utf-8 -*-
"""
直接检测头
从CAM热图 + 图像特征直接预测框，无需阈值检测
类似FCOS/YOLO的anchor-free检测器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math


class DirectDetectionHead(nn.Module):
    """
    直接检测头
    
    输入: CAM热图 + 图像特征（patch features）
    输出: 每个位置的框预测和置信度
    
    架构:
    1. 融合CAM和图像特征
    2. 使用CNN提取特征
    3. 直接预测框坐标和置信度
    """
    
    def __init__(self, num_classes: int, 
                 feature_dim: int = 768,  # 图像特征维度
                 hidden_dim: int = 256,
                 cam_resolution: int = 7,
                 use_image_features: bool = True):
        """
        Args:
            num_classes: 类别数
            feature_dim: 图像特征维度（patch features）
            hidden_dim: 隐藏层维度
            cam_resolution: CAM分辨率
            use_image_features: 是否使用图像特征
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.cam_resolution = cam_resolution
        self.use_image_features = use_image_features
        
        # 如果使用图像特征，需要将patch features投影并上采样到CAM分辨率
        if use_image_features:
            # 投影层：将图像特征维度降到hidden_dim
            self.feature_proj = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            )
            
            # 输入通道：CAM (C) + 图像特征 (hidden_dim)
            input_channels = num_classes + hidden_dim
        else:
            # 只使用CAM
            input_channels = num_classes
        
        # 特征融合和提取
        self.conv1 = nn.Conv2d(
            input_channels, hidden_dim,
            kernel_size=3, padding=1
        )
        self.gn1 = nn.GroupNorm(32, hidden_dim)
        
        self.conv2 = nn.Conv2d(
            hidden_dim, hidden_dim,
            kernel_size=3, padding=1
        )
        self.gn2 = nn.GroupNorm(32, hidden_dim)
        
        self.conv3 = nn.Conv2d(
            hidden_dim, hidden_dim,
            kernel_size=3, padding=1
        )
        self.gn3 = nn.GroupNorm(32, hidden_dim)
        
        # 框回归头：每个位置预测框坐标
        # 输出: [B, C, H, W, 4] (xmin, ymin, xmax, ymax) 归一化坐标
        self.box_head = nn.Conv2d(
            hidden_dim, num_classes * 4,
            kernel_size=1
        )
        
        # 置信度头：每个位置预测置信度
        # 输出: [B, C, H, W] 置信度分数
        self.conf_head = nn.Conv2d(
            hidden_dim, num_classes,
            kernel_size=1
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, cam: torch.Tensor, 
                image_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            cam: [B, C, H, W] CAM热图
            image_features: [B, N², D] 图像特征（patch features），可选
        
        Returns:
            Dict with:
                - boxes: [B, C, H, W, 4] 预测框坐标（归一化）
                - confidences: [B, C, H, W] 置信度分数
                - features: [B, hidden_dim, H, W] 融合后的特征（可选）
        """
        B, C, H, W = cam.shape
        
        # 准备输入特征
        if self.use_image_features and image_features is not None:
            # image_features: [B, N², D]
            # 需要reshape并上采样到CAM分辨率
            
            # 1. 投影到hidden_dim
            feat_proj = self.feature_proj(image_features)  # [B, N², hidden_dim]
            
            # 2. Reshape到空间维度
            N = int(math.sqrt(image_features.shape[1]))  # 假设是正方形
            feat_spatial = feat_proj.view(B, N, N, -1).permute(0, 3, 1, 2)  # [B, hidden_dim, N, N]
            
            # 3. 上采样到CAM分辨率
            if N != H or N != W:
                feat_spatial = F.interpolate(
                    feat_spatial, 
                    size=(H, W), 
                    mode='bilinear', 
                    align_corners=False
                )  # [B, hidden_dim, H, W]
            
            # 4. 融合CAM和图像特征
            x = torch.cat([cam, feat_spatial], dim=1)  # [B, C + hidden_dim, H, W]
        else:
            # 只使用CAM
            x = cam  # [B, C, H, W]
        
        # 特征提取
        x = F.relu(self.gn1(self.conv1(x)))  # [B, hidden_dim, H, W]
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.relu(self.gn3(self.conv3(x)))
        
        # 框回归
        box_logits = self.box_head(x)  # [B, C*4, H, W]
        box_logits = box_logits.view(B, C, 4, H, W)
        box_logits = box_logits.permute(0, 1, 3, 4, 2)  # [B, C, H, W, 4]
        
        # 解码框坐标（相对于网格位置）
        boxes = self._decode_boxes(box_logits, H, W)  # [B, C, H, W, 4]
        
        # 置信度预测
        conf_logits = self.conf_head(x)  # [B, C, H, W]
        confidences = torch.sigmoid(conf_logits)  # [B, C, H, W]
        
        # 结合CAM作为额外的置信度信号
        cam_conf = torch.sigmoid(cam)  # [B, C, H, W]
        final_confidences = confidences * cam_conf  # 融合预测置信度和CAM
        
        return {
            'boxes': boxes,
            'confidences': final_confidences,
            'raw_confidences': confidences,
            'cam_confidences': cam_conf,
            'features': x  # 用于可视化或进一步处理
        }
    
    def _decode_boxes(self, box_logits: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        解码框坐标
        
        Args:
            box_logits: [B, C, H, W, 4] 框参数logits
            H, W: 空间分辨率
        
        Returns:
            boxes: [B, C, H, W, 4] (xmin, ymin, xmax, ymax) 归一化坐标
        """
        B, C, H_out, W_out, _ = box_logits.shape
        
        # 生成网格（每个位置的中心坐标）
        device = box_logits.device
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, H_out, device=device),
            torch.linspace(0, 1, W_out, device=device),
            indexing='ij'
        )
        grid_x = grid_x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        grid_y = grid_y.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # 解析框参数
        # 方案1: 直接预测相对于网格中心的偏移和尺寸
        delta_x = torch.tanh(box_logits[..., 0]) * 0.5  # [-0.5, 0.5]
        delta_y = torch.tanh(box_logits[..., 1]) * 0.5
        w = torch.sigmoid(box_logits[..., 2])  # [0, 1]
        h = torch.sigmoid(box_logits[..., 3])  # [0, 1]
        
        # 预测的中心位置
        cx = grid_x + delta_x  # [B, C, H, W]
        cy = grid_y + delta_y
        
        # 限制在[0, 1]范围内
        cx = torch.clamp(cx, 0, 1)
        cy = torch.clamp(cy, 0, 1)
        
        # 限制w和h，避免框太大
        w = torch.clamp(w, 0.01, 1.0)
        h = torch.clamp(h, 0.01, 1.0)
        
        # 转换为框坐标
        xmin = (cx - w / 2).clamp(0, 1)
        ymin = (cy - h / 2).clamp(0, 1)
        xmax = (cx + w / 2).clamp(0, 1)
        ymax = (cy + h / 2).clamp(0, 1)
        
        boxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1)  # [B, C, H, W, 4]
        
        return boxes
    
    def inference(self, cam: torch.Tensor,
                  image_features: Optional[torch.Tensor] = None,
                  conf_threshold: float = 0.3,
                  nms_threshold: float = 0.5,
                  topk: int = 100) -> Dict:
        """
        推理接口
        
        Args:
            cam: [B, C, H, W] CAM
            image_features: [B, N², D] 图像特征
            conf_threshold: 置信度阈值
            nms_threshold: NMS阈值
            topk: 保留的top-k检测
        
        Returns:
            detections: List[List[dict]] 每个图像的检测结果
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(cam, image_features)
            boxes = outputs['boxes']  # [B, C, H, W, 4]
            confidences = outputs['confidences']  # [B, C, H, W]
            
            B, C, H, W = confidences.shape
            all_detections = []
            
            for b in range(B):
                detections = []
                
                # 收集所有位置的检测
                for c in range(C):
                    conf_class = confidences[b, c]  # [H, W]
                    boxes_class = boxes[b, c]  # [H, W, 4]
                    
                    # 找到置信度超过阈值的位置
                    mask = conf_class > conf_threshold
                    if mask.sum() == 0:
                        continue
                    
                    # 提取检测
                    conf_values = conf_class[mask]  # [N]
                    box_values = boxes_class[mask]  # [N, 4]
                    positions = mask.nonzero(as_tuple=False)  # [N, 2]
                    
                    for i, (pos, conf, box) in enumerate(zip(positions, conf_values, box_values)):
                        detections.append({
                            'box': box.cpu(),
                            'confidence': conf.item(),
                            'class': c,
                            'position': (pos[0].item(), pos[1].item())
                        })
                
                # 按置信度排序
                detections.sort(key=lambda x: x['confidence'], reverse=True)
                
                # NMS（简化版，按类别分别处理）
                if len(detections) > 0:
                    import torchvision.ops
                    
                    boxes_tensor = torch.stack([d['box'] for d in detections])
                    scores_tensor = torch.tensor([d['confidence'] for d in detections])
                    
                    keep = torchvision.ops.nms(boxes_tensor, scores_tensor, nms_threshold)
                    detections = [detections[i] for i in keep.tolist()]
                    
                    # Top-k
                    detections = detections[:topk]
                
                all_detections.append(detections)
            
            return all_detections


