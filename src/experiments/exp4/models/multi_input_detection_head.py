# -*- coding: utf-8 -*-
"""
多输入检测头
接收原图特征、融合CAM、多层特征，预测框和置信度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import math


class MultiInputDetectionHead(nn.Module):
    """
    多输入检测头
    
    输入组织（分组处理）:
    ├─ 原图特征 [B, 128, 7, 7]
    ├─ 融合CAM [B, 20, 7, 7]
    └─ 多层特征 [B, 256*3, 7, 7]
    
    输出:
    ├─ 框坐标 [B, C, H, W, 4]
    └─ 置信度 [B, C, H, W]
    """
    
    def __init__(
        self, 
        num_classes: int = 20,
        img_feat_dim: int = 128,
        cam_dim: int = 20,
        layer_feat_dim: int = 256,
        num_layers: int = 3,
        hidden_dim: int = 256,
        cam_resolution: int = 7
    ):
        """
        Args:
            num_classes: 类别数
            img_feat_dim: 原图特征维度
            cam_dim: CAM维度（等于num_classes）
            layer_feat_dim: 每层特征维度
            num_layers: 层数
            hidden_dim: 隐藏层维度
            cam_resolution: CAM分辨率
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.cam_resolution = cam_resolution
        
        # ===== 分组投影 =====
        
        # 原图特征投影
        self.img_proj = nn.Sequential(
            nn.Conv2d(img_feat_dim, hidden_dim, 1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # CAM投影
        self.cam_proj = nn.Sequential(
            nn.Conv2d(cam_dim, hidden_dim, 1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 多层特征投影（先融合多层，再投影）
        multi_layer_input_dim = layer_feat_dim * num_layers
        self.multi_layer_proj = nn.Sequential(
            nn.Conv2d(multi_layer_input_dim, hidden_dim * 2, 1),  # 先压缩
            nn.GroupNorm(32, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1),  # 再投影
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # ===== 最终融合 =====
        # 输入: hidden_dim * 3 (img + cam + multi)
        fusion_input_dim = hidden_dim * 3
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_input_dim, hidden_dim * 2, 3, padding=1),
            nn.GroupNorm(32, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # ===== 输出预测 =====
        
        # 框回归头
        self.box_head = nn.Conv2d(hidden_dim, num_classes * 4, 1)
        
        # 置信度头
        self.conf_head = nn.Conv2d(hidden_dim, num_classes, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        img_features: torch.Tensor,
        fused_cam: torch.Tensor,
        multi_layer_features: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            img_features: [B, img_feat_dim, H, W] 原图特征
            fused_cam: [B, C, H, W] 融合后的CAM
            multi_layer_features: List[[B, layer_feat_dim, H, W]] (num_layers个)
        
        Returns:
            Dict with:
                - boxes: [B, C, H, W, 4] 预测框坐标
                - confidences: [B, C, H, W] 置信度分数
                - raw_confidences: [B, C, H, W] 原始置信度（未与CAM相乘）
        """
        B = img_features.shape[0]
        H, W = img_features.shape[2], img_features.shape[3]
        
        # ===== 分组处理 =====
        
        # 原图特征投影
        img_feat = self.img_proj(img_features)  # [B, hidden_dim, H, W]
        
        # CAM投影
        cam_feat = self.cam_proj(fused_cam)  # [B, hidden_dim, H, W]
        
        # 多层特征融合和投影
        # multi_layer_features是List[[B, N², D]]，需要转换为空间格式
        multi_feats_aligned = []
        for layer_feat in multi_layer_features:
            # layer_feat是 [B, N², D]
            if layer_feat.dim() == 3:
                N_sq = layer_feat.shape[1]
                D = layer_feat.shape[2]
                N = int(math.sqrt(N_sq))
                
                # 验证N²是否匹配
                if N * N != N_sq:
                    raise ValueError(f"N² mismatch: {N_sq} is not a perfect square")
                
                # [B, N², D] -> [B, D, N, N]
                layer_feat_spatial = layer_feat.view(B, N, N, D).permute(0, 3, 1, 2)
            else:
                # 已经是空间格式 [B, D, H', W']
                layer_feat_spatial = layer_feat
            
            # 上采样到CAM分辨率（如果需要）
            if layer_feat_spatial.shape[2] != H or layer_feat_spatial.shape[3] != W:
                layer_feat_spatial = F.interpolate(
                    layer_feat_spatial, size=(H, W), 
                    mode='bilinear', align_corners=False
                )
            
            multi_feats_aligned.append(layer_feat_spatial)
        
        # 融合多层特征
        multi_stack = torch.cat(multi_feats_aligned, dim=1)  # [B, layer_feat_dim*num_layers, H, W]
        multi_feat = self.multi_layer_proj(multi_stack)  # [B, hidden_dim, H, W]
        
        # ===== 最终融合 =====
        x = torch.cat([img_feat, cam_feat, multi_feat], dim=1)  # [B, hidden_dim*3, H, W]
        x = self.fusion_conv(x)  # [B, hidden_dim, H, W]
        
        # ===== 预测 =====
        
        # 框回归
        box_logits = self.box_head(x)  # [B, C*4, H, W]
        box_logits = box_logits.view(B, self.num_classes, 4, H, W)
        box_logits = box_logits.permute(0, 1, 3, 4, 2)  # [B, C, H, W, 4]
        
        # 解码框坐标
        boxes = self._decode_boxes(box_logits, H, W)  # [B, C, H, W, 4]
        
        # 置信度预测
        conf_logits = self.conf_head(x)  # [B, C, H, W]
        raw_confidences = torch.sigmoid(conf_logits)  # [B, C, H, W]
        
        # CAM增强置信度
        cam_conf = torch.sigmoid(fused_cam)  # [B, C, H, W]
        final_confidences = raw_confidences * cam_conf  # [B, C, H, W]
        
        return {
            'boxes': boxes,
            'confidences': final_confidences,
            'raw_confidences': raw_confidences,
            'cam_confidences': cam_conf,
            'features': x  # 用于可视化
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
    
    def get_param_count(self) -> int:
        """获取参数量"""
        return sum(p.numel() for p in self.parameters())

