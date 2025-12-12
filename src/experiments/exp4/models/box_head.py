# -*- coding: utf-8 -*-
"""
BoxHead Regression Head
从CAM预测框参数 (Δcx, Δcy, w, h)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class BoxHead(nn.Module):
    """
    BoxHead: 从CAM回归框参数
    
    输入: CAM [B, C, H, W]
    输出: box参数 [B, C, H, W, 4] (Δcx, Δcy, w, h)
    """
    
    def __init__(self, num_classes: int, hidden_dim: int = 256, 
                 cam_resolution: int = 7, upsample: bool = False):
        """
        Args:
            num_classes: Number of classes (C)
            hidden_dim: Hidden dimension for CNN
            cam_resolution: Original CAM resolution (e.g., 7 for ViT-B/32)
            upsample: Whether to upsample CAM before regression
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.cam_resolution = cam_resolution
        self.upsample = upsample
        
        # Optional upsampling layer
        if upsample:
            self.upsample_layer = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False
            )
            input_res = cam_resolution * 2
        else:
            input_res = cam_resolution
        
        # CNN layers for box regression
        self.conv1 = nn.Conv2d(
            num_classes, hidden_dim, 
            kernel_size=3, padding=1
        )
        self.gn1 = nn.GroupNorm(32, hidden_dim)
        
        self.conv2 = nn.Conv2d(
            hidden_dim, hidden_dim,
            kernel_size=3, padding=1
        )
        self.gn2 = nn.GroupNorm(32, hidden_dim)
        
        # Output layer: 4 parameters per class (Δcx, Δcy, w, h)
        self.box_head = nn.Conv2d(
            hidden_dim, num_classes * 4,
            kernel_size=1
        )
        
        # Optional: objectness head (confidence score)
        self.obj_head = nn.Conv2d(
            num_classes, num_classes,
            kernel_size=1
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, cam: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            cam: [B, C, H, W] Class Activation Maps
        
        Returns:
            Dict with:
                - box_params: [B, C, H, W, 4] (Δcx, Δcy, w, h)
                - objectness: [B, C, H, W] (optional)
        """
        B, C, H, W = cam.shape
        
        # Optional upsampling
        if self.upsample:
            cam = self.upsample_layer(cam)
            H, W = cam.shape[2], cam.shape[3]
        
        # Feature extraction
        x = F.relu(self.gn1(self.conv1(cam)))
        x = F.relu(self.gn2(self.conv2(x)))
        
        # Box regression
        box_params = self.box_head(x)  # [B, C*4, H, W]
        box_params = box_params.view(B, C, 4, H, W)
        box_params = box_params.permute(0, 1, 3, 4, 2)  # [B, C, H, W, 4]
        
        # Parse parameters
        delta_cx = box_params[..., 0]  # [B, C, H, W]
        delta_cy = box_params[..., 1]
        w = torch.sigmoid(box_params[..., 2])  # [0, 1]
        h = torch.sigmoid(box_params[..., 3])   # [0, 1]
        
        # Optional: objectness score
        objectness = torch.sigmoid(self.obj_head(cam))  # [B, C, H, W]
        
        return {
            'box_params': box_params,
            'delta_cx': delta_cx,
            'delta_cy': delta_cy,
            'w': w,
            'h': h,
            'objectness': objectness
        }
    
    def decode_boxes(self, box_params: torch.Tensor, 
                    cam_resolution: int = None) -> torch.Tensor:
        """
        Decode box parameters to actual boxes
        
        Args:
            box_params: [B, C, H, W, 4] (Δcx, Δcy, w, h)
            cam_resolution: CAM resolution (for grid generation)
        
        Returns:
            boxes: [B, C, H, W, 4] (xmin, ymin, xmax, ymax) in normalized coordinates
        """
        B, C, H, W, _ = box_params.shape
        
        if cam_resolution is None:
            cam_resolution = H
        
        # Generate grid (anchor centers)
        device = box_params.device
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, H, device=device),
            torch.linspace(0, 1, W, device=device),
            indexing='ij'
        )
        # grid_x, grid_y: [H, W]
        
        # Expand to batch and class dimensions
        grid_x = grid_x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        grid_y = grid_y.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Extract parameters
        delta_cx = box_params[..., 0]  # [B, C, H, W]
        delta_cy = box_params[..., 1]
        w = torch.sigmoid(box_params[..., 2])  # [B, C, H, W]
        h = torch.sigmoid(box_params[..., 3])
        
        # Predicted center = anchor center + offset
        cx = grid_x + delta_cx  # [B, C, H, W]
        cy = grid_y + delta_cy
        
        # Clamp centers to [0, 1]
        cx = torch.clamp(cx, 0, 1)
        cy = torch.clamp(cy, 0, 1)
        
        # Convert to box coordinates
        xmin = (cx - w / 2).clamp(0, 1)
        ymin = (cy - h / 2).clamp(0, 1)
        xmax = (cx + w / 2).clamp(0, 1)
        ymax = (cy + h / 2).clamp(0, 1)
        
        # Stack to [B, C, H, W, 4]
        boxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1)
        
        return boxes

