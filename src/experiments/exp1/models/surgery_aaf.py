# -*- coding: utf-8 -*-
"""
SurgeryCLIP + AAF + p2p Propagation Model

Main model that integrates:
1. Pre-trained SurgeryCLIP (dual-path)
2. AAF layer to fuse multi-layer attentions
3. p2p propagation for CAM generation
"""

import torch
import torch.nn as nn
import sys
import os
from typing import List, Tuple, Dict

# Add path to surgeryclip - will be used in create_surgery_aaf_model
# We delay the import to avoid relative import issues

from .aaf import AAF
from .cam_generator import CAMGenerator


class SurgeryAAF(nn.Module):
    """
    SurgeryCLIP + AAF + p2p propagation
    
    Architecture:
    1. Use pre-trained SurgeryCLIP (dual-path)
    2. AAF fuses attention from last 6 layers
    3. p2p propagation optimizes CAM
    """
    
    def __init__(self, surgery_clip_model, num_layers=6):
        """
        Args:
            surgery_clip_model: Pre-loaded SurgeryCLIP model
            num_layers: Number of layers to collect attention from (default 6)
        """
        super().__init__()
        
        # SurgeryCLIP model
        self.clip = surgery_clip_model
        
        # AAF layer (trainable)
        self.aaf = AAF(num_layers=num_layers)
        
        # CAM generator
        self.cam_generator = CAMGenerator()
        
        # Storage for collecting attention
        self.vv_attentions = []
        self.ori_attentions = []
        
        # Track if hooks are registered (SurgeryCLIP uses lazy initialization)
        self._hooks_registered = False
        self._hook_handles = []
        
    def _register_hooks(self):
        """
        Register forward hooks for SurgeryCLIP's last 6 layers to collect attention weights.
        Note: This is called lazily because SurgeryCLIP replaces attention modules during first forward.
        We register hooks after the first forward pass when attention modules are replaced.
        """
        if self._hooks_registered:
            return
        
        def make_vv_hook(storage):
            def hook(module, input, output):
                # Get VV attention from Attention module
                if hasattr(module, 'attn_weights_vv'):
                    # attn_weights_vv: [B, num_heads, 1+N², 1+N²]
                    storage.append(module.attn_weights_vv)
            return hook
        
        def make_ori_hook(storage):
            def hook(module, input, output):
                # Get original path attention from Attention module
                if hasattr(module, 'attn_weights_ori'):
                    # attn_weights_ori: [B, num_heads, 1+N², 1+N²]
                    storage.append(module.attn_weights_ori)
            return hook
        
        # Register hooks for last 6 layers
        # Note: SurgeryCLIP applies surgery to last 6 layers (indices -6 to -1)
        visual = self.clip.visual
        
        if hasattr(visual, 'transformer') and hasattr(visual.transformer, 'resblocks'):
            blocks = visual.transformer.resblocks
            # Import Attention class to check type
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from src.competitors.clip_methods.surgeryclip.clip_surgery_model import Attention
            
            for i in range(1, 7):  # Last 6 layers
                block = blocks[-i]
                if hasattr(block, 'attn'):
                    # Check if it's an Attention instance (after first forward)
                    if isinstance(block.attn, Attention):
                        handle1 = block.attn.register_forward_hook(make_vv_hook(self.vv_attentions))
                        handle2 = block.attn.register_forward_hook(make_ori_hook(self.ori_attentions))
                        self._hook_handles.extend([handle1, handle2])
                        self._hooks_registered = True
    
    def forward(self, images, text_queries):
        """
        Args:
            images: [B, 3, H, W]
            text_queries: List[str] - class names, e.g., ["airplane", "ship", "car"]
        
        Returns:
            cam: [B, C, N, N]
            aux_outputs: dict - auxiliary outputs (for visualization, debugging)
        """
        # Clear attention storage from previous forward pass
        self.vv_attentions.clear()
        self.ori_attentions.clear()
        
        # ===== Step 1: Encode text =====
        with torch.no_grad():
            # Use CLIP's text encoder
            # Import tokenize from surgeryclip
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from src.competitors.clip_methods.surgeryclip.clip import tokenize
            text_tokens = tokenize(text_queries).to(images.device)
            text_features = self.clip.encode_text(text_tokens)
            text_features = text_features.float()  # [C, D]
        
        # ===== Step 2: Encode image (will automatically collect attention via hooks) =====
        # Get full tokens (cls + patches)
        # This will trigger SurgeryCLIP's lazy initialization and replace attention modules
        # Note: CLIP is frozen, so we use no_grad to avoid inplace operation issues
        # But we need to detach and require_grad for patch_features to flow gradients through AAF
        with torch.no_grad():
            if hasattr(self.clip.visual, 'encode_image_with_all_tokens'):
                image_features_all = self.clip.visual.encode_image_with_all_tokens(images)
                # image_features_all: [B, 1+N², D]
            else:
                # Fallback: use standard encode_image and extract patches manually
                # This is a workaround if encode_image_with_all_tokens is not available
                raise NotImplementedError("encode_image_with_all_tokens not available. Please ensure SurgeryCLIP is properly modified.")
        
        # Register hooks after first forward (when attention modules are replaced)
        self._register_hooks()
        
        # Extract patch tokens (remove cls token) and enable gradients
        # We detach from CLIP computation graph but allow gradients for AAF
        patch_features = image_features_all[:, 1:, :].detach().requires_grad_(True)  # [B, N², D]
        
        # ===== Step 3: AAF fuse attention =====
        # If hooks weren't registered yet, we need to do another forward pass
        # This happens on the first call
        if len(self.vv_attentions) == 0 or len(self.ori_attentions) == 0:
            # Re-run to collect attention with hooks registered
            with torch.no_grad():
                image_features_all = self.clip.visual.encode_image_with_all_tokens(images)
            patch_features = image_features_all[:, 1:, :].detach().requires_grad_(True)
        
        # Ensure we collected attention from all 6 layers
        if len(self.vv_attentions) != 6 or len(self.ori_attentions) != 6:
            raise RuntimeError(f"Expected 6 attention tensors, got {len(self.vv_attentions)} VV and {len(self.ori_attentions)} original")
        
        attn_p2p = self.aaf(self.vv_attentions, self.ori_attentions)
        # attn_p2p: [B, N², N²]
        
        # ===== Step 4: Generate CAM =====
        # 获取CLIP的视觉投影层（如果存在）
        clip_visual_proj = None
        if hasattr(self.clip.visual, 'proj'):
            clip_visual_proj = self.clip.visual.proj.data  # [D_img, D_text]
        
        cam = self.cam_generator(
            patch_features,
            text_features,  # [C, D]
            attn_p2p,
            clip_visual_proj=clip_visual_proj
        )
        # cam: [B, C, N, N]
        
        # Auxiliary outputs
        aux_outputs = {
            'attn_p2p': attn_p2p,
            'patch_features': patch_features,
            'text_features': text_features,
            'vv_attentions': self.vv_attentions,
            'ori_attentions': self.ori_attentions
        }
        
        return cam, aux_outputs


def create_surgery_aaf_model(checkpoint_path, device='cuda', num_layers=6):
    """
    Factory function to create SurgeryAAF model
    
    Args:
        checkpoint_path: Path to SurgeryCLIP checkpoint
        device: Device to load model on
        num_layers: Number of layers for AAF (default 6)
    
    Returns:
        SurgeryAAF model instance
    """
    # Import build_model here to avoid relative import issues
    # Add project root to path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Import as a package module from project root
    from src.competitors.clip_methods.surgeryclip import build_model
    
    # Load SurgeryCLIP model (build_model is a function, not a module)
    surgery_clip, preprocess = build_model(
        model_name='surgeryclip',
        checkpoint_path=checkpoint_path,
        device=device
    )
    
    # Create SurgeryAAF
    model = SurgeryAAF(surgery_clip, num_layers=num_layers)
    model = model.to(device)
    
    return model, preprocess

