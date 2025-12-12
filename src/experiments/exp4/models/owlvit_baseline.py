# -*- coding: utf-8 -*-
"""
OWL-ViT Baseline Model
使用transformers库的OWL-ViT模型作为baseline
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import sys
import os
from pathlib import Path

try:
    from transformers import OwlViTProcessor, OwlViTForObjectDetection
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. OWL-ViT baseline will not work.")


class OWLViTBaseline(nn.Module):
    """
    OWL-ViT Baseline Model for Object Detection
    
    封装transformers库的OWL-ViT模型，提供统一的检测接口
    """
    
    def __init__(self, model_name: str = "google/owlvit-base-patch32", 
                 device: str = "cuda"):
        """
        Args:
            model_name: HuggingFace model name
            device: Device to load model on
        """
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required for OWL-ViT. "
                "Install with: pip install transformers"
            )
        
        self.device = device
        self.model_name = model_name
        
        # Load processor and model
        print(f"Loading OWL-ViT model: {model_name}")
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.model.eval()
        
        # Image size for preprocessing
        self.image_size = 224
    
    def forward(self, images: torch.Tensor, text_queries: List[str]):
        """
        Forward pass
        
        Args:
            images: [B, 3, H, W] normalized images
            text_queries: List of class names (same for all images in batch)
        
        Returns:
            Dict with:
                - pred_boxes: List of [N, 4] boxes (normalized)
                - pred_scores: List of [N] scores
                - pred_labels: List of [N] labels
        """
        B = images.shape[0]
        
        # Convert tensor to PIL for processor
        # Note: images are already normalized, need to denormalize
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(images.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(images.device)
        
        images_denorm = images * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
        images_denorm = torch.clamp(images_denorm, 0, 1)
        
        # Convert to PIL
        from PIL import Image
        import torchvision.transforms.functional as F
        
        pil_images = []
        for i in range(B):
            img_tensor = images_denorm[i]
            img_pil = F.to_pil_image(img_tensor)
            pil_images.append(img_pil)
        
        # Process inputs
        inputs = self.processor(
            text=text_queries,
            images=pil_images,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract predictions
        pred_boxes = []
        pred_scores = []
        pred_labels = []
        
        # Get target sizes (original image sizes)
        target_sizes = torch.tensor([[self.image_size, self.image_size]] * B).to(self.device)
        
        # Post-process outputs
        results = self.processor.post_process_object_detection(
            outputs,
            threshold=0.0,  # We'll filter later
            target_sizes=target_sizes
        )
        
        for result in results:
            boxes = result['boxes']  # [N, 4] in (xmin, ymin, xmax, ymax) format
            scores = result['scores']  # [N]
            labels = result['labels']  # [N]
            
            # Normalize boxes to [0, 1]
            boxes = boxes / self.image_size
            boxes = torch.clamp(boxes, 0, 1)
            
            # Convert to (xmin, ymin, xmax, ymax) format if needed
            # OWL-ViT already outputs in this format
            
            pred_boxes.append(boxes)
            pred_scores.append(scores)
            pred_labels.append(labels)
        
        return {
            'pred_boxes': pred_boxes,
            'pred_scores': pred_scores,
            'pred_labels': pred_labels
        }
    
    def inference(self, images: torch.Tensor, text_queries: List[str],
                  conf_threshold: float = 0.3, nms_threshold: float = 0.5):
        """
        Inference with NMS
        
        Args:
            images: [B, 3, H, W] normalized images
            text_queries: List of class names
            conf_threshold: Confidence threshold
            nms_threshold: NMS IoU threshold
        
        Returns:
            Dict with filtered predictions
        """
        outputs = self.forward(images, text_queries)
        
        # Apply NMS per image
        from torchvision.ops import nms
        
        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []
        
        for boxes, scores, labels in zip(
            outputs['pred_boxes'],
            outputs['pred_scores'],
            outputs['pred_labels']
        ):
            # Filter by confidence
            mask = scores >= conf_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]
            
            if len(boxes) == 0:
                filtered_boxes.append(torch.zeros((0, 4), device=boxes.device))
                filtered_scores.append(torch.zeros((0,), device=scores.device))
                filtered_labels.append(torch.zeros((0,), dtype=torch.int64, device=labels.device))
                continue
            
            # Apply NMS per class
            keep_all = []
            for class_id in labels.unique():
                class_mask = labels == class_id
                if not class_mask.any():
                    continue
                
                boxes_class = boxes[class_mask]
                scores_class = scores[class_mask]
                
                keep = nms(boxes_class, scores_class, nms_threshold)
                keep_all.extend(torch.where(class_mask)[0][keep].tolist())
            
            if len(keep_all) > 0:
                keep_all = torch.tensor(keep_all, device=boxes.device)
                filtered_boxes.append(boxes[keep_all])
                filtered_scores.append(scores[keep_all])
                filtered_labels.append(labels[keep_all])
            else:
                filtered_boxes.append(torch.zeros((0, 4), device=boxes.device))
                filtered_scores.append(torch.zeros((0,), device=scores.device))
                filtered_labels.append(torch.zeros((0,), dtype=torch.int64, device=labels.device))
        
        return {
            'pred_boxes': filtered_boxes,
            'pred_scores': filtered_scores,
            'pred_labels': filtered_labels
        }


def create_owlvit_model(model_name: str = "google/owlvit-base-patch32",
                        device: str = "cuda") -> OWLViTBaseline:
    """
    Factory function to create OWL-ViT model
    
    Args:
        model_name: HuggingFace model name
        device: Device to load model on
    
    Returns:
        OWLViTBaseline instance
    """
    return OWLViTBaseline(model_name=model_name, device=device)


