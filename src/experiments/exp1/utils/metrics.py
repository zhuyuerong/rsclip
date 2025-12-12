# -*- coding: utf-8 -*-
"""
Evaluation metrics utilities
"""

import numpy as np
import torch


def compute_metrics(cam, labels, text_queries, threshold=0.5):
    """
    Compute evaluation metrics
    
    Args:
        cam: [C, H, W] or [B, C, H, W] - CAM
        labels: [C] or [B, C] - ground truth labels (multi-label)
        text_queries: List[str] - class names
        threshold: Threshold for binarizing CAM scores
    
    Returns:
        metrics: dict with accuracy, precision, recall, f1
    """
    # Handle batch dimension
    if cam.ndim == 4:
        # [B, C, H, W] -> process first sample
        cam = cam[0]
        labels = labels[0] if labels.ndim == 2 else labels
    
    # Convert to numpy if tensor
    if isinstance(cam, torch.Tensor):
        cam = cam.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Extract prediction scores from CAM (max pooling)
    cam_scores = cam.reshape(cam.shape[0], -1).max(axis=1)  # [C]
    
    # Binarize predictions
    predictions = (cam_scores > threshold).astype(int)
    labels = labels.astype(int)
    
    # Compute metrics
    metrics = {}
    
    # Accuracy (exact match)
    accuracy = (predictions == labels).mean()
    metrics['accuracy'] = float(accuracy)
    
    # Precision
    if predictions.sum() > 0:
        precision = (predictions * labels).sum() / predictions.sum()
    else:
        precision = 0.0
    metrics['precision'] = float(precision)
    
    # Recall
    if labels.sum() > 0:
        recall = (predictions * labels).sum() / labels.sum()
    else:
        recall = 0.0
    metrics['recall'] = float(recall)
    
    # F1 score
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    metrics['f1'] = float(f1)
    
    # Per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(text_queries):
        if i < len(predictions):
            per_class_metrics[class_name] = {
                'predicted': int(predictions[i]),
                'ground_truth': int(labels[i]),
                'score': float(cam_scores[i])
            }
    
    metrics['per_class'] = per_class_metrics
    
    return metrics


def compute_ap(cam, labels, text_queries):
    """
    Compute Average Precision (AP) for each class
    
    Args:
        cam: [C, H, W] - CAM
        labels: [C] - ground truth labels
        text_queries: List[str] - class names
    
    Returns:
        ap_per_class: dict mapping class name to AP
        mean_ap: mean AP across all classes
    """
    if isinstance(cam, torch.Tensor):
        cam = cam.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Extract scores
    cam_scores = cam.reshape(cam.shape[0], -1).max(axis=1)  # [C]
    
    ap_per_class = {}
    aps = []
    
    for i, class_name in enumerate(text_queries):
        if i >= len(cam_scores):
            continue
        
        score = cam_scores[i]
        label = int(labels[i])
        
        # For binary classification, AP = precision at threshold where recall = 1
        # Simplified: use score as confidence
        if label == 1:
            # Positive class: higher score is better
            ap = score
        else:
            # Negative class: lower score is better (1 - score)
            ap = 1.0 - score
        
        ap_per_class[class_name] = float(ap)
        aps.append(ap)
    
    mean_ap = np.mean(aps) if aps else 0.0
    
    return ap_per_class, float(mean_ap)





