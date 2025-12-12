# -*- coding: utf-8 -*-
"""
CLIP方法统一接口基类

所有CLIP方法（surgeryclip, declip, diffclip）都应实现此接口
用于统一推理、训练和评估
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from abc import ABC, abstractmethod
from PIL import Image


class BaseCLIPMethod(ABC):
    """
    CLIP方法统一接口基类
    
    所有CLIP方法都应继承此类并实现必要的方法
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        """
        初始化CLIP方法
        
        Args:
            model_name: 模型名称
            device: 设备
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.preprocess = None
    
    @abstractmethod
    def load_model(self, checkpoint_path: Optional[str] = None):
        """
        加载模型
        
        Args:
            checkpoint_path: 检查点路径（可选）
        """
        pass
    
    @abstractmethod
    def encode_image(self, image: Union[torch.Tensor, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        编码图像
        
        Args:
            image: 输入图像（Tensor/PIL/ndarray）
        
        Returns:
            image_features: [B, D] 图像特征
        """
        pass
    
    @abstractmethod
    def encode_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        编码文本
        
        Args:
            text: 文本或文本列表
        
        Returns:
            text_features: [B, D] 文本特征
        """
        pass
    
    @abstractmethod
    def compute_similarity(self, image: Union[torch.Tensor, Image.Image, np.ndarray], 
                          text: Union[str, List[str]]) -> torch.Tensor:
        """
        计算图像-文本相似度
        
        Args:
            image: 输入图像
            text: 文本或文本列表
        
        Returns:
            similarity: [B, N] 相似度矩阵（B=图像数，N=文本数）
        """
        pass
    
    def generate_heatmap(self, image: Union[torch.Tensor, Image.Image, np.ndarray],
                        text: Union[str, List[str]],
                        return_features: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        生成热图（用于目标检测）
        
        Args:
            image: 输入图像
            text: 文本或文本列表
            return_features: 是否返回中间特征
        
        Returns:
            heatmap: [H, W] 热图，范围[0, 1]
            或 (heatmap, features_dict) 如果return_features=True
        """
        # 默认实现：使用相似度生成热图
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        # 计算相似度
        similarity = (image_features @ text_features.T)  # [B, N]
        
        # 如果图像特征有空间维度，可以生成热图
        # 这里需要子类实现具体逻辑
        raise NotImplementedError("子类需要实现generate_heatmap方法")
    
    def inference(self, image_path: str, text_queries: List[str],
                 threshold: float = 0.5) -> Dict:
        """
        推理接口（统一接口）
        
        Args:
            image_path: 图像路径
            text_queries: 文本查询列表
            threshold: 相似度阈值
        
        Returns:
            results: {
                'heatmap': np.ndarray,  # [H, W] 热图
                'bboxes': List[Dict],   # 边界框列表
                'similarities': np.ndarray,  # [N] 相似度
                'predictions': List[Dict]    # 预测结果
            }
        """
        # 加载图像
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        # 生成热图
        heatmap = self.generate_heatmap(image, text_queries)
        
        # 从热图生成边界框（使用eval模块）
        from .eval import heatmap_to_bboxes
        bboxes = heatmap_to_bboxes(heatmap, threshold=threshold)
        
        # 计算相似度
        image_features = self.encode_image(image)
        text_features = self.encode_text(text_queries)
        similarities = (image_features @ text_features.T).detach().cpu().numpy()
        
        # 确保similarities是numpy数组
        import numpy as np
        if not isinstance(similarities, np.ndarray):
            similarities = np.array(similarities)
        
        # 处理similarities的形状
        if similarities.ndim == 2:
            # [batch_size, num_texts]
            similarities_flat = similarities[0] if similarities.shape[0] > 0 else similarities.flatten()
        else:
            similarities_flat = similarities.flatten()
        
        return {
            'heatmap': heatmap,
            'bboxes': bboxes,
            'similarities': similarities,
            'predictions': [
                {
                    'text': text_queries[i],
                    'similarity': float(similarities_flat[i]) if i < len(similarities_flat) else 0.0,
                    'bboxes': [b for b in bboxes if b.get('class_id') == i]
                }
                for i in range(len(text_queries))
            ]
        }
    
    def train_step(self, batch: Dict) -> Dict:
        """
        训练步骤（统一接口）
        
        Args:
            batch: 批次数据
        
        Returns:
            loss_dict: 损失字典
        """
        raise NotImplementedError("子类需要实现train_step方法")
    
    def evaluate(self, dataset, metrics: List[str] = ['mAP']) -> Dict:
        """
        评估接口（统一接口）
        
        Args:
            dataset: 数据集
            metrics: 评估指标列表
        
        Returns:
            metrics_dict: 指标字典
        """
        from .eval import evaluate_bboxes_with_gt, multi_threshold_evaluation
        
        all_predictions = []
        all_ground_truths = []
        
        for idx in range(len(dataset)):
            sample = dataset[idx]
            # 推理
            results = self.inference(
                sample['image'],
                sample.get('text_queries', sample.get('classes', [])),
                threshold=0.5
            )
            all_predictions.append(results['bboxes'])
            all_ground_truths.append(sample.get('gt_boxes', []))
        
        # 评估
        if 'mAP' in metrics:
            eval_results = evaluate_bboxes_with_gt(
                all_predictions, all_ground_truths, iou_threshold=0.5
            )
            return eval_results
        
        return {}

