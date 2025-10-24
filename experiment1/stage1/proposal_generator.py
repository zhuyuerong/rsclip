#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage1: 候选框生成模块
基于采样区域生成候选检测框
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
import open_clip
from PIL import Image
import os
import sys

# 添加父目录到路径以导入原有模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class ProposalGenerator:
    """候选框生成器"""
    
    def __init__(self, model_name: str = 'RN50', device: str = 'cuda'):
        """
        初始化候选框生成器
        
        参数:
            model_name: RemoteCLIP模型名称
            device: 计算设备
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        
        # 初始化模型
        self._load_model()
    
    def _load_model(self):
        """加载RemoteCLIP模型"""
        print(f"🔄 加载RemoteCLIP模型: {self.model_name}")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.model_name)
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        
        checkpoint_path = f"checkpoints/RemoteCLIP-{self.model_name}.pt"
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(ckpt)
        self.model = self.model.to(self.device).eval()
        
        print(f"✅ 模型已加载到 {self.device}")
    
    def generate_proposals_from_regions(self, image: np.ndarray, regions: List[Dict]) -> List[Dict]:
        """
        从采样区域生成候选框
        
        参数:
            image: 输入图像
            regions: 采样区域列表
        
        返回:
            候选框列表
        """
        print(f"\n🔧 从 {len(regions)} 个区域生成候选框...")
        
        proposals = []
        
        for idx, region in enumerate(regions):
            # 获取区域边界框
            x1, y1, x2, y2 = region['bbox']
            
            # 确保坐标在有效范围内
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(image.shape[1], int(x2)), min(image.shape[0], int(y2))
            
            # 裁剪区域
            crop = image[y1:y2, x1:x2]
            
            # 检查有效性
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue
            
            # 创建候选框
            proposal = {
                'proposal_id': idx,
                'bbox': (x1, y1, x2, y2),
                'region_info': region,
                'crop': crop,
                'area': (x2 - x1) * (y2 - y1),
                'aspect_ratio': (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 1.0,
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'confidence': 0.0,  # 将在后续阶段计算
                'features': None,   # 将在后续阶段计算
                'category': 'unknown'  # 将在分类阶段确定
            }
            
            proposals.append(proposal)
        
        print(f"✅ 生成 {len(proposals)} 个候选框")
        
        return proposals
    
    def refine_proposals(self, proposals: List[Dict], 
                        min_area: int = 100,
                        max_area: int = 50000,
                        min_aspect_ratio: float = 0.1,
                        max_aspect_ratio: float = 10.0) -> List[Dict]:
        """
        精化候选框
        
        参数:
            proposals: 候选框列表
            min_area: 最小面积
            max_area: 最大面积
            min_aspect_ratio: 最小宽高比
            max_aspect_ratio: 最大宽高比
        
        返回:
            精化后的候选框列表
        """
        print(f"\n🔧 精化候选框...")
        
        refined_proposals = []
        
        for proposal in proposals:
            area = proposal['area']
            aspect_ratio = proposal['aspect_ratio']
            
            # 面积过滤
            if area < min_area or area > max_area:
                continue
            
            # 宽高比过滤
            if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                continue
            
            refined_proposals.append(proposal)
        
        print(f"✅ 精化后保留: {len(proposals)} -> {len(refined_proposals)} 个候选框")
        
        return refined_proposals
    
    def extract_proposal_features(self, proposals: List[Dict]) -> List[Dict]:
        """
        提取候选框特征
        
        参数:
            proposals: 候选框列表
        
        返回:
            带特征的候选框列表
        """
        print(f"\n🔧 提取 {len(proposals)} 个候选框的特征...")
        
        for proposal in proposals:
            crop = proposal['crop']
            
            # 转换为PIL图像并预处理
            crop_pil = Image.fromarray(crop)
            crop_tensor = self.preprocess(crop_pil).unsqueeze(0)
            
            # 提取特征
            with torch.no_grad():
                features = self.model.encode_image(crop_tensor.to(self.device))
                features /= features.norm(dim=-1, keepdim=True)
                
                proposal['features'] = features.cpu().numpy()
        
        print(f"✅ 特征提取完成")
        
        return proposals
    
    def compute_proposal_similarities(self, proposals: List[Dict], 
                                    query_classes: List[str]) -> List[Dict]:
        """
        计算候选框与查询类别的相似度
        
        参数:
            proposals: 候选框列表
            query_classes: 查询类别列表
        
        返回:
            带相似度的候选框列表
        """
        print(f"\n🔧 计算与 {len(query_classes)} 个类别的相似度...")
        
        # 编码查询类别
        query_tokens = self.tokenizer(query_classes)
        with torch.no_grad():
            query_features = self.model.encode_text(query_tokens.to(self.device))
            query_features /= query_features.norm(dim=-1, keepdim=True)
        
        # 计算相似度
        for proposal in proposals:
            if proposal['features'] is None:
                continue
            
            features = torch.from_numpy(proposal['features']).to(self.device)
            
            # 计算相似度
            similarities = (features @ query_features.T).squeeze(0).cpu().numpy()
            
            # 找到最相似的类别
            best_idx = similarities.argmax()
            best_class = query_classes[best_idx]
            best_similarity = similarities[best_idx]
            
            proposal['similarities'] = similarities
            proposal['best_class'] = best_class
            proposal['best_similarity'] = best_similarity
            proposal['confidence'] = best_similarity
        
        print(f"✅ 相似度计算完成")
        
        return proposals
    
    def generate_proposals_pipeline(self, image: np.ndarray, regions: List[Dict],
                                  query_classes: List[str] = None,
                                  **kwargs) -> List[Dict]:
        """
        完整的候选框生成流水线
        
        参数:
            image: 输入图像
            regions: 采样区域列表
            query_classes: 查询类别列表
            **kwargs: 其他参数
        
        返回:
            完整的候选框列表
        """
        print(f"\n🚀 开始候选框生成流水线...")
        
        # 1. 从区域生成候选框
        proposals = self.generate_proposals_from_regions(image, regions)
        
        # 2. 精化候选框
        proposals = self.refine_proposals(proposals, **kwargs)
        
        # 3. 提取特征
        proposals = self.extract_proposal_features(proposals)
        
        # 4. 计算相似度（如果提供了查询类别）
        if query_classes:
            proposals = self.compute_proposal_similarities(proposals, query_classes)
        
        print(f"✅ 候选框生成流水线完成，得到 {len(proposals)} 个候选框")
        
        return proposals
    
    def get_proposal_statistics(self, proposals: List[Dict]) -> Dict:
        """
        获取候选框统计信息
        
        参数:
            proposals: 候选框列表
        
        返回:
            统计信息字典
        """
        if not proposals:
            return {}
        
        areas = [p['area'] for p in proposals]
        aspect_ratios = [p['aspect_ratio'] for p in proposals]
        confidences = [p.get('confidence', 0) for p in proposals]
        
        stats = {
            'total_proposals': len(proposals),
            'area_stats': {
                'mean': np.mean(areas),
                'std': np.std(areas),
                'min': np.min(areas),
                'max': np.max(areas)
            },
            'aspect_ratio_stats': {
                'mean': np.mean(aspect_ratios),
                'std': np.std(aspect_ratios),
                'min': np.min(aspect_ratios),
                'max': np.max(aspect_ratios)
            },
            'confidence_stats': {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            }
        }
        
        return stats


def main():
    """测试候选框生成器"""
    print("=" * 70)
    print("测试候选框生成器")
    print("=" * 70)
    
    # 测试图像
    test_image_path = "assets/airport.jpg"
    if not os.path.exists(test_image_path):
        print(f"❌ 测试图像不存在: {test_image_path}")
        return
    
    # 加载测试图像
    image = cv2.imread(test_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 创建模拟区域
    h, w = image.shape[:2]
    mock_regions = [
        {'bbox': (100, 100, 200, 200), 'score': 0.8, 'saliency': 0.7},
        {'bbox': (300, 200, 400, 300), 'score': 0.6, 'saliency': 0.5},
        {'bbox': (150, 250, 250, 350), 'score': 0.9, 'saliency': 0.8},
    ]
    
    # 创建候选框生成器
    generator = ProposalGenerator()
    
    # 查询类别
    query_classes = ['airplane', 'building', 'runway', 'vehicle']
    
    # 运行完整流水线
    proposals = generator.generate_proposals_pipeline(
        image, 
        mock_regions, 
        query_classes
    )
    
    # 获取统计信息
    stats = generator.get_proposal_statistics(proposals)
    print(f"\n📊 候选框统计:")
    print(f"  总候选框数: {stats.get('total_proposals', 0)}")
    print(f"  面积范围: {stats.get('area_stats', {}).get('min', 0)} - {stats.get('area_stats', {}).get('max', 0)}")
    print(f"  置信度范围: {stats.get('confidence_stats', {}).get('min', 0):.3f} - {stats.get('confidence_stats', {}).get('max', 0):.3f}")
    
    # 显示每个候选框的信息
    print(f"\n📋 候选框详情:")
    for i, proposal in enumerate(proposals):
        print(f"  候选框 {i+1}:")
        print(f"    位置: {proposal['bbox']}")
        print(f"    面积: {proposal['area']}")
        print(f"    最佳类别: {proposal.get('best_class', 'unknown')}")
        print(f"    置信度: {proposal.get('confidence', 0):.3f}")
    
    print("\n✅ 候选框生成器测试完成!")


if __name__ == "__main__":
    main()
