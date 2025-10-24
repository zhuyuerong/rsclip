#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage1: 候选框分类模块
对生成的候选框进行分类和置信度评估
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import open_clip
from PIL import Image
import os
import sys

# 添加父目录到路径以导入原有模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wordnet_vocabulary import WORDNET_REMOTE_SENSING_CLASSES, get_synonyms


class ProposalClassifier:
    """候选框分类器"""
    
    def __init__(self, model_name: str = 'RN50', device: str = 'cuda'):
        """
        初始化候选框分类器
        
        参数:
            model_name: RemoteCLIP模型名称
            device: 计算设备
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.class_list = None
        self.text_features = None
        
        # 初始化模型
        self._load_model()
        self._setup_classification_vocabulary()
    
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
    
    def _setup_classification_vocabulary(self, custom_classes: List[str] = None):
        """
        设置分类词表
        
        参数:
            custom_classes: 自定义类别列表
        """
        if custom_classes:
            self.class_list = custom_classes
        else:
            # 使用默认的遥感类别
            self.class_list = WORDNET_REMOTE_SENSING_CLASSES.copy()
        
        print(f"📋 设置分类词表: {len(self.class_list)} 个类别")
        
        # 预编码文本特征
        print("🔄 编码文本特征...")
        text_tokens = self.tokenizer(self.class_list)
        with torch.no_grad():
            self.text_features = self.model.encode_text(text_tokens.to(self.device))
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        
        print("✅ 文本特征编码完成")
    
    def classify_proposals(self, proposals: List[Dict]) -> List[Dict]:
        """
        对候选框进行分类
        
        参数:
            proposals: 候选框列表
        
        返回:
            分类后的候选框列表
        """
        print(f"\n🔧 对 {len(proposals)} 个候选框进行分类...")
        
        classified_proposals = []
        
        for proposal in proposals:
            if proposal.get('features') is None:
                continue
            
            # 获取特征
            features = torch.from_numpy(proposal['features']).to(self.device)
            
            # 计算与所有类别的相似度
            with torch.no_grad():
                similarities = (features @ self.text_features.T).squeeze(0).cpu().numpy()
            
            # 获取Top-K预测
            top_k = min(5, len(self.class_list))
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            # 构建分类结果
            classification_results = []
            for idx in top_indices:
                classification_results.append({
                    'class': self.class_list[idx],
                    'confidence': float(similarities[idx]),
                    'rank': len(classification_results) + 1
                })
            
            # 更新候选框信息
            proposal['classification'] = classification_results
            proposal['predicted_class'] = classification_results[0]['class']
            proposal['prediction_confidence'] = classification_results[0]['confidence']
            proposal['top_k_predictions'] = classification_results
            
            classified_proposals.append(proposal)
        
        print(f"✅ 分类完成")
        
        return classified_proposals
    
    def filter_proposals_by_confidence(self, proposals: List[Dict], 
                                     min_confidence: float = 0.1) -> List[Dict]:
        """
        按置信度过滤候选框
        
        参数:
            proposals: 候选框列表
            min_confidence: 最小置信度阈值
        
        返回:
            过滤后的候选框列表
        """
        print(f"\n🔧 按置信度过滤候选框 (阈值: {min_confidence})...")
        
        filtered_proposals = []
        
        for proposal in proposals:
            confidence = proposal.get('prediction_confidence', 0)
            
            if confidence >= min_confidence:
                filtered_proposals.append(proposal)
        
        print(f"✅ 置信度过滤: {len(proposals)} -> {len(filtered_proposals)} 个候选框")
        
        return filtered_proposals
    
    def filter_proposals_by_class(self, proposals: List[Dict], 
                                target_classes: List[str]) -> List[Dict]:
        """
        按类别过滤候选框
        
        参数:
            proposals: 候选框列表
            target_classes: 目标类别列表
        
        返回:
            过滤后的候选框列表
        """
        print(f"\n🔧 按类别过滤候选框 (目标类别: {target_classes})...")
        
        filtered_proposals = []
        
        for proposal in proposals:
            predicted_class = proposal.get('predicted_class', '')
            
            # 检查是否匹配目标类别或其同义词
            if predicted_class in target_classes:
                filtered_proposals.append(proposal)
                continue
            
            # 检查同义词
            synonyms = get_synonyms(predicted_class)
            if any(syn in target_classes for syn in synonyms):
                filtered_proposals.append(proposal)
                continue
            
            # 检查目标类别的同义词
            for target_class in target_classes:
                target_synonyms = get_synonyms(target_class)
                if predicted_class in target_synonyms:
                    filtered_proposals.append(proposal)
                    break
        
        print(f"✅ 类别过滤: {len(proposals)} -> {len(filtered_proposals)} 个候选框")
        
        return filtered_proposals
    
    def rank_proposals_by_confidence(self, proposals: List[Dict]) -> List[Dict]:
        """
        按置信度排序候选框
        
        参数:
            proposals: 候选框列表
        
        返回:
            排序后的候选框列表
        """
        print(f"\n🔧 按置信度排序候选框...")
        
        sorted_proposals = sorted(
            proposals, 
            key=lambda x: x.get('prediction_confidence', 0), 
            reverse=True
        )
        
        print(f"✅ 排序完成")
        
        return sorted_proposals
    
    def classify_proposals_pipeline(self, proposals: List[Dict],
                                  target_classes: List[str] = None,
                                  min_confidence: float = 0.1,
                                  top_k: int = 10) -> List[Dict]:
        """
        完整的候选框分类流水线
        
        参数:
            proposals: 候选框列表
            target_classes: 目标类别列表
            min_confidence: 最小置信度阈值
            top_k: 返回前K个结果
        
        返回:
            分类后的候选框列表
        """
        print(f"\n🚀 开始候选框分类流水线...")
        
        # 1. 分类
        classified_proposals = self.classify_proposals(proposals)
        
        # 2. 按置信度过滤
        filtered_proposals = self.filter_proposals_by_confidence(
            classified_proposals, 
            min_confidence
        )
        
        # 3. 按类别过滤（如果指定了目标类别）
        if target_classes:
            filtered_proposals = self.filter_proposals_by_class(
                filtered_proposals, 
                target_classes
            )
        
        # 4. 按置信度排序
        ranked_proposals = self.rank_proposals_by_confidence(filtered_proposals)
        
        # 5. 返回Top-K
        final_proposals = ranked_proposals[:top_k]
        
        print(f"✅ 候选框分类流水线完成，返回 {len(final_proposals)} 个候选框")
        
        return final_proposals
    
    def get_classification_statistics(self, proposals: List[Dict]) -> Dict:
        """
        获取分类统计信息
        
        参数:
            proposals: 候选框列表
        
        返回:
            统计信息字典
        """
        if not proposals:
            return {}
        
        # 统计类别分布
        class_counts = {}
        confidences = []
        
        for proposal in proposals:
            predicted_class = proposal.get('predicted_class', 'unknown')
            confidence = proposal.get('prediction_confidence', 0)
            
            class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
            confidences.append(confidence)
        
        stats = {
            'total_proposals': len(proposals),
            'class_distribution': class_counts,
            'confidence_stats': {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            },
            'top_classes': sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }
        
        return stats


def main():
    """测试候选框分类器"""
    print("=" * 70)
    print("测试候选框分类器")
    print("=" * 70)
    
    # 创建模拟候选框
    mock_proposals = [
        {
            'proposal_id': 0,
            'bbox': (100, 100, 200, 200),
            'area': 10000,
            'features': np.random.randn(1, 512)  # 模拟特征
        },
        {
            'proposal_id': 1,
            'bbox': (300, 200, 400, 300),
            'area': 10000,
            'features': np.random.randn(1, 512)
        },
        {
            'proposal_id': 2,
            'bbox': (150, 250, 250, 350),
            'area': 10000,
            'features': np.random.randn(1, 512)
        }
    ]
    
    # 创建候选框分类器
    classifier = ProposalClassifier()
    
    # 目标类别
    target_classes = ['airplane', 'building', 'ship']
    
    # 运行完整流水线
    classified_proposals = classifier.classify_proposals_pipeline(
        mock_proposals,
        target_classes=target_classes,
        min_confidence=0.05,
        top_k=5
    )
    
    # 获取统计信息
    stats = classifier.get_classification_statistics(classified_proposals)
    print(f"\n📊 分类统计:")
    print(f"  总候选框数: {stats.get('total_proposals', 0)}")
    print(f"  置信度范围: {stats.get('confidence_stats', {}).get('min', 0):.3f} - {stats.get('confidence_stats', {}).get('max', 0):.3f}")
    print(f"  类别分布: {stats.get('class_distribution', {})}")
    
    # 显示每个候选框的分类结果
    print(f"\n📋 分类结果:")
    for i, proposal in enumerate(classified_proposals):
        print(f"  候选框 {i+1}:")
        print(f"    位置: {proposal['bbox']}")
        print(f"    预测类别: {proposal.get('predicted_class', 'unknown')}")
        print(f"    置信度: {proposal.get('prediction_confidence', 0):.3f}")
        
        # 显示Top-3预测
        top_predictions = proposal.get('top_k_predictions', [])[:3]
        print(f"    Top-3预测:")
        for pred in top_predictions:
            print(f"      {pred['rank']}. {pred['class']}: {pred['confidence']:.3f}")
    
    print("\n✅ 候选框分类器测试完成!")


if __name__ == "__main__":
    main()
