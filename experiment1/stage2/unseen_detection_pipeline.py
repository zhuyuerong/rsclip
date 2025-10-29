#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
未见目标检测Pipeline
基于RemoteCLIP和对比学习的未见目标检测系统
"""

import torch
import torch.nn.functional as F
import open_clip
from PIL import Image
import numpy as np
import cv2
from typing import List, Dict, Tuple
import argparse

from sampling import sample_regions
from wordnet_vocabulary import get_full_class_list, get_expansion_words, WORDNET_80_CLASSES
from bbox_refinement import BBoxRefinement, compute_saliency_map


class UnseenDetectionPipeline:
    """未见目标检测Pipeline"""
    
    def __init__(self, model_name='RN50', device='cuda'):
        """
        初始化Pipeline
        
        参数:
            model_name: RemoteCLIP模型名称
            device: 计算设备
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        
        # 加载模型
        print(f"🔄 加载RemoteCLIP模型: {model_name}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        checkpoint_path = f"checkpoints/RemoteCLIP-{model_name}.pt"
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(ckpt)
        self.model = self.model.to(self.device).eval()
        print(f"✅ 模型已加载到 {self.device}")
        
        # 词表和特征缓存
        self.class_list = None
        self.text_features = None
        self.expansion_indices = None
        
        # 框微调器
        self.bbox_refiner = BBoxRefinement(self.model, self.preprocess, self.device)
        
    def setup_vocabulary(self, unseen_class=None):
        """
        设置词表
        
        参数:
            unseen_class: 未见类别名称（可选）
        """
        # 构建完整类别列表
        self.class_list = get_full_class_list(unseen_class)
        
        if unseen_class:
            # 记录扩展词的索引
            expansion_words = get_expansion_words(unseen_class, num_words=5)
            self.expansion_indices = [
                self.class_list.index(word) for word in expansion_words
            ]
            print(f"📊 词表设置: 80基础类 + 5扩展词 + 1未见类 = {len(self.class_list)}类")
            print(f"   未见类别: {unseen_class}")
            print(f"   扩展词: {expansion_words}")
        else:
            self.expansion_indices = None
            print(f"📊 词表设置: {len(self.class_list)}个基础类别")
        
        # 预编码文本特征
        print("🔄 编码文本特征...")
        text_tokens = self.tokenizer(self.class_list)
        with torch.no_grad():
            self.text_features = self.model.encode_text(text_tokens.to(self.device))
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        print("✅ 文本特征编码完成")
    
    def step1_density_map(self, image, strategy='multi_threshold_saliency', max_regions=50):
        """
        Step 1: 密度图计算 → 候选区域
        
        参数:
            image: 输入图像 (numpy array, RGB)
            strategy: 采样策略
            max_regions: 最大区域数
        
        返回:
            候选区域列表
        """
        print("\n" + "="*70)
        print("📍 Step 1: 密度图计算 → 候选区域")
        print("="*70)
        
        regions = sample_regions(image, strategy=strategy, max_regions=max_regions)
        print(f"✅ 提取到 {len(regions)} 个候选区域")
        
        return regions
    
    def step2_intelligent_crop(self, image, regions):
        """
        Step 2: 智能切割 → 预选框（crops）
        
        参数:
            image: 原始图像
            regions: 候选区域列表
        
        返回:
            crops列表和对应的区域信息
        """
        print("\n" + "="*70)
        print("📍 Step 2: 智能切割 → 预选框")
        print("="*70)
        
        crops = []
        valid_regions = []
        
        for idx, region in enumerate(regions):
            x1, y1, x2, y2 = region['bbox']
            
            # 确保坐标有效
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            
            # 裁剪区域
            crop = image[y1:y2, x1:x2]
            
            # 检查有效性
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue
            
            crops.append(crop)
            valid_regions.append(region)
        
        print(f"✅ 生成 {len(crops)} 个有效预选框")
        return crops, valid_regions
    
    def step3_batch_inference(self, crops):
        """
        Step 3: RemoteCLIP批量推理
        
        参数:
            crops: 预选框列表
        
        返回:
            相似度矩阵 [N × 86]
        """
        print("\n" + "="*70)
        print("📍 Step 3: RemoteCLIP批量推理")
        print("="*70)
        
        if self.text_features is None:
            raise ValueError("请先调用 setup_vocabulary() 设置词表")
        
        similarity_matrix = []
        
        print(f"🔄 处理 {len(crops)} 个crops...")
        for idx, crop in enumerate(crops):
            # 转换为PIL图像并预处理
            crop_pil = Image.fromarray(crop)
            crop_tensor = self.preprocess(crop_pil).unsqueeze(0)
            
            # 编码图像
            with torch.no_grad():
                image_features = self.model.encode_image(crop_tensor.to(self.device))
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # 计算相似度
                similarities = (image_features @ self.text_features.T).squeeze(0)
                similarity_matrix.append(similarities.cpu().numpy())
            
            if (idx + 1) % 10 == 0:
                print(f"   已处理: {idx + 1}/{len(crops)}")
        
        similarity_matrix = np.array(similarity_matrix)  # [N, 86]
        print(f"✅ 相似度矩阵: {similarity_matrix.shape}")
        
        return similarity_matrix
    
    def step4_initial_filtering(self, similarity_matrix, score_threshold=0.15):
        """
        Step 4: 初步筛选
        
        参数:
            similarity_matrix: 相似度矩阵 [N × 86]
            score_threshold: 最低分数阈值
        
        返回:
            正样本索引、负样本索引、有效索引
        """
        print("\n" + "="*70)
        print("📍 Step 4: 初步筛选")
        print("="*70)
        
        N = similarity_matrix.shape[0]
        
        # 4.1 移除低分噪声
        max_scores = similarity_matrix.max(axis=1)
        valid_mask = max_scores > score_threshold
        valid_indices = np.where(valid_mask)[0]
        
        print(f"📊 噪声过滤:")
        print(f"   - 原始样本数: {N}")
        print(f"   - 分数阈值: {score_threshold}")
        print(f"   - 保留样本数: {len(valid_indices)}")
        print(f"   - 移除噪声: {N - len(valid_indices)}")
        
        # 4.2 根据扩展词标注正负样本
        positive_samples = []
        negative_samples = []
        
        if self.expansion_indices is not None:
            # 计算扩展词的平均分数
            expansion_scores = similarity_matrix[:, self.expansion_indices].mean(axis=1)
            
            # 计算基础类别的最大分数
            base_scores = similarity_matrix[:, :80].max(axis=1)
            
            # 正样本: 扩展词分数 > 基础类别分数
            for idx in valid_indices:
                if expansion_scores[idx] > base_scores[idx]:
                    positive_samples.append(idx)
                else:
                    negative_samples.append(idx)
            
            print(f"\n📊 正负样本标注:")
            print(f"   - 正样本（可能的未见目标）: {len(positive_samples)}")
            print(f"   - 负样本（干扰背景）: {len(negative_samples)}")
        else:
            print("\n⚠️  未设置未见类别，跳过正负样本标注")
        
        return positive_samples, negative_samples, valid_indices
    
    def step5_contrastive_refinement(self, image, crops, regions, similarity_matrix, 
                                     positive_samples, negative_samples,
                                     refine_bbox=True, refine_method='both'):
        """
        Step 5: 对比学习精化 + 框微调
        
        参数:
            image: 原始图像
            crops: 预选框列表
            regions: 区域信息
            similarity_matrix: 相似度矩阵
            positive_samples: 正样本索引
            negative_samples: 负样本索引
            refine_bbox: 是否进行框微调
            refine_method: 框微调方法 ('position', 'scale', 'both', 'boundary')
        
        返回:
            精化后的检测结果
        """
        print("\n" + "="*70)
        print("📍 Step 5: 对比学习精化" + (" + 框微调" if refine_bbox else ""))
        print("="*70)
        
        if len(positive_samples) == 0:
            print("⚠️  没有正样本，跳过精化")
            return []
        
        # 5.1 构建正负样本原型
        print("🔄 构建特征原型...")
        
        # 编码正样本特征
        positive_features = []
        for idx in positive_samples:
            crop_pil = Image.fromarray(crops[idx])
            crop_tensor = self.preprocess(crop_pil).unsqueeze(0)
            with torch.no_grad():
                feat = self.model.encode_image(crop_tensor.to(self.device))
                feat /= feat.norm(dim=-1, keepdim=True)
                positive_features.append(feat)
        
        positive_prototype = torch.cat(positive_features).mean(dim=0, keepdim=True)
        
        # 编码负样本特征（如果有）
        if len(negative_samples) > 0:
            negative_features = []
            for idx in negative_samples[:min(10, len(negative_samples))]:  # 限制数量
                crop_pil = Image.fromarray(crops[idx])
                crop_tensor = self.preprocess(crop_pil).unsqueeze(0)
                with torch.no_grad():
                    feat = self.model.encode_image(crop_tensor.to(self.device))
                    feat /= feat.norm(dim=-1, keepdim=True)
                    negative_features.append(feat)
            
            negative_prototype = torch.cat(negative_features).mean(dim=0, keepdim=True)
        else:
            negative_prototype = None
        
        print("✅ 原型构建完成")
        
        # 5.2 计算显著性图（用于框微调）
        if refine_bbox:
            print("🔄 计算显著性图用于框微调...")
            saliency_map = compute_saliency_map(image)
        else:
            saliency_map = None
        
        # 5.3 重新评分 + 框微调
        print(f"🔄 基于原型重新评分" + (" + 框微调..." if refine_bbox else "..."))
        refined_results = []
        
        for idx in positive_samples:
            crop_pil = Image.fromarray(crops[idx])
            crop_tensor = self.preprocess(crop_pil).unsqueeze(0)
            
            with torch.no_grad():
                feat = self.model.encode_image(crop_tensor.to(self.device))
                feat /= feat.norm(dim=-1, keepdim=True)
                
                # 与正原型的相似度
                pos_sim = (feat @ positive_prototype.T).item()
                
                # 与负原型的相似度
                if negative_prototype is not None:
                    neg_sim = (feat @ negative_prototype.T).item()
                    # 对比分数
                    contrast_score = pos_sim - neg_sim
                else:
                    contrast_score = pos_sim
                
                # 获取最可能的类别
                similarities = similarity_matrix[idx]
                top_class_idx = similarities.argmax()
                top_class = self.class_list[top_class_idx]
                top_score = similarities[top_class_idx]
                
                # 扩展词平均分数
                if self.expansion_indices:
                    expansion_score = similarities[self.expansion_indices].mean()
                else:
                    expansion_score = 0.0
                
                # 框微调
                initial_bbox = regions[idx]['bbox']
                if refine_bbox and saliency_map is not None:
                    refine_result = self.bbox_refiner.refine_bbox_hybrid(
                        image=image,
                        bbox=initial_bbox,
                        saliency_map=saliency_map,
                        positive_prototype=positive_prototype,
                        negative_prototype=negative_prototype,
                        method=refine_method
                    )
                    refined_bbox = refine_result['bbox']
                    bbox_refined = refine_result.get('refined', False)
                    saliency_score = refine_result.get('saliency_score', 0.0)
                    composite_score = refine_result.get('composite_score', contrast_score)
                else:
                    refined_bbox = initial_bbox
                    bbox_refined = False
                    saliency_score = 0.0
                    composite_score = contrast_score
                
                refined_results.append({
                    'index': idx,
                    'bbox': refined_bbox,
                    'initial_bbox': initial_bbox,
                    'bbox_refined': bbox_refined,
                    'top_class': top_class,
                    'top_score': float(top_score),
                    'expansion_score': float(expansion_score),
                    'contrast_score': float(contrast_score),
                    'saliency_score': float(saliency_score),
                    'composite_score': float(composite_score),
                    'is_unseen': contrast_score > 0.1  # 简单阈值判断
                })
        
        # 按综合分数排序
        refined_results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # 统计框微调情况
        if refine_bbox:
            n_refined = sum(1 for r in refined_results if r['bbox_refined'])
            print(f"   框微调统计: {n_refined}/{len(refined_results)} 个框被优化")
        
        print(f"✅ 精化完成，得到 {len(refined_results)} 个候选目标")
        
        return refined_results
    
    def run_pipeline(self, image, unseen_class=None, strategy='multi_threshold_saliency',
                     max_regions=50, score_threshold=0.15, top_k=10,
                     refine_bbox=True, refine_method='both'):
        """
        运行完整Pipeline
        
        参数:
            image: 输入图像 (numpy array, RGB)
            unseen_class: 未见类别名称
            strategy: 采样策略
            max_regions: 最大区域数
            score_threshold: 分数阈值
            top_k: 返回前K个结果
            refine_bbox: 是否进行框微调
            refine_method: 框微调方法 ('position', 'scale', 'both', 'boundary')
        
        返回:
            检测结果列表
        """
        print("\n" + "="*70)
        print("🚀 未见目标检测Pipeline")
        print("="*70)
        
        # 设置词表
        self.setup_vocabulary(unseen_class)
        
        # Step 1: 密度图计算
        regions = self.step1_density_map(image, strategy, max_regions)
        
        # Step 2: 智能切割
        crops, valid_regions = self.step2_intelligent_crop(image, regions)
        
        # Step 3: 批量推理
        similarity_matrix = self.step3_batch_inference(crops)
        
        # Step 4: 初步筛选
        positive_samples, negative_samples, valid_indices = self.step4_initial_filtering(
            similarity_matrix, score_threshold
        )
        
        # Step 5: 对比学习精化 + 框微调
        refined_results = self.step5_contrastive_refinement(
            image, crops, valid_regions, similarity_matrix, 
            positive_samples, negative_samples,
            refine_bbox=refine_bbox, refine_method=refine_method
        )
        
        # 返回Top-K结果
        final_results = refined_results[:top_k]
        
        print("\n" + "="*70)
        print(f"✅ Pipeline完成! 返回前{len(final_results)}个结果")
        print("="*70)
        
        return final_results


def main():
    parser = argparse.ArgumentParser(description='未见目标检测Pipeline')
    parser.add_argument('--image', type=str, default='assets/airport.jpg',
                        help='输入图像路径')
    parser.add_argument('--unseen-class', type=str, default=None,
                        help='未见类别名称（例如: wind turbine）')
    parser.add_argument('--model', type=str, default='RN50',
                        choices=['RN50', 'ViT-B-32', 'ViT-L-14'],
                        help='模型选择')
    parser.add_argument('--strategy', type=str, default='multi_threshold_saliency',
                        choices=['layered', 'pyramid', 'multi_threshold_saliency'],
                        help='采样策略')
    parser.add_argument('--max-regions', type=int, default=50,
                        help='最大区域数')
    parser.add_argument('--threshold', type=float, default=0.15,
                        help='分数阈值')
    parser.add_argument('--top-k', type=int, default=10,
                        help='返回前K个结果')
    parser.add_argument('--refine-bbox', action='store_true', default=True,
                        help='启用框微调（默认启用）')
    parser.add_argument('--no-refine-bbox', action='store_true',
                        help='禁用框微调')
    parser.add_argument('--refine-method', type=str, default='both',
                        choices=['position', 'scale', 'both', 'boundary'],
                        help='框微调方法: position(位置), scale(尺寸), both(两者), boundary(边界)')
    
    args = parser.parse_args()
    
    # 加载图像
    image = cv2.imread(args.image)
    if image is None:
        print(f"❌ 无法加载图像: {args.image}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 创建Pipeline
    pipeline = UnseenDetectionPipeline(model_name=args.model)
    
    # 运行Pipeline
    refine_bbox = args.refine_bbox and not args.no_refine_bbox
    
    results = pipeline.run_pipeline(
        image=image,
        unseen_class=args.unseen_class,
        strategy=args.strategy,
        max_regions=args.max_regions,
        score_threshold=args.threshold,
        top_k=args.top_k,
        refine_bbox=refine_bbox,
        refine_method=args.refine_method
    )
    
    # 打印结果
    print("\n" + "="*70)
    print("🎯 检测结果")
    print("="*70)
    
    for i, result in enumerate(results, 1):
        print(f"\n结果 {i}:")
        if result.get('bbox_refined', False):
            print(f"  初始位置: {result['initial_bbox']}")
            print(f"  优化位置: {result['bbox']} ✓")
        else:
            print(f"  位置: {result['bbox']}")
        print(f"  最可能类别: {result['top_class']} ({result['top_score']:.3f})")
        print(f"  扩展词分数: {result['expansion_score']:.3f}")
        print(f"  对比分数: {result['contrast_score']:.3f}")
        print(f"  显著性分数: {result.get('saliency_score', 0.0):.3f}")
        print(f"  综合分数: {result.get('composite_score', 0.0):.3f}")
        print(f"  是否未见目标: {'✓' if result['is_unseen'] else '✗'}")


if __name__ == "__main__":
    main()

