#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理引擎：统一接口
提供不同模块的推理功能调用接口
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import cv2

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from inference.utils.model_loader import create_model_loader


class InferenceEngine:
    """推理引擎"""
    
    def __init__(self, model_name: str = 'RN50', device: str = 'cuda'):
        """
        初始化推理引擎
        
        参数:
            model_name: 模型名称
            device: 计算设备
        """
        self.model_name = model_name
        self.device = device
        self.model_loader = create_model_loader(model_name, device)
        
        # 加载模型
        self.model, self.preprocess, self.tokenizer = self.model_loader.load_model()
    
    def run_stage1_pipeline(self, image_path: str, **kwargs):
        """
        运行Stage1流水线
        
        参数:
            image_path: 图像路径
            **kwargs: 其他参数
        
        返回:
            Stage1结果
        """
        print(f"\n🚀 运行Stage1流水线...")
        
        # 加载图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 1. 数据加载
        print("📍 Step 1.1: 数据加载")
        # 这里可以添加数据加载逻辑
        
        # 2. 区域采样
        print("📍 Step 1.2: 区域采样")
        from experiment1.stage1.sampling.region_sampler import ExperimentRegionSampler
        
        sampler = ExperimentRegionSampler(kwargs.get('sampling_strategy', 'multi_threshold_saliency'))
        regions = sampler.sample_regions(image, max_regions=kwargs.get('max_regions', 50))
        
        # 3. 候选框生成
        print("📍 Step 1.3: 候选框生成")
        from experiment1.stage1.proposal_generation.proposal_generator import ProposalGenerator
        
        generator = ProposalGenerator(self.model_name, self.device)
        proposals = generator.generate_proposals_from_regions(image, regions)
        
        # 4. 候选框分类
        print("📍 Step 1.4: 候选框分类")
        from experiment1.stage1.proposal_classification.proposal_classifier import ProposalClassifier
        
        classifier = ProposalClassifier(self.model_name, self.device)
        classified_proposals = classifier.classify_proposals_pipeline(proposals)
        
        # 保存Stage1结果
        stage1_results = {
            'regions': regions,
            'proposals': classified_proposals,
            'image_shape': image.shape
        }
        
        print(f"✅ Stage1流水线完成")
        
        return stage1_results
    
    def run_stage2_pipeline(self, image_path: str, stage1_results: dict, **kwargs):
        """
        运行Stage2流水线
        
        参数:
            image_path: 图像路径
            stage1_results: Stage1结果
            **kwargs: 其他参数
        
        返回:
            Stage2结果
        """
        print(f"\n🚀 运行Stage2流水线...")
        
        # 加载图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        proposals = stage1_results['proposals']
        
        # 1. 候选框打分
        print("📍 Step 2.1: 候选框打分")
        from experiment1.stage2.scoring.proposal_scorer import ProposalScorer
        
        scorer = ProposalScorer(kwargs.get('scoring_method', 'composite'))
        scored_proposals = scorer.score_proposals_pipeline(proposals, image)
        
        # 2. 目标检测
        print("📍 Step 2.2: 目标检测")
        from experiment1.stage2.target_detection.target_detector import ExperimentTargetDetector
        
        detector = ExperimentTargetDetector(self.model_name, self.device)
        target_classes = kwargs.get('target_classes', ['airplane', 'building', 'ship'])
        detection_results = detector.detect_multiple_targets(image, target_classes)
        
        # 3. WordNet增强
        print("📍 Step 2.3: WordNet增强")
        from experiment1.stage2.wordnet_enhancement.wordnet_enhancer import WordNetEnhancer
        
        enhancer = WordNetEnhancer()
        enhanced_vocab = enhancer.create_enhanced_vocabulary(target_classes)
        
        # 4. 边界框微调
        print("📍 Step 2.4: 边界框微调")
        from experiment1.stage2.bbox_refinement.bbox_refiner import ExperimentBBoxRefiner
        
        refiner = ExperimentBBoxRefiner(self.model, self.preprocess, self.device)
        refined_proposals = refiner.refine_proposals(
            image, scored_proposals, 
            refinement_method=kwargs.get('refinement_method', 'both')
        )
        
        # 保存Stage2结果
        stage2_results = {
            'scored_proposals': scored_proposals,
            'detection_results': detection_results,
            'enhanced_vocabulary': enhanced_vocab,
            'refined_proposals': refined_proposals
        }
        
        print(f"✅ Stage2流水线完成")
        
        return stage2_results
    
    def run_full_pipeline(self, image_path: str, **kwargs):
        """
        运行完整流水线
        
        参数:
            image_path: 图像路径
            **kwargs: 其他参数
        
        返回:
            完整结果
        """
        print(f"\n🚀 运行完整流水线...")
        
        # Stage1
        stage1_results = self.run_stage1_pipeline(image_path, **kwargs)
        
        # Stage2
        stage2_results = self.run_stage2_pipeline(image_path, stage1_results, **kwargs)
        
        # 合并结果
        full_results = {
            'stage1': stage1_results,
            'stage2': stage2_results,
            'image_path': image_path,
            'parameters': kwargs
        }
        
        print(f"✅ 完整流水线完成")
        
        return full_results
    
    def run_single_module(self, module_name: str, image_path: str, **kwargs):
        """
        运行单个模块
        
        参数:
            module_name: 模块名称
            image_path: 图像路径
            **kwargs: 其他参数
        
        返回:
            模块结果
        """
        print(f"\n🚀 运行单个模块: {module_name}")
        
        # 加载图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if module_name == 'data_loading':
            # 数据加载模块
            from experiment1.stage1.data_loading.data_loader import RemoteSensingDataLoader
            
            loader = RemoteSensingDataLoader(kwargs.get('data_dir', 'assets'))
            return {'image_info': loader.get_image_info(image_path)}
        
        elif module_name == 'sampling':
            # 区域采样模块
            from experiment1.stage1.sampling.region_sampler import ExperimentRegionSampler
            
            sampler = ExperimentRegionSampler(kwargs.get('strategy', 'multi_threshold_saliency'))
            regions = sampler.sample_regions(image, max_regions=kwargs.get('max_regions', 50))
            
            return {'regions': regions}
        
        elif module_name == 'proposal_generation':
            # 候选框生成模块
            from experiment1.stage1.proposal_generation.proposal_generator import ProposalGenerator
            
            generator = ProposalGenerator(self.model_name, self.device)
            # 需要先有区域，这里创建模拟区域
            mock_regions = [{'bbox': (100, 100, 200, 200), 'score': 0.8}]
            proposals = generator.generate_proposals_from_regions(image, mock_regions)
            
            return {'proposals': proposals}
        
        elif module_name == 'proposal_classification':
            # 候选框分类模块
            from experiment1.stage1.proposal_classification.proposal_classifier import ProposalClassifier
            
            classifier = ProposalClassifier(self.model_name, self.device)
            # 需要先有候选框，这里创建模拟候选框
            mock_proposals = [{'bbox': (100, 100, 200, 200), 'features': np.random.randn(1, 512)}]
            classified_proposals = classifier.classify_proposals(mock_proposals)
            
            return {'classified_proposals': classified_proposals}
        
        elif module_name == 'scoring':
            # 候选框打分模块
            from experiment1.stage2.scoring.proposal_scorer import ProposalScorer
            
            scorer = ProposalScorer(kwargs.get('method', 'composite'))
            # 需要先有候选框，这里创建模拟候选框
            mock_proposals = [{'bbox': (100, 100, 200, 200), 'predicted_class': 'airplane', 'prediction_confidence': 0.8}]
            scored_proposals = scorer.score_proposals(mock_proposals, image)
            
            return {'scored_proposals': scored_proposals}
        
        elif module_name == 'target_detection':
            # 目标检测模块
            from experiment1.stage2.target_detection.target_detector import ExperimentTargetDetector
            
            detector = ExperimentTargetDetector(self.model_name, self.device)
            target_class = kwargs.get('target_class', 'airplane')
            detection_results = detector.detect_target_with_contrastive_learning(image, target_class)
            
            return {'detection_results': detection_results}
        
        elif module_name == 'wordnet_enhancement':
            # WordNet增强模块
            from experiment1.stage2.wordnet_enhancement.wordnet_enhancer import WordNetEnhancer
            
            enhancer = WordNetEnhancer()
            target_classes = kwargs.get('target_classes', ['airplane', 'building', 'ship'])
            enhanced_vocab = enhancer.create_enhanced_vocabulary(target_classes)
            
            return {'enhanced_vocabulary': enhanced_vocab}
        
        elif module_name == 'bbox_refinement':
            # 边界框微调模块
            from experiment1.stage2.bbox_refinement.bbox_refiner import ExperimentBBoxRefiner
            
            refiner = ExperimentBBoxRefiner(self.model, self.preprocess, self.device)
            # 需要先有候选框，这里创建模拟候选框
            mock_proposals = [{'bbox': (100, 100, 200, 200), 'score': 0.8}]
            refined_proposals = refiner.refine_proposals(image, mock_proposals)
            
            return {'refined_proposals': refined_proposals}
        
        else:
            raise ValueError(f"未知的模块名称: {module_name}")
    
    def get_available_modules(self) -> list:
        """
        获取可用模块列表
        
        返回:
            可用模块列表
        """
        return [
            'data_loading',
            'sampling',
            'proposal_generation',
            'proposal_classification',
            'scoring',
            'target_detection',
            'wordnet_enhancement',
            'bbox_refinement'
        ]


def main():
    """测试推理引擎"""
    parser = argparse.ArgumentParser(description='推理引擎测试')
    parser.add_argument('--image', type=str, default='assets/airport.jpg',
                        help='输入图像路径')
    parser.add_argument('--model', type=str, default='RN50',
                        choices=['RN50', 'ViT-B-32', 'ViT-L-14'],
                        help='模型选择')
    parser.add_argument('--module', type=str, default=None,
                        help='运行单个模块')
    parser.add_argument('--pipeline', type=str, default='stage1',
                        choices=['stage1', 'stage2', 'full'],
                        help='运行流水线类型')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("推理引擎测试")
    print("=" * 70)
    
    # 创建推理引擎
    engine = InferenceEngine(args.model)
    
    # 显示可用模块
    available_modules = engine.get_available_modules()
    print(f"\n📋 可用模块: {available_modules}")
    
    if args.module:
        # 运行单个模块
        print(f"\n🔧 运行单个模块: {args.module}")
        
        result = engine.run_single_module(
            args.module, 
            args.image,
            max_regions=30,
            target_class='airplane'
        )
        
        print(f"✅ 模块运行完成")
        print(f"   结果类型: {type(result)}")
        print(f"   结果键: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
    
    elif args.pipeline == 'stage1':
        # 运行Stage1流水线
        print(f"\n🔧 运行Stage1流水线")
        
        result = engine.run_stage1_pipeline(
            args.image,
            sampling_strategy='multi_threshold_saliency',
            max_regions=30
        )
        
        print(f"✅ Stage1流水线完成")
        print(f"   区域数: {len(result['regions'])}")
        print(f"   候选框数: {len(result['proposals'])}")
    
    elif args.pipeline == 'stage2':
        # 运行Stage2流水线（需要先运行Stage1）
        print(f"\n🔧 运行Stage2流水线")
        
        # 先运行Stage1
        stage1_results = engine.run_stage1_pipeline(args.image, max_regions=30)
        
        # 再运行Stage2
        result = engine.run_stage2_pipeline(
            args.image, 
            stage1_results,
            target_classes=['airplane', 'building']
        )
        
        print(f"✅ Stage2流水线完成")
        print(f"   打分候选框数: {len(result['scored_proposals'])}")
        print(f"   检测结果数: {len(result['detection_results'])}")
    
    elif args.pipeline == 'full':
        # 运行完整流水线
        print(f"\n🔧 运行完整流水线")
        
        result = engine.run_full_pipeline(
            args.image,
            sampling_strategy='multi_threshold_saliency',
            max_regions=30,
            target_classes=['airplane', 'building']
        )
        
        print(f"✅ 完整流水线完成")
        print(f"   Stage1区域数: {len(result['stage1']['regions'])}")
        print(f"   Stage2检测结果数: {len(result['stage2']['detection_results'])}")
    
    print("\n✅ 推理引擎测试完成!")


if __name__ == "__main__":
    main()
