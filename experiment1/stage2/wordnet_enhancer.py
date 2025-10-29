#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage2: WordNet增强模块
基于原有wordnet_vocabulary.py，专门用于实验中的词汇增强
"""

import os
import sys
from typing import List, Dict, Optional

# 添加父目录到路径以导入原有模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wordnet_vocabulary import (
    WORDNET_REMOTE_SENSING_CLASSES, 
    get_synonyms, 
    get_expansion_words,
    get_hypernyms,
    build_vocabulary,
    get_full_class_list
)


class WordNetEnhancer:
    """WordNet增强器"""
    
    def __init__(self):
        """初始化WordNet增强器"""
        self.base_classes = WORDNET_REMOTE_SENSING_CLASSES.copy()
        self.enhanced_vocabulary = {}
        self.expansion_cache = {}
    
    def enhance_class_with_synonyms(self, target_class: str, max_synonyms: int = 5) -> List[str]:
        """
        使用同义词增强类别
        
        参数:
            target_class: 目标类别
            max_synonyms: 最大同义词数量
        
        返回:
            增强后的类别列表
        """
        print(f"\n🔧 使用同义词增强类别: {target_class}")
        
        # 获取同义词
        synonyms = get_synonyms(target_class)
        
        # 限制同义词数量
        enhanced_synonyms = synonyms[:max_synonyms]
        
        # 构建增强类别列表
        enhanced_classes = [target_class] + enhanced_synonyms
        
        print(f"✅ 增强类别: {enhanced_classes}")
        
        return enhanced_classes
    
    def enhance_class_with_expansion(self, target_class: str, num_expansion: int = 5) -> List[str]:
        """
        使用扩展词增强类别
        
        参数:
            target_class: 目标类别
            num_expansion: 扩展词数量
        
        返回:
            增强后的类别列表
        """
        print(f"\n🔧 使用扩展词增强类别: {target_class}")
        
        # 获取扩展词
        expansion_words = get_expansion_words(target_class, num_words=num_expansion)
        
        # 构建增强类别列表
        enhanced_classes = [target_class] + expansion_words
        
        print(f"✅ 增强类别: {enhanced_classes}")
        
        return enhanced_classes
    
    def enhance_class_with_hierarchy(self, target_class: str) -> List[str]:
        """
        使用层次结构增强类别
        
        参数:
            target_class: 目标类别
        
        返回:
            增强后的类别列表
        """
        print(f"\n🔧 使用层次结构增强类别: {target_class}")
        
        # 获取上位词
        hypernyms = get_hypernyms(target_class)
        
        # 构建增强类别列表
        enhanced_classes = [target_class] + hypernyms
        
        print(f"✅ 增强类别: {enhanced_classes}")
        
        return enhanced_classes
    
    def create_enhanced_vocabulary(self, target_classes: List[str], 
                                 enhancement_methods: List[str] = None) -> Dict[str, List[str]]:
        """
        创建增强词表
        
        参数:
            target_classes: 目标类别列表
            enhancement_methods: 增强方法列表
        
        返回:
            增强词表字典
        """
        if enhancement_methods is None:
            enhancement_methods = ['synonyms', 'expansion', 'hierarchy']
        
        print(f"\n🔧 创建增强词表 (目标类别: {len(target_classes)}个, 方法: {enhancement_methods})")
        
        enhanced_vocab = {}
        
        for target_class in target_classes:
            enhanced_classes = [target_class]
            
            # 应用不同的增强方法
            if 'synonyms' in enhancement_methods:
                synonyms = get_synonyms(target_class)
                enhanced_classes.extend(synonyms[:3])  # 最多3个同义词
            
            if 'expansion' in enhancement_methods:
                expansion_words = get_expansion_words(target_class, num_words=3)
                enhanced_classes.extend(expansion_words)
            
            if 'hierarchy' in enhancement_methods:
                hypernyms = get_hypernyms(target_class)
                enhanced_classes.extend(hypernyms)
            
            # 去重并保持顺序
            enhanced_classes = list(dict.fromkeys(enhanced_classes))
            
            enhanced_vocab[target_class] = enhanced_classes
            
            print(f"  {target_class}: {len(enhanced_classes)} 个增强词")
        
        print(f"✅ 增强词表创建完成")
        
        return enhanced_vocab
    
    def build_comprehensive_vocabulary(self, target_classes: List[str],
                                     include_base_classes: bool = True,
                                     enhancement_methods: List[str] = None) -> List[str]:
        """
        构建综合词表
        
        参数:
            target_classes: 目标类别列表
            include_base_classes: 是否包含基础类别
            enhancement_methods: 增强方法列表
        
        返回:
            综合词表列表
        """
        print(f"\n🔧 构建综合词表...")
        
        # 获取增强词表
        enhanced_vocab = self.create_enhanced_vocabulary(target_classes, enhancement_methods)
        
        # 构建综合词表
        comprehensive_vocab = []
        
        # 添加增强的目标类别
        for target_class, enhanced_classes in enhanced_vocab.items():
            comprehensive_vocab.extend(enhanced_classes)
        
        # 添加基础类别（如果需要）
        if include_base_classes:
            # 排除已经在增强词表中的类别
            enhanced_classes_set = set(comprehensive_vocab)
            base_classes = [c for c in self.base_classes if c not in enhanced_classes_set]
            comprehensive_vocab.extend(base_classes)
        
        # 去重并保持顺序
        comprehensive_vocab = list(dict.fromkeys(comprehensive_vocab))
        
        print(f"✅ 综合词表构建完成: {len(comprehensive_vocab)} 个类别")
        
        return comprehensive_vocab
    
    def get_vocabulary_statistics(self, vocabulary: List[str]) -> Dict:
        """
        获取词表统计信息
        
        参数:
            vocabulary: 词表列表
        
        返回:
            统计信息字典
        """
        stats = {
            'total_classes': len(vocabulary),
            'base_classes_count': len(set(vocabulary) & set(self.base_classes)),
            'enhanced_classes_count': len(vocabulary) - len(set(vocabulary) & set(self.base_classes))
        }
        
        # 按类别分组统计
        category_stats = {}
        for cls in vocabulary:
            # 简单的类别分组逻辑
            if any(word in cls for word in ['building', 'house', 'apartment']):
                category = 'buildings'
            elif any(word in cls for word in ['ship', 'boat', 'vessel']):
                category = 'vessels'
            elif any(word in cls for word in ['airplane', 'aircraft', 'helicopter']):
                category = 'aircraft'
            elif any(word in cls for word in ['road', 'highway', 'street']):
                category = 'transportation'
            elif any(word in cls for word in ['forest', 'tree', 'vegetation']):
                category = 'vegetation'
            elif any(word in cls for word in ['water', 'lake', 'river']):
                category = 'water'
            else:
                category = 'other'
            
            category_stats[category] = category_stats.get(category, 0) + 1
        
        stats['category_distribution'] = category_stats
        
        return stats
    
    def save_enhanced_vocabulary(self, vocabulary: List[str], 
                               target_classes: List[str],
                               output_path: str):
        """
        保存增强词表
        
        参数:
            vocabulary: 词表列表
            target_classes: 目标类别列表
            output_path: 输出路径
        """
        print(f"\n💾 保存增强词表到: {output_path}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Enhanced WordNet Vocabulary\n")
            f.write(f"# Generated for target classes: {', '.join(target_classes)}\n")
            f.write(f"# Total classes: {len(vocabulary)}\n\n")
            
            for i, cls in enumerate(vocabulary, 1):
                f.write(f"{i:3d}. {cls}\n")
        
        print(f"✅ 增强词表已保存")
    
    def load_enhanced_vocabulary(self, input_path: str) -> List[str]:
        """
        加载增强词表
        
        参数:
            input_path: 输入路径
        
        返回:
            词表列表
        """
        print(f"\n📖 加载增强词表: {input_path}")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"词表文件不存在: {input_path}")
        
        vocabulary = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # 提取类别名称（去掉序号）
                    if '. ' in line:
                        cls = line.split('. ', 1)[1]
                    else:
                        cls = line
                    vocabulary.append(cls)
        
        print(f"✅ 加载词表: {len(vocabulary)} 个类别")
        
        return vocabulary


def main():
    """测试WordNet增强器"""
    print("=" * 70)
    print("测试WordNet增强器")
    print("=" * 70)
    
    # 创建WordNet增强器
    enhancer = WordNetEnhancer()
    
    # 测试目标类别
    target_classes = ['ship', 'airplane', 'building']
    
    # 测试不同增强方法
    print(f"\n{'='*50}")
    print("测试同义词增强")
    print(f"{'='*50}")
    
    for target_class in target_classes:
        enhanced = enhancer.enhance_class_with_synonyms(target_class, max_synonyms=3)
        print(f"{target_class}: {enhanced}")
    
    print(f"\n{'='*50}")
    print("测试扩展词增强")
    print(f"{'='*50}")
    
    for target_class in target_classes:
        enhanced = enhancer.enhance_class_with_expansion(target_class, num_expansion=3)
        print(f"{target_class}: {enhanced}")
    
    print(f"\n{'='*50}")
    print("测试层次结构增强")
    print(f"{'='*50}")
    
    for target_class in target_classes:
        enhanced = enhancer.enhance_class_with_hierarchy(target_class)
        print(f"{target_class}: {enhanced}")
    
    # 测试综合增强
    print(f"\n{'='*50}")
    print("测试综合增强")
    print(f"{'='*50}")
    
    comprehensive_vocab = enhancer.build_comprehensive_vocabulary(
        target_classes, 
        include_base_classes=True,
        enhancement_methods=['synonyms', 'expansion']
    )
    
    # 获取统计信息
    stats = enhancer.get_vocabulary_statistics(comprehensive_vocab)
    print(f"\n📊 词表统计:")
    print(f"  总类别数: {stats['total_classes']}")
    print(f"  基础类别数: {stats['base_classes_count']}")
    print(f"  增强类别数: {stats['enhanced_classes_count']}")
    print(f"  类别分布: {stats['category_distribution']}")
    
    # 保存词表
    output_path = "outputs/enhanced_vocabulary.txt"
    enhancer.save_enhanced_vocabulary(comprehensive_vocab, target_classes, output_path)
    
    print("\n✅ WordNet增强器测试完成!")


if __name__ == "__main__":
    main()
