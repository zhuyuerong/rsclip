#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建小数据集

功能：
1. 从3个数据集各选20张图片（共60张）
2. 提供不同的seen/unseen分割比例
3. 生成配置文件供实验使用
"""

import os
import shutil
import random
import json
from pathlib import Path
from typing import Dict, List, Tuple


class MiniDatasetCreator:
    """小数据集创建器"""
    
    def __init__(self, output_dir: str = 'datasets/mini_dataset'):
        """
        参数:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据集配置
        self.datasets = {
            'hrsc2016': {
                'images_dir': Path('datasets/hrsc2016/images'),
                'annotations_dir': None,  # 暂无标注
                'image_ext': '.bmp',
                'num_samples': 20,
                'classes': ['ship']
            },
            'DIOR': {
                'images_dir': Path('datasets/DIOR/images/trainval'),
                'annotations_dir': Path('datasets/DIOR/annotations/horizontal'),
                'image_ext': '.jpg',
                'num_samples': 20,
                'classes': [
                    'airplane', 'ship', 'bridge', 'harbor', 'vehicle',
                    'storage-tank', 'baseball-field', 'tennis-court',
                    'basketball-court', 'stadium'
                ]
            },
            'DOTA': {
                'images_dir': None,  # 图片未下载
                'annotations_dir': Path('datasets/DOTA/DOTA-v2.0/annotations/train'),
                'image_ext': '.png',
                'num_samples': 0,  # 暂不包含
                'classes': [
                    'plane', 'ship', 'storage-tank', 'baseball-diamond',
                    'tennis-court', 'basketball-court', 'harbor', 'bridge'
                ]
            }
        }
        
        # Seen/Unseen类别配置
        self.class_splits = {
            'seen_classes': [
                'airplane', 'ship', 'vehicle', 'bridge', 'harbor'
            ],
            'unseen_classes': [
                'storage-tank', 'baseball-field', 'tennis-court',
                'basketball-court', 'stadium'
            ]
        }
    
    def sample_images(self, dataset_name: str) -> List[Path]:
        """
        从数据集中采样图片
        
        参数:
            dataset_name: 数据集名称
        
        返回:
            采样的图片路径列表
        """
        config = self.datasets[dataset_name]
        images_dir = config['images_dir']
        num_samples = config['num_samples']
        
        if images_dir is None or not images_dir.exists() or num_samples == 0:
            return []
        
        # 获取所有图片
        image_ext = config['image_ext']
        all_images = list(images_dir.glob(f'*{image_ext}'))
        
        # 随机采样
        if len(all_images) >= num_samples:
            sampled = random.sample(all_images, num_samples)
        else:
            sampled = all_images
        
        return sorted(sampled)
    
    def create_mini_dataset(self):
        """创建小数据集"""
        print("=" * 70)
        print("创建小数据集（60张图片）")
        print("=" * 70)
        
        # 创建输出目录结构
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'annotations').mkdir(exist_ok=True)
        
        all_samples = []
        
        # 从每个数据集采样
        for dataset_name, config in self.datasets.items():
            print(f"\n📦 处理 {dataset_name}...")
            
            sampled_images = self.sample_images(dataset_name)
            
            if not sampled_images:
                print(f"  ⏭️  跳过（无可用图片）")
                continue
            
            print(f"  采样 {len(sampled_images)} 张图片")
            
            # 复制图片和标注
            for img_path in sampled_images:
                # 复制图片
                new_name = f"{dataset_name}_{img_path.stem}{img_path.suffix}"
                target_img = self.output_dir / 'images' / new_name
                
                shutil.copy2(img_path, target_img)
                
                # 复制标注（如果有）
                anno_dir = config['annotations_dir']
                if anno_dir and anno_dir.exists():
                    anno_path = anno_dir / f"{img_path.stem}.xml"
                    if anno_path.exists():
                        target_anno = self.output_dir / 'annotations' / f"{dataset_name}_{anno_path.name}"
                        shutil.copy2(anno_path, target_anno)
                
                # 记录样本信息
                all_samples.append({
                    'dataset': dataset_name,
                    'image_name': new_name,
                    'original_path': str(img_path)
                })
            
            print(f"  ✅ 已复制到 mini_dataset/")
        
        # 保存样本列表
        samples_file = self.output_dir / 'samples.json'
        with open(samples_file, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 总计: {len(all_samples)} 张图片")
        print(f"  样本列表已保存: {samples_file}")
        
        return all_samples
    
    def create_splits(
        self,
        all_samples: List[Dict],
        seen_ratio: float = 0.7
    ) -> Dict:
        """
        创建seen/unseen分割
        
        参数:
            all_samples: 所有样本列表
            seen_ratio: seen类别的比例
        
        返回:
            分割配置
        """
        print(f"\n📊 创建seen/unseen分割（seen比例: {seen_ratio:.0%}）...")
        
        # 根据比例分配类别
        all_classes = self.class_splits['seen_classes'] + self.class_splits['unseen_classes']
        num_seen = int(len(all_classes) * seen_ratio)
        
        # 随机分配seen/unseen
        random.shuffle(all_classes)
        seen_classes = all_classes[:num_seen]
        unseen_classes = all_classes[num_seen:]
        
        split_config = {
            'seen_ratio': seen_ratio,
            'seen_classes': seen_classes,
            'unseen_classes': unseen_classes,
            'num_seen': len(seen_classes),
            'num_unseen': len(unseen_classes)
        }
        
        print(f"  Seen类别 ({len(seen_classes)}): {seen_classes}")
        print(f"  Unseen类别 ({len(unseen_classes)}): {unseen_classes}")
        
        return split_config
    
    def save_split_configs(self, all_samples: List[Dict]):
        """
        保存多种分割配置
        
        参数:
            all_samples: 所有样本列表
        """
        print("\n📝 生成多种seen/unseen分割配置...")
        
        # 不同的分割比例
        split_ratios = [0.5, 0.6, 0.7, 0.8]
        
        configs = {}
        
        for ratio in split_ratios:
            split_name = f"seen_{int(ratio*100)}"
            split_config = self.create_splits(all_samples, ratio)
            configs[split_name] = split_config
            
            # 保存单独的配置文件
            config_file = self.output_dir / f'split_config_{split_name}.json'
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(split_config, f, indent=2, ensure_ascii=False)
            
            print(f"  ✅ {split_name}: {config_file}")
        
        # 保存所有配置
        all_configs_file = self.output_dir / 'all_split_configs.json'
        with open(all_configs_file, 'w', encoding='utf-8') as f:
            json.dump(configs, f, indent=2, ensure_ascii=False)
        
        print(f"\n  ✅ 所有配置已保存: {all_configs_file}")
        
        return configs
    
    def create_readme(self, all_samples: List[Dict], configs: Dict):
        """创建README文档"""
        readme_path = self.output_dir / 'README.md'
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("# Mini Dataset（小数据集）\n\n")
            f.write("用于快速实验的小规模数据集。\n\n")
            
            f.write("## 📊 数据集统计\n\n")
            f.write(f"- **总图片数**: {len(all_samples)}\n")
            
            # 按来源统计
            dataset_counts = {}
            for sample in all_samples:
                ds = sample['dataset']
                dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
            
            f.write("- **来源分布**:\n")
            for ds, count in dataset_counts.items():
                f.write(f"  - {ds}: {count}张\n")
            
            f.write("\n## 🎯 Seen/Unseen分割配置\n\n")
            f.write("提供4种分割比例，可通过参数选择：\n\n")
            
            for split_name, config in configs.items():
                ratio = config['seen_ratio']
                f.write(f"### {split_name} (seen: {ratio:.0%})\n")
                f.write(f"- Seen类别 ({config['num_seen']}个): {', '.join(config['seen_classes'])}\n")
                f.write(f"- Unseen类别 ({config['num_unseen']}个): {', '.join(config['unseen_classes'])}\n\n")
            
            f.write("## 🚀 使用方式\n\n")
            f.write("### 在Experiment1中使用\n\n")
            f.write("```bash\n")
            f.write("# 使用seen类别训练\n")
            f.write("python experiment1/stage2/target_detection.py \\\n")
            f.write("  --image datasets/mini_dataset/images/DIOR_00001.jpg \\\n")
            f.write("  --target airplane\n")
            f.write("```\n\n")
            
            f.write("### 在Experiment2中使用\n\n")
            f.write("```bash\n")
            f.write("# 加载分割配置\n")
            f.write("python experiment2/scripts/train.py \\\n")
            f.write("  --mini-dataset datasets/mini_dataset \\\n")
            f.write("  --split-config split_config_seen_70.json\n")
            f.write("```\n\n")
            
            f.write("### Python API\n\n")
            f.write("```python\n")
            f.write("import json\n")
            f.write("from pathlib import Path\n\n")
            f.write("# 加载分割配置\n")
            f.write("config_path = 'datasets/mini_dataset/split_config_seen_70.json'\n")
            f.write("with open(config_path, 'r') as f:\n")
            f.write("    split_config = json.load(f)\n\n")
            f.write("seen_classes = split_config['seen_classes']\n")
            f.write("unseen_classes = split_config['unseen_classes']\n\n")
            f.write("print(f'Seen: {seen_classes}')\n")
            f.write("print(f'Unseen: {unseen_classes}')\n")
            f.write("```\n\n")
            
            f.write("## 📁 目录结构\n\n")
            f.write("```\n")
            f.write("mini_dataset/\n")
            f.write("├── images/                  # 60张图片\n")
            f.write("├── annotations/             # 对应的标注文件\n")
            f.write("├── samples.json             # 样本列表\n")
            f.write("├── split_config_seen_50.json  # 50%配置\n")
            f.write("├── split_config_seen_60.json  # 60%配置\n")
            f.write("├── split_config_seen_70.json  # 70%配置\n")
            f.write("├── split_config_seen_80.json  # 80%配置\n")
            f.write("├── all_split_configs.json   # 所有配置\n")
            f.write("└── README.md                # 本文档\n")
            f.write("```\n\n")
            
            f.write("## 🎯 实验建议\n\n")
            f.write("1. **seen_50**: 对半分，测试零样本泛化能力\n")
            f.write("2. **seen_60**: 轻微倾向seen，平衡测试\n")
            f.write("3. **seen_70**: 推荐配置，足够的seen类别训练\n")
            f.write("4. **seen_80**: 大部分seen，少量unseen测试\n")
        
        print(f"✅ README已创建: {readme_path}")


def main():
    """主函数"""
    print("=" * 70)
    print("创建小数据集（用于快速实验）")
    print("=" * 70)
    
    # 设置随机种子
    random.seed(42)
    
    # 创建小数据集
    creator = MiniDatasetCreator()
    
    # 采样图片
    all_samples = creator.create_mini_dataset()
    
    # 创建多种分割配置
    configs = creator.save_split_configs(all_samples)
    
    # 创建README
    creator.create_readme(all_samples, configs)
    
    print("\n" + "=" * 70)
    print("✅ 小数据集创建完成！")
    print("=" * 70)
    print(f"\n📍 位置: {creator.output_dir}")
    print(f"📊 图片: {len(all_samples)} 张")
    print(f"📋 配置: {len(configs)} 种seen/unseen分割")


if __name__ == "__main__":
    main()

