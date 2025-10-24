#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扩充 mini_dataset 到 100 个 DIOR 样本

功能：
1. 删除所有 hrsc2016 图片（缺少标注）
2. 从 DIOR 数据集随机选择 100 个样本
3. 复制图片和标注文件
4. 更新 samples.json
"""

import os
import shutil
import json
import random
from pathlib import Path
import xml.etree.ElementTree as ET


def delete_hrsc2016_files(mini_dataset_dir):
    """删除所有 hrsc2016 文件"""
    print("=" * 70)
    print("删除 hrsc2016 文件")
    print("=" * 70)
    
    images_dir = mini_dataset_dir / 'images'
    
    # 删除 hrsc2016 图片
    deleted_count = 0
    for img_file in images_dir.glob('hrsc2016_*.bmp'):
        print(f"删除: {img_file.name}")
        img_file.unlink()
        deleted_count += 1
    
    print(f"\n✅ 删除了 {deleted_count} 个 hrsc2016 文件")
    return deleted_count


def get_dior_samples(dior_dir, num_samples=100):
    """从 DIOR 数据集获取样本"""
    print("\n" + "=" * 70)
    print(f"从 DIOR 数据集选择 {num_samples} 个样本")
    print("=" * 70)
    
    trainval_dir = dior_dir / 'images' / 'trainval'
    annotations_dir = dior_dir / 'annotations' / 'horizontal'
    
    # 获取所有有标注的图片
    all_images = []
    for img_file in trainval_dir.glob('*.jpg'):
        img_id = img_file.stem
        xml_file = annotations_dir / f'{img_id}.xml'
        
        if xml_file.exists():
            # 检查是否有有效的目标
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                objects = root.findall('object')
                
                if len(objects) > 0:  # 至少有一个目标
                    all_images.append({
                        'image_file': img_file,
                        'xml_file': xml_file,
                        'image_id': img_id,
                        'num_objects': len(objects)
                    })
            except:
                continue
    
    print(f"找到 {len(all_images)} 个有效的 DIOR 样本")
    
    # 随机选择
    if len(all_images) >= num_samples:
        selected = random.sample(all_images, num_samples)
    else:
        print(f"⚠️ 可用样本不足，使用全部 {len(all_images)} 个样本")
        selected = all_images
    
    # 按目标数量排序，确保多样性
    selected.sort(key=lambda x: x['num_objects'])
    
    print(f"\n✅ 选择了 {len(selected)} 个样本")
    print(f"   目标数量范围: {selected[0]['num_objects']} - {selected[-1]['num_objects']}")
    
    return selected


def copy_samples_to_mini_dataset(selected_samples, mini_dataset_dir):
    """复制样本到 mini_dataset"""
    print("\n" + "=" * 70)
    print("复制样本到 mini_dataset")
    print("=" * 70)
    
    images_dir = mini_dataset_dir / 'images'
    annotations_dir = mini_dataset_dir / 'annotations'
    
    # 确保目录存在
    images_dir.mkdir(exist_ok=True, parents=True)
    annotations_dir.mkdir(exist_ok=True, parents=True)
    
    samples_info = []
    
    for i, sample in enumerate(selected_samples, 1):
        # 生成新文件名
        new_image_name = f"DIOR_{sample['image_id']}.jpg"
        new_xml_name = f"DIOR_{sample['image_id']}.xml"
        
        # 复制图片
        dst_img = images_dir / new_image_name
        if not dst_img.exists():
            shutil.copy2(sample['image_file'], dst_img)
        
        # 复制标注
        dst_xml = annotations_dir / new_xml_name
        if not dst_xml.exists():
            shutil.copy2(sample['xml_file'], dst_xml)
        
        # 记录信息
        samples_info.append({
            'dataset': 'DIOR',
            'image_name': new_image_name,
            'original_path': str(sample['image_file'].relative_to(sample['image_file'].parents[3]))
        })
        
        if i % 20 == 0:
            print(f"  已复制 {i}/{len(selected_samples)} 个样本")
    
    print(f"\n✅ 复制完成！共 {len(samples_info)} 个样本")
    
    return samples_info


def update_samples_json(samples_info, mini_dataset_dir):
    """更新 samples.json"""
    print("\n" + "=" * 70)
    print("更新 samples.json")
    print("=" * 70)
    
    samples_file = mini_dataset_dir / 'samples.json'
    
    # 保存
    with open(samples_file, 'w', encoding='utf-8') as f:
        json.dump(samples_info, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 已保存 {len(samples_info)} 个样本信息到 samples.json")


def update_split_configs(mini_dataset_dir, num_samples):
    """更新分割配置文件"""
    print("\n" + "=" * 70)
    print("更新分割配置")
    print("=" * 70)
    
    # 不同的 seen 比例
    seen_ratios = [50, 60, 70, 80]
    
    all_configs = {}
    
    for seen_ratio in seen_ratios:
        num_seen = int(num_samples * seen_ratio / 100)
        num_unseen = num_samples - num_seen
        
        config = {
            "name": f"seen_{seen_ratio}",
            "total_samples": num_samples,
            "num_seen_classes": 10,
            "num_unseen_classes": 10,
            "train_samples": num_seen,
            "test_samples": num_unseen
        }
        
        all_configs[f"seen_{seen_ratio}"] = config
        
        # 保存单独的配置文件
        config_file = mini_dataset_dir / f'split_config_seen_{seen_ratio}.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 已保存 split_config_seen_{seen_ratio}.json")
    
    # 保存所有配置
    all_configs_file = mini_dataset_dir / 'all_split_configs.json'
    with open(all_configs_file, 'w', encoding='utf-8') as f:
        json.dump(all_configs, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 已保存 all_split_configs.json")


def main():
    """主函数"""
    print("\n" + "🎯" * 35)
    print("扩充 mini_dataset 到 100 个 DIOR 样本")
    print("🎯" * 35 + "\n")
    
    # 设置路径
    mini_dataset_dir = Path(__file__).parent
    dior_dir = mini_dataset_dir.parent / 'DIOR'
    
    print(f"Mini Dataset 目录: {mini_dataset_dir}")
    print(f"DIOR 数据集目录: {dior_dir}")
    
    # 设置随机种子
    random.seed(42)
    
    # 1. 删除 hrsc2016 文件
    delete_hrsc2016_files(mini_dataset_dir)
    
    # 2. 获取 DIOR 样本
    selected_samples = get_dior_samples(dior_dir, num_samples=100)
    
    # 3. 复制样本
    samples_info = copy_samples_to_mini_dataset(selected_samples, mini_dataset_dir)
    
    # 4. 更新 samples.json
    update_samples_json(samples_info, mini_dataset_dir)
    
    # 5. 更新分割配置
    update_split_configs(mini_dataset_dir, len(samples_info))
    
    # 6. 统计信息
    print("\n" + "=" * 70)
    print("最终统计")
    print("=" * 70)
    
    images_count = len(list((mini_dataset_dir / 'images').glob('*.jpg')))
    annotations_count = len(list((mini_dataset_dir / 'annotations').glob('*.xml')))
    
    print(f"\n✅ 扩充完成！")
    print(f"   图片数量: {images_count}")
    print(f"   标注数量: {annotations_count}")
    print(f"   样本信息: {len(samples_info)} 条记录")
    
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()

