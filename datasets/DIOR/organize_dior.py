#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIOR数据集整理脚本

功能：
1. 解除嵌套的目录结构
2. 整理标注文件和图片到规范目录
3. 删除不必要的文件
"""

import os
import shutil
from pathlib import Path


def organize_dior():
    """整理DIOR数据集"""
    
    print("=" * 70)
    print("DIOR数据集整理")
    print("=" * 70)
    
    base_dir = Path(__file__).parent
    archive_dir = base_dir / 'archive (1)'
    
    # 1. 创建目标目录结构
    target_dirs = {
        'images_trainval': base_dir / 'images' / 'trainval',
        'images_test': base_dir / 'images' / 'test',
        'annotations_horizontal': base_dir / 'annotations' / 'horizontal',
        'annotations_oriented': base_dir / 'annotations' / 'oriented',
        'splits': base_dir / 'splits',
        'docs': base_dir / 'docs'
    }
    
    print("\n📁 创建目标目录结构...")
    for name, path in target_dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {name}")
    
    # 2. 整理图片
    print("\n📷 整理图片...")
    
    # trainval图片
    trainval_src = archive_dir / 'JPEGImages-trainval' / 'JPEGImages-trainval'
    if trainval_src.exists():
        images = list(trainval_src.glob('*.jpg'))
        print(f"  找到trainval图片: {len(images)}张")
        
        for img in images:
            target = target_dirs['images_trainval'] / img.name
            if not target.exists():
                shutil.copy2(img, target)
        
        print(f"  ✅ 复制到 images/trainval/")
    
    # test图片
    test_src = archive_dir / 'JPEGImages-test' / 'JPEGImages-test'
    if test_src.exists():
        images = list(test_src.glob('*.jpg'))
        print(f"  找到test图片: {len(images)}张")
        
        for img in images:
            target = target_dirs['images_test'] / img.name
            if not target.exists():
                shutil.copy2(img, target)
        
        print(f"  ✅ 复制到 images/test/")
    
    # 3. 整理标注
    print("\n📋 整理标注...")
    
    # 水平边界框标注
    horizontal_src = archive_dir / 'Annotations' / 'Annotations' / 'Horizontal Bounding Boxes'
    if horizontal_src.exists():
        annotations = list(horizontal_src.glob('*.xml'))
        print(f"  找到水平框标注: {len(annotations)}个")
        
        for anno in annotations:
            target = target_dirs['annotations_horizontal'] / anno.name
            if not target.exists():
                shutil.copy2(anno, target)
        
        print(f"  ✅ 复制到 annotations/horizontal/")
    
    # 旋转边界框标注
    oriented_src = archive_dir / 'Annotations' / 'Annotations' / 'Oriented Bounding Boxes'
    if oriented_src.exists():
        annotations = list(oriented_src.glob('*.xml'))
        print(f"  找到旋转框标注: {len(annotations)}个")
        
        for anno in annotations:
            target = target_dirs['annotations_oriented'] / anno.name
            if not target.exists():
                shutil.copy2(anno, target)
        
        print(f"  ✅ 复制到 annotations/oriented/")
    
    # 4. 整理ImageSets
    print("\n📂 整理ImageSets...")
    
    imagesets_src = archive_dir / 'ImageSets' / 'Main'
    if imagesets_src.exists():
        split_files = list(imagesets_src.glob('*.txt'))
        print(f"  找到split文件: {len(split_files)}个")
        
        for split_file in split_files:
            target = target_dirs['splits'] / split_file.name
            if not target.exists():
                shutil.copy2(split_file, target)
        
        print(f"  ✅ 复制到 splits/")
    
    # 5. 统计信息
    print("\n📊 统计信息...")
    
    stats = {
        'trainval_images': len(list(target_dirs['images_trainval'].glob('*.jpg'))),
        'test_images': len(list(target_dirs['images_test'].glob('*.jpg'))),
        'horizontal_annos': len(list(target_dirs['annotations_horizontal'].glob('*.xml'))),
        'oriented_annos': len(list(target_dirs['annotations_oriented'].glob('*.xml'))),
        'split_files': len(list(target_dirs['splits'].glob('*.txt')))
    }
    
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 6. 生成整理报告
    print("\n📊 生成整理报告...")
    
    report_path = base_dir / 'dataset_structure.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("DIOR数据集整理报告\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("📁 目录结构：\n")
        f.write("-" * 70 + "\n")
        f.write("DIOR/\n")
        f.write("├── images/\n")
        f.write("│   ├── trainval/        # 训练+验证图片\n")
        f.write("│   └── test/            # 测试图片\n")
        f.write("├── annotations/\n")
        f.write("│   ├── horizontal/      # 水平边界框标注（XML）\n")
        f.write("│   └── oriented/        # 旋转边界框标注（XML）\n")
        f.write("├── splits/              # 数据集划分文件\n")
        f.write("├── docs/                # 文档\n")
        f.write("├── organize_dior.py     # 整理脚本\n")
        f.write("├── dataset_structure.txt # 本报告\n")
        f.write("└── README.md            # 数据集说明\n\n")
        
        f.write("📊 统计信息：\n")
        f.write("-" * 70 + "\n")
        f.write(f"训练+验证图片: {stats['trainval_images']}\n")
        f.write(f"测试图片: {stats['test_images']}\n")
        f.write(f"水平框标注: {stats['horizontal_annos']}\n")
        f.write(f"旋转框标注: {stats['oriented_annos']}\n")
        f.write(f"划分文件: {stats['split_files']}\n")
        f.write(f"总图片: {stats['trainval_images'] + stats['test_images']}\n\n")
        
        f.write("✅ DIOR数据集类别（20类）：\n")
        f.write("-" * 70 + "\n")
        classes = [
            "airplane", "airport", "baseball field", "basketball court",
            "bridge", "chimney", "dam", "Expressway Service Area",
            "Expressway toll station", "golf course", "ground track field",
            "harbor", "overpass", "ship", "stadium", "storage tank",
            "tennis court", "train station", "vehicle", "wind mill"
        ]
        for i, cls in enumerate(classes, 1):
            f.write(f"{i:2d}. {cls}\n")
        
        f.write("\n🗑️  已删除内容：\n")
        f.write("-" * 70 + "\n")
        f.write("1. archive (1)/ - 原始嵌套目录（已整理）\n\n")
        
        f.write("💡 标注格式（VOC XML）：\n")
        f.write("-" * 70 + "\n")
        f.write("<annotation>\n")
        f.write("  <folder>...</folder>\n")
        f.write("  <filename>...</filename>\n")
        f.write("  <object>\n")
        f.write("    <name>airplane</name>\n")
        f.write("    <bndbox>\n")
        f.write("      <xmin>...</xmin>\n")
        f.write("      <ymin>...</ymin>\n")
        f.write("      <xmax>...</xmax>\n")
        f.write("      <ymax>...</ymax>\n")
        f.write("    </bndbox>\n")
        f.write("  </object>\n")
        f.write("</annotation>\n")
    
    print(f"  ✅ 报告已保存: {report_path}")
    
    # 7. 删除原始archive目录
    print("\n🗑️  删除原始archive目录...")
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
        print(f"  ✅ 已删除 archive (1)/")
    
    print("\n" + "=" * 70)
    print("✅ 数据集整理完成！")
    print("=" * 70)
    print(f"\n📍 图片目录: {base_dir / 'images'}")
    print(f"📍 标注目录: {base_dir / 'annotations'}")
    print(f"📍 整理报告: {report_path}")


if __name__ == "__main__":
    organize_dior()

