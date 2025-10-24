#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRSC2016数据集整理脚本

功能：
1. 合并所有part中的图片到统一目录
2. 删除重复和不必要的文件
3. 创建规范的目录结构
"""

import os
import shutil
from pathlib import Path


def organize_hrsc2016():
    """整理HRSC2016数据集"""
    
    print("=" * 70)
    print("HRSC2016数据集整理")
    print("=" * 70)
    
    base_dir = Path(__file__).parent
    
    # 1. 创建目标目录结构
    target_structure = {
        'images': base_dir / 'images',
        'annotations': base_dir / 'annotations',
        'splits': base_dir / 'splits',
        'docs': base_dir / 'docs'
    }
    
    print("\n📁 创建目标目录结构...")
    for name, path in target_structure.items():
        path.mkdir(exist_ok=True)
        print(f"  ✅ {name}/")
    
    # 2. 收集所有图片
    print("\n📷 收集图片...")
    
    all_images = []
    part_dirs = [d for d in base_dir.glob('HRSC2016.part*') if d.is_dir()]
    
    for part_dir in sorted(part_dirs):
        print(f"\n  处理 {part_dir.name}...")
        
        # 查找AllImages目录
        all_images_dir = part_dir / 'HRSC2016' / 'FullDataSet' / 'AllImages'
        
        if all_images_dir.exists():
            images = list(all_images_dir.glob('*.bmp'))
            print(f"    找到 {len(images)} 张图片")
            all_images.extend(images)
        else:
            # 尝试其他可能的路径
            for subdir in part_dir.rglob('AllImages'):
                images = list(subdir.glob('*.bmp'))
                if images:
                    print(f"    在 {subdir} 找到 {len(images)} 张图片")
                    all_images.extend(images)
    
    print(f"\n  总计找到 {len(all_images)} 张图片")
    
    # 3. 复制图片到目标目录
    print("\n📋 复制图片到 images/...")
    
    copied = 0
    skipped = 0
    
    for img_path in all_images:
        target_path = target_structure['images'] / img_path.name
        
        if target_path.exists():
            skipped += 1
        else:
            shutil.copy2(img_path, target_path)
            copied += 1
    
    print(f"  ✅ 复制: {copied} 张")
    print(f"  ⏭️  跳过（已存在）: {skipped} 张")
    
    # 4. 整理文档
    print("\n📚 整理文档...")
    
    # 移动PDF到docs（如果不在）
    pdf_file = base_dir / 'ShipTeam_HRSC2016_Introduction.pdf'
    if pdf_file.exists():
        target_pdf = target_structure['docs'] / pdf_file.name
        if not target_pdf.exists():
            shutil.move(str(pdf_file), str(target_pdf))
            print(f"  ✅ 移动: {pdf_file.name}")
    
    # 5. 评估文件必要性
    print("\n🔍 评估文件必要性...")
    
    evaluation = {
        'AnnotationTool_v2': '❌ 不需要 - C#标注工具，本项目不需要',
        'dev-tools': '❌ 不需要 - C++开发工具，本项目不需要',
        'State_Of_The_Art_Codes': '❌ 不需要 - C++代码，本项目用Python',
        'SOA_Results': '⚠️  参考用 - 其他算法结果，可选保留',
        'HRSC2016_dataset.zip': '❌ 可删除 - 已解压的数据',
        'HRSC2016.part*': '✅ 保留 - 包含原始图片',
        'docs': '✅ 保留 - 数据集说明文档'
    }
    
    for item, status in evaluation.items():
        print(f"  {status}: {item}")
    
    # 6. 生成整理报告
    print("\n📊 生成整理报告...")
    
    report_path = base_dir / 'dataset_structure.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("HRSC2016数据集整理报告\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("📁 推荐的目录结构：\n")
        f.write("-" * 70 + "\n")
        f.write("hrsc2016/\n")
        f.write("├── images/              # 所有图片（.bmp）\n")
        f.write("├── annotations/         # 标注文件（待添加）\n")
        f.write("├── splits/              # 训练/验证/测试划分\n")
        f.write("├── docs/                # 数据集说明文档\n")
        f.write("└── dataset_structure.txt # 本报告\n\n")
        
        f.write("📊 统计信息：\n")
        f.write("-" * 70 + "\n")
        f.write(f"总图片数: {len(list(target_structure['images'].glob('*.bmp')))}\n")
        f.write(f"文档数: {len(list(target_structure['docs'].glob('*')))}\n\n")
        
        f.write("🗑️  可以安全删除的内容：\n")
        f.write("-" * 70 + "\n")
        f.write("1. AnnotationTool_v2/ - C#标注工具（本项目用Python）\n")
        f.write("2. dev-tools/ - C++开发工具（本项目用Python）\n")
        f.write("3. State_Of_The_Art_Codes/ - C++算法代码（本项目用Python）\n")
        f.write("4. HRSC2016_dataset.zip - 原始压缩包（已解压）\n")
        f.write("5. SOA_Results/ - 可选删除（其他算法的结果）\n\n")
        
        f.write("✅ 需要保留的内容：\n")
        f.write("-" * 70 + "\n")
        f.write("1. images/ - 图片数据\n")
        f.write("2. HRSC2016.part*/ - 包含原始数据的分卷\n")
        f.write("3. docs/ - 数据集说明文档\n\n")
        
        f.write("💡 后续建议：\n")
        f.write("-" * 70 + "\n")
        f.write("1. 创建Python标注解析脚本\n")
        f.write("2. 生成训练/验证/测试划分文件\n")
        f.write("3. 删除不需要的C#/C++工具\n")
        f.write("4. 保留核心图片和标注数据\n")
    
    print(f"  ✅ 报告已保存: {report_path}")
    
    print("\n" + "=" * 70)
    print("✅ 数据集整理完成！")
    print("=" * 70)
    print(f"\n📍 图片目录: {target_structure['images']}")
    print(f"📍 整理报告: {report_path}")


if __name__ == "__main__":
    organize_hrsc2016()

