#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试文件组织
验证所有文件是否正确组织，输出文件是否正确保存
"""

import os
import sys
from output_manager import get_output_manager


def test_file_organization():
    """测试文件组织"""
    print("=" * 70)
    print("测试文件组织")
    print("=" * 70)
    
    # 1. 检查核心Python模块
    print("\n📋 检查核心Python模块:")
    core_modules = [
        'target_detection.py',
        'unseen_detection_pipeline.py', 
        'bbox_refinement.py',
        'wordnet_vocabulary.py',
        'sampling.py',
        'visualize_regions.py',
        'output_manager.py',
        'demo_simple.py',
        'test_bbox_refinement.py',
        'test_remoteclip.py',
        'download_hrsc2016.py',
        'retrieval.py'
    ]
    
    for module in core_modules:
        if os.path.exists(module):
            print(f"  ✅ {module}")
        else:
            print(f"  ❌ {module}")
    
    # 2. 检查extensions目录结构
    print("\n📁 检查extensions目录结构:")
    extensions_dirs = [
        'extensions',
        'extensions/docs',
        'extensions/scripts', 
        'extensions/outputs',
        'extensions/outputs/detection_results',
        'extensions/outputs/visualizations',
        'extensions/outputs/test_images',
        'extensions/outputs/notebooks',
        'extensions/outputs/logs',
        'extensions/outputs/temp'
    ]
    
    for dir_path in extensions_dirs:
        if os.path.exists(dir_path):
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path}")
    
    # 3. 检查文档文件
    print("\n📚 检查文档文件:")
    docs_dir = 'extensions/docs'
    if os.path.exists(docs_dir):
        docs = os.listdir(docs_dir)
        print(f"  📖 找到 {len(docs)} 个文档文件:")
        for doc in sorted(docs):
            print(f"    - {doc}")
    else:
        print("  ❌ 文档目录不存在")
    
    # 4. 检查脚本文件
    print("\n🎬 检查脚本文件:")
    scripts_dir = 'extensions/scripts'
    if os.path.exists(scripts_dir):
        scripts = os.listdir(scripts_dir)
        print(f"  🔧 找到 {len(scripts)} 个脚本文件:")
        for script in sorted(scripts):
            print(f"    - {script}")
    else:
        print("  ❌ 脚本目录不存在")
    
    # 5. 检查输出文件
    print("\n📊 检查输出文件:")
    om = get_output_manager()
    outputs = om.list_outputs()
    
    for dir_name, files in outputs.items():
        print(f"  📁 {dir_name}: {len(files)} 个文件")
        for file in sorted(files)[:5]:  # 只显示前5个
            print(f"    - {file}")
        if len(files) > 5:
            print(f"    ... 还有 {len(files) - 5} 个文件")
    
    # 6. 检查RemoteCLIP原始文件
    print("\n🎯 检查RemoteCLIP原始文件:")
    remoteclip_files = [
        'checkpoints',
        'datasets',
        'assets',
        'remoteclip',
        'README.md',
        'README_CN.md'
    ]
    
    for file_path in remoteclip_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
    
    # 7. 测试输出管理器
    print("\n🔧 测试输出管理器:")
    try:
        # 测试各种路径生成
        detection_path = om.get_detection_result_path('ship', 'RN50')
        vis_path = om.get_visualization_path('pyramid', 'airport')
        test_path = om.get_test_image_path('bbox_refinement')
        log_path = om.get_log_path('test')
        
        print(f"  ✅ 检测结果路径: {detection_path}")
        print(f"  ✅ 可视化路径: {vis_path}")
        print(f"  ✅ 测试图像路径: {test_path}")
        print(f"  ✅ 日志路径: {log_path}")
        
    except Exception as e:
        print(f"  ❌ 输出管理器测试失败: {e}")
    
    print("\n" + "=" * 70)
    print("✅ 文件组织测试完成！")
    print("=" * 70)


def test_output_generation():
    """测试输出文件生成"""
    print("\n🧪 测试输出文件生成:")
    
    om = get_output_manager()
    
    # 创建一些测试文件
    test_files = [
        ('detection_results', 'test_ship_detection.jpg'),
        ('visualizations', 'test_visualization.jpg'),
        ('test_images', 'test_image.jpg'),
        ('logs', 'test.log')
    ]
    
    for subdir, filename in test_files:
        file_path = os.path.join(om.dirs[subdir], filename)
        try:
            with open(file_path, 'w') as f:
                f.write(f"Test file for {subdir}\n")
            print(f"  ✅ 创建测试文件: {file_path}")
        except Exception as e:
            print(f"  ❌ 创建测试文件失败: {e}")
    
    # 清理测试文件
    print("\n🧹 清理测试文件:")
    for subdir, filename in test_files:
        file_path = os.path.join(om.dirs[subdir], filename)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"  ✅ 删除测试文件: {filename}")
        except Exception as e:
            print(f"  ❌ 删除测试文件失败: {e}")


if __name__ == "__main__":
    test_file_organization()
    test_output_generation()
