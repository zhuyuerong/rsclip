#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查代码实现的5个关键点
"""
import sys
import os
from pathlib import Path
import re

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*70)
print("🔍 检查代码实现的5个关键点")
print("="*70)

# 1️⃣ 检查主循环逻辑
print("\n1️⃣ 检查主循环逻辑 (inference_mini_dior.py)")
print("-"*70)

with open('inference_mini_dior.py', 'r') as f:
    content = f.read()
    
# 检查是否有类别数量判断
if 'len(actual_classes)' in content:
    print("✅ 找到 len(actual_classes) 判断")
    
    # 查找判断逻辑
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'len(actual_classes)' in line:
            print(f"   行 {i+1}: {line.strip()}")
            # 显示上下文
            for j in range(max(0, i-2), min(len(lines), i+3)):
                if j != i:
                    print(f"   {j+1:4d}: {lines[j]}")
            break
else:
    print("❌ 未找到 len(actual_classes) 判断")

# 检查是否有单类别分支
if 'generate_heatmap_overlay' in content and '单类别' in content:
    print("\n✅ 找到单类别分支调用 generate_heatmap_overlay")
else:
    print("\n❌ 未找到单类别分支")

# 检查是否有多类别分支
if 'generate_heatmap_overlay_multi' in content and '多类别' in content:
    print("✅ 找到多类别分支调用 generate_heatmap_overlay_multi")
else:
    print("❌ 未找到多类别分支")

# 2️⃣ 检查单类别函数
print("\n2️⃣ 检查单类别函数 (generate_heatmap_overlay)")
print("-"*70)

# 查找函数定义
pattern = r'def generate_heatmap_overlay\([^)]+\):.*?(?=\n\ndef|\n\nclass|\Z)'
match = re.search(pattern, content, re.DOTALL)
if match:
    func_content = match.group(0)
    if '[class_name]' in func_content:
        print("✅ 单类别函数传入 [class_name] (单个类别列表)")
    else:
        print("❌ 单类别函数未传入 [class_name]")
    
    if 'model.generate_heatmap(image, [class_name]' in func_content:
        print("✅ 调用 model.generate_heatmap(image, [class_name])")
    else:
        print("⚠️  检查调用方式...")
        if 'model.generate_heatmap' in func_content:
            print("   找到 generate_heatmap 调用，但参数可能不正确")
else:
    print("❌ 未找到 generate_heatmap_overlay 函数")

# 3️⃣ 检查多类别函数
print("\n3️⃣ 检查多类别函数 (generate_heatmap_overlay_multi)")
print("-"*70)

pattern = r'def generate_heatmap_overlay_multi\([^)]+\):.*?(?=\n\ndef|\n\nclass|\Z)'
match = re.search(pattern, content, re.DOTALL)
if match:
    func_content = match.group(0)
    print("✅ 找到 generate_heatmap_overlay_multi 函数")
    
    if 'all_classes' in func_content:
        print("✅ 函数接收 all_classes 参数")
    else:
        print("❌ 函数未接收 all_classes 参数")
    
    if 'model.generate_heatmap(image, all_classes' in func_content:
        print("✅ 调用 model.generate_heatmap(image, all_classes, ...)")
    else:
        print("❌ 未找到正确的 generate_heatmap 调用")
    
    if 'return_features=True' in func_content:
        print("✅ 使用 return_features=True")
    else:
        print("❌ 未使用 return_features=True")
    
    if 'similarity_maps' in func_content and 'target_idx' in func_content:
        print("✅ 从 similarity_maps 提取目标类别")
    else:
        print("❌ 未从 similarity_maps 提取目标类别")
else:
    print("❌ 未找到 generate_heatmap_overlay_multi 函数")

# 4️⃣ 检查 model_wrapper.py
print("\n4️⃣ 检查 model_wrapper.py (generate_heatmap)")
print("-"*70)

with open('src/competitors/clip_methods/surgeryclip/model_wrapper.py', 'r') as f:
    wrapper_content = f.read()

if 'def generate_heatmap' in wrapper_content:
    print("✅ 找到 generate_heatmap 函数")
    
    if 'return_features' in wrapper_content:
        print("✅ 函数支持 return_features 参数")
    else:
        print("❌ 函数不支持 return_features 参数")
    
    if 'similarity_maps' in wrapper_content and 'return_features' in wrapper_content:
        # 检查是否返回 similarity_maps
        if "'similarity_maps': similarity_maps" in wrapper_content or '"similarity_maps": similarity_maps' in wrapper_content:
            print("✅ return_features=True 时返回 similarity_maps")
        else:
            print("⚠️  检查返回内容...")
    else:
        print("❌ 未找到 similarity_maps 返回")
else:
    print("❌ 未找到 generate_heatmap 函数")

# 5️⃣ 检查数据集中的多类别图像
print("\n5️⃣ 检查数据集中的多类别图像")
print("-"*70)

from inference_mini_dior import load_mini_dior_split, load_mini_dior_annotations

split_file = 'datasets/mini-DIOR/splits/val.txt'
annotation_dir = 'datasets/mini-DIOR/annotations'

image_ids = load_mini_dior_split(split_file)
single_class_images = []
multi_class_images = []

for image_id in image_ids:
    gt_boxes = load_mini_dior_annotations(annotation_dir, image_id)
    if len(gt_boxes) == 0:
        continue
    
    actual_classes = sorted(list(set([box['class'] for box in gt_boxes])))
    
    if len(actual_classes) == 1:
        single_class_images.append((image_id, actual_classes[0]))
    elif len(actual_classes) > 1:
        multi_class_images.append((image_id, actual_classes))

print(f"✅ 单类别图像: {len(single_class_images)} 张")
if len(single_class_images) > 0:
    print(f"   示例: {single_class_images[0]}")

print(f"✅ 多类别图像: {len(multi_class_images)} 张")
if len(multi_class_images) > 0:
    print(f"   示例: {multi_class_images[0]}")
    print(f"   前3个多类别图像:")
    for img_id, classes in multi_class_images[:3]:
        print(f"     {img_id}: {classes}")
else:
    print("   ⚠️  数据集中没有多类别图像！")

print("\n" + "="*70)
print("✅ 检查完成！")
print("="*70)






