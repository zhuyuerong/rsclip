# -*- coding: utf-8 -*-
"""
运行示例脚本 - 生成符合标准的多组样图

展示如何使用统一热图生成器生成各种类型的样图
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """运行命令并显示描述"""
    print(f"\n{'='*60}")
    print(f"运行: {description}")
    print(f"命令: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✓ 成功完成")
        if result.stdout:
            print("输出:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 执行失败: {e}")
        if e.stdout:
            print("标准输出:")
            print(e.stdout)
        if e.stderr:
            print("错误输出:")
            print(e.stderr)
        return False

def main():
    """运行各种示例"""
    
    print("=" * 80)
    print("exp3_text_guided_vvt - 运行示例脚本")
    print("=" * 80)
    print("本脚本将演示如何使用统一热图生成器生成各种类型的样图")
    
    # 获取脚本目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent.parent
    
    # 基础命令前缀
    base_cmd = f"cd {project_root} && PYTHONPATH=. ovadetr_env/bin/python3.9 {script_dir}/unified_heatmap_generator.py"
    
    # 示例1: 生成多类别4模式对比样图
    print(f"\n{'='*80}")
    print("示例1: 多类别4模式对比样图")
    print(f"{'='*80}")
    print("生成3个多类别图像的4模式对比热图")
    print("每个图像为每个类别生成独立的4模式对比图")
    print("布局: 4行×13列 (4模式 × 1原图+12层)")
    
    cmd1 = f"{base_cmd} --mode multi_class --max-samples 3"
    success1 = run_command(cmd1, "多类别4模式对比")
    
    # 示例2: GT边界框调试样图
    print(f"\n{'='*80}")
    print("示例2: GT边界框调试样图")
    print(f"{'='*80}")
    print("生成2个样本的GT边界框调试可视化")
    print("布局: 2列 (原图+坐标信息)")
    print("显示: 原始坐标、缩放坐标、边界框信息")
    
    cmd2 = f"{base_cmd} --mode debug_gt --debug-samples 2"
    success2 = run_command(cmd2, "GT边界框调试")
    
    # 示例3: 全面对比样图
    print(f"\n{'='*80}")
    print("示例3: 全面对比样图")
    print(f"{'='*80}")
    print("生成3个样本的全面对比热图")
    print("布局: 4行×13列 (4模式 × 1原图+12层)")
    print("对比: With Surgery, Without Surgery, With VV, Complete Surgery")
    
    cmd3 = f"{base_cmd} --mode comprehensive --max-samples 3"
    success3 = run_command(cmd3, "全面对比实验")
    
    # 示例4: 自定义层分析
    print(f"\n{'='*80}")
    print("示例4: 自定义层分析")
    print(f"{'='*80}")
    print("只分析关键层: L1, L6, L9, L12")
    print("生成更紧凑的对比图")
    
    cmd4 = f"{base_cmd} --mode multi_class --max-samples 2 --layers 1 6 9 12"
    success4 = run_command(cmd4, "自定义层分析")
    
    # 示例5: 运行所有功能
    print(f"\n{'='*80}")
    print("示例5: 运行所有功能")
    print(f"{'='*80}")
    print("一次性运行所有功能，生成完整的样图集合")
    
    cmd5 = f"{base_cmd} --mode all --max-samples 2 --debug-samples 2"
    success5 = run_command(cmd5, "运行所有功能")
    
    # 总结
    print(f"\n{'='*80}")
    print("运行结果总结")
    print(f"{'='*80}")
    
    results = [
        ("多类别4模式对比", success1),
        ("GT边界框调试", success2),
        ("全面对比实验", success3),
        ("自定义层分析", success4),
        ("运行所有功能", success5)
    ]
    
    for name, success in results:
        status = "✓ 成功" if success else "✗ 失败"
        print(f"{name:20} : {status}")
    
    # 检查输出文件
    print(f"\n{'='*80}")
    print("输出文件检查")
    print(f"{'='*80}")
    
    output_dirs = [
        "results/multi_class_4mode",
        "results/debug_gt_boxes", 
        "results/comprehensive_comparison"
    ]
    
    for output_dir in output_dirs:
        dir_path = script_dir / output_dir
        if dir_path.exists():
            png_files = list(dir_path.glob("*.png"))
            print(f"{output_dir:30} : {len(png_files)} 个PNG文件")
            if png_files:
                total_size = sum(f.stat().st_size for f in png_files)
                print(f"{'':30} : 总大小 {total_size/1024/1024:.1f} MB")
        else:
            print(f"{output_dir:30} : 目录不存在")
    
    print(f"\n{'='*80}")
    print("样图生成完成！")
    print(f"{'='*80}")
    print("查看结果:")
    print(f"  - 多类别4模式对比: results/multi_class_4mode/")
    print(f"  - GT边界框调试: results/debug_gt_boxes/")
    print(f"  - 全面对比实验: results/comprehensive_comparison/")
    print(f"  - 文档说明: docs/")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
