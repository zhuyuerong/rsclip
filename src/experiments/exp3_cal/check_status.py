#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查实验状态"""
import json
from pathlib import Path

output_dir = Path('outputs/exp3_cal')
summary_file = output_dir / 'experiments_summary.json'

print("=" * 80)
print("📊 CAL实验状态检查")
print("=" * 80)

# 检查热图文件
png_files = list(output_dir.rglob("*.png"))
print(f"\n🖼️  已生成热图: {len(png_files)} 张")

if png_files:
    print("\n📁 文件列表:")
    for png in sorted(png_files)[:10]:
        print(f"   - {png.relative_to(output_dir)}")
    if len(png_files) > 10:
        print(f"   ... 还有 {len(png_files) - 10} 张")

# 检查实验记录
if summary_file.exists():
    with open(summary_file, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    print(f"\n📄 实验记录文件: {summary_file}")
    print(f"   总实验数: {summary['results']['total']}")
    print(f"   完成: {summary['results']['completed']}")
    print(f"   跳过: {summary['results']['skipped']}")
    print(f"   失败: {summary['results']['failed']}")
else:
    print(f"\n⚠️  实验记录文件不存在: {summary_file}")

# 检查实验目录
config_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
print(f"\n📦 实验配置目录: {len(config_dirs)} 个")
for config_dir in sorted(config_dirs):
    png_count = len(list(config_dir.glob("*.png")))
    print(f"   - {config_dir.name}: {png_count} 张热图")

print("\n" + "=" * 80)
