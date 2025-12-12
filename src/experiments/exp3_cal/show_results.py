#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""显示实验结果"""
from pathlib import Path
import json

output_dir = Path('outputs/exp3_cal')

print("=" * 80)
print("📊 CAL实验结果")
print("=" * 80)

# 统计所有热图
png_files = sorted(output_dir.rglob("*.png"))
print(f"\n✅ 已保存热图: {len(png_files)} 张")
print(f"📁 总大小: {sum(f.stat().st_size for f in png_files) / 1024 / 1024:.2f} MB\n")

# 按配置分组
configs = {}
for png in png_files:
    config_name = png.parent.name
    if config_name not in configs:
        configs[config_name] = []
    configs[config_name].append(png.name)

print("📋 实验配置完成情况:\n")
for config_name in sorted(configs.keys()):
    files = configs[config_name]
    print(f"  ✅ {config_name}: {len(files)} 张")
    for f in sorted(files)[:3]:
        print(f"     - {f}")
    if len(files) > 3:
        print(f"     ... 还有 {len(files) - 3} 张")

# 检查实验记录
summary_file = output_dir / 'experiments_summary.json'
if summary_file.exists():
    print(f"\n📄 实验记录: {summary_file}")
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    print(f"   完成: {summary['results']['completed']}")
    print(f"   失败: {summary['results']['failed']}")
else:
    print(f"\n⏳ 实验记录: 实验完成后将生成 {summary_file}")

print("\n" + "=" * 80)
