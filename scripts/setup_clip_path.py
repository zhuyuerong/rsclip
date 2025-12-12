#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
设置CLIP模块路径

将本地CLIP Surgery实现添加到sys.path，以便使用CS模型
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
clip_surgery_path = project_root / "src/legacy_experiments/experiment6/CLIP_Surgery-master"

if clip_surgery_path.exists():
    if str(clip_surgery_path) not in sys.path:
        sys.path.insert(0, str(clip_surgery_path))
    print(f"✅ 已添加CLIP Surgery路径: {clip_surgery_path}")
else:
    print(f"⚠️  CLIP Surgery路径不存在: {clip_surgery_path}")

