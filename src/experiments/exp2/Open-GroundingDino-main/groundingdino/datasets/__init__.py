# 为了兼容 tools/inference_on_a_image.py 中的导入
# from groundingdino.datasets import transforms
import sys
import os
import importlib.util

# 添加 datasets 和 util 目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(current_dir))
datasets_dir = os.path.join(repo_root, "datasets")
util_dir = os.path.join(repo_root, "util")
if datasets_dir not in sys.path:
    sys.path.insert(0, datasets_dir)
if util_dir not in sys.path:
    sys.path.insert(0, util_dir)

# 直接导入 transforms 模块
transforms_path = os.path.join(datasets_dir, "transforms.py")
spec = importlib.util.spec_from_file_location("transforms", transforms_path)
transforms_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transforms_module)

# 将 transforms 模块的内容暴露为 transforms
transforms = transforms_module

__all__ = ['transforms']

