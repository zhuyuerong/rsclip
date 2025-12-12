# 为了兼容 tools/inference_on_a_image.py 中的导入
# from groundingdino.models import build_model
import sys
import os

# 添加 models 目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(current_dir))
models_dir = os.path.join(repo_root, "models")
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

# 导入 build_model
from models import build_model

__all__ = ['build_model']

