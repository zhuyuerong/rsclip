#!/bin/bash
# 完整功能验证脚本

set -e

echo "=================================================================================="
echo "SurgeryCLIP项目完整功能验证"
echo "=================================================================================="

# 激活环境
if [ -d "remoteclip" ]; then
    source remoteclip/bin/activate
    echo "✅ 已激活remoteclip环境"
else
    echo "❌ 未找到remoteclip环境"
    exit 1
fi

# 检查Python版本
echo ""
echo "Python版本:"
python --version

# 检查关键依赖
echo ""
echo "=================================================================================="
echo "1. 检查依赖"
echo "=================================================================================="
python -c "
import sys
from pathlib import Path
project_root = Path.cwd()
sys.path.insert(0, str(project_root / 'src/legacy_experiments/experiment6/CLIP_Surgery-master'))

try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'   CUDA可用: {torch.cuda.is_available()}')
except ImportError:
    print('❌ PyTorch未安装')

try:
    import clip
    print(f'✅ CLIP: 可用（本地实现）')
    print(f'   可用模型数: {len(clip.available_models())}')
except ImportError:
    print('❌ CLIP未安装')

try:
    import numpy
    print(f'✅ NumPy: {numpy.__version__}')
except ImportError:
    print('❌ NumPy未安装')

try:
    from PIL import Image
    print('✅ PIL/Pillow: 可用')
except ImportError:
    print('❌ PIL未安装')
"

# 运行验证脚本
echo ""
echo "=================================================================================="
echo "2. 运行功能验证"
echo "=================================================================================="
python scripts/verify_surgeryclip.py <<< "n"

# 测试训练功能
echo ""
echo "=================================================================================="
echo "3. 测试训练功能"
echo "=================================================================================="
python scripts/test_surgeryclip_training.py

# 测试推理脚本
echo ""
echo "=================================================================================="
echo "4. 测试推理脚本"
echo "=================================================================================="
python -c "
import sys
from pathlib import Path
project_root = Path.cwd()
sys.path.insert(0, str(project_root / 'src/legacy_experiments/experiment6/CLIP_Surgery-master'))

from src.methods.surgeryclip_rs_det.inference_rs import SurgeryCLIPInference

config_path = project_root / 'configs/methods/surgeryclip_rs_det.yaml'
if config_path.exists():
    try:
        inferencer = SurgeryCLIPInference(
            config_path=str(config_path),
            device='cuda' if __import__('torch').cuda.is_available() else 'cpu'
        )
        print('✅ 推理器初始化成功')
        print(f'  模式: {inferencer.config.mode}')
        print(f'  骨干网络: {inferencer.config.backbone}')
    except Exception as e:
        print(f'❌ 初始化失败: {e}')
else:
    print(f'⚠️  配置文件不存在')
"

echo ""
echo "=================================================================================="
echo "验证完成"
echo "=================================================================================="

