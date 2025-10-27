# -*- coding: utf-8 -*-
"""
实验4 - 安装验证脚本
快速检查所有组件是否正常
"""

import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_header(text):
    """打印标题"""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def test_imports():
    """测试模块导入"""
    print_header("测试1: 模块导入")
    
    tests = []
    
    # 测试Python标准库
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        tests.append(True)
    except Exception as e:
        print(f"✗ PyTorch导入失败: {e}")
        tests.append(False)
    
    try:
        import torchvision
        print(f"✓ torchvision: {torchvision.__version__}")
        tests.append(True)
    except Exception as e:
        print(f"✗ torchvision导入失败: {e}")
        tests.append(False)
    
    try:
        import clip
        print(f"✓ CLIP")
        tests.append(True)
    except Exception as e:
        print(f"✗ CLIP导入失败: {e}")
        tests.append(False)
    
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
        tests.append(True)
    except Exception as e:
        print(f"✗ NumPy导入失败: {e}")
        tests.append(False)
    
    try:
        from PIL import Image
        print(f"✓ Pillow")
        tests.append(True)
    except Exception as e:
        print(f"✗ Pillow导入失败: {e}")
        tests.append(False)
    
    try:
        import matplotlib
        print(f"✓ matplotlib: {matplotlib.__version__}")
        tests.append(True)
    except Exception as e:
        print(f"✗ matplotlib导入失败: {e}")
        tests.append(False)
    
    try:
        from tqdm import tqdm
        print(f"✓ tqdm")
        tests.append(True)
    except Exception as e:
        print(f"✗ tqdm导入失败: {e}")
        tests.append(False)
    
    return all(tests)


def test_experiment4_modules():
    """测试实验4模块"""
    print_header("测试2: 实验4模块")
    
    tests = []
    
    try:
        from experiment4.config import get_config
        config = get_config()
        print(f"✓ config.py (device: {config.device})")
        tests.append(True)
    except Exception as e:
        print(f"✗ config.py导入失败: {e}")
        tests.append(False)
    
    try:
        from experiment4.models.clip_surgery import CLIPSurgery
        print(f"✓ clip_surgery.py")
        tests.append(True)
    except Exception as e:
        print(f"✗ clip_surgery.py导入失败: {e}")
        tests.append(False)
    
    try:
        from experiment4.models.noise_filter import RuleBasedDenoiser
        print(f"✓ noise_filter.py")
        tests.append(True)
    except Exception as e:
        print(f"✗ noise_filter.py导入失败: {e}")
        tests.append(False)
    
    try:
        from experiment4.models.decomposer import TextGuidedDecomposer, ImageOnlyDecomposer
        print(f"✓ decomposer.py")
        tests.append(True)
    except Exception as e:
        print(f"✗ decomposer.py导入失败: {e}")
        tests.append(False)
    
    try:
        from experiment4 import losses
        print(f"✓ losses.py")
        tests.append(True)
    except Exception as e:
        print(f"✗ losses.py导入失败: {e}")
        tests.append(False)
    
    try:
        from experiment4.data.dataset import MiniDataset
        print(f"✓ dataset.py")
        tests.append(True)
    except Exception as e:
        print(f"✗ dataset.py导入失败: {e}")
        tests.append(False)
    
    try:
        from experiment4.data.wordnet_utils import get_wordnet_words
        print(f"✓ wordnet_utils.py")
        tests.append(True)
    except Exception as e:
        print(f"✗ wordnet_utils.py导入失败: {e}")
        tests.append(False)
    
    return all(tests)


def test_directories():
    """测试目录结构"""
    print_header("测试3: 目录结构")
    
    required_dirs = [
        'experiment4',
        'experiment4/models',
        'experiment4/data',
        'experiment4/checkpoints',
        'experiment4/outputs',
        'experiment4/logs',
    ]
    
    tests = []
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}")
            tests.append(True)
        else:
            print(f"✗ {dir_path} 不存在")
            tests.append(False)
    
    return all(tests)


def test_files():
    """测试关键文件"""
    print_header("测试4: 关键文件")
    
    required_files = [
        'experiment4/config.py',
        'experiment4/losses.py',
        'experiment4/train_seen.py',
        'experiment4/inference_seen.py',
        'experiment4/inference_unseen.py',
        'experiment4/demo.py',
        'experiment4/quick_start.sh',
        'experiment4/README.md',
        'experiment4/models/clip_surgery.py',
        'experiment4/models/decomposer.py',
        'experiment4/models/noise_filter.py',
        'experiment4/data/dataset.py',
        'experiment4/data/wordnet_utils.py',
    ]
    
    tests = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✓ {file_path} ({size} bytes)")
            tests.append(True)
        else:
            print(f"✗ {file_path} 不存在")
            tests.append(False)
    
    return all(tests)


def test_cuda():
    """测试CUDA"""
    print_header("测试5: CUDA可用性")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print(f"✓ CUDA可用")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            print(f"  当前GPU: {torch.cuda.current_device()}")
            print(f"  GPU名称: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA版本: {torch.version.cuda}")
        else:
            print(f"⚠ CUDA不可用，将使用CPU（速度较慢）")
        
        return True
    except Exception as e:
        print(f"✗ CUDA测试失败: {e}")
        return False


def test_simple_model():
    """测试简单模型创建"""
    print_header("测试6: 模型创建")
    
    try:
        import torch
        from experiment4.config import get_config
        from experiment4.models.decomposer import ImageOnlyDecomposer
        
        config = get_config()
        
        # 创建模型
        model = ImageOnlyDecomposer(config)
        print(f"✓ 创建ImageOnlyDecomposer")
        
        # 测试前向传播
        dummy_input = torch.randn(2, 196, 512)
        output = model(dummy_input)
        print(f"✓ 前向传播成功: {output.shape}")
        
        # 统计参数
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ 模型参数: {num_params:,}")
        
        return True
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "🔍" * 30)
    print("实验4 - 安装验证")
    print("🔍" * 30)
    
    results = []
    
    # 运行所有测试
    results.append(("依赖库导入", test_imports()))
    results.append(("实验4模块", test_experiment4_modules()))
    results.append(("目录结构", test_directories()))
    results.append(("关键文件", test_files()))
    results.append(("CUDA可用性", test_cuda()))
    results.append(("模型创建", test_simple_model()))
    
    # 总结
    print_header("测试总结")
    
    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{status} - {test_name}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！实验4已正确安装。")
        print("\n下一步:")
        print("  1. 运行快速启动脚本:")
        print("     ./experiment4/quick_start.sh")
        print("\n  2. 或运行Demo:")
        print("     python experiment4/demo.py assets/airport.jpg")
    else:
        print("⚠️  部分测试失败，请检查上述错误信息。")
        print("\n可能的解决方案:")
        print("  1. 安装缺失的依赖: pip install torch torchvision clip-by-openai")
        print("  2. 检查Python版本: python --version (需要3.7+)")
        print("  3. 查看详细文档: experiment4/使用指南.md")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

