#!/usr/bin/env python3
import torch
import os

print("=" * 70)
print("CUDA 状态检查")
print("=" * 70)

print(f"\nPyTorch版本: {torch.__version__}")
print(f"CUDA编译版本: {torch.version.cuda}")
print(f"cuDNN版本: {torch.backends.cudnn.version()}")

print("\n尝试初始化CUDA...")

# 设置环境变量
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

try:
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.current_device()}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("❌ CUDA不可用")
        print("\n可能的原因:")
        print("  1. NVIDIA驱动版本不匹配")
        print("  2. CUDA toolkit版本不兼容")
        print("  3. 需要重启系统")
except Exception as e:
    print(f"❌ CUDA初始化失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)

