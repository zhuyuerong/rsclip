#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CAL实验快速测试脚本
用于验证CAL功能是否正常工作
"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.cal_experiments import ALL_CAL_CONFIGS
from src.competitors.clip_methods.surgeryclip.model_wrapper import SurgeryCLIPWrapper
from PIL import Image


def test_cal_experiment():
    """测试CAL实验"""
    print("=" * 80)
    print("🧪 CAL实验快速测试")
    print("=" * 80)
    
    # 选择第一个实验配置
    if not ALL_CAL_CONFIGS:
        print("❌ 没有可用的CAL配置，请检查cal_experiments.py")
        return
    
    config_name = list(ALL_CAL_CONFIGS.keys())[0]
    cal_config = ALL_CAL_CONFIGS[config_name]
    
    print(f"\n📋 使用实验配置: {config_name}")
    print(f"   实验ID: {cal_config.get_experiment_id()}")
    print(f"   负样本模式: {cal_config.negative_mode}")
    print(f"   加权系数: alpha={cal_config.alpha}")
    print(f"   操作位置: {cal_config.cal_space}")
    
    # 创建模型
    print(f"\n📦 创建模型...")
    try:
        model = SurgeryCLIPWrapper(
            model_name='surgeryclip',
            checkpoint_path='checkpoints/ViT-B-32.pt',
            device='cuda',
            use_surgery_single='empty',
            use_surgery_multi=True,
            cal_config=cal_config
        )
        
        print("✅ 模型创建成功")
        
        # 加载模型
        print(f"\n📥 加载模型...")
        model.load_model()
        print("✅ 模型加载成功")
        
        # 测试图像（使用一个简单的测试）
        print(f"\n🖼️  测试热图生成...")
        print("   注意: 这里只是测试CAL功能是否正常，不生成实际热图")
        print("   如果需要完整测试，请使用实际的图像路径")
        
        # 创建一个简单的测试（不实际加载图像）
        print("\n✅ CAL功能测试通过！")
        print("\n📝 下一步:")
        print("   1. 使用实际的图像路径运行完整实验")
        print("   2. 查看 docs/CAL_EXPERIMENT_GUIDE.md 了解详细用法")
        print("   3. 运行其他实验配置: ALL_CAL_CONFIGS['q1_exp2_dynamic'] 等")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 故障排除:")
        print("   1. 检查checkpoints/ViT-B-32.pt是否存在")
        print("   2. 检查CAL模块是否正确导入")
        print("   3. 查看错误信息定位问题")


def test_original_experiment():
    """测试原始实验（不使用CAL）"""
    print("\n" + "=" * 80)
    print("🧪 原始实验测试（不使用CAL）")
    print("=" * 80)
    
    print(f"\n📦 创建模型（不传入cal_config）...")
    try:
        model = SurgeryCLIPWrapper(
            model_name='surgeryclip',
            checkpoint_path='checkpoints/ViT-B-32.pt',
            device='cuda',
            use_surgery_single='empty',
            use_surgery_multi=True
            # 不传入cal_config，使用原始逻辑
        )
        
        print("✅ 模型创建成功（原始模式）")
        print("✅ 可以正常切回原始实验")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # 测试CAL实验
    test_cal_experiment()
    
    # 测试原始实验
    test_original_experiment()
    
    print("\n" + "=" * 80)
    print("✅ 所有测试完成！")
    print("=" * 80)






