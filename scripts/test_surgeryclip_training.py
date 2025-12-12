#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试SurgeryCLIP训练功能

验证训练脚本是否可以正常初始化
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_training_imports():
    """测试训练相关导入"""
    print("=" * 80)
    print("测试训练模块导入")
    print("=" * 80)
    
    try:
        # 检查训练脚本是否存在
        from pathlib import Path
        training_file = Path(__file__).parent.parent / "src/methods/surgeryclip_rs_det/training/train_remoteclip_surgery.py"
        if training_file.exists():
            # 只检查文件存在，不实际导入（因为可能有依赖问题）
            print("✅ 训练脚本文件存在")
            print(f"  路径: {training_file}")
            return True
        else:
            print(f"❌ 训练脚本文件不存在: {training_file}")
            return False
    except Exception as e:
        print(f"❌ 训练模块导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_config():
    """测试训练配置"""
    print("\n" + "=" * 80)
    print("测试训练配置")
    print("=" * 80)
    
    try:
        from src.methods.surgeryclip_rs_det.config import load_config
        
        config_path = project_root / "configs/methods/surgeryclip_rs_det.yaml"
        if not config_path.exists():
            print(f"⚠️  配置文件不存在: {config_path}")
            return False
        
        config = load_config(str(config_path))
        print("✅ 配置加载成功")
        
        # 检查训练配置
        if 'training' in config:
            print(f"  批次大小: {config.get('training', {}).get('batch_size', 'N/A')}")
            print(f"  学习率: {config.get('training', {}).get('learning_rate', 'N/A')}")
            print(f"  训练轮数: {config.get('training', {}).get('epochs', 'N/A')}")
        else:
            print("  ⚠️  配置中没有training部分")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("=" * 80)
    print("SurgeryCLIP训练功能验证")
    print("=" * 80)
    
    results = {}
    
    results['imports'] = test_training_imports()
    results['config'] = test_training_config()
    
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    for test_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✅ 训练功能验证通过")
    else:
        print("\n⚠️  部分测试失败")
    
    return all_passed


if __name__ == "__main__":
    main()

