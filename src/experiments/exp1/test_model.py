#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本：验证模型是否能正常创建和运行
"""

import torch
import sys
import os
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_model_creation():
    """测试模型创建"""
    print("=" * 80)
    print("测试1: 模型创建")
    print("=" * 80)
    
    try:
        from models.surgery_aaf import create_surgery_aaf_model
        
        # 查找检查点
        project_root = Path(__file__).parent.parent.parent.parent
        possible_checkpoints = [
            project_root / "checkpoints" / "RemoteCLIP-ViT-B-32.pt",
            project_root / "checkpoints" / "ViT-B-32.pt",
        ]
        
        checkpoint_path = None
        for cp in possible_checkpoints:
            if cp.exists():
                checkpoint_path = str(cp)
                print(f"找到检查点: {checkpoint_path}")
                break
        
        if checkpoint_path is None:
            print("⚠️  未找到CLIP检查点，跳过模型创建测试")
            return False
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        model, preprocess = create_surgery_aaf_model(
            checkpoint_path=checkpoint_path,
            device=device,
            num_layers=6
        )
        
        print("✅ 模型创建成功")
        print(f"   模型类型: {type(model)}")
        print(f"   AAF参数数量: {sum(p.numel() for p in model.aaf.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """测试前向传播"""
    print("\n" + "=" * 80)
    print("测试2: 前向传播")
    print("=" * 80)
    
    try:
        from models.surgery_aaf import create_surgery_aaf_model
        
        # 查找检查点
        project_root = Path(__file__).parent.parent.parent.parent
        possible_checkpoints = [
            project_root / "checkpoints" / "RemoteCLIP-ViT-B-32.pt",
            project_root / "checkpoints" / "ViT-B-32.pt",
        ]
        
        checkpoint_path = None
        for cp in possible_checkpoints:
            if cp.exists():
                checkpoint_path = str(cp)
                break
        
        if checkpoint_path is None:
            print("⚠️  未找到CLIP检查点，跳过前向传播测试")
            return False
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = create_surgery_aaf_model(
            checkpoint_path=checkpoint_path,
            device=device,
            num_layers=6
        )
        
        # 创建测试输入
        batch_size = 1
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        text_queries = ["airplane", "ship", "car"]
        
        print(f"输入图像形状: {images.shape}")
        print(f"文本查询: {text_queries}")
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            cam, aux = model(images, text_queries)
        
        print("✅ 前向传播成功")
        print(f"   CAM形状: {cam.shape}")
        print(f"   辅助输出键: {list(aux.keys())}")
        
        if 'attn_p2p' in aux:
            print(f"   p2p注意力形状: {aux['attn_p2p'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """测试数据加载"""
    print("\n" + "=" * 80)
    print("测试3: 数据加载")
    print("=" * 80)
    
    try:
        from utils.data import get_dataloader
        
        # 尝试加载数据
        project_root = Path(__file__).parent.parent.parent.parent
        possible_dior_paths = [
            project_root / "datasets" / "DIOR",
            project_root.parent / "datasets" / "DIOR",
        ]
        
        dior_path = None
        for path in possible_dior_paths:
            if path.exists():
                dior_path = str(path)
                print(f"找到DIOR数据集: {dior_path}")
                break
        
        if dior_path is None:
            print("⚠️  未找到DIOR数据集，跳过数据加载测试")
            print("   提示: 数据集应位于 datasets/DIOR/")
            return False
        
        # 尝试加载一个小批次
        dataloader = get_dataloader(
            dataset_name='DIOR',
            root=dior_path,
            split='trainval',
            batch_size=2,
            num_workers=0,  # 使用0避免多进程问题
            shuffle=False
        )
        
        print(f"数据集大小: {len(dataloader.dataset)}")
        
        # 加载一个批次
        batch = next(iter(dataloader))
        print("✅ 数据加载成功")
        print(f"   批次键: {list(batch.keys())}")
        print(f"   图像形状: {batch['images'].shape}")
        print(f"   标签形状: {batch['labels'].shape}")
        print(f"   类别数量: {len(batch['text_queries'][0])}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_imports():
    """测试导入"""
    print("=" * 80)
    print("测试0: 模块导入")
    print("=" * 80)
    
    try:
        from models import AAF, CAMGenerator, SurgeryAAF
        print("✅ 模型模块导入成功")
        
        from utils import DIORDataset, get_dataloader, visualize_cam, compute_metrics
        print("✅ 工具模块导入成功")
        
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("开始测试 SurgeryCLIP + AAF + p2p 实验代码")
    print("=" * 80 + "\n")
    
    results = []
    
    # 测试导入
    results.append(("模块导入", test_imports()))
    
    # 测试模型创建
    results.append(("模型创建", test_model_creation()))
    
    # 测试前向传播
    results.append(("前向传播", test_forward_pass()))
    
    # 测试数据加载
    results.append(("数据加载", test_data_loading()))
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(r for _, r in results)
    if all_passed:
        print("\n🎉 所有测试通过！")
    else:
        print("\n⚠️  部分测试失败，请检查上述错误信息")
    
    print("=" * 80)





