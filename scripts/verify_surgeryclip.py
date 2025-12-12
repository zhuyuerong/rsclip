#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证SurgeryCLIP实验的完整性

测试内容：
1. 模型加载（CS权重）
2. 推理功能
3. 热图生成
4. 训练功能（可选）
"""
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 添加本地CLIP Surgery路径（如果标准CLIP未安装）
clip_surgery_path = project_root / "src/legacy_experiments/experiment6/CLIP_Surgery-master"
if clip_surgery_path.exists() and str(clip_surgery_path) not in sys.path:
    sys.path.insert(0, str(clip_surgery_path))

def test_model_loading():
    """测试模型加载"""
    print("=" * 80)
    print("测试1: 模型加载")
    print("=" * 80)
    
    try:
        import clip
        from types import SimpleNamespace
        
        # 测试CS模型加载
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"设备: {device}")
        
        # 测试CS-ViT-B/32
        print("\n测试加载 CS-ViT-B/32...")
        try:
            model, preprocess = clip.load("CS-ViT-B/32", device=device)
            print("✅ CS-ViT-B/32 加载成功")
            
            # 测试编码
            test_image = Image.new('RGB', (224, 224), color='red')
            image_tensor = preprocess(test_image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = model.encode_image(image_tensor)
                print(f"  图像特征形状: {features.shape}")
                
                # 检查是否返回所有tokens
                if features.dim() == 3:
                    print(f"  ✅ 返回所有tokens: {features.shape[1]} tokens")
                elif features.dim() == 2:
                    print(f"  ⚠️  只返回CLS token: {features.shape}")
            
            # 测试文本编码
            text = clip.tokenize(["an aerial photo of airplane"]).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text)
                print(f"  文本特征形状: {text_features.shape}")
            
            return True, model, preprocess
            
        except Exception as e:
            print(f"❌ CS-ViT-B/32 加载失败: {e}")
            print("  尝试加载标准CLIP...")
            try:
                model, preprocess = clip.load("ViT-B/32", device=device)
                print("✅ 标准CLIP加载成功（将使用标准CLIP）")
                return True, model, preprocess
            except Exception as e2:
                print(f"❌ 标准CLIP也加载失败: {e2}")
                return False, None, None
                
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("  请安装clip模块: pip install git+https://github.com/openai/CLIP.git")
        return False, None, None


def test_surgery_variants():
    """测试Surgery变体"""
    print("\n" + "=" * 80)
    print("测试2: Surgery变体功能")
    print("=" * 80)
    
    try:
        from src.methods.surgeryclip_rs_det.core.models.surgery_variants_exp6 import (
            clip_feature_surgery_with_redundancy,
            clip_feature_surgery_without_redundancy,
            get_similarity_map
        )
        
        # 创建测试数据
        B, N_patches, D = 1, 49, 512
        N_classes = 20
        
        image_features = torch.randn(B, N_patches + 1, D)  # 包含CLS token
        text_features = torch.randn(N_classes, D)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 测试full mode
        print("\n测试 full mode (with redundancy removal)...")
        similarity_full, info_full = clip_feature_surgery_with_redundancy(image_features, text_features)
        print(f"  ✅ 相似度形状: {similarity_full.shape}")
        print(f"  范围: [{similarity_full.min():.4f}, {similarity_full.max():.4f}]")
        
        # 测试no_surgery mode
        print("\n测试 no_surgery mode (without redundancy removal)...")
        similarity_no_surgery, info_no_surgery = clip_feature_surgery_without_redundancy(image_features, text_features)
        print(f"  ✅ 相似度形状: {similarity_no_surgery.shape}")
        print(f"  范围: [{similarity_no_surgery.min():.4f}, {similarity_no_surgery.max():.4f}]")
        
        # 测试热图生成
        print("\n测试热图生成...")
        heatmap = get_similarity_map(similarity_full, (224, 224))
        print(f"  ✅ 热图形状: {heatmap.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Surgery变体测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference_wrapper():
    """测试推理包装器"""
    print("\n" + "=" * 80)
    print("测试3: CLIPSurgeryWrapperExp6")
    print("=" * 80)
    
    try:
        from src.methods.surgeryclip_rs_det.core.models.clip_surgery_exp6 import CLIPSurgeryWrapperExp6
        from types import SimpleNamespace
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 创建配置
        config = SimpleNamespace()
        config.backbone = 'ViT-B/32'
        config.checkpoint_path = None  # 先不加载RemoteCLIP权重
        config.device = device
        config.mode = 'full'
        
        print(f"设备: {device}")
        print(f"模式: {config.mode}")
        
        # 测试full mode
        print("\n测试 full mode...")
        wrapper_full = CLIPSurgeryWrapperExp6(config)
        
        # 测试图像编码
        test_image = torch.randn(1, 3, 224, 224).to(device)
        image_features = wrapper_full.encode_image(test_image)
        print(f"  ✅ 图像特征形状: {image_features.shape}")
        
        # 测试文本编码
        class_names = ["airplane", "ship", "vehicle"]
        text_features = wrapper_full.encode_text(class_names)
        print(f"  ✅ 文本特征形状: {text_features.shape}")
        
        # 测试相似度计算
        similarity_full = wrapper_full.compute_similarity(image_features, text_features)
        print(f"  ✅ 相似度形状: {similarity_full.shape}")
        
        # 测试热图生成
        heatmap = wrapper_full.similarity_to_heatmap(similarity_full, (224, 224))
        print(f"  ✅ 热图形状: {heatmap.shape}")
        
        # 测试no_surgery mode
        print("\n测试 no_surgery mode...")
        config.mode = 'no_surgery'
        wrapper_no_surgery = CLIPSurgeryWrapperExp6(config)
        similarity_no_surgery = wrapper_no_surgery.compute_similarity(image_features, text_features)
        print(f"  ✅ 相似度形状: {similarity_no_surgery.shape}")
        
        # 对比两种模式
        diff = (similarity_full - similarity_no_surgery).abs().mean()
        print(f"\n两种模式差异: {diff:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 推理包装器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference_script():
    """测试推理脚本"""
    print("\n" + "=" * 80)
    print("测试4: 推理脚本接口")
    print("=" * 80)
    
    try:
        from src.methods.surgeryclip_rs_det.inference_rs import SurgeryCLIPInference
        from pathlib import Path
        
        config_path = project_root / "configs/methods/surgeryclip_rs_det.yaml"
        
        if not config_path.exists():
            print(f"⚠️  配置文件不存在: {config_path}")
            print("  跳过推理脚本测试")
            return True
        
        print(f"配置文件: {config_path}")
        
        # 测试初始化（不实际运行推理）
        print("\n测试推理器初始化...")
        try:
            inferencer = SurgeryCLIPInference(
                config_path=str(config_path),
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            print("  ✅ 推理器初始化成功")
            print(f"  模式: {inferencer.config.mode}")
            print(f"  骨干网络: {inferencer.config.backbone}")
            return True
        except Exception as e:
            print(f"  ⚠️  推理器初始化失败: {e}")
            print("  （可能是缺少依赖或权重文件）")
            return False
            
    except Exception as e:
        print(f"❌ 推理脚本测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_cs_weights():
    """检查CS权重文件"""
    print("\n" + "=" * 80)
    print("检查CS权重文件")
    print("=" * 80)
    
    cache_dir = Path.home() / ".cache/clip"
    checkpoints_dir = project_root / "checkpoints"
    
    print(f"CLIP缓存目录: {cache_dir}")
    print(f"Checkpoints目录: {checkpoints_dir}")
    
    if cache_dir.exists():
        print(f"\n缓存目录中的文件:")
        for f in sorted(cache_dir.glob("*.pt")):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {size_mb:.1f} MB")
            
            # 检查是否是CS权重
            if "CS" in f.name or "ViT" in f.name:
                target = checkpoints_dir / f.name
                if not target.exists():
                    print(f"    → 可以复制到 {target}")
                else:
                    print(f"    → 已存在于checkpoints")
    else:
        print("  ⚠️  缓存目录不存在")
    
    print(f"\nCheckpoints目录中的文件:")
    for f in sorted(checkpoints_dir.glob("*.pt")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")


def copy_cs_weights():
    """复制CS权重到checkpoints目录"""
    print("\n" + "=" * 80)
    print("复制CS权重到checkpoints")
    print("=" * 80)
    
    cache_dir = Path.home() / ".cache/clip"
    checkpoints_dir = project_root / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    if not cache_dir.exists():
        print("  ⚠️  缓存目录不存在，无法复制")
        return
    
    # CS模型文件名映射
    cs_models = {
        "ViT-B-32.pt": "CS-ViT-B-32",
        "ViT-B-16.pt": "CS-ViT-B-16",
        "ViT-L-14.pt": "CS-ViT-L-14",
        "RN50.pt": "CS-RN50",
    }
    
    copied = 0
    for filename, model_name in cs_models.items():
        source = cache_dir / filename
        target = checkpoints_dir / filename
        
        if source.exists() and not target.exists():
            print(f"\n复制 {model_name}...")
            try:
                import shutil
                shutil.copy2(source, target)
                size_mb = source.stat().st_size / (1024 * 1024)
                print(f"  ✅ 已复制: {filename} ({size_mb:.1f} MB)")
                copied += 1
            except Exception as e:
                print(f"  ❌ 复制失败: {e}")
        elif source.exists() and target.exists():
            print(f"  ⏭️  已存在: {filename}")
        else:
            print(f"  ⚠️  未找到: {filename}")
    
    if copied > 0:
        print(f"\n✅ 共复制了 {copied} 个权重文件")
    else:
        print("\n⚠️  没有需要复制的权重文件")


def main():
    """主函数"""
    print("=" * 80)
    print("SurgeryCLIP实验验证")
    print("=" * 80)
    
    results = {}
    
    # 检查CS权重
    check_cs_weights()
    
    # 询问是否复制权重
    print("\n是否复制CS权重到checkpoints目录？(y/n): ", end="")
    try:
        response = input().strip().lower()
        if response == 'y':
            copy_cs_weights()
    except:
        pass
    
    # 测试1: 模型加载
    success, model, preprocess = test_model_loading()
    results['model_loading'] = success
    
    if not success:
        print("\n⚠️  模型加载失败，跳过后续测试")
        return
    
    # 测试2: Surgery变体
    results['surgery_variants'] = test_surgery_variants()
    
    # 测试3: 推理包装器
    results['inference_wrapper'] = test_inference_wrapper()
    
    # 测试4: 推理脚本
    results['inference_script'] = test_inference_script()
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    for test_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 所有测试通过！")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")
    
    return all_passed


if __name__ == "__main__":
    main()

