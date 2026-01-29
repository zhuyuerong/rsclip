#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
确认SurgeryCLIP + CAL(scene_neg)热图格式

热图来源: SurgeryCLIP baseline + CAL(scene_neg)
"""

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("检查 SurgeryCLIP + CAL(scene_neg) 热图格式")
print("=" * 70)


def check_heatmap_format():
    """检查热图格式"""
    
    # 尝试导入
    try:
        from src.competitors.clip_methods.surgeryclip.model_wrapper import SurgeryCLIPWrapper
        from src.competitors.clip_methods.surgeryclip.cal_config import CALConfig
        print("✅ 模块导入成功")
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return None
    
    # 查找checkpoint
    checkpoint_path = None
    for path in [
        project_root / "checkpoints/RemoteCLIP-ViT-B-32.pt",
        project_root / "checkpoints/ViT-B-32.pt",
    ]:
        if path.exists():
            checkpoint_path = str(path)
            break
    
    if checkpoint_path is None:
        print("❌ 未找到checkpoint")
        return None
    
    print(f"✅ Checkpoint: {checkpoint_path}")
    
    # 查找测试图像
    image_path = None
    for path in [
        project_root / "datasets/DIOR/images/trainval/00053.jpg",
        project_root / "datasets/mini-DIOR/images/00053.jpg",
        project_root / "datasets/DIOR/images/test/00053.jpg",
    ]:
        if path.exists():
            image_path = str(path)
            break
    
    if image_path is None:
        print("⚠️ 未找到测试图像，使用随机图像")
        # 创建随机测试图像
        test_image = Image.fromarray(np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8))
    else:
        print(f"✅ 测试图像: {image_path}")
        test_image = Image.open(image_path).convert('RGB')
    
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ 设备: {device}")
    
    # 创建CAL(scene_neg)配置
    cal_scene_neg_config = CALConfig(
        enable_cal=True,
        negative_mode='fixed',
        fixed_negatives=["aerial view", "satellite image", "remote sensing scene"],
        alpha=2.0,
        cal_space='similarity',
        experiment_name='cal_scene_neg',
        verbose=False
    )
    
    # 创建模型
    print("\n📥 加载 SurgeryCLIP + CAL(scene_neg) 模型...")
    model = SurgeryCLIPWrapper(
        model_name="surgeryclip",
        checkpoint_path=checkpoint_path,
        device=device,
        use_surgery_single="empty",  # SurgeryCLIP baseline
        use_surgery_multi=True,
        cal_config=cal_scene_neg_config  # + CAL(scene_neg)
    )
    model.load_model()
    print("✅ 模型加载完成")
    
    # 生成热图
    print("\n🔥 生成热图...")
    class_name = "baseballfield"
    
    heatmap = model.generate_heatmap(test_image, [class_name])
    
    # 分析热图格式
    print("\n" + "=" * 70)
    print("📊 热图格式分析")
    print("=" * 70)
    
    print(f"\n1️⃣  类型: {type(heatmap)}")
    print(f"2️⃣  dtype: {heatmap.dtype}")
    print(f"3️⃣  shape: {heatmap.shape}")
    print(f"4️⃣  值域: [{heatmap.min():.6f}, {heatmap.max():.6f}]")
    print(f"5️⃣  均值: {heatmap.mean():.6f}")
    print(f"6️⃣  标准差: {heatmap.std():.6f}")
    print(f"7️⃣  是否有NaN: {np.isnan(heatmap).any()}")
    print(f"8️⃣  是否有Inf: {np.isinf(heatmap).any()}")
    
    # 检查是否在[0,1]范围
    in_range = (heatmap >= 0).all() and (heatmap <= 1).all()
    print(f"9️⃣  在[0,1]范围内: {in_range}")
    
    # 检查空间分布
    print(f"\n🔟 空间分布:")
    h, w = heatmap.shape
    print(f"   左上角 (0:10, 0:10): mean={heatmap[0:10, 0:10].mean():.4f}")
    print(f"   右下角 ({h-10}:{h}, {w-10}:{w}): mean={heatmap[-10:, -10:].mean():.4f}")
    print(f"   中心 ({h//2-5}:{h//2+5}, {w//2-5}:{w//2+5}): mean={heatmap[h//2-5:h//2+5, w//2-5:w//2+5].mean():.4f}")
    
    # 分位数
    print(f"\n📈 分位数:")
    for q in [0, 25, 50, 75, 90, 95, 99, 100]:
        print(f"   {q}%: {np.percentile(heatmap, q):.6f}")
    
    print("\n" + "=" * 70)
    print("✅ 热图格式检查完成")
    print("=" * 70)
    
    # 返回热图用于进一步分析
    return heatmap, test_image


def check_compatibility_with_query_gen():
    """检查热图与Query Generator的兼容性"""
    
    result = check_heatmap_format()
    if result is None:
        return
    
    heatmap, test_image = result
    
    print("\n" + "=" * 70)
    print("📐 与 HeatmapQueryGenerator 兼容性检查")
    print("=" * 70)
    
    # 导入HeatmapQueryGenerator
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from models.heatmap_query_gen import HeatmapQueryGenerator
        print("✅ HeatmapQueryGenerator 导入成功")
    except ImportError as e:
        print(f"❌ HeatmapQueryGenerator 导入失败: {e}")
        return
    
    # 模拟输入
    B, C, d = 1, 256, 256
    num_levels = 4
    H, W = heatmap.shape
    
    # 转换为tensor并添加batch维度
    heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).float()  # [1, H, W]
    
    print(f"\n🔄 转换后的热图:")
    print(f"   shape: {heatmap_tensor.shape}")
    print(f"   dtype: {heatmap_tensor.dtype}")
    print(f"   device: {heatmap_tensor.device}")
    
    # 模拟多尺度特征 (FPN输出)
    # 假设输入图像resize到800x800，FPN stride为[8, 16, 32, 64]
    input_size = 800
    strides = [8, 16, 32, 64]
    
    srcs = []
    for stride in strides:
        feat_size = input_size // stride
        feat = torch.randn(B, C, feat_size, feat_size)
        srcs.append(feat)
        print(f"   Level stride={stride}: feature shape = {feat.shape}")
    
    spatial_shapes = torch.tensor([[src.shape[2], src.shape[3]] for src in srcs])
    
    # 创建Query Generator
    gen = HeatmapQueryGenerator(
        hidden_dim=d,
        num_queries=100,
        num_feature_levels=num_levels,
        pool_mode='heatmap_weighted',
        pool_window=3,
    )
    
    # 测试生成
    print(f"\n🚀 测试 HeatmapQueryGenerator...")
    
    # 需要将热图resize到合适的尺寸 (与最大特征图对齐)
    # HeatmapQueryGenerator期望的热图尺寸应该与原图或某个参考尺寸对应
    # 这里热图已经是原图尺寸(800x800)，可以直接使用
    
    try:
        output = gen(srcs, spatial_shapes, heatmap_tensor)
        
        print(f"\n✅ 生成成功!")
        print(f"   query_embed: {output['query_embed'].shape}")
        print(f"   query_content: {output['query_content'].shape}")
        print(f"   query_pos: {output['query_pos'].shape}")
        print(f"   reference_points: {output['reference_points'].shape}")
        print(f"   heatmap_scores: {output['heatmap_scores'].shape}")
        
        # 检查reference_points范围
        ref_min = output['reference_points'].min().item()
        ref_max = output['reference_points'].max().item()
        print(f"   reference_points范围: [{ref_min:.4f}, {ref_max:.4f}]")
        
        # 检查scores分布
        scores = output['heatmap_scores'][0]
        print(f"   heatmap_scores分布: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ 兼容性检查完成")
    print("=" * 70)
    
    # 返回总结
    print("\n" + "=" * 70)
    print("📋 热图格式总结 (用于Pseudo Query)")
    print("=" * 70)
    print(f"""
热图来源: SurgeryCLIP baseline + CAL(scene_neg)
生成方式: model.generate_heatmap(image, [class_name])

格式规范:
- 类型: numpy.ndarray
- dtype: float32
- shape: (H, W) 与原图尺寸一致 (例如 800x800)
- 值域: [0, 1] (已归一化)
- 坐标系: 原图像素坐标

使用方法:
```python
# 1. 生成热图
heatmap = model.generate_heatmap(image, [class_name])  # np.ndarray [H, W]

# 2. 转换为tensor
heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).float()  # [B, H, W]

# 3. 输入Query Generator
output = query_gen(srcs, spatial_shapes, heatmap_tensor)
```

注意事项:
- 热图尺寸与原图一致，HeatmapQueryGenerator内部会处理坐标映射
- 热图值域已归一化到[0,1]，无需额外处理
- 高响应区域值接近1，低响应区域值接近0
""")


if __name__ == '__main__':
    check_compatibility_with_query_gen()
