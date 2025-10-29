#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查实验4的数据结构是否符合VV机制的预期格式

检查点：
1. 模型输出格式：应该是 [B, N+1, 512]（包含CLS token）
2. CLS token提取：features[:, 0, :] → [B, 512]
3. Patch tokens提取：features[:, 1:, :] → [B, N, 512]
4. VV机制是否应用
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from experiment4.config import Config

# 直接使用open_clip加载RemoteCLIP（不通过Surgery）
try:
    import open_clip
except ImportError:
    print("需要安装open_clip")
    sys.exit(1)


def load_remoteclip_direct():
    """直接加载RemoteCLIP，检查输出格式"""
    checkpoint_path = "checkpoints/RemoteCLIP-ViT-B-32.pt"
    if not os.path.exists(checkpoint_path):
        alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 "checkpoints", "RemoteCLIP-ViT-B-32.pt")
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
        else:
            raise FileNotFoundError(f"找不到RemoteCLIP权重: {checkpoint_path}")
    
    print(f"加载RemoteCLIP权重: {checkpoint_path}")
    
    config = Config()
    
    # 创建模型
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', device=config.device)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(config.device)
    model.eval()
    
    return model, preprocess, config


def check_model_output_format(model, config):
    """检查模型输出格式"""
    print("\n" + "="*70)
    print("检查模型输出格式")
    print("="*70)
    
    # 创建测试输入
    images = torch.randn(2, 3, 224, 224).to(config.device)
    
    print(f"\n输入图像形状: {images.shape}")
    
    with torch.no_grad():
        # 方法1：使用标准encode_image（获取全局特征）
        global_features = model.encode_image(images)
        print(f"\n【方法1】model.encode_image(images):")
        print(f"  输出形状: {global_features.shape}")
        print(f"  说明: 这是CLS token（全局特征）")
        
        # 方法2：手动提取所有tokens（包括CLS和patches）
        visual = model.visual
        
        # 获取patch embeddings
        x = visual.conv1(images)  # [B, 768, 7, 7] for ViT-B-32
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)  # [B, 49, 768]
        
        # 添加CLS token
        class_embedding = visual.class_embedding.to(x.dtype)
        cls_tokens = class_embedding.unsqueeze(0).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 50, 768]
        
        # 位置编码
        pos_embed = visual.positional_embedding.to(x.dtype)
        x = x + pos_embed
        
        # Layer norm
        x = visual.ln_pre(x)
        
        # Transformer
        x = x.permute(1, 0, 2)  # [50, B, 768]
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)  # [B, 50, 768]
        
        # Layer norm
        x = visual.ln_post(x)
        
        # 投影到512维（如果存在）
        if hasattr(visual, 'proj') and visual.proj is not None:
            B, N, D = x.shape
            x = x.reshape(B * N, D)
            x = x @ visual.proj
            x = x.reshape(B, N, 512)
        
        print(f"\n【方法2】手动提取所有tokens:")
        print(f"  输出形状: {x.shape}")
        print(f"  说明: [B, 50, 512] = CLS token + 49个patches")
        
        # 提取CLS token
        cls_features = x[:, 0, :]  # [B, 512]
        print(f"\n【CLS Token】x[:, 0, :]:")
        print(f"  形状: {cls_features.shape}")
        print(f"  与全局特征是否相同: {torch.allclose(cls_features, global_features, atol=1e-5)}")
        
        # 提取patch tokens
        patch_features = x[:, 1:, :]  # [B, 49, 512]
        print(f"\n【Patch Tokens】x[:, 1:, :]:")
        print(f"  形状: {patch_features.shape}")
        print(f"  网格大小: {int(patch_features.shape[1] ** 0.5)}×{int(patch_features.shape[1] ** 0.5)}")
        
    return x, cls_features, patch_features


def check_vv_mechanism(model):
    """检查是否使用了VV机制"""
    print("\n" + "="*70)
    print("检查VV机制")
    print("="*70)
    
    visual = model.visual
    
    # 检查transformer层
    if hasattr(visual, 'transformer') and hasattr(visual.transformer, 'resblocks'):
        print(f"\nTransformer层数: {len(visual.transformer.resblocks)}")
        
        # 检查最后几层的注意力机制
        print(f"\n最后3层的注意力类型:")
        for i in range(1, min(4, len(visual.transformer.resblocks)) + 1):
            block = visual.transformer.resblocks[-i]
            if hasattr(block, 'attn'):
                attn = block.attn
                attn_type = type(attn).__name__
                print(f"  第{len(visual.transformer.resblocks) - i + 1}层: {attn_type}")
                
                # 检查是否有VV机制的特征
                if hasattr(attn, 'qkv'):
                    print(f"    ✓ 有qkv权重")
                if hasattr(attn, 'scale_multiplier'):
                    print(f"    ✓ 检测到VV机制（scale_multiplier）")
            else:
                print(f"  第{len(visual.transformer.resblocks) - i + 1}层: 无attn属性")
    else:
        print("\n⚠️ 无法访问transformer resblocks")
    
    print(f"\n说明:")
    print(f"  标准RemoteCLIP使用标准的MultiheadAttention")
    print(f"  如果使用VV机制，需要替换为VVAttention")


def check_experiment4_usage():
    """检查实验4中如何使用模型"""
    print("\n" + "="*70)
    print("检查实验4的用法")
    print("="*70)
    
    # 检查train_seen.py中的用法
    train_seen_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_seen.py")
    
    if os.path.exists(train_seen_path):
        with open(train_seen_path, 'r') as f:
            content = f.read()
        
        # 查找关键用法
        if "get_patch_features" in content:
            print(f"\n✅ 实验4使用了 get_patch_features()")
            # 查找具体用法
            import re
            matches = re.findall(r'get_patch_features\([^)]+\)', content)
            if matches:
                print(f"  用法示例: {matches[0] if len(matches) > 0 else 'N/A'}")
        
        if "encode_image" in content:
            print(f"\n✅ 实验4使用了 encode_image()")
        
        if "CLIPSurgeryWrapper" in content:
            print(f"\n✅ 实验4使用了 CLIPSurgeryWrapper")
    else:
        print(f"\n⚠️ 未找到train_seen.py")


def check_expected_vs_actual():
    """对比期望格式和实际格式"""
    print("\n" + "="*70)
    print("期望格式 vs 实际格式对比")
    print("="*70)
    
    print(f"\n【期望格式】（VV机制）:")
    print(f"  model(images) → [B, N+1, 512]  # CLS + N patches")
    print(f"  cls_features = features[:, 0, :]  # [B, 512]")
    print(f"  patch_features = features[:, 1:, :]  # [B, N, 512]")
    
    print(f"\n【实验4当前格式】:")
    print(f"  CLIPSurgeryWrapper.get_patch_features() → [B, N, 512]  # 只有patches，没有CLS")
    print(f"  ⚠️ 缺少CLS token的输出")
    
    print(f"\n【问题】:")
    print(f"  1. 实验4的get_patch_features()只返回patch tokens，不包含CLS token")
    print(f"  2. 需要修改以匹配VV机制的格式：[B, N+1, 512]")
    print(f"  3. 或者单独提供CLS token的提取方法")


def check_clip_surgery_wrapper():
    """检查CLIPSurgeryWrapper的实际实现"""
    print("\n" + "="*70)
    print("检查CLIPSurgeryWrapper实现")
    print("="*70)
    
    clip_surgery_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                      "models", "clip_surgery.py")
    
    if os.path.exists(clip_surgery_path):
        with open(clip_surgery_path, 'r') as f:
            content = f.read()
        
        # 查找get_patch_features
        if 'def get_patch_features' in content:
            # 提取函数定义
            import re
            match = re.search(r'def get_patch_features.*?return.*?', content, re.DOTALL)
            if match:
                func_code = match.group(0)
                print(f"\n【get_patch_features实现】:")
                print(f"{func_code[:500]}...")  # 显示前500字符
                
                if 'features[:, 1:, :]' in func_code:
                    print(f"\n  ✓ 确实去掉了CLS token（features[:, 1:, :]）")
                elif 'features[:, 0, :]' in func_code:
                    print(f"\n  ⚠️ 只返回了CLS token")
                else:
                    print(f"\n  ? 需要检查具体实现")
        
        # 查找encode_image
        if 'def encode_image' in content:
            match = re.search(r'def encode_image.*?return.*?', content, re.DOTALL)
            if match:
                func_code = match.group(0)
                if 'features[:, 0, :]' in func_code:
                    print(f"\n【encode_image实现】:")
                    print(f"  ✓ 返回CLS token（features[:, 0, :]）")
    else:
        print(f"\n⚠️ 未找到clip_surgery.py")


def main():
    """主函数"""
    print("="*70)
    print("检查实验4数据结构是否符合VV机制格式")
    print("="*70)
    
    # 1. 加载模型
    print("\n加载RemoteCLIP模型...")
    model, preprocess, config = load_remoteclip_direct()
    print("✅ 模型加载完成")
    
    # 2. 检查输出格式
    all_features, cls_features, patch_features = check_model_output_format(model, config)
    
    # 3. 检查VV机制
    check_vv_mechanism(model)
    
    # 4. 检查实验4的用法
    check_experiment4_usage()
    
    # 5. 检查CLIPSurgeryWrapper
    check_clip_surgery_wrapper()
    
    # 6. 对比期望和实际
    check_expected_vs_actual()
    
    # 总结
    print("\n" + "="*70)
    print("总结和建议")
    print("="*70)
    
    print(f"\n✅ 当前状态:")
    print(f"  • RemoteCLIP可以提取完整特征 [B, N+1, 512]（包含CLS + patches）")
    print(f"  • CLS token: features[:, 0, :] → [B, 512]")
    print(f"  • Patch tokens: features[:, 1:, :] → [B, N, 512]")
    
    print(f"\n⚠️ 实验4的问题:")
    print(f"  • CLIPSurgeryWrapper.get_patch_features()只返回patches，不包含CLS")
    print(f"  • 需要修改以匹配VV机制的完整输出格式")
    
    print(f"\n💡 建议:")
    print(f"  1. 修改get_patch_features()返回完整特征 [B, N+1, 512]")
    print(f"  2. 或者添加get_all_features()返回完整特征")
    print(f"  3. 确保能够分别提取CLS和patch tokens")


if __name__ == "__main__":
    main()

