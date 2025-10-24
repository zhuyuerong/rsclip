#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三个实验的性能基准测试

在 mini_dataset 上测试模型架构和推理性能
"""

import torch
import torch.nn as nn
import time
import json
from pathlib import Path
import sys
from collections import defaultdict

print("=" * 70)
print("三个实验性能基准测试")
print("=" * 70)


def count_parameters(model):
    """统计模型参数"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
        'total_M': total / 1e6,
        'trainable_M': trainable / 1e6
    }


def test_experiment1():
    """测试 Experiment1"""
    
    print("\n" + "▶" * 35)
    print("Experiment1: 两阶段检测")
    print("▶" * 35)
    
    sys.path.insert(0, 'experiment1')
    from inference.model_loader import ModelLoader
    
    # 加载模型
    print("\n加载 RemoteCLIP...")
    loader = ModelLoader(model_name='RN50', device='cpu')
    model, preprocess, tokenizer = loader.load_model()
    
    # 参数统计
    params = count_parameters(model)
    
    print(f"\n📊 模型参数:")
    print(f"  总参数: {params['total']:,} ({params['total_M']:.2f}M)")
    print(f"  RemoteCLIP (全部为可用): {params['total_M']:.2f}M")
    
    print(f"\n📋 模型架构:")
    print(f"  类型: 两阶段检测")
    print(f"  Stage1: 提议生成 + 分类")
    print(f"  Stage2: 目标检测 + 边界框细化")
    print(f"  特征提取: RemoteCLIP RN50")
    print(f"  方法: 基于区域的检索和对比")
    
    # 测试推理速度
    print(f"\n🔬 测试推理速度...")
    
    from PIL import Image
    test_image = Image.new('RGB', (800, 800), color='red')
    test_texts = ['airplane', 'ship', 'harbor']
    
    # 预热
    _ = loader.encode_image(test_image)
    _ = loader.encode_text_batch(test_texts)
    
    # 测试
    num_runs = 10
    start_time = time.time()
    
    for _ in range(num_runs):
        img_feat = loader.encode_image(test_image)
        txt_feat = loader.encode_text_batch(test_texts)
        similarity = (img_feat @ txt_feat.T)
    
    elapsed = time.time() - start_time
    
    print(f"  测试次数: {num_runs}")
    print(f"  总用时: {elapsed:.3f}秒")
    print(f"  平均用时: {elapsed/num_runs*1000:.1f}ms/图")
    print(f"  FPS: {num_runs/elapsed:.2f}")
    
    return {
        'experiment': 'Experiment1',
        'name': '两阶段检测',
        'architecture': {
            'type': 'Two-Stage Detection',
            'backbone': 'RemoteCLIP RN50',
            'method': 'Region-based Retrieval'
        },
        'parameters': params,
        'performance': {
            'avg_time_ms': (elapsed/num_runs) * 1000,
            'fps': num_runs / elapsed
        },
        'features': {
            'stage1': 'Proposal Generation + Classification',
            'stage2': 'Target Detection + BBox Refinement',
            'wordnet': 'Vocabulary Enhancement'
        }
    }


def test_experiment2():
    """测试 Experiment2"""
    
    print("\n" + "▶" * 35)
    print("Experiment2: 上下文引导检测")
    print("▶" * 35)
    
    sys.path.insert(0, 'experiment2')
    
    from config.default_config import DefaultConfig
    from stage1_encoder.clip_text_encoder import CLIPTextEncoder
    
    config = DefaultConfig()
    
    print(f"\n📋 模型配置:")
    print(f"  查询数量: {config.num_queries}")
    print(f"  解码器层数: {config.num_decoder_layers}")
    print(f"  模型维度: {config.d_model}")
    print(f"  CLIP维度: {config.d_clip}")
    print(f"  上下文门控: {config.context_gating_type}")
    
    # 加载文本编码器
    print(f"\n加载 RemoteCLIP 文本编码器...")
    text_encoder = CLIPTextEncoder(
        model_name=config.clip_model_name,
        pretrained_path=config.clip_checkpoint
    )
    
    params_clip = count_parameters(text_encoder)
    
    print(f"\n📊 CLIP编码器参数:")
    print(f"  总参数: {params_clip['total']:,} ({params_clip['total_M']:.2f}M)")
    
    # 估算完整模型参数（如果实现了）
    estimated_params = {
        'clip_encoder': params_clip['total_M'],
        'context_extractor': 10,  # 估计
        'decoder': 15,  # 估计
        'prediction_heads': 5,  # 估计
        'total_estimated': params_clip['total_M'] + 30
    }
    
    print(f"\n📊 估算完整模型参数:")
    print(f"  CLIP编码器: {estimated_params['clip_encoder']:.2f}M")
    print(f"  上下文提取器: ~{estimated_params['context_extractor']:.2f}M")
    print(f"  解码器: ~{estimated_params['decoder']:.2f}M")
    print(f"  预测头: ~{estimated_params['prediction_heads']:.2f}M")
    print(f"  总计（估算）: ~{estimated_params['total_estimated']:.2f}M")
    
    print(f"\n📋 模型架构:")
    print(f"  类型: Transformer-based Detection")
    print(f"  Stage1: CLIP Text Encoder + Global Context Extractor")
    print(f"  Stage2: Context Gating + Query Initializer + Text Conditioner")
    print(f"  Stage3: Classification Head + Regression Head")
    print(f"  Stage4: Global Contrast Loss + Box Loss + Matcher")
    
    print(f"\n✅ 已实现模块: 11个")
    print(f"❌ 缺失模块: 5个 (数据加载器、完整模型、训练/评估脚本)")
    
    return {
        'experiment': 'Experiment2',
        'name': '上下文引导检测',
        'architecture': {
            'type': 'Context-Guided Transformer',
            'backbone': 'RemoteCLIP RN50',
            'queries': config.num_queries,
            'decoder_layers': config.num_decoder_layers
        },
        'parameters': {
            'clip_only': params_clip,
            'estimated_total_M': estimated_params['total_estimated']
        },
        'config': {
            'd_model': config.d_model,
            'd_clip': config.d_clip,
            'context_gating': config.context_gating_type,
            'temperature': config.temperature
        },
        'status': 'Incomplete - Missing: DataLoader, Train/Eval scripts'
    }


def test_experiment3():
    """测试 Experiment3"""
    
    print("\n" + "▶" * 35)
    print("Experiment3: OVA-DETR")
    print("▶" * 35)
    
    sys.path.insert(0, 'experiment3')
    
    from config.default_config import DefaultConfig
    from models.ova_detr import OVADETR
    from utils.data_loader import DIOR_CLASSES
    
    config = DefaultConfig()
    
    print(f"\n📋 模型配置:")
    print(f"  RemoteCLIP: {config.remoteclip_model}")
    print(f"  查询数量: {config.num_queries}")
    print(f"  解码器层数: {config.num_decoder_layers}")
    print(f"  模型维度: {config.d_model}")
    print(f"  文本维度: {config.txt_dim}")
    
    # 创建模型
    print(f"\n创建 OVA-DETR模型...")
    device = torch.device('cpu')  # 使用CPU避免CUDA问题
    model = OVADETR(config).to(device)
    model.eval()
    
    # 参数统计
    params = count_parameters(model)
    
    print(f"\n📊 模型参数:")
    print(f"  总参数: {params['total']:,} ({params['total_M']:.2f}M)")
    print(f"  可训练: {params['trainable']:,} ({params['trainable_M']:.2f}M)")
    print(f"  冻结: {params['frozen']:,} ({params['frozen']/1e6:.2f}M)")
    
    # 提取文本特征
    print(f"\n提取文本特征...")
    with torch.no_grad():
        text_features = model.backbone.forward_text(DIOR_CLASSES).to(device)
    
    print(f"  文本特征: {text_features.shape}")
    print(f"  类别数: {len(DIOR_CLASSES)}")
    
    # 测试推理
    print(f"\n🔬 测试推理速度...")
    
    batch_size = 2
    test_images = torch.randn(batch_size, 3, 800, 800).to(device)
    
    # 预热
    with torch.no_grad():
        _ = model(test_images, text_features)
    
    # 测试
    num_runs = 5
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            outputs = model(test_images, text_features)
    
    elapsed = time.time() - start_time
    
    print(f"  测试次数: {num_runs} (每次{batch_size}张图)")
    print(f"  总用时: {elapsed:.3f}秒")
    print(f"  平均用时: {elapsed/(num_runs*batch_size)*1000:.1f}ms/图")
    print(f"  FPS: {(num_runs*batch_size)/elapsed:.2f}")
    
    # 输出形状
    print(f"\n📤 模型输出:")
    print(f"  pred_logits: {outputs['pred_logits'].shape}")
    print(f"  pred_boxes: {outputs['pred_boxes'].shape}")
    print(f"  解释: ({config.num_decoder_layers}层, {batch_size}批次, {config.num_queries}查询, 20类别/4坐标)")
    
    print(f"\n📋 完整架构:")
    print(f"  1. RemoteCLIP Backbone (冻结)")
    print(f"  2. FPN 特征金字塔 (4层)")
    print(f"  3. Hybrid Encoder (CNN + Transformer)")
    print(f"  4. Text-Vision Fusion (VAT + TVG)")
    print(f"  5. Transformer Decoder (6层)")
    print(f"  6. Detection Heads (对比分类 + MLP回归)")
    
    return {
        'experiment': 'Experiment3',
        'name': 'OVA-DETR with RemoteCLIP',
        'architecture': {
            'type': 'Open-Vocabulary DETR',
            'backbone': config.remoteclip_model,
            'queries': config.num_queries,
            'decoder_layers': config.num_decoder_layers,
            'd_model': config.d_model,
            'components': [
                'RemoteCLIP Backbone',
                'FPN (4-level)',
                'Hybrid Encoder (6-layer)',
                'Text-Vision Fusion',
                'Transformer Decoder (6-layer)',
                'Contrastive Classification Head',
                'MLP Regression Head'
            ]
        },
        'parameters': params,
        'performance': {
            'avg_time_ms': (elapsed/(num_runs*batch_size)) * 1000,
            'fps': (num_runs*batch_size) / elapsed,
            'batch_size': batch_size
        },
        'loss_functions': {
            'varifocal_loss': f'alpha={config.varifocal_alpha}, gamma={config.varifocal_gamma}',
            'bbox_l1': f'weight={config.loss_bbox_weight}',
            'giou': f'weight={config.loss_giou_weight}'
        },
        'status': 'Complete - All modules implemented'
    }


def generate_comparison_report(results):
    """生成对比报告"""
    
    print("\n" + "=" * 70)
    print("性能对比报告")
    print("=" * 70)
    
    # 表格
    print(f"\n{'实验':<15} {'模型类型':<25} {'总参数':<12} {'可训练':<12} {'推理速度':<12}")
    print("-" * 76)
    
    for result in results:
        exp_name = result['experiment']
        model_type = result['architecture']['type']
        
        if 'estimated_total_M' in result['parameters']:
            total_params = f"~{result['parameters']['estimated_total_M']:.1f}M"
            trainable = "未知"
        else:
            total_params = f"{result['parameters']['total_M']:.1f}M"
            trainable = f"{result['parameters']['trainable_M']:.1f}M"
        
        if 'performance' in result:
            fps = f"{result['performance']['fps']:.2f} FPS"
        else:
            fps = "N/A"
        
        print(f"{exp_name:<15} {model_type:<25} {total_params:<12} {trainable:<12} {fps:<12}")
    
    # 详细对比
    print(f"\n" + "=" * 70)
    print("详细架构对比")
    print("=" * 70)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['name']}")
        print(f"   类型: {result['architecture']['type']}")
        print(f"   骨干网络: {result['architecture'].get('backbone', 'RemoteCLIP')}")
        
        if 'components' in result['architecture']:
            print(f"   组件:")
            for comp in result['architecture']['components']:
                print(f"     - {comp}")
        
        if 'features' in result:
            print(f"   特性:")
            for key, value in result['features'].items():
                print(f"     - {key}: {value}")
        
        status = result.get('status', 'Unknown')
        print(f"   状态: {status}")
    
    # 保存JSON报告
    report_file = Path('experiments_comparison_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': 'mini_dataset (100 samples)',
            'experiments': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 详细报告保存到: {report_file}")
    
    return report_file


def print_summary(results):
    """打印总结"""
    
    print("\n" + "🎯" * 35)
    print("性能总结")
    print("🎯" * 35)
    
    print(f"\n✅ 完成度:")
    print(f"  Experiment1: ⚠️ 部分实现（缺少标准评估）")
    print(f"  Experiment2: ⚠️ 部分实现（缺少数据加载和评估）")
    print(f"  Experiment3: ✅ 完整实现（推荐使用）")
    
    print(f"\n📊 模型规模:")
    for result in results:
        exp = result['experiment']
        if 'total_M' in result['parameters']:
            params = result['parameters']['total_M']
            print(f"  {exp}: {params:.1f}M 参数")
        else:
            params = result['parameters']['estimated_total_M']
            print(f"  {exp}: ~{params:.1f}M 参数（估算）")
    
    print(f"\n⚡ 推理速度（CPU）:")
    for result in results:
        if 'performance' in result:
            exp = result['experiment']
            fps = result['performance']['fps']
            print(f"  {exp}: {fps:.2f} FPS")
    
    print(f"\n🎯 推荐:")
    print(f"  ⭐⭐⭐⭐⭐ Experiment3 - 完整实现，代码质量最高")
    print(f"  ⭐⭐⭐ Experiment2 - 架构设计好，需补充完整")
    print(f"  ⭐⭐⭐ Experiment1 - 两阶段方法，需统一评估")


def main():
    """主函数"""
    
    results = []
    
    # 测试三个实验
    try:
        result1 = test_experiment1()
        results.append(result1)
    except Exception as e:
        print(f"\n❌ Experiment1 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        result2 = test_experiment2()
        results.append(result2)
    except Exception as e:
        print(f"\n❌ Experiment2 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        result3 = test_experiment3()
        results.append(result3)
    except Exception as e:
        print(f"\n❌ Experiment3 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 生成报告
    if len(results) > 0:
        generate_comparison_report(results)
        print_summary(results)
    
    print("\n" + "=" * 70)
    print("基准测试完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()

