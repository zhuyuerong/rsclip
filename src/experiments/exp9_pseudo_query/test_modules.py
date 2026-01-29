"""
测试所有pseudo query模块

验证:
1. HeatmapQueryGenerator输出shape正确
2. QueryMixer各种模式正常工作
3. Loss函数可计算
4. 与Deformable DETR的query_embed接口兼容
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from models.heatmap_query_gen import (
    HeatmapQueryGenerator,
    TeacherQueryGenerator,
    FusionQueryGenerator,
    build_query_generator,
    PositionalEncoding2D
)
from models.query_injection import (
    QueryMixer,
    QueryAlignmentLoss,
    AttentionPriorLoss,
    PseudoQueryCriterion
)


def test_positional_encoding():
    """测试2D位置编码"""
    print("\n" + "="*50)
    print("Testing PositionalEncoding2D...")
    
    B, K, d = 2, 100, 256
    coords = torch.rand(B, K, 2)  # [B, K, 2] in [0, 1]
    
    pos_encoder = PositionalEncoding2D(hidden_dim=d)
    pos_embed = pos_encoder(coords)
    
    assert pos_embed.shape == (B, K, d), f"Expected {(B, K, d)}, got {pos_embed.shape}"
    print(f"  ✓ Output shape: {pos_embed.shape}")
    
    # 验证不同位置有不同编码
    diff = (pos_embed[0, 0] - pos_embed[0, 1]).abs().sum()
    assert diff > 0, "Different positions should have different encodings"
    print(f"  ✓ Different positions have different encodings (diff={diff.item():.4f})")


def test_heatmap_query_generator():
    """测试热图Query生成器"""
    print("\n" + "="*50)
    print("Testing HeatmapQueryGenerator...")
    
    B, C, H, W = 2, 256, 20, 20
    num_levels = 4
    K = 100
    
    # 模拟多尺度特征
    srcs = [torch.randn(B, C, H // (2**i), W // (2**i)) for i in range(num_levels)]
    spatial_shapes = torch.tensor([[H // (2**i), W // (2**i)] for i in range(num_levels)])
    
    # 模拟热图 (原图尺度)
    heatmap = torch.rand(B, H * 2, W * 2)
    
    for pool_mode in ['mean', 'heatmap_weighted']:
        print(f"\n  Testing pool_mode={pool_mode}...")
        
        gen = HeatmapQueryGenerator(
            hidden_dim=C,
            num_queries=K,
            num_feature_levels=num_levels,
            pool_mode=pool_mode,
            pool_window=3
        )
        
        output = gen(srcs, spatial_shapes, heatmap)
        
        # 验证输出shape
        assert output['query_embed'].shape == (B, K, C * 2), \
            f"query_embed: expected {(B, K, C*2)}, got {output['query_embed'].shape}"
        assert output['query_content'].shape == (B, K, C), \
            f"query_content: expected {(B, K, C)}, got {output['query_content'].shape}"
        assert output['query_pos'].shape == (B, K, C), \
            f"query_pos: expected {(B, K, C)}, got {output['query_pos'].shape}"
        assert output['reference_points'].shape == (B, K, 2), \
            f"reference_points: expected {(B, K, 2)}, got {output['reference_points'].shape}"
        assert output['heatmap_scores'].shape == (B, K), \
            f"heatmap_scores: expected {(B, K)}, got {output['heatmap_scores'].shape}"
        
        # 验证reference_points在[0,1]范围内
        assert output['reference_points'].min() >= 0 and output['reference_points'].max() <= 1, \
            "reference_points should be in [0, 1]"
        
        print(f"    ✓ All output shapes correct")
        print(f"    ✓ reference_points in [0, 1]")


def test_query_mixer():
    """测试Query混合器"""
    print("\n" + "="*50)
    print("Testing QueryMixer...")
    
    B, K, d = 2, 100, 256
    num_learnable = 300
    
    # 模拟pseudo queries
    pseudo_queries = {
        'query_embed': torch.randn(B, K, d * 2),
        'query_content': torch.randn(B, K, d),
        'query_pos': torch.randn(B, K, d),
        'reference_points': torch.rand(B, K, 2),
    }
    
    for mix_mode in ['replace', 'concat', 'ratio', 'attention']:
        print(f"\n  Testing mix_mode={mix_mode}...")
        
        mixer = QueryMixer(
            hidden_dim=d,
            num_learnable_queries=num_learnable,
            num_pseudo_queries=K,
            mix_mode=mix_mode
        )
        
        mixed_embed, mixed_ref = mixer(pseudo_queries, batch_size=B)
        
        print(f"    mixed_embed shape: {mixed_embed.shape}")
        if mixed_ref is not None:
            print(f"    mixed_ref shape: {mixed_ref.shape}")
        
        # 验证维度正确
        assert mixed_embed.shape[-1] == d * 2, "Last dim should be 2*hidden_dim"
        print(f"    ✓ Dimension correct")


def test_deformable_detr_compatibility():
    """测试与Deformable DETR的兼容性"""
    print("\n" + "="*50)
    print("Testing Deformable DETR Compatibility...")
    
    B, K, d = 2, 100, 256
    num_queries = 300
    
    # 模拟pseudo query输出
    pseudo_queries = {
        'query_embed': torch.randn(B, K, d * 2),
        'reference_points': torch.rand(B, K, 2),
    }
    
    # 混合
    mixer = QueryMixer(
        hidden_dim=d,
        num_learnable_queries=num_queries,
        num_pseudo_queries=K,
        mix_mode='concat'
    )
    
    mixed_embed, mixed_ref = mixer(pseudo_queries, batch_size=B)
    
    # 模拟Deformable DETR transformer的query处理
    # 原始代码: query_embed, tgt = torch.split(query_embed, c, dim=1)
    # 这里query_embed是[Q, 2*d], 需要扩展到[B, Q, 2*d]
    
    c = d
    query_embed, tgt = torch.split(mixed_embed, c, dim=-1)  # 注意这里用dim=-1
    
    print(f"  mixed_embed: {mixed_embed.shape}")
    print(f"  After split:")
    print(f"    tgt (query content): {tgt.shape}")
    print(f"    query_embed (query pos): {query_embed.shape}")
    
    assert tgt.shape == (B, num_queries, d), f"tgt: expected {(B, num_queries, d)}, got {tgt.shape}"
    assert query_embed.shape == (B, num_queries, d), f"query_embed: expected {(B, num_queries, d)}, got {query_embed.shape}"
    
    # 模拟reference_points计算
    # 原始代码: reference_points = self.reference_points(query_embed).sigmoid()
    ref_linear = nn.Linear(d, 2)
    reference_points = ref_linear(query_embed).sigmoid()
    
    print(f"  reference_points: {reference_points.shape}")
    assert reference_points.shape == (B, num_queries, 2)
    
    print(f"  ✓ Fully compatible with Deformable DETR!")


def test_losses():
    """测试Loss函数"""
    print("\n" + "="*50)
    print("Testing Loss Functions...")
    
    B, K, d = 2, 100, 256
    
    # Alignment Loss
    print("\n  Testing QueryAlignmentLoss...")
    decoder_queries = torch.randn(B, K, d)
    pseudo_content = torch.randn(B, K, d)
    
    for loss_type in ['l2', 'cosine', 'infonce']:
        loss_fn = QueryAlignmentLoss(loss_type=loss_type, loss_weight=1.0)
        loss = loss_fn(decoder_queries, pseudo_content)
        print(f"    {loss_type}: {loss.item():.4f}")
        assert not torch.isnan(loss), f"{loss_type} loss is NaN"
    
    print("  ✓ All alignment losses computed successfully")
    
    # Prior Loss
    print("\n  Testing AttentionPriorLoss...")
    pred_boxes = torch.rand(B, K, 4)
    heatmap = torch.rand(B, 32, 32)
    
    prior_loss = AttentionPriorLoss(loss_type='center', loss_weight=1.0)
    loss = prior_loss(pred_boxes, heatmap)
    print(f"    center prior: {loss.item():.4f}")
    assert not torch.isnan(loss), "Prior loss is NaN"
    
    print("  ✓ Prior loss computed successfully")


def test_end_to_end():
    """端到端测试"""
    print("\n" + "="*50)
    print("Testing End-to-End Pipeline...")
    
    B, C, d = 2, 256, 256
    H, W = 40, 40  # 原图尺度热图
    num_levels = 4
    K = 100
    num_queries = 300
    
    # 1. 准备输入
    print("\n  1. Preparing inputs...")
    srcs = [torch.randn(B, C, H // (2**(i+1)), W // (2**(i+1))) for i in range(num_levels)]
    spatial_shapes = torch.tensor([[src.shape[2], src.shape[3]] for src in srcs])
    heatmap = torch.rand(B, H, W)
    print(f"     srcs: {[s.shape for s in srcs]}")
    print(f"     heatmap: {heatmap.shape}")
    
    # 2. 生成pseudo queries
    print("\n  2. Generating pseudo queries...")
    query_gen = HeatmapQueryGenerator(
        hidden_dim=d,
        num_queries=K,
        num_feature_levels=num_levels,
        pool_mode='heatmap_weighted'
    )
    pseudo_output = query_gen(srcs, spatial_shapes, heatmap)
    print(f"     query_embed: {pseudo_output['query_embed'].shape}")
    print(f"     reference_points: {pseudo_output['reference_points'].shape}")
    
    # 3. 混合queries
    print("\n  3. Mixing queries...")
    mixer = QueryMixer(
        hidden_dim=d,
        num_learnable_queries=num_queries,
        num_pseudo_queries=K,
        mix_mode='concat'
    )
    mixed_embed, mixed_ref = mixer(pseudo_output, batch_size=B)
    print(f"     mixed_embed: {mixed_embed.shape}")
    
    # 4. 模拟decoder输出
    print("\n  4. Simulating decoder output...")
    decoder_output = torch.randn(B, num_queries, d)
    pred_boxes = torch.rand(B, num_queries, 4)
    
    # 5. 计算loss
    print("\n  5. Computing losses...")
    criterion = PseudoQueryCriterion(
        use_alignment_loss=True,
        use_prior_loss=True,
        alignment_config={'loss_type': 'l2', 'loss_weight': 1.0},
        prior_config={'loss_type': 'center', 'loss_weight': 0.5},
    )
    
    outputs = {
        'decoder_queries': decoder_output,
        'pred_boxes': pred_boxes,
    }
    
    losses = criterion(outputs, pseudo_output, heatmap)
    print(f"     losses: {losses}")
    
    print("\n  ✓ End-to-end pipeline works!")


def main():
    print("="*60)
    print("Pseudo Query Module Tests")
    print("="*60)
    
    test_positional_encoding()
    test_heatmap_query_generator()
    test_query_mixer()
    test_deformable_detr_compatibility()
    test_losses()
    test_end_to_end()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)


if __name__ == '__main__':
    main()
