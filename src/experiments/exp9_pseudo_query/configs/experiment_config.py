"""
实验配置文件

定义了所有Phase A/B/C/D的实验配置
按照论文写作顺序组织
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum


class QueryGenType(Enum):
    """Q-Gen来源类型 (C1消融)"""
    TEACHER = "teacher"           # 来自teacher detector
    HEATMAP = "heatmap"           # 来自vv-attention热图
    FUSION = "fusion"             # 融合两者


class PoolMode(Enum):
    """特征聚合方式 (C3消融)"""
    MEAN = "mean"                 # 最近邻/均值
    HEATMAP_WEIGHTED = "heatmap_weighted"  # 热图加权
    ATTN_POOL = "attn_pool"       # Attention池化


class QueryUseMode(Enum):
    """Query使用方式 (C4消融)"""
    INIT_REPLACE = "init_replace"     # Use-1: 完全替换
    INIT_CONCAT = "init_concat"       # Use-1: 拼接混合
    PLUS_ALIGN = "plus_align"         # Use-1 + Use-2: 加对齐loss
    PLUS_PRIOR = "plus_prior"         # Use-1 + Use-2 + Use-3: 加prior loss


class MixMode(Enum):
    """Query混合模式"""
    REPLACE = "replace"       # 100%替换
    CONCAT = "concat"         # 拼接
    RATIO = "ratio"           # 按比例软混合
    ATTENTION = "attention"   # Attention融合


@dataclass
class BaseConfig:
    """基础配置"""
    # 数据
    dataset: str = "DIOR"
    data_root: str = "./datasets/DIOR"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    
    # 模型基础
    backbone: str = "resnet50"
    hidden_dim: int = 256
    num_feature_levels: int = 4
    
    # 训练
    batch_size: int = 2
    num_workers: int = 4
    epochs: int = 50
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    
    # 设备
    device: str = "cuda"
    seed: int = 42


@dataclass
class DeformableDETRConfig(BaseConfig):
    """Deformable DETR配置"""
    # Transformer
    enc_layers: int = 6
    dec_layers: int = 6
    nheads: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    
    # Query
    num_queries: int = 300
    
    # Loss
    aux_loss: bool = True
    with_box_refine: bool = True
    two_stage: bool = False
    
    # Loss weights
    cls_loss_coef: float = 2.0
    bbox_loss_coef: float = 5.0
    giou_loss_coef: float = 2.0


@dataclass
class PseudoQueryConfig:
    """Pseudo Query配置"""
    # Q-Gen
    gen_type: QueryGenType = QueryGenType.HEATMAP
    num_pseudo_queries: int = 100
    pool_mode: PoolMode = PoolMode.HEATMAP_WEIGHTED
    pool_window: int = 3
    min_score_thresh: float = 0.1
    
    # Q-Use
    use_mode: QueryUseMode = QueryUseMode.INIT_CONCAT
    mix_mode: MixMode = MixMode.CONCAT
    
    # Losses
    use_alignment_loss: bool = False
    use_prior_loss: bool = False
    alignment_loss_type: str = "l2"  # 'l2', 'cosine', 'infonce'
    alignment_weight: float = 1.0
    prior_loss_type: str = "center"  # 'center', 'attn_map'
    prior_weight: float = 1.0


# ==================== Phase A: MVP可行性实验 ====================

@dataclass
class PhaseA0_Baseline(DeformableDETRConfig):
    """A0: 标准Deformable DETR baseline (无pseudo query)"""
    exp_name: str = "A0_baseline_no_pseudo"
    use_pseudo_query: bool = False


@dataclass  
class PhaseA2_TeacherProposal(DeformableDETRConfig):
    """A2: Teacher proposals → pseudo query"""
    exp_name: str = "A2_teacher_proposal"
    use_pseudo_query: bool = True
    pseudo_config: PseudoQueryConfig = field(default_factory=lambda: PseudoQueryConfig(
        gen_type=QueryGenType.TEACHER,
        num_pseudo_queries=100,
        use_mode=QueryUseMode.INIT_CONCAT,
    ))


@dataclass
class PhaseA3_HeatmapPseudo(DeformableDETRConfig):
    """A3: vv-attention → pseudo query (核心方法)"""
    exp_name: str = "A3_heatmap_pseudo"
    use_pseudo_query: bool = True
    pseudo_config: PseudoQueryConfig = field(default_factory=lambda: PseudoQueryConfig(
        gen_type=QueryGenType.HEATMAP,
        num_pseudo_queries=100,
        pool_mode=PoolMode.HEATMAP_WEIGHTED,
        use_mode=QueryUseMode.INIT_CONCAT,
    ))


# ==================== Phase B: 证伪实验 ====================

@dataclass
class PhaseB1_RandomQuery(DeformableDETRConfig):
    """B1: 随机query (证明不是"多加query就行")"""
    exp_name: str = "B1_random_query"
    use_pseudo_query: bool = True
    use_random_query: bool = True  # 特殊标记
    pseudo_config: PseudoQueryConfig = field(default_factory=lambda: PseudoQueryConfig(
        num_pseudo_queries=100,
    ))


@dataclass
class PhaseB2_ShuffledHeatmap(DeformableDETRConfig):
    """B2: 打乱热图-图像对应 (证明是图像相关的空间证据)"""
    exp_name: str = "B2_shuffled_heatmap"
    use_pseudo_query: bool = True
    shuffle_heatmap: bool = True  # 特殊标记
    pseudo_config: PseudoQueryConfig = field(default_factory=lambda: PseudoQueryConfig(
        gen_type=QueryGenType.HEATMAP,
        num_pseudo_queries=100,
    ))


# ==================== Phase C: 核心消融实验 ====================

# C1: Q-Gen来源消融
C1_QGEN_CONFIGS = {
    "teacher": PseudoQueryConfig(gen_type=QueryGenType.TEACHER),
    "heatmap": PseudoQueryConfig(gen_type=QueryGenType.HEATMAP),
    "fusion": PseudoQueryConfig(gen_type=QueryGenType.FUSION),
}

# C2: K (query数量) 消融
C2_K_VALUES = [50, 100, 200, 300, 500]

# C3: Q-Pool (聚合方式) 消融
C3_POOL_CONFIGS = {
    "mean": PseudoQueryConfig(pool_mode=PoolMode.MEAN),
    "heatmap_weighted": PseudoQueryConfig(pool_mode=PoolMode.HEATMAP_WEIGHTED),
    "attn_pool": PseudoQueryConfig(pool_mode=PoolMode.ATTN_POOL),
}

# C4: Q-Use (使用方式) 消融
C4_USE_CONFIGS = {
    "init_only": PseudoQueryConfig(
        use_mode=QueryUseMode.INIT_CONCAT,
        use_alignment_loss=False,
        use_prior_loss=False,
    ),
    "init_plus_align": PseudoQueryConfig(
        use_mode=QueryUseMode.PLUS_ALIGN,
        use_alignment_loss=True,
        use_prior_loss=False,
    ),
    "init_plus_align_prior": PseudoQueryConfig(
        use_mode=QueryUseMode.PLUS_PRIOR,
        use_alignment_loss=True,
        use_prior_loss=True,
    ),
}


# ==================== 辅助函数 ====================

def get_experiment_config(phase: str, variant: str = "default") -> BaseConfig:
    """
    获取实验配置
    
    Args:
        phase: 'A0', 'A2', 'A3', 'B1', 'B2', 'C1', 'C2', 'C3', 'C4'
        variant: 变体名称
    """
    configs = {
        'A0': PhaseA0_Baseline,
        'A2': PhaseA2_TeacherProposal,
        'A3': PhaseA3_HeatmapPseudo,
        'B1': PhaseB1_RandomQuery,
        'B2': PhaseB2_ShuffledHeatmap,
    }
    
    if phase in configs:
        return configs[phase]()
    
    # 消融实验需要基于A3修改
    base = PhaseA3_HeatmapPseudo()
    
    if phase == 'C1' and variant in C1_QGEN_CONFIGS:
        base.pseudo_config = C1_QGEN_CONFIGS[variant]
        base.exp_name = f"C1_qgen_{variant}"
    
    elif phase == 'C2':
        k = int(variant) if variant.isdigit() else 100
        base.pseudo_config.num_pseudo_queries = k
        base.exp_name = f"C2_K_{k}"
    
    elif phase == 'C3' and variant in C3_POOL_CONFIGS:
        base.pseudo_config = C3_POOL_CONFIGS[variant]
        base.exp_name = f"C3_pool_{variant}"
    
    elif phase == 'C4' and variant in C4_USE_CONFIGS:
        base.pseudo_config = C4_USE_CONFIGS[variant]
        base.exp_name = f"C4_use_{variant}"
    
    return base


def print_experiment_summary():
    """打印所有实验配置摘要"""
    print("=" * 60)
    print("Pseudo Query Experiment Configurations")
    print("=" * 60)
    
    print("\n📌 Phase A: MVP可行性实验")
    print("  A0: Baseline (无pseudo) - 对照组")
    print("  A2: Teacher proposals → pseudo query")
    print("  A3: vv-attention → pseudo query ⭐核心方法")
    
    print("\n📌 Phase B: 证伪实验")
    print("  B1: 随机query - 证明不是'多加query就行'")
    print("  B2: 打乱热图 - 证明是图像相关的空间证据")
    print("  B3: 阈值box→query - 复现'box级别不稳定'")
    print("  B4: CLIP crop teacher - 引用已有负结果")
    
    print("\n📌 Phase C: 核心消融实验")
    print("  C1: Q-Gen来源 - teacher vs heatmap vs fusion")
    print("  C2: K数量 - 50/100/200/300/500")
    print("  C3: Q-Pool方式 - mean vs weighted vs attn")
    print("  C4: Q-Use方式 - init → +align → +prior")
    
    print("\n📌 Phase D: Open-vocab扩展 (最后做)")
    print("  D1: 接入open-vocab分类头")
    print("  D2: Seen/Unseen拆分评估")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    print_experiment_summary()
    
    print("\n\n示例配置:")
    config = get_experiment_config('A3')
    print(f"  实验名: {config.exp_name}")
    print(f"  num_pseudo_queries: {config.pseudo_config.num_pseudo_queries}")
    print(f"  pool_mode: {config.pseudo_config.pool_mode}")
