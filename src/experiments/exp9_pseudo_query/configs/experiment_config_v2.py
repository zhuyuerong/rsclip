"""
实验配置文件 v2 - 修复消融不干净的问题

关键改进:
1. 训练预算显式化 (epochs, eval_epochs, warmup)
2. 指标目标/预期现象写进config
3. Query注入方式拆成正交开关
4. 使用inherit()保证消融干净
5. 固定total_queries策略
"""

from dataclasses import dataclass, field, replace
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import copy


# ==================== 枚举类型 (正交设计) ====================

class QueryGenType(Enum):
    """Q-Gen来源类型 (C1消融)"""
    TEACHER = "teacher"
    HEATMAP = "heatmap"
    FUSION = "fusion"


class PoolMode(Enum):
    """特征聚合方式 (C3消融)"""
    MEAN = "mean"
    HEATMAP_WEIGHTED = "heatmap_weighted"
    ATTN_POOL = "attn_pool"


class InitMode(Enum):
    """Query初始化模式 - 与loss正交"""
    REPLACE = "replace"      # 100%替换learnable
    CONCAT = "concat"        # 拼接 [pseudo, learnable]
    RATIO = "ratio"          # 按比例软混合
    ATTENTION = "attention"  # Attention融合


class AlignLossType(Enum):
    """对齐Loss类型"""
    NONE = "none"
    L2 = "l2"
    COSINE = "cosine"
    INFONCE = "infonce"


class PriorLossType(Enum):
    """Prior Loss类型"""
    NONE = "none"
    CENTER = "center"
    ATTN_MAP = "attn_map"


class DebugMode(Enum):
    """调试模式 (用于B1/B2证伪实验)"""
    NONE = "none"
    RANDOM_QUERY = "random_query"      # B1: 随机query
    SHUFFLE_HEATMAP = "shuffle_heatmap"  # B2: 打乱热图


# ==================== 预期结果模板 ====================

@dataclass
class ExpectedBehavior:
    """实验预期现象 - 用于自动验证和报告"""
    
    # 正常预期现象描述
    normal_phenomena: List[str] = field(default_factory=list)
    
    # 目标指标 (相对/绝对)
    targets: Dict[str, Any] = field(default_factory=dict)
    # 例如: {"recall_0.5_at_epoch_10": (">", 0.3), "ap_small_vs_baseline": (">", 0)}
    
    # 如果没出现预期现象，可能的原因
    failure_modes: Dict[str, str] = field(default_factory=dict)
    # 例如: {"no_improvement": "检查query注入是否生效"}
    
    # 用于自动判断的阈值
    sanity_checks: Dict[str, Tuple[str, float]] = field(default_factory=dict)
    # 例如: {"loss_epoch_1": ("<", 10.0), "recall_epoch_5": (">", 0.1)}


# ==================== 训练预算配置 ====================

@dataclass
class TrainingBudget:
    """训练预算 - 确保公平对比"""
    max_epochs: int = 50
    warmup_epochs: int = 1
    
    # 固定评估点 (用于收敛速度对比)
    eval_epochs: Tuple[int, ...] = (1, 5, 10, 20, 30, 50)
    
    # 保存checkpoint的epoch
    save_epochs: Tuple[int, ...] = (10, 20, 30, 50)
    
    # early stopping (可选)
    patience: Optional[int] = None
    min_delta: float = 0.001


# ==================== Pseudo Query配置 (正交设计) ====================

@dataclass
class PseudoQueryConfig:
    """
    Pseudo Query配置 - 正交设计
    
    三个维度:
    1. Q-Gen: gen_type, num_pseudo, pool_mode, pool_window
    2. Q-Init: init_mode
    3. Q-Loss: align_loss_type, prior_loss_type
    """
    
    # === Q-Gen 配置 ===
    gen_type: QueryGenType = QueryGenType.HEATMAP
    num_pseudo_queries: int = 100
    pool_mode: PoolMode = PoolMode.HEATMAP_WEIGHTED
    pool_window: int = 3
    min_score_thresh: float = 0.1
    nms_radius: int = 2  # top-k选点时的NMS半径
    
    # === Q-Init 配置 (与loss正交) ===
    init_mode: InitMode = InitMode.CONCAT
    
    # === Q-Loss 配置 (与init正交) ===
    align_loss_type: AlignLossType = AlignLossType.NONE
    align_loss_weight: float = 1.0
    
    prior_loss_type: PriorLossType = PriorLossType.NONE
    prior_loss_weight: float = 0.5
    
    # === 总Query数策略 ===
    fixed_total_queries: bool = True  # 是否固定总数
    total_queries: int = 300  # 总query数 (fixed_total时learnable = total - pseudo)
    
    # === 调试模式 ===
    debug_mode: DebugMode = DebugMode.NONE


def inherit(base: PseudoQueryConfig, **kwargs) -> PseudoQueryConfig:
    """
    从base配置继承，只修改指定字段
    保证消融实验的干净性
    """
    return replace(base, **kwargs)


# ==================== 基础配置 ====================

@dataclass
class BaseConfig:
    """基础配置"""
    # 实验标识
    exp_name: str = "unnamed"
    exp_version: str = "v1"
    
    # 数据
    dataset: str = "DIOR"
    data_root: str = "/path/to/DIOR"
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
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    clip_max_norm: float = 0.1
    
    # 训练预算
    budget: TrainingBudget = field(default_factory=TrainingBudget)
    
    # 设备
    device: str = "cuda"
    seed: int = 42
    deterministic: bool = True


@dataclass
class DeformableDETRConfig(BaseConfig):
    """Deformable DETR配置"""
    # Transformer
    enc_layers: int = 6
    dec_layers: int = 6
    nheads: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    
    # Query (总数)
    num_queries: int = 300
    
    # Loss
    aux_loss: bool = True
    with_box_refine: bool = True
    two_stage: bool = False
    
    # Loss weights
    cls_loss_coef: float = 2.0
    bbox_loss_coef: float = 5.0
    giou_loss_coef: float = 2.0
    
    # Pseudo Query
    use_pseudo_query: bool = False
    pseudo_config: Optional[PseudoQueryConfig] = None
    
    # 预期结果
    expected: ExpectedBehavior = field(default_factory=ExpectedBehavior)


# ==================== A3 基准配置 (所有消融的base) ====================

# 定义A3的pseudo config作为消融基准
A3_PSEUDO_CONFIG = PseudoQueryConfig(
    # Q-Gen
    gen_type=QueryGenType.HEATMAP,
    num_pseudo_queries=100,
    pool_mode=PoolMode.HEATMAP_WEIGHTED,
    pool_window=3,
    min_score_thresh=0.1,
    nms_radius=2,
    # Q-Init
    init_mode=InitMode.CONCAT,
    # Q-Loss (默认不加)
    align_loss_type=AlignLossType.NONE,
    prior_loss_type=PriorLossType.NONE,
    # 总数
    fixed_total_queries=True,
    total_queries=300,
    # Debug
    debug_mode=DebugMode.NONE,
)


# ==================== Phase A: MVP实验配置 ====================

def create_A0_baseline() -> DeformableDETRConfig:
    """A0: 标准Deformable DETR baseline (无pseudo query)"""
    return DeformableDETRConfig(
        exp_name="A0_baseline_no_pseudo",
        use_pseudo_query=False,
        expected=ExpectedBehavior(
            normal_phenomena=[
                "训练loss平稳下降",
                "decoder输出的box分布逐渐从'全图乱飘'到'目标附近聚集'",
                "Recall@K在前5-10个epoch有明显上升",
            ],
            targets={
                "role": "对照组",
                "note": "A2/A3至少在收敛速度或small objects上超过它",
            },
            failure_modes={
                "loss_not_decreasing": "检查box normalize/gt format、matcher/target组装",
                "boxes_at_edge": "检查box encoding/decoding",
            },
            sanity_checks={
                "loss_epoch_1": ("<", 20.0),
                "recall_300_epoch_5": (">", 0.05),
            }
        )
    )


def create_A2_teacher() -> DeformableDETRConfig:
    """A2: Teacher proposals → pseudo query (管线自检)"""
    return DeformableDETRConfig(
        exp_name="A2_teacher_proposal",
        use_pseudo_query=True,
        pseudo_config=inherit(A3_PSEUDO_CONFIG,
            gen_type=QueryGenType.TEACHER,
        ),
        expected=ExpectedBehavior(
            normal_phenomena=[
                "前期收敛更快 (最重要)",
                "同样epoch下Recall@K更早抬头",
                "matched queries占比前期更高",
            ],
            targets={
                "recall_0.5_at_epoch_10": (">", "A0"),  # 相对目标
                "ap_small_at_epoch_10": (">=", "A0"),
                "note": "后期可能趋同,这不算失败",
            },
            failure_modes={
                "no_speedup": "检查pseudo content/pos/reference是否对齐",
                "worse_than_A0": "检查teacher proposals坐标映射(原图vs feature)",
                "diversity_issue": "检查混合策略是否把learnable全替掉",
            },
            sanity_checks={
                "recall_300_epoch_5": (">", 0.1),
            }
        )
    )


def create_A3_heatmap() -> DeformableDETRConfig:
    """A3: vv-attention → pseudo query (核心方法)"""
    return DeformableDETRConfig(
        exp_name="A3_heatmap_pseudo",
        use_pseudo_query=True,
        pseudo_config=A3_PSEUDO_CONFIG,
        expected=ExpectedBehavior(
            normal_phenomena=[
                "比A0更快收敛",
                "密集小目标(ship/vehicle)Recall上升更明显",
                "可能带来FP(背景高响应)→mAP未必立刻涨,这是正常的",
            ],
            targets={
                "vs_A0": "至少在AP_small或Recall@0.5之一有稳定提升",
                "vs_A2": "允许略弱,但不能全指标明显劣于A2",
            },
            failure_modes={
                "no_improvement": "heatmap坐标系没对齐(patch vs 原图)",
                "same_as_random": "top-k全挤在一个连通域(没有NMS)",
                "unstable": "pool_window太小(噪声)或太大(吃背景)",
            },
            sanity_checks={
                "recall_300_epoch_5": (">", 0.08),
            }
        )
    )


# ==================== Phase B: 证伪实验配置 ====================

def create_B1_random() -> DeformableDETRConfig:
    """B1: 随机query (必须显著差)"""
    return DeformableDETRConfig(
        exp_name="B1_random_query",
        use_pseudo_query=True,
        pseudo_config=inherit(A3_PSEUDO_CONFIG,
            debug_mode=DebugMode.RANDOM_QUERY,
        ),
        expected=ExpectedBehavior(
            normal_phenomena=[
                "明显劣于A2/A3 (尤其early epoch)",
                "甚至可能比A0还差",
            ],
            targets={
                "recall_0.5": ("<", "A3"),
                "ap_small": ("<", "A3"),
                "note": "如果B1≈A3,说明A3增益只是'多了queries'而非空间证据",
            },
            failure_modes={
                "same_as_A3": "pseudo注入可能没生效(被mask/没喂进decoder)",
            },
        )
    )


def create_B2_shuffled() -> DeformableDETRConfig:
    """B2: 打乱热图 (必须明显掉)"""
    return DeformableDETRConfig(
        exp_name="B2_shuffled_heatmap",
        use_pseudo_query=True,
        pseudo_config=inherit(A3_PSEUDO_CONFIG,
            debug_mode=DebugMode.SHUFFLE_HEATMAP,
        ),
        expected=ExpectedBehavior(
            normal_phenomena=[
                "相对A3有显著下降 (early epoch更明显)",
            ],
            targets={
                "ap_small": ("<", "A3"),
                "recall_0.5": ("<", "A3"),
                "note": "如果不下降,A3的因果链不成立",
            },
            failure_modes={
                "no_drop": "heatmap信息被弱化(mixing比例太小)",
            },
        )
    )


# ==================== Phase C: 消融实验配置 ====================

def create_C1_ablations() -> Dict[str, DeformableDETRConfig]:
    """C1: Q-Gen来源消融"""
    return {
        "teacher": DeformableDETRConfig(
            exp_name="C1_qgen_teacher",
            use_pseudo_query=True,
            pseudo_config=inherit(A3_PSEUDO_CONFIG, gen_type=QueryGenType.TEACHER),
        ),
        "heatmap": DeformableDETRConfig(
            exp_name="C1_qgen_heatmap",
            use_pseudo_query=True,
            pseudo_config=inherit(A3_PSEUDO_CONFIG, gen_type=QueryGenType.HEATMAP),
        ),
        "fusion": DeformableDETRConfig(
            exp_name="C1_qgen_fusion",
            use_pseudo_query=True,
            pseudo_config=inherit(A3_PSEUDO_CONFIG, gen_type=QueryGenType.FUSION),
        ),
    }


def create_C2_ablations() -> Dict[str, DeformableDETRConfig]:
    """
    C2: K (query数量) 消融
    
    关键: 固定total_queries=300, 只变pseudo数量
    """
    K_values = [50, 100, 150, 200, 300]
    configs = {}
    
    for K in K_values:
        configs[f"K{K}"] = DeformableDETRConfig(
            exp_name=f"C2_K_{K}",
            use_pseudo_query=True,
            pseudo_config=inherit(A3_PSEUDO_CONFIG,
                num_pseudo_queries=K,
                fixed_total_queries=True,
                total_queries=300,  # 固定!
            ),
            expected=ExpectedBehavior(
                normal_phenomena=[
                    "性能随K增长先升后平/下降 (U型或饱和)",
                ],
                targets={
                    "curve_shape": "找到甜点区间 (通常50~200)",
                },
                failure_modes={
                    "monotonic_increase": "可能只是总queries变多(检查是否固定total)",
                },
            )
        )
    
    return configs


def create_C3_ablations() -> Dict[str, DeformableDETRConfig]:
    """C3: Q-Pool (聚合方式) 消融"""
    return {
        "mean": DeformableDETRConfig(
            exp_name="C3_pool_mean",
            use_pseudo_query=True,
            pseudo_config=inherit(A3_PSEUDO_CONFIG, pool_mode=PoolMode.MEAN),
            expected=ExpectedBehavior(
                normal_phenomena=["通常最弱但最稳"],
            )
        ),
        "heatmap_weighted": DeformableDETRConfig(
            exp_name="C3_pool_weighted",
            use_pseudo_query=True,
            pseudo_config=inherit(A3_PSEUDO_CONFIG, pool_mode=PoolMode.HEATMAP_WEIGHTED),
            expected=ExpectedBehavior(
                normal_phenomena=["一般是最稳且最强的默认"],
            )
        ),
        "attn_pool": DeformableDETRConfig(
            exp_name="C3_pool_attn",
            use_pseudo_query=True,
            pseudo_config=inherit(A3_PSEUDO_CONFIG, pool_mode=PoolMode.ATTN_POOL),
            expected=ExpectedBehavior(
                normal_phenomena=["可能更强但更容易不稳 (波动/对seed敏感)"],
            )
        ),
    }


def create_C4_ablations() -> Dict[str, DeformableDETRConfig]:
    """
    C4: Q-Use (使用方式) 消融
    
    正交设计: init_mode 固定, 只变 loss
    """
    return {
        # 只有init, 无额外loss
        "init_only": DeformableDETRConfig(
            exp_name="C4_use_init_only",
            use_pseudo_query=True,
            pseudo_config=inherit(A3_PSEUDO_CONFIG,
                align_loss_type=AlignLossType.NONE,
                prior_loss_type=PriorLossType.NONE,
            ),
            expected=ExpectedBehavior(
                normal_phenomena=["baseline"],
            )
        ),
        # init + alignment loss
        "plus_align_l2": DeformableDETRConfig(
            exp_name="C4_use_plus_align_l2",
            use_pseudo_query=True,
            pseudo_config=inherit(A3_PSEUDO_CONFIG,
                align_loss_type=AlignLossType.L2,
                align_loss_weight=1.0,
                prior_loss_type=PriorLossType.NONE,
            ),
            expected=ExpectedBehavior(
                normal_phenomena=["小幅稳定提升 (尤其early epoch/small objects)"],
            )
        ),
        # init + alignment + prior
        "plus_align_prior": DeformableDETRConfig(
            exp_name="C4_use_plus_align_prior",
            use_pseudo_query=True,
            pseudo_config=inherit(A3_PSEUDO_CONFIG,
                align_loss_type=AlignLossType.L2,
                align_loss_weight=1.0,
                prior_loss_type=PriorLossType.CENTER,
                prior_loss_weight=0.5,
            ),
            expected=ExpectedBehavior(
                normal_phenomena=[
                    "可能再涨",
                    "也可能引入FP或训练不稳 (这都正常,关键是解释)",
                ],
            )
        ),
    }


# ==================== 辅助函数 ====================

def get_all_experiments() -> Dict[str, DeformableDETRConfig]:
    """获取所有实验配置"""
    experiments = {
        # Phase A
        "A0": create_A0_baseline(),
        "A2": create_A2_teacher(),
        "A3": create_A3_heatmap(),
        # Phase B
        "B1": create_B1_random(),
        "B2": create_B2_shuffled(),
    }
    
    # Phase C
    experiments.update({f"C1_{k}": v for k, v in create_C1_ablations().items()})
    experiments.update({f"C2_{k}": v for k, v in create_C2_ablations().items()})
    experiments.update({f"C3_{k}": v for k, v in create_C3_ablations().items()})
    experiments.update({f"C4_{k}": v for k, v in create_C4_ablations().items()})
    
    return experiments


def print_experiment_card(config: DeformableDETRConfig):
    """打印单个实验的配置卡片"""
    print(f"\n{'='*60}")
    print(f"实验: {config.exp_name}")
    print(f"{'='*60}")
    
    print(f"\n📌 基本配置:")
    print(f"   use_pseudo_query: {config.use_pseudo_query}")
    
    if config.pseudo_config:
        pc = config.pseudo_config
        print(f"\n📌 Pseudo Query配置:")
        print(f"   Q-Gen:")
        print(f"     gen_type: {pc.gen_type.value}")
        print(f"     num_pseudo: {pc.num_pseudo_queries}")
        print(f"     pool_mode: {pc.pool_mode.value}")
        print(f"   Q-Init:")
        print(f"     init_mode: {pc.init_mode.value}")
        print(f"   Q-Loss:")
        print(f"     align: {pc.align_loss_type.value} (w={pc.align_loss_weight})")
        print(f"     prior: {pc.prior_loss_type.value} (w={pc.prior_loss_weight})")
        print(f"   Query总数:")
        print(f"     fixed_total: {pc.fixed_total_queries}")
        print(f"     total: {pc.total_queries}")
        if pc.debug_mode != DebugMode.NONE:
            print(f"   ⚠️ Debug: {pc.debug_mode.value}")
    
    if config.expected.normal_phenomena:
        print(f"\n📌 预期现象:")
        for p in config.expected.normal_phenomena:
            print(f"   • {p}")
    
    if config.expected.failure_modes:
        print(f"\n📌 失败排查:")
        for k, v in config.expected.failure_modes.items():
            print(f"   • {k}: {v}")


def print_all_experiments_summary():
    """打印所有实验摘要"""
    print("="*70)
    print("Pseudo Query Experiments Summary (v2)")
    print("="*70)
    
    experiments = get_all_experiments()
    
    # Phase A
    print("\n📌 Phase A: MVP可行性实验")
    for name in ["A0", "A2", "A3"]:
        cfg = experiments[name]
        pseudo = "无" if not cfg.use_pseudo_query else cfg.pseudo_config.gen_type.value
        print(f"   {name}: {cfg.exp_name} (pseudo={pseudo})")
    
    # Phase B
    print("\n📌 Phase B: 证伪实验")
    for name in ["B1", "B2"]:
        cfg = experiments[name]
        debug = cfg.pseudo_config.debug_mode.value
        print(f"   {name}: {cfg.exp_name} (debug={debug})")
    
    # Phase C
    print("\n📌 Phase C: 消融实验")
    c1_keys = [k for k in experiments if k.startswith("C1_")]
    c2_keys = [k for k in experiments if k.startswith("C2_")]
    c3_keys = [k for k in experiments if k.startswith("C3_")]
    c4_keys = [k for k in experiments if k.startswith("C4_")]
    
    print(f"   C1 Q-Gen: {len(c1_keys)} variants - {c1_keys}")
    print(f"   C2 K:     {len(c2_keys)} variants - {c2_keys}")
    print(f"   C3 Pool:  {len(c3_keys)} variants - {c3_keys}")
    print(f"   C4 Use:   {len(c4_keys)} variants - {c4_keys}")
    
    print(f"\n总计: {len(experiments)} 个实验配置")
    print("="*70)


if __name__ == '__main__':
    print_all_experiments_summary()
    
    # 打印A3详细配置作为示例
    print_experiment_card(create_A3_heatmap())
