"""
Pseudo Query Models

核心模块:
1. heatmap_query_gen.py - Q-Gen: heatmap → pseudo queries
2. query_injection.py - Q-Use: query混合与loss
3. deformable_detr_pseudo.py - 改进的Deformable DETR
"""

from .heatmap_query_gen import (
    HeatmapQueryGenerator,
    TeacherQueryGenerator,
    FusionQueryGenerator,
    build_query_generator,
)

from .query_injection import (
    QueryMixer,
    QueryAlignmentLoss,
    AttentionPriorLoss,
    PseudoQueryCriterion,
)

from .deformable_detr_pseudo import (
    DeformableDETRPseudo,
    DeformableDETRPseudoCriterion,
    build_deformable_detr_pseudo,
)

__all__ = [
    # Q-Gen
    'HeatmapQueryGenerator',
    'TeacherQueryGenerator', 
    'FusionQueryGenerator',
    'build_query_generator',
    # Q-Use
    'QueryMixer',
    'QueryAlignmentLoss',
    'AttentionPriorLoss',
    'PseudoQueryCriterion',
    # Model
    'DeformableDETRPseudo',
    'DeformableDETRPseudoCriterion',
    'build_deformable_detr_pseudo',
]
