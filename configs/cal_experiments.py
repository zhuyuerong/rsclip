# -*- coding: utf-8 -*-
"""
CAL实验配置定义
所有实验配置都在这里，方便管理和切换
"""
try:
    from src.competitors.clip_methods.surgeryclip.cal_config import CALConfig
except ImportError:
    print("⚠️  CAL模块未找到，请确保cal_config.py存在")
    CALConfig = None


# ============ Q1: 负样本策略实验 ============

# Q1-Exp1: 固定负样本
cal_q1_exp1_fixed = CALConfig(
    enable_cal=True,
    negative_mode='fixed',
    fixed_negatives=["background", "irrelevant objects"],
    alpha=1.0,
    cal_space='similarity',
    experiment_name='q1_exp1_fixed',
    verbose=True
) if CALConfig else None

# Q1-Exp2: 动态负样本
cal_q1_exp2_dynamic = CALConfig(
    enable_cal=True,
    negative_mode='dynamic',
    num_dynamic_negatives=3,
    alpha=1.0,
    cal_space='similarity',
    experiment_name='q1_exp2_dynamic',
    verbose=True
) if CALConfig else None

# Q1-Exp3: 随机负样本
cal_q1_exp3_random = CALConfig(
    enable_cal=True,
    negative_mode='random',
    num_random_negatives=3,
    alpha=1.0,
    cal_space='similarity',
    experiment_name='q1_exp3_random',
    verbose=True
) if CALConfig else None

# Q1-Exp4: 组合负样本（固定+动态）
cal_q1_exp4_combined = CALConfig(
    enable_cal=True,
    negative_mode='combined',
    fixed_negatives=["background"],
    num_dynamic_negatives=2,
    alpha=1.0,
    cal_space='similarity',
    experiment_name='q1_exp4_combined',
    verbose=True
) if CALConfig else None

# ============ Q2: 加权减法实验 ============

# Q2-Exp1: alpha=0.5
cal_q2_exp1_alpha05 = CALConfig(
    enable_cal=True,
    negative_mode='fixed',
    alpha=0.5,
    cal_space='similarity',
    experiment_name='q2_exp1_alpha05',
    verbose=True
) if CALConfig else None

# Q2-Exp2: alpha=1.0 (baseline)
cal_q2_exp2_alpha10 = CALConfig(
    enable_cal=True,
    negative_mode='fixed',
    alpha=1.0,
    cal_space='similarity',
    experiment_name='q2_exp2_alpha10',
    verbose=True
) if CALConfig else None

# Q2-Exp3: alpha=1.5
cal_q2_exp3_alpha15 = CALConfig(
    enable_cal=True,
    negative_mode='fixed',
    alpha=1.5,
    cal_space='similarity',
    experiment_name='q2_exp3_alpha15',
    verbose=True
) if CALConfig else None

# Q2-Exp4: alpha=2.0
cal_q2_exp4_alpha20 = CALConfig(
    enable_cal=True,
    negative_mode='fixed',
    alpha=2.0,
    cal_space='similarity',
    experiment_name='q2_exp4_alpha20',
    verbose=True
) if CALConfig else None

# ============ Q3: 操作位置实验 ============

# Q3-Exp1: 特征空间
cal_q3_exp1_feature = CALConfig(
    enable_cal=True,
    negative_mode='fixed',
    alpha=1.0,
    cal_space='feature',
    experiment_name='q3_exp1_feature',
    verbose=True
) if CALConfig else None

# Q3-Exp2: 相似度空间
cal_q3_exp2_similarity = CALConfig(
    enable_cal=True,
    negative_mode='fixed',
    alpha=1.0,
    cal_space='similarity',
    experiment_name='q3_exp2_similarity',
    verbose=True
) if CALConfig else None

# Q3-Exp3: 双重操作
cal_q3_exp3_both = CALConfig(
    enable_cal=True,
    negative_mode='fixed',
    alpha=1.0,
    cal_space='both',
    experiment_name='q3_exp3_both',
    verbose=True
) if CALConfig else None

# ============ 组合实验 ============

# 最佳组合: Q1-combined + Q2-alpha1.0 + Q3-similarity
cal_best_combination = CALConfig(
    enable_cal=True,
    negative_mode='combined',
    fixed_negatives=["background"],
    num_dynamic_negatives=2,
    alpha=1.0,
    cal_space='similarity',
    experiment_name='best_combination',
    verbose=True
) if CALConfig else None

# 所有实验配置字典
ALL_CAL_CONFIGS = {}
if CALConfig:
    ALL_CAL_CONFIGS = {
        # Q1实验
        'q1_exp1_fixed': cal_q1_exp1_fixed,
        'q1_exp2_dynamic': cal_q1_exp2_dynamic,
        'q1_exp3_random': cal_q1_exp3_random,
        'q1_exp4_combined': cal_q1_exp4_combined,
        
        # Q2实验
        'q2_exp1_alpha05': cal_q2_exp1_alpha05,
        'q2_exp2_alpha10': cal_q2_exp2_alpha10,
        'q2_exp3_alpha15': cal_q2_exp3_alpha15,
        'q2_exp4_alpha20': cal_q2_exp4_alpha20,
        
        # Q3实验
        'q3_exp1_feature': cal_q3_exp1_feature,
        'q3_exp2_similarity': cal_q3_exp2_similarity,
        'q3_exp3_both': cal_q3_exp3_both,
        
        # 组合实验
        'best_combination': cal_best_combination,
    }






