#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
智能运行实验脚本
- 检查哪些实验已完成
- 只运行未完成的实验
- 支持断点续传
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import yaml

# 设置Python路径
PYTHON_ENV = "/home/ubuntu22/.cursor/worktrees/RemoteCLIP-main/nvVcv/remoteclip/bin/python"
WORK_DIR = Path("/media/ubuntu22/新加卷1/Projects/RemoteCLIP-main/src/experiments/exp4")

# 实验配置
EXPERIMENTS = {
    "1.1": {
        "name": "损失组件诊断",
        "script": "diagnose_loss_components.py",
        "args": ["--checkpoint", "checkpoints/best_simple_model.pth", "--num-batches", "50"],
        "output": "outputs/diagnosis/loss_components_report.json",
        "type": "diagnosis"
    },
    "1.2": {
        "name": "梯度流诊断",
        "script": "check_gradients.py",
        "args": ["--checkpoint", "checkpoints/best_simple_model.pth"],
        "output": "outputs/diagnosis/gradient_report.json",
        "type": "diagnosis"
    },
    "2.1a": {
        "name": "增加CAM损失权重",
        "script": "train_simple_surgery_cam.py",
        "args": ["--config", "configs/exp2.1a_increase_cam_loss.yaml"],
        "checkpoint": "checkpoints/exp2.1a/best_exp2.1a_model.pth",
        "type": "training",
        "epochs": 20
    },
    "2.1b": {
        "name": "降低峰值阈值",
        "script": "train_simple_surgery_cam.py",
        "args": ["--config", "configs/exp2.1b_lower_peak_threshold.yaml"],
        "checkpoint": "checkpoints/exp2.1b/best_exp2.1b_model.pth",
        "type": "training",
        "epochs": 20
    },
    "2.1c": {
        "name": "增加CAM生成器学习率",
        "script": "train_simple_surgery_cam.py",
        "args": ["--config", "configs/exp2.1c_increase_cam_lr.yaml"],
        "checkpoint": "checkpoints/exp2.1c/best_exp2.1c_model.pth",
        "type": "training",
        "epochs": 20
    },
    "2.1d": {
        "name": "调整损失权重比例",
        "script": "train_simple_surgery_cam.py",
        "args": ["--config", "configs/exp2.1d_adjust_loss_weights.yaml"],
        "checkpoint": "checkpoints/exp2.1d/best_exp2.1d_model.pth",
        "type": "training",
        "epochs": 20
    },
    "2.2": {
        "name": "改进正样本分配",
        "script": "train_exp2.2_improved_matching.py",
        "args": ["--config", "configs/surgery_cam_config.yaml"],
        "checkpoint": "checkpoints/exp2.2/best_exp2.2_model.pth",
        "type": "training",
        "epochs": 30
    }
}


def check_experiment_completed(exp_id, exp_config):
    """检查实验是否已完成"""
    if exp_config["type"] == "diagnosis":
        output_path = WORK_DIR / exp_config["output"]
        return output_path.exists()
    elif exp_config["type"] == "training":
        checkpoint_path = WORK_DIR / exp_config["checkpoint"]
        return checkpoint_path.exists()
    return False


def run_experiment(exp_id, exp_config):
    """运行单个实验"""
    print(f"\n{'='*80}")
    print(f"运行实验 {exp_id}: {exp_config['name']}")
    print(f"{'='*80}")
    
    script_path = WORK_DIR / exp_config["script"]
    if not script_path.exists():
        print(f"❌ 脚本不存在: {script_path}")
        return False
    
    cmd = [PYTHON_ENV, str(script_path)] + exp_config["args"]
    
    print(f"执行命令: {' '.join(cmd)}")
    print(f"工作目录: {WORK_DIR}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(WORK_DIR),
            check=True,
            capture_output=False
        )
        print(f"✅ 实验 {exp_id} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 实验 {exp_id} 失败: {e}")
        return False


def main():
    print("="*80)
    print("智能实验运行脚本")
    print("="*80)
    print(f"Python环境: {PYTHON_ENV}")
    print(f"工作目录: {WORK_DIR}")
    print()
    
    # 检查实验状态
    print("检查实验状态...")
    completed = []
    pending = []
    
    for exp_id, exp_config in EXPERIMENTS.items():
        if check_experiment_completed(exp_id, exp_config):
            completed.append(exp_id)
            print(f"✅ {exp_id}: {exp_config['name']} - 已完成")
        else:
            pending.append(exp_id)
            print(f"⏳ {exp_id}: {exp_config['name']} - 待运行")
    
    print(f"\n已完成: {len(completed)}/{len(EXPERIMENTS)}")
    print(f"待运行: {len(pending)}/{len(EXPERIMENTS)}")
    
    if not pending:
        print("\n所有实验都已完成！")
        return
    
    # 询问是否继续
    print(f"\n准备运行 {len(pending)} 个待完成的实验...")
    print("注意: 训练实验可能需要很长时间（每个20-100个epoch）")
    
    # 运行待完成的实验
    for exp_id in pending:
        exp_config = EXPERIMENTS[exp_id]
        
        if exp_config["type"] == "training":
            print(f"\n⚠️  训练实验 {exp_id} 需要 {exp_config.get('epochs', '?')} 个epoch")
            print("   这可能需要较长时间，建议在后台运行")
        
        success = run_experiment(exp_id, exp_config)
        
        if not success:
            print(f"\n实验 {exp_id} 失败，是否继续运行下一个实验？")
            # 这里可以添加用户交互，但为了自动化，我们继续运行
    
    print("\n" + "="*80)
    print("所有实验运行完成！")
    print("="*80)


if __name__ == "__main__":
    main()


