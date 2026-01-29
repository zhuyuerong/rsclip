# Exp9 Pseudo Query 快速参考卡片

## 🚀 一键命令

```bash
# 激活环境
conda activate samrs

# 验证环境
bash scripts/verify_environment.sh

# 运行实验
bash scripts/run_a0.sh      # A0: Baseline
bash scripts/run_a2_teacher.sh  # A2: Teacher
bash scripts/run_a3_heatmap.sh  # A3: Heatmap (核心)
bash scripts/run_b1_random.sh   # B1: Random
bash scripts/run_b2_shuffled.sh # B2: Shuffled

# 监控训练
tail -f outputs/exp9_pseudo_query/a0_training.log
watch -n 1 nvidia-smi

# 对比分析
python scripts/compare_experiments.py --exp_dirs outputs/exp9_pseudo_query/a*
```

---

## 📊 实验矩阵

| ID | 名称 | K (pseudo) | 来源 | 预期 | 命令 |
|----|------|------------|------|------|------|
| A0 | Baseline | 0 | - | 对照组 | `run_a0.sh` |
| A2 | Teacher | 100 | GT boxes | 更快收敛 | `run_a2_teacher.sh` |
| A3 | Heatmap | 100 | 热图top-k | 小目标↑ | `run_a3_heatmap.sh` |
| B1 | Random | 100 | 随机 | 显著差 | `run_b1_random.sh` |
| B2 | Shuffled | 100 | 打乱热图 | 显著差 | `run_b2_shuffled.sh` |

---

## 📁 关键路径

```
代码:     src/experiments/exp9_pseudo_query/
数据:     datasets/DIOR/
输出:     outputs/exp9_pseudo_query/
热图:     outputs/heatmap_cache/
权重:     checkpoints/RemoteCLIP-ViT-B-32.pt
外部:     external/Deformable-DETR/
```

---

## 🔧 环境变量

```bash
export LD_LIBRARY_PATH=/home/ubuntu22/anaconda3/envs/samrs/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTHONPATH="/media/ubuntu22/新加卷1/Projects/RemoteCLIP-main:/media/ubuntu22/新加卷1/Projects/RemoteCLIP-main/external/Deformable-DETR:${PYTHONPATH}"
```

或使用:
```bash
source scripts/setup_env.sh
```

---

## 📊 成功判据

| 实验 | 判据 |
|------|------|
| A0 | Loss↓, Epoch 1 Recall>0.02 |
| A2 | Epoch 10 Recall > A0 + 3~10% |
| A3 | Recall ≥ A2 |
| B1 | Recall < A3 * 0.9 |
| B2 | Recall < A3 * 0.9 |

---

## 🐛 常见问题速查

| 问题 | 解决 |
|------|------|
| ImportError: libc10.so | `export LD_LIBRARY_PATH=...` |
| CUDA OOM | `--batch_size 1` |
| Loss不降 | 检查lr/数据/box格式 |
| Recall很低 | 检查matcher/坐标归一化 |
| A2不如A0 | 检查query_embed格式 |

---

## 📈 监控指标

```bash
# GPU使用
nvidia-smi

# 训练进度
tail -f outputs/exp9_pseudo_query/a0_training.log | grep "Epoch:"

# Loss曲线
grep "loss:" outputs/exp9_pseudo_query/a0_training.log | tail -20

# Recall
grep "Recall@100" outputs/exp9_pseudo_query/a0_training.log
```

---

## 📝 快速笔记模板

```markdown
## 实验: A2 Teacher
- 日期: 2026-01-29
- 状态: ✅ 完成 / ⏳ 运行中 / ❌ 失败
- Epoch 10 Recall: 0.089 (A0: 0.080, +11.3%)
- 最终 mAP@0.5: 0.158
- 结论: A2成功，可进行A3
```

---

## 🎯 下一步检查清单

- [ ] A0完成 (约14小时)
- [ ] 分析A0曲线
- [ ] 运行A2
- [ ] A2成功 → 运行A3
- [ ] A2失败 → Debug
- [ ] 运行B1/B2
- [ ] 对比分析

---

**工作目录**: `/media/ubuntu22/新加卷1/Projects/RemoteCLIP-main`
**Conda环境**: `samrs`
**当前状态**: A0运行中 (PID: 84120)
