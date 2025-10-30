# Experiment 5: 层级注意力搜索（Hierarchical Attention Search）

## 概述

基于递归细化检测的层级搜索方法，解决RemoteCLIP 7×7 patch导致的小目标响应弱化问题。

## 核心算法

### 递归细化流程
```
初始: 224×224整图 (7×7 patch)
  ↓ 生成热图，找top-k高响应区域
  ├─ 区域1 (margin=0.65) → 超分2x → 重新检测
  │   ↓ 递归分区
  │   ├─ 子区域1.1 (margin=0.72) → 继续细化...
  │   └─ 子区域1.2 (margin=0.58) → 停止（margin下降）
  └─ 区域2 (margin=0.42) → 停止（margin低）

输出: 最高margin的叶子节点 = 最精确的目标定位
```

### Prediction Margin
```python
margin = top1_prob - top2_prob
```
- margin高 → 模型有信心 → 可能包含清晰目标
- margin低 → 模型犹豫 → 纯背景或噪声

## 4种分区策略

### 1. Grid分区
- **策略**: 2×2均匀网格划分
- **优点**: 简单、保证全覆盖
- **缺点**: 可能切碎目标
- **适用**: 目标位置未知

### 2. Threshold分区
- **策略**: 连通域分析（热图 > mean + 0.5*std）
- **优点**: 聚焦高响应区域
- **缺点**: 对阈值敏感
- **适用**: 热图清晰时

### 3. Peaks分区
- **策略**: 局部极大值检测
- **优点**: 直接定位峰值
- **缺点**: 可能漏掉弱响应
- **适用**: 稀疏目标

### 4. Hybrid分区（推荐）
- **策略**: Grid + Threshold + Peaks + NMS
- **优点**: 鲁棒性最好，综合3种策略
- **缺点**: 计算量稍大
- **适用**: 通用场景

## 终止条件

搜索在以下情况停止：
1. `depth >= max_depth`（达到最大深度）
2. `margin < margin_threshold`（置信度太低）
3. `region_size < min_region_size`（区域太小）
4. 无有效子区域（分区失败）

## 使用方法

### 快速测试（2样本×3层）
```bash
cd /media/ubuntu22/新加卷1/Projects/RemoteCLIP-main

PYTHONPATH=. ovadetr_env/bin/python3.9 experiment5/hierarchical_search/hierarchical_searcher.py \
  --max-samples 2 \
  --layers 6 9 12 \
  --max-depth 2 \
  --top-k 2
```

### 完整运行（10样本×5层）
```bash
PYTHONPATH=. ovadetr_env/bin/python3.9 experiment5/hierarchical_search/hierarchical_searcher.py \
  --max-samples 10 \
  --layers 1 3 6 9 12 \
  --max-depth 2 \
  --top-k 2
```

### 参数说明
- `--max-samples`: 最大样本数（默认10）
- `--layers`: 要分析的层（默认1,3,6,9,12）
- `--max-depth`: 最大递归深度（默认2）
- `--top-k`: 每层探索前k个区域（默认2）

## 输出结果

### 目录结构
```
experiment5/hierarchical_search/
├── hierarchical_searcher.py    # 主脚本
├── README.md                   # 本文档
└── results/
    ├── 4strategy_comparison/   # 4策略对比（待实现）
    └── search_trees/           # 搜索树可视化
        ├── DIOR_03135_vehicle_tree.png
        ├── DIOR_05386_overpass_tree.png
        └── ...
```

### 搜索树可视化

每个PNG包含2列：
- **左图**: 所有探索的区域
  - 不同颜色 = 不同深度
  - 虚线 = 非最优路径
  - 实线 = 最优路径
- **右图**: 最优路径高亮
  - 绿色粗线 = 最优路径
  - 标注每层的margin值

## 技术实现

### SearchNode数据结构
```python
@dataclass
class SearchNode:
    bbox: (y, x, h, w)      # 区域位置
    margin: float           # prediction margin
    confidence: float       # top-1概率
    class_idx: int         # 预测类别
    depth: int             # 深度
    parent: SearchNode     # 父节点
    children: List[SearchNode]  # 子节点列表
    heatmap: np.ndarray    # 热图
```

### HierarchicalAttentionSearch核心方法
- `search()`: 执行搜索，返回best_node和root_node
- `_recursive_search()`: DFS递归细化
- `_partition_*()`: 4种分区策略实现
- `_super_resolve()`: 双三次超分
- `_compute_margin()`: 计算prediction margin
- `_generate_heatmap()`: 生成热图
- `_find_best_leaf()`: 找最高margin的叶子节点

## 实验结果示例

### Sample: DIOR_05386 (overpass)
```
Strategy: grid
  Nodes: 7 (1 root + 4 depth-1 + 2 depth-2)
  Best margin: 0.6489
  
Strategy: threshold  
  Nodes: 4 (1 root + 2 depth-1 + 1 depth-2)
  Best margin: 0.6489
  
Strategy: peaks
  Nodes: 7
  Best margin: 0.6489
  
Strategy: hybrid
  Nodes: 4 (NMS去重后）
  Best margin: 0.6489
```

**观察**: 不同策略探索节点数不同，但最终margin相近

## 与伪OLA的对比

### ❌ 伪OLA（已废弃）
```python
# 错误流程
features = model(image_224)  # → 7×7 patch
heatmap_224 = upsample(features)  # → 224×224插值
tiles = 切片(heatmap_224)  # 切碎低分辨率图
stitched = OLA拼接(tiles)  # 粘回去 = 毫无意义
```

### ✅ 层级搜索（正确）
```python
# 正确流程
heatmap_root = model(image_224)  # → 7×7
regions = 找高响应区(heatmap_root)  # → 定位可疑区域
for region in regions:
    crop_sr = 超分(region)  # → 放大2x
    heatmap_sr = model(crop_sr)  # → 重新检测（更高分辨率）
    递归细化...
```

**关键差异**: 每次超分后**重新推理**，获得更高分辨率的特征，而非简单插值。

## 性能统计

### 计算复杂度
- **Root节点**: 1次推理
- **Depth-1**: top_k次推理（每个2x超分）
- **Depth-2**: top_k^2次推理（每个4x超分）
- **总推理次数**: 1 + k + k^2 (k=2时，最多7次)

### 示例统计
- **DIOR_03135**: 1个节点（margin太低，未递归）
- **DIOR_05386**: 4-7个节点（递归2层）

## 下一步改进

### 1. 实现4策略对比可视化
生成4行×6列的对比图（4策略 × 1原图+5层）

### 2. 早停优化
```python
# margin连续不增长则停止
if node.margin <= parent.margin * 1.05:
    return  # 增长<5%，停止递归
```

### 3. Beam Search变体
改为广度优先，每层保留top-k节点同时扩展

### 4. 自适应scale_factor
```python
# 大区域用2x，小区域用4x
scale = 4.0 if area < 5000 else 2.0
```

## 相关文件

- `experiment4/core/models/clip_surgery.py`: 模型包装器
- `experiment4/experiments/surgery_clip/exp3_text_guided_vvt/`: 4模式对比（无递归）
- `experiment5/hierarchical_search/`: 层级搜索（本实验）

---
创建时间: 2024-10-30
总搜索树: 4个PNG (~250KB each)
策略对比: Grid / Threshold / Peaks / Hybrid

