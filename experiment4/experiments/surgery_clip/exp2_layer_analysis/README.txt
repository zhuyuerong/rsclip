实验2: 多层特征分析 (Layers 1/6/9/12)
=======================================

代码文件:
- layer_analysis.py (本目录)

依赖代码:
- experiment4/core/models/clip_surgery.py
  - CLIPSurgeryWrapper类
  - get_layer_features方法 (第561-629行)
  - clip_feature_surgery函数 (第15-59行)
- ../utils/seen_unseen_split.py
  - SeenUnseenDataset类

运行命令:
python layer_analysis.py \
    --dataset datasets/mini_dataset \
    --layers 1 6 9 12 \
    --max-samples 10 \
    --use-surgery

输出结果:
- layer_comparison_heatmaps.png (609KB)
  - 4x3热图网格 (4层 x 3种方法)
  - 方法: 余弦相似度/Feature Surgery/VV^T重要性
- layer_statistics.json (501B)
  - 各层统计数据

实验发现:
- 余弦相似度: L1(0.47) → L12(0.58) +23%
- VV^T重要性: 所有层稳定在0.56-0.65
- Surgery相似度: 全NaN (待诊断)
- 结论: 深层特征(L12)与文本相似度最高

