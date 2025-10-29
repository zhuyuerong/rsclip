实验1: Patch相似度矩阵分析 (49x49)
=====================================

代码文件:
- patch_similarity_matrix.py (本目录)

依赖代码:
- experiment4/core/models/clip_surgery.py
  - CLIPSurgeryWrapper类
  - get_layer_features方法 (第561-629行)

运行命令:
python patch_similarity_matrix.py --dataset datasets/mini_dataset --layer 12

输出结果:
- surgery_comparison_layer12.png (83KB)
  - 3张对比图: 标准特征/Surgery特征/差异
  - 49x49热力图矩阵

实验发现:
- 标准特征相似度: 均值0.66, std0.15
- Surgery特征相似度: 均值0.01, std0.34
- Surgery影响: -98.7%均值, +122%多样性
- 结论: Surgery显著降低patch冗余

