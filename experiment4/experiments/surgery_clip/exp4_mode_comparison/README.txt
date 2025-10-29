实验4: 3种模式对比 (Seen/Unseen数据集)
=========================================

代码文件:
- compare_three_modes.py (本目录)

依赖代码:
- experiment4/core/models/clip_surgery.py
  - CLIPSurgeryWrapper类 (支持3种模式切换)
  - clip_feature_surgery函数 (第15-59行)
  - get_similarity_map函数 (第62-100行)
- experiment4/core/utils/map_calculator.py
  - calculate_map函数
- ../config_experiments.py
  - ExperimentConfig类 (3种模式配置)
- ../utils/seen_unseen_split.py
  - SeenUnseenDataset类 (seen/unseen划分)

3种模式:
1. 标准RemoteCLIP (use_surgery=False, use_vv=False)
2. Surgery去冗余 (use_surgery=True, use_vv=False)
3. Surgery+VV机制 (use_surgery=True, use_vv=True)

运行命令:
python compare_three_modes.py --quick-test

输出结果 (待生成):
- mode_comparison_seen.png
  - 3种模式热图网格 (seen数据集)
- mode_comparison_unseen.png
  - 3种模式热图网格 (unseen数据集)
- map_comparison.json
  - 3种模式mAP对比表

实验状态: 待运行

