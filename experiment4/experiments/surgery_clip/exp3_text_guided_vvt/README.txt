实验3: 文本引导VV^T热图 (Layers 1/3/6/9)
==========================================

代码文件:
- text_guided_vvt.py (本目录)

依赖代码:
- experiment4/core/models/clip_surgery.py
  - CLIPSurgeryWrapper类
  - clip_feature_surgery函数 (第15-59行)
  - get_similarity_map函数 (第62-100行)
  - get_layer_features方法 (第561-629行)
- ../utils/seen_unseen_split.py
  - SeenUnseenDataset类

运行命令:
python text_guided_vvt.py \
    --dataset datasets/mini_dataset \
    --layers 1 3 6 9 \
    --max-samples 5

输出结果:
- text_guided_vvt_sample0-4.png (5个, 共2.2MB)
  - 每个样本显示: 原图 + 4层热图
  - 热图使用Feature Surgery计算
- gt_responses.json (82B)
  - 各层GT区域响应强度

实验发现:
- GT响应: 所有层均为NaN (bbox格式问题)
- 热图生成: 成功 (5个样本 x 4层)
- 可视化: 原图+热图叠加显示
- 待修复: bbox坐标处理

