实验3: 文本引导VV^T热图 (Layers 1/3/6/9)
==========================================

代码文件:
- text_guided_vvt.py (本目录)

依赖代码:
- experiment4/core/models/clip_surgery.py
  - CLIPSurgeryWrapper类
  - clip_feature_surgery函数 (第15-59行) **需要多个类别**
  - get_similarity_map函数 (第62-100行)
  - get_layer_features方法 (第561-629行)
- ../utils/seen_unseen_split.py
  - SeenUnseenDataset类

运行命令:
cd /media/ubuntu22/新加卷1/Projects/RemoteCLIP-main
PYTHONPATH=. ovadetr_env/bin/python3.9 \
  experiment4/experiments/surgery_clip/exp3_text_guided_vvt/text_guided_vvt.py \
  --dataset datasets/mini_dataset \
  --layers 1 3 6 9 \
  --max-samples 5

输出结果:
- text_guided_vvt_sample0-4.png (5个, 共2.3MB)
  - 每个样本显示: 原图 + 4层热图
  - 热图使用Feature Surgery计算
- gt_responses.json (139B)
  - 各层GT区域响应强度

实验发现:
- 相似度范围: -2.24 ~ 5.24 (Surgery后有正负值)
- 热图范围: 0.01 ~ 0.99 (归一化后)
- 热图唯一值: 2943个 (丰富的颜色渐变)
- GT响应:
  * Layer 1: 0.61 (最高)
  * Layer 3: 0.36
  * Layer 6: 0.44
  * Layer 9: 0.40

关键修复 (2025-10-29):
- 问题: clip_feature_surgery需要多个类别才能正确工作
- 原因: 单类别时 redundant = feats → feats-redundant = 0
- 解决: 使用DIOR所有20个类别，提取目标类别热图
- 结果: 热图从NaN变为有效数值

结论:
- Layer 1响应最强 (0.61)
- 深层(L9)响应反而降低 (0.40)
- 说明浅层特征对Surgery更敏感
