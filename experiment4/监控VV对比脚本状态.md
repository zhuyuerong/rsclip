# VV机制对比脚本运行状态

## ✅ 问题修复总结

### 已修复的问题：

1. **DIOR_CLASSES未定义** ✅
   - 已在模块级别定义DIOR_CLASSES

2. **数据集加载失败** ✅
   - 修复了XML文件路径问题（位于`annotations/horizontal/`子目录）
   - 改进了图像文件名匹配逻辑
   - 添加了类别名称标准化（处理变体）

3. **数据集验证** ✅
   - 成功加载1000个有效样本
   - 每个样本包含图像、bboxes和labels

## 📊 脚本功能

脚本正在对比：
- **正常机制**：标准MultiheadAttention
- **VV机制**：Attention(V,V,V)，最后6层使用VV注意力

## 📈 评估指标

1. **Patch-Text对齐度**：
   - Top-1 patch命中率（最相似patch是否在bbox内）
   - Top-5 patch命中率
   - 平均最大相似度

2. **定位精度**：
   - 平均Top-1 IoU
   - 平均Top-5 IoU

## 🚀 运行状态

脚本已在后台运行，PID: 992451

### 监控命令：

```bash
# 查看实时日志
tail -f /tmp/vv_comparison.log

# 检查进程状态
ps aux | grep compare_vv_vs_normal_patch_alignment

# 检查结果文件
cat experiment4/outputs/diagnosis/vv_vs_normal_alignment.json
```

## 📝 预计完成时间

- 数据集大小：1000个样本（限制前1000个以加快速度）
- 每个样本需要：提取特征（正常+VV）+ 对齐分析
- 预计时间：10-20分钟（取决于GPU性能）

## ✅ 下一步

等待脚本完成后，将生成详细对比报告在：
`experiment4/outputs/diagnosis/vv_vs_normal_alignment.json`

报告将包含：
- 正常机制的完整指标
- VV机制的完整指标
- 两者的对比分析
- 性能提升/下降的量化结果

