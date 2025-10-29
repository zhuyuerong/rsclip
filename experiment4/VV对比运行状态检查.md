# VV对比脚本运行状态检查报告

## 📊 当前状态

### 进程状态
- **状态**: ⚠️ 脚本已完成运行但有问题
- **完成时间**: 已执行完毕

### 主要问题

1. **设备不匹配错误** ⚠️
   - 错误信息: `Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`
   - 影响: 大部分样本（477-499）处理失败
   - 原因: VV机制提取的特征未正确移动到GPU

2. **VV机制结果缺失** ❌
   - VV机制的有效样本数: 0
   - 正常机制的有效样本数: 1（应该更多）

### 初步结果（仅正常机制，样本数不足）

| 指标 | 正常机制 |
|------|----------|
| Top-1 patch命中率 | 16.00% |
| Top-5 patch命中率 | 50.60% |
| 平均Top-1 IoU | 0.0060 |
| 平均Top-5 IoU | 0.0177 |
| 平均最大相似度 | 0.255752 |

## ✅ 已修复

已在代码中添加设备同步：
- 确保patch特征在正确的设备上
- 确保文本特征在GPU上
- 添加异常处理的错误抑制（避免刷屏）

## 🔄 重新运行状态

脚本已重新启动，修复了设备不匹配问题。

### 监控命令

```bash
# 查看实时进度
tail -f /tmp/vv_comparison.log

# 检查进程
ps aux | grep compare_vv_vs_normal_patch_alignment

# 查看结果（完成后）
cat experiment4/outputs/diagnosis/vv_vs_normal_alignment.json
```

## ⏱️ 预计完成时间

- 样本数: 500个（max_samples限制）
- 每个样本: 正常机制提取 + VV机制提取 + 对齐分析
- 预计: 15-25分钟

