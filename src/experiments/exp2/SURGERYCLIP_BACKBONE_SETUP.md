# SurgeryCLIP Backbone 集成指南

## ✅ 已完成的工作

### 1. 创建了 SurgeryCLIPBackbone 类
- **文件**: `src/experiments/exp2/surgeryclip_backbone.py`
- **功能**: 
  - 使用 SurgeryCLIP 的 visual encoder 作为 GroundingDINO 的 backbone
  - 单尺度输出（只返回最后一层 patch features）
  - 支持冻结/解冻 backbone 参数

### 2. 修改了 GroundingDINO 的 backbone.py
- **文件**: `src/experiments/exp2/Open-GroundingDino-main/models/GroundingDINO/backbone/backbone.py`
- **修改内容**:
  - 添加了 SurgeryCLIP backbone 的导入
  - 在 `build_backbone` 函数中添加了 `surgeryclip` 分支
  - 支持单尺度配置

### 3. 创建了配置文件
- **文件**: `src/experiments/exp2/Open-GroundingDino-main/tools/GroundingDINO_SurgeryCLIP_cfg.py`
- **配置要点**:
  - `backbone = "surgeryclip"`
  - `return_interm_indices = [3]` (单尺度)
  - `num_feature_levels = 1`
  - `surgeryclip_ckpt` 需要设置为你自己的 checkpoint 路径

### 4. 创建了测试脚本
- **文件**: `src/experiments/exp2/test_surgeryclip_backbone_only.py`
- **功能**: 独立测试 SurgeryCLIPBackbone 是否能正常工作

## 📋 使用步骤

### Step 1: 准备 SurgeryCLIP Checkpoint

确保你有 SurgeryCLIP 的 checkpoint 文件（.pt 格式）。

### Step 2: 修改配置文件

编辑 `GroundingDINO_SurgeryCLIP_cfg.py`，设置 checkpoint 路径：

```python
surgeryclip_ckpt = "/absolute/path/to/your/surgeryclip/checkpoint.pt"
```

### Step 3: 测试 Backbone（可选）

先单独测试 backbone 是否能工作：

```bash
cd /media/ubuntu22/新加卷1/Projects/RemoteCLIP-main
source remoteclip/bin/activate
python src/experiments/exp2/test_surgeryclip_backbone_only.py
```

### Step 4: 运行完整测试

修改 `run_gdino_sanity.py` 使用新的配置文件：

```python
config_path = os.path.join(
    open_gdino_root,
    "tools",
    "GroundingDINO_SurgeryCLIP_cfg.py"  # 改为这个
)
```

然后运行：

```bash
cd /media/ubuntu22/新加卷1/Projects/RemoteCLIP-main
source remoteclip/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
python src/experiments/exp2/run_gdino_sanity.py
```

## 🔍 关键接口说明

### SurgeryCLIPBackbone.forward()

**输入**: `NestedTensor(images, mask)`
- `images`: `[B, 3, H, W]` - 图像张量（应该已经做过 CLIP 预处理）
- `mask`: `[B, H, W]` - 可选的 mask

**输出**: `Dict[str, NestedTensor]`
- 返回 `{"0": NestedTensor(features, mask)}`
- `features`: `[B, D, N, N]` - patch features，其中 D 是 embed_dim，N 是 patch 数量

### build_surgeryclip_backbone(args)

**参数**:
- `args.surgeryclip_ckpt`: SurgeryCLIP checkpoint 路径（必需）
- `args.device`: 设备（默认 "cuda"）
- `args.train_surgeryclip_backbone`: 是否训练 backbone（默认 False）

**返回**: `(backbone, num_channels)`
- `backbone`: SurgeryCLIPBackbone 实例
- `num_channels`: `[embed_dim]` - 通道数列表

## ⚠️ 注意事项

1. **输入图像尺寸**: CLIP 通常使用 224x224，确保输入图像尺寸正确
2. **预处理**: 图像应该已经做过 CLIP 的预处理（Resize + CenterCrop + Normalize）
3. **单尺度限制**: 当前实现只支持单尺度输出，`return_interm_indices` 应该是 `[3]`
4. **Checkpoint 路径**: 必须使用绝对路径或相对于项目根目录的路径

## 🐛 调试建议

如果遇到问题：

1. **检查 checkpoint 路径**: 确保路径正确且文件存在
2. **检查 embed_dim**: 确认 `self.visual.embed_dim` 存在
3. **检查 encode_image_with_all_tokens**: 确认方法存在且返回正确的形状
4. **检查 mask**: 确保 mask 的形状和 features 匹配

## 📝 下一步优化方向

1. **多尺度支持**: 如果需要，可以扩展支持多尺度特征
2. **预处理集成**: 在 backbone 内部集成 CLIP 预处理
3. **部分解冻**: 支持只解冻部分层进行训练
4. **性能优化**: 优化 token 提取和 reshape 操作


