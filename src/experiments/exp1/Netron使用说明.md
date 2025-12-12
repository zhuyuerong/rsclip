# Netron模型可视化使用说明

## 导出的模型文件

已成功导出以下文件，可直接在Netron中打开：

### 1. 完整模型

#### `model_for_netron.onnx` (推荐) ⭐
- **格式**: ONNX
- **大小**: ~1.08 MB
- **内容**: 完整SurgeryAAF模型（包含CLIP + AAF + CAM生成器）
- **优点**: 轻量级，结构清晰，Netron支持最好
- **推荐用于**: 查看整体模型架构

#### `model_for_netron.pt`
- **格式**: TorchScript
- **大小**: ~577.61 MB
- **内容**: 完整模型（包含权重）
- **优点**: 包含完整权重，可以查看参数值
- **推荐用于**: 需要查看权重值的场景

### 2. AAF层（轻量级）

#### `aaf_for_netron.onnx`
- **格式**: ONNX
- **大小**: ~37.29 KB
- **内容**: 仅AAF层（自适应注意力融合）
- **优点**: 非常轻量，结构简单
- **推荐用于**: 详细查看AAF层的内部结构

## 使用方法

### 方法1: 在线使用（推荐）

1. 打开浏览器，访问：https://netron.app/
2. 点击 "Open Model" 或直接拖拽文件到页面
3. 选择导出的 `.onnx` 或 `.pt` 文件
4. 即可查看模型结构图

### 方法2: 桌面版

1. 下载Netron桌面版：
   - Windows/Mac/Linux: https://github.com/lutzroeder/netron/releases
2. 安装并打开Netron
3. 拖拽模型文件到Netron窗口
4. 查看模型结构

## 文件位置

所有导出的文件位于：
```
src/experiments/exp1/
├── model_for_netron.onnx    # 完整模型（ONNX，推荐）
├── model_for_netron.pt       # 完整模型（TorchScript）
└── aaf_for_netron.onnx       # AAF层（ONNX，轻量级）
```

## 模型结构说明

### 完整模型 (`model_for_netron.onnx`)

```
输入: images [batch_size, 3, 224, 224]
  ↓
SurgeryCLIP Visual Encoder (冻结)
  ├── Conv2d (patch embedding)
  ├── Transformer × 12层
  │   └── 后6层使用Surgery Attention
  └── LayerNorm + Projection
  ↓
Patch Features [batch_size, 49, 768]
  ↓
AAF层 (可训练，13个参数)
  ├── 融合VV路径注意力 (6层)
  ├── 融合原始路径注意力 (6层)
  └── 混合系数 alpha
  ↓
Patch-to-Patch Attention [batch_size, 49, 49]
  ↓
CAM生成器
  ├── 初始CAM (patch-text相似度)
  └── p2p传播
  ↓
输出: CAM [batch_size, 20, 7, 7]
```

### AAF层 (`aaf_for_netron.onnx`)

```
输入: 
  - vv_attn_0 ~ vv_attn_5: [batch, 12, 50, 50] × 6
  - ori_attn_0 ~ ori_attn_5: [batch, 12, 50, 50] × 6
  ↓
AAF层
  ├── vv_layer_weights: [6] - 学习权重
  ├── ori_layer_weights: [6] - 学习权重
  └── alpha: [] - 混合系数
  ↓
输出: attn_p2p [batch, 49, 49]
```

## Netron功能

在Netron中可以：

1. **查看模型结构**
   - 节点图：每个操作节点
   - 数据流：张量流动路径
   - 参数：权重和偏置

2. **查看节点详情**
   - 点击节点查看详细信息
   - 输入/输出形状
   - 操作类型和参数

3. **搜索和过滤**
   - 搜索特定节点
   - 过滤节点类型
   - 高亮相关节点

4. **导出信息**
   - 导出为图片
   - 导出节点信息
   - 复制节点信息

## 推荐查看顺序

1. **首先查看** `aaf_for_netron.onnx`
   - 结构简单，容易理解AAF层的工作原理
   - 可以看到13个可训练参数的位置

2. **然后查看** `model_for_netron.onnx`
   - 查看完整模型架构
   - 理解数据流动
   - 查看各模块的连接关系

3. **如需查看权重** `model_for_netron.pt`
   - 查看实际参数值
   - 分析权重分布

## 重新导出

如果需要重新导出模型（例如修改了模型结构），运行：

```bash
cd src/experiments/exp1
source ../../../remoteclip/bin/activate
python export_for_netron.py
```

## 注意事项

1. **ONNX文件**：不包含权重，只包含结构，适合查看架构
2. **TorchScript文件**：包含完整权重，文件较大
3. **AAF层**：独立导出，便于详细分析
4. **动态输入**：模型支持动态batch_size

## 故障排除

如果Netron无法打开文件：

1. **检查文件格式**：确保是 `.onnx` 或 `.pt` 格式
2. **更新Netron**：使用最新版本的Netron
3. **在线版本**：尝试使用 https://netron.app/ 在线版本
4. **重新导出**：运行 `export_for_netron.py` 重新生成

## 技术细节

- **ONNX版本**: opset 14
- **PyTorch版本**: 2.9.0
- **导出方式**: torch.onnx.export (完整模型) / torch.jit.trace (TorchScript)
- **输入格式**: 图像 [batch_size, 3, 224, 224]
- **输出格式**: CAM [batch_size, 20, 7, 7]





