# DeCLIP

从 `external/DeCLIP-main/` 提取的核心代码。

## 核心文件

- `declip.py` - DeCLIP训练核心实现
- `region_clip.py` - 区域CLIP实现
- `data.py` - 数据处理
- `open_clip/` - OpenCLIP实现（包含DeCLIP的修改）

## 说明

此目录包含从external提取的DeCLIP核心代码，用于在competitors中实现。

DeCLIP是一个解耦学习的开放词汇密集感知框架，通过解耦CLIP的自注意力模块来获得"内容"和"上下文"特征。

下一步需要验证图像是否正常输入模型,是否可以正常推理和训练,输出的形式,是否可以和eval里的函数连通