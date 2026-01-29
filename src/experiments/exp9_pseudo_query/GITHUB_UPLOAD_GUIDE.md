# Exp9 GitHub上传指南

## 📋 概述

本文档说明如何将实验9完整上传到GitHub，包括必需的依赖文件。

---

## 📦 上传内容清单

### 1. ✅ 实验9核心文件 (已上传)

```
src/experiments/exp9_pseudo_query/
├── models/                        # ✅ 已上传
│   ├── __init__.py
│   ├── deformable_detr_pseudo.py
│   ├── heatmap_query_gen.py
│   └── query_injection.py
├── datasets/                      # ✅ 已上传
│   ├── __init__.py
│   ├── dior_deformable.py
│   └── dior_with_heatmap.py
├── scripts/                       # ✅ 已上传
│   ├── train_a0_baseline.py
│   ├── train_pseudo_query.py
│   ├── run_a0.sh
│   ├── run_a2_teacher.sh
│   ├── run_a3_heatmap.sh
│   ├── run_b1_random.sh
│   ├── run_b2_shuffled.sh
│   ├── setup_env.sh
│   └── verify_environment.sh
├── configs/                       # ✅ 已上传
│   ├── experiment_config.py
│   └── experiment_config_v2.py
├── utils/                         # ✅ 已上传
│   ├── run_manager.py
│   └── check_heatmap_format.py
├── test_modules.py                # ✅ 已上传
├── requirements.txt               # ✅ 已上传
├── README.md                      # ✅ 已上传 (已更新依赖说明)
├── DEPENDENCIES.md                # ✅ 已上传 (新增)
├── GITHUB_UPLOAD_GUIDE.md         # ✅ 本文件
├── EXPERIMENT_CHECKLIST.md        # ✅ 已上传
├── FILES_INVENTORY.md             # ✅ 已上传 (需要更新)
├── NEXT_STEPS.md                  # ✅ 已上传
├── QUICK_REFERENCE.md             # ✅ 已上传
└── SETUP_SUMMARY.md               # ✅ 已上传
```

### 2. ⚠️ Deformable DETR依赖 (需要处理)

**问题**: 实验9依赖`external/Deformable-DETR/`，但这是第三方代码库。

**解决方案**: 使用Git Submodule

```
external/
└── Deformable-DETR/               # ⚠️ 通过Git Submodule管理
    ├── models/
    │   ├── __init__.py
    │   ├── deformable_detr.py
    │   ├── deformable_transformer.py
    │   ├── matcher.py
    │   ├── backbone.py
    │   ├── position_encoding.py
    │   ├── segmentation.py
    │   └── ops/                   # CUDA算子
    └── util/
        ├── misc.py
        └── box_ops.py
```

---

## 🔧 推荐方案: 使用Git Submodule

### 为什么使用Submodule?

1. **版权清晰**: 不直接复制第三方代码
2. **更新方便**: 可以跟踪官方更新
3. **体积小**: 不增加仓库大小
4. **标准做法**: 管理外部依赖的标准方式

### 操作步骤

#### Step 1: 添加Submodule

```bash
cd /path/to/RemoteCLIP-main

# 如果external/Deformable-DETR/已经存在，先删除
rm -rf external/Deformable-DETR

# 添加为submodule
git submodule add https://github.com/fundamentalvision/Deformable-DETR.git external/Deformable-DETR

# 提交submodule配置
git add .gitmodules external/Deformable-DETR
git commit -m "Add Deformable-DETR as submodule for exp9"
```

#### Step 2: 更新README说明

在项目根目录的`README.md`中添加:

```markdown
## Dependencies

This project uses Git Submodules for external dependencies. After cloning, run:

\`\`\`bash
# Clone with submodules
git clone --recursive https://github.com/your-username/RemoteCLIP.git

# Or if already cloned, initialize submodules
git submodule update --init --recursive
\`\`\`

### Compile Deformable DETR CUDA Operators

\`\`\`bash
cd external/Deformable-DETR/models/ops
bash make.sh
python test.py
\`\`\`
```

#### Step 3: 用户克隆时的操作

```bash
# 方法1: 克隆时自动拉取submodule
git clone --recursive https://github.com/your-username/RemoteCLIP.git

# 方法2: 克隆后手动初始化submodule
git clone https://github.com/your-username/RemoteCLIP.git
cd RemoteCLIP
git submodule update --init --recursive

# 编译CUDA算子
cd external/Deformable-DETR/models/ops
bash make.sh
```

---

## 🔄 备选方案: 提供安装脚本

如果不想用submodule，可以提供自动安装脚本:

### 创建 `scripts/setup_deformable_detr.sh`

```bash
#!/bin/bash
# 自动安装Deformable DETR依赖

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EXTERNAL_DIR="$PROJECT_ROOT/external"

echo "=== Exp9: 安装Deformable DETR依赖 ==="

# 检查是否已存在
if [ -d "$EXTERNAL_DIR/Deformable-DETR" ]; then
    echo "✅ Deformable-DETR 已存在: $EXTERNAL_DIR/Deformable-DETR"
    read -p "是否重新克隆? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$EXTERNAL_DIR/Deformable-DETR"
    else
        echo "跳过克隆，直接编译CUDA算子"
        cd "$EXTERNAL_DIR/Deformable-DETR/models/ops"
        bash make.sh
        python test.py
        exit 0
    fi
fi

# 克隆官方仓库
echo "📥 克隆Deformable-DETR官方仓库..."
mkdir -p "$EXTERNAL_DIR"
cd "$EXTERNAL_DIR"
git clone https://github.com/fundamentalvision/Deformable-DETR.git

# 编译CUDA算子
echo "🔧 编译CUDA算子..."
cd Deformable-DETR/models/ops
bash make.sh

# 测试
echo "✅ 测试CUDA算子..."
python test.py

echo ""
echo "=== 安装完成! ==="
echo "Deformable-DETR 已安装到: $EXTERNAL_DIR/Deformable-DETR"
```

### 在README中说明

```markdown
## Setup for Exp9

Exp9 requires Deformable DETR. Run the setup script:

\`\`\`bash
bash src/experiments/exp9_pseudo_query/scripts/setup_deformable_detr.sh
\`\`\`

Or manually:

\`\`\`bash
cd external/
git clone https://github.com/fundamentalvision/Deformable-DETR.git
cd Deformable-DETR/models/ops
bash make.sh
python test.py
\`\`\`
```

---

## 📝 更新 `FILES_INVENTORY.md`

需要更新文件清单，说明依赖情况:

```markdown
### 8. 外部依赖 (不直接包含在本仓库)

| 文件 | 来源 | 获取方式 | 状态 |
|------|------|----------|------|
| `external/Deformable-DETR/` | [官方仓库](https://github.com/fundamentalvision/Deformable-DETR) | Git Submodule 或 手动克隆 | ⚠️ 必需 |

**说明**:
- Deformable DETR是标准的第三方实现，不包含在本仓库
- 使用前需要克隆并编译CUDA算子
- 详见 [`DEPENDENCIES.md`](DEPENDENCIES.md)
```

---

## 🎯 推荐上传流程

### Step 1: 检查现有文件

```bash
cd src/experiments/exp9_pseudo_query

# 查看git状态
git status

# 应该看到:
# - DEPENDENCIES.md (新文件)
# - GITHUB_UPLOAD_GUIDE.md (新文件)
# - README.md (已修改)
```

### Step 2: 添加Submodule

```bash
cd /path/to/RemoteCLIP-main

# 添加Deformable DETR为submodule
git submodule add https://github.com/fundamentalvision/Deformable-DETR.git external/Deformable-DETR
```

### Step 3: 更新文档

```bash
# 更新 FILES_INVENTORY.md
# 添加外部依赖说明

# 更新项目根目录的README.md
# 添加submodule使用说明
```

### Step 4: 提交所有更改

```bash
# 提交实验9的新文档
git add src/experiments/exp9_pseudo_query/DEPENDENCIES.md
git add src/experiments/exp9_pseudo_query/GITHUB_UPLOAD_GUIDE.md
git add src/experiments/exp9_pseudo_query/README.md

# 提交submodule配置
git add .gitmodules
git add external/Deformable-DETR

# 提交
git commit -m "Exp9: Add dependency documentation and Deformable-DETR submodule

- Add DEPENDENCIES.md: Detailed explanation of Deformable DETR dependency
- Add GITHUB_UPLOAD_GUIDE.md: Guide for uploading to GitHub
- Update README.md: Add dependency warning and reference
- Add Deformable-DETR as git submodule

Changes:
- Clarify that exp9 uses standard Deformable DETR from external/
- Provide setup instructions for users
- Use git submodule to manage external dependency
"
```

### Step 5: 推送到GitHub

```bash
# 推送主分支
git push origin main

# 推送submodule (如果需要)
git push --recurse-submodules=on-demand
```

---

## ✅ 用户使用指南

上传后，用户应该这样使用:

### 1. 克隆仓库

```bash
# 推荐: 自动拉取submodule
git clone --recursive https://github.com/your-username/RemoteCLIP.git

# 或: 克隆后手动初始化
git clone https://github.com/your-username/RemoteCLIP.git
cd RemoteCLIP
git submodule update --init --recursive
```

### 2. 编译依赖

```bash
cd external/Deformable-DETR/models/ops
bash make.sh
python test.py
```

### 3. 运行实验9

```bash
conda activate samrs
cd src/experiments/exp9_pseudo_query
bash scripts/run_a0.sh
```

---

## 📚 相关文档

上传后，仓库应该包含以下文档链接:

```
src/experiments/exp9_pseudo_query/
├── README.md                   # 项目总览 + 依赖警告
├── DEPENDENCIES.md             # ⭐ 详细依赖说明
├── GITHUB_UPLOAD_GUIDE.md      # ⭐ 本文件 (上传指南)
├── FILES_INVENTORY.md          # 文件清单 (需要更新)
├── EXPERIMENT_CHECKLIST.md     # 实验清单
├── QUICK_REFERENCE.md          # 快速参考
└── SETUP_SUMMARY.md            # 环境配置
```

**用户阅读顺序**:
1. `README.md` - 看到依赖警告
2. `DEPENDENCIES.md` - 了解依赖详情
3. `SETUP_SUMMARY.md` - 配置环境
4. `QUICK_REFERENCE.md` - 开始实验

---

## 🔍 验证清单

上传前检查:

- [ ] `DEPENDENCIES.md` 已创建
- [ ] `GITHUB_UPLOAD_GUIDE.md` 已创建
- [ ] `README.md` 已更新 (添加依赖说明)
- [ ] `FILES_INVENTORY.md` 已更新 (添加外部依赖说明)
- [ ] Deformable-DETR已添加为submodule
- [ ] `.gitmodules` 文件已创建
- [ ] 根目录README已更新 (submodule使用说明)

上传后检查:

- [ ] 用户可以通过`git clone --recursive`获取完整代码
- [ ] `DEPENDENCIES.md`在GitHub上可以正常查看
- [ ] Submodule链接正确指向官方仓库
- [ ] 所有文档内的链接可以正常跳转

---

## 💡 最佳实践建议

### 1. 在README顶部添加醒目提示

```markdown
# Exp9: Pseudo Query for Object Detection

> ⚠️ **重要依赖**: 本实验依赖Deformable DETR。克隆后请运行:
> ```bash
> git submodule update --init --recursive
> cd external/Deformable-DETR/models/ops && bash make.sh
> ```
> 详见 [DEPENDENCIES.md](DEPENDENCIES.md)
```

### 2. 提供一键安装脚本

```bash
# scripts/setup_exp9.sh
#!/bin/bash
git submodule update --init --recursive
cd external/Deformable-DETR/models/ops
bash make.sh
cd -
conda activate samrs
source scripts/setup_env.sh
bash scripts/verify_environment.sh
```

### 3. 在CI中自动检查

如果有CI/CD，添加检查:

```yaml
# .github/workflows/test_exp9.yml
- name: Check Deformable DETR
  run: |
    git submodule update --init --recursive
    cd external/Deformable-DETR/models/ops
    bash make.sh
    python test.py
```

---

## 📝 补充说明

### 关于许可证

- **Deformable DETR**: Apache License 2.0
- **本项目**: (根据你的项目许可证)

在根目录`README.md`中添加:

```markdown
## License

This project is licensed under [YOUR LICENSE].

External dependencies:
- Deformable DETR: Apache License 2.0 (https://github.com/fundamentalvision/Deformable-DETR)
```

### 关于引用

在论文/README中引用:

```bibtex
@inproceedings{deformable_detr,
  title={Deformable DETR: Deformable Transformers for End-to-End Object Detection},
  author={Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin and Wang, Xiaogang and Dai, Jifeng},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```

---

## 🎯 总结

**推荐方案**: 使用Git Submodule管理Deformable DETR依赖

**关键文件**:
1. `DEPENDENCIES.md` - 详细说明依赖
2. `README.md` - 添加醒目的依赖警告
3. `.gitmodules` - Submodule配置
4. `FILES_INVENTORY.md` - 更新外部依赖说明

**用户体验**:
```bash
# 一条命令搞定
git clone --recursive https://github.com/your-username/RemoteCLIP.git
cd RemoteCLIP/external/Deformable-DETR/models/ops && bash make.sh
```

**维护成本**: 低 (submodule自动跟踪官方更新)

---

**更新日期**: 2026-01-29  
**维护者**: Exp9 Team
