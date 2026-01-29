# Exp9 GitHub上传检查清单

## 📋 总览

本清单帮助你确保实验9的所有必要文件和文档都已准备完毕，可以安全上传到GitHub。

**上传日期**: 2026-01-29  
**检查者**: ______  
**状态**: ⬜ 准备中 / ⬜ 已完成

---

## ✅ 核心文件检查 (必须上传)

### 1. 模型文件 (4个)

- [ ] `models/__init__.py`
- [ ] `models/deformable_detr_pseudo.py`
- [ ] `models/heatmap_query_gen.py`
- [ ] `models/query_injection.py`

**说明**: 实验9的核心创新模块

### 2. 数据集文件 (3个)

- [ ] `datasets/__init__.py`
- [ ] `datasets/dior_deformable.py`
- [ ] `datasets/dior_with_heatmap.py`

**说明**: DIOR数据集加载器，支持热图

### 3. 训练脚本 (2个)

- [ ] `scripts/train_a0_baseline.py`
- [ ] `scripts/train_pseudo_query.py`

**说明**: A0 baseline和A2/A3/B1/B2的训练脚本

### 4. Shell脚本 (7个)

- [ ] `scripts/run_a0.sh`
- [ ] `scripts/run_a2_teacher.sh`
- [ ] `scripts/run_a3_heatmap.sh`
- [ ] `scripts/run_b1_random.sh`
- [ ] `scripts/run_b2_shuffled.sh`
- [ ] `scripts/setup_env.sh`
- [ ] `scripts/verify_environment.sh`
- [ ] `scripts/setup_deformable_detr.sh` ⭐ 新增

**说明**: 实验运行和环境配置脚本

### 5. 配置文件 (2个)

- [ ] `configs/experiment_config.py`
- [ ] `configs/experiment_config_v2.py`

**说明**: 实验配置参数

### 6. 工具文件 (3个)

- [ ] `utils/run_manager.py`
- [ ] `utils/check_heatmap_format.py`
- [ ] `scripts/compare_experiments.py`

**说明**: 训练管理和结果分析工具

### 7. 测试文件 (1个)

- [ ] `test_modules.py`

**说明**: 单元测试

### 8. 依赖文件 (1个)

- [ ] `requirements.txt`

**说明**: Python依赖清单

---

## 📚 文档文件检查 (强烈推荐)

### 核心文档 (8个)

- [ ] `README.md` ⭐ 已更新 (添加依赖说明)
- [ ] `DEPENDENCIES.md` ⭐ 新增 (依赖详细说明)
- [ ] `GITHUB_UPLOAD_GUIDE.md` ⭐ 新增 (上传指南)
- [ ] `MISSING_FILES_SUMMARY.md` ⭐ 新增 (缺失文件说明)
- [ ] `UPLOAD_CHECKLIST.md` ⭐ 本文件
- [ ] `FILES_INVENTORY.md` ✏️ 已更新
- [ ] `EXPERIMENT_CHECKLIST.md`
- [ ] `NEXT_STEPS.md`
- [ ] `QUICK_REFERENCE.md`
- [ ] `SETUP_SUMMARY.md`

**重点检查**:
- `README.md`: 是否添加了依赖警告?
- `DEPENDENCIES.md`: 是否清楚说明了Deformable DETR依赖?
- `FILES_INVENTORY.md`: 是否更新了外部依赖说明?

---

## 🔗 外部依赖处理 (关键!)

### 选项1: 使用Git Submodule (推荐) ⭐

- [ ] 在项目根目录运行:
  ```bash
  git submodule add https://github.com/fundamentalvision/Deformable-DETR.git external/Deformable-DETR
  ```

- [ ] 提交submodule配置:
  ```bash
  git add .gitmodules external/Deformable-DETR
  git commit -m "Add Deformable-DETR as submodule"
  ```

- [ ] 更新项目根目录的`README.md`:
  - [ ] 添加submodule使用说明
  - [ ] 添加编译CUDA算子的说明

### 选项2: 只提供安装说明

如果不使用submodule:

- [ ] 确保`DEPENDENCIES.md`说明了如何手动克隆
- [ ] 确保`scripts/setup_deformable_detr.sh`可以正常工作
- [ ] 在`README.md`中清楚说明安装步骤

---

## 📝 文档内容检查

### README.md

- [ ] 包含项目简介
- [ ] ⭐ 包含依赖警告 (Deformable DETR)
- [ ] 包含实验设计说明 (A0/A2/A3/B1/B2)
- [ ] 包含运行示例
- [ ] 链接到其他关键文档 (DEPENDENCIES.md等)

### DEPENDENCIES.md (新增)

- [ ] 清楚说明Deformable DETR是什么
- [ ] 说明为什么需要它 (A0/A2/A3都依赖)
- [ ] 提供获取方式 (submodule/手动克隆)
- [ ] 说明编译CUDA算子的步骤
- [ ] 提供故障排查建议

### GITHUB_UPLOAD_GUIDE.md (新增)

- [ ] 说明submodule的使用方法
- [ ] 提供完整的上传步骤
- [ ] 说明用户克隆后的操作
- [ ] 包含验证清单

### FILES_INVENTORY.md

- [ ] 列出所有核心文件
- [ ] ⭐ 添加外部依赖说明
- [ ] 说明文件用途
- [ ] 更新文件统计

---

## 🔧 脚本功能检查

### setup_deformable_detr.sh (新增)

测试脚本:

```bash
cd src/experiments/exp9_pseudo_query
bash scripts/setup_deformable_detr.sh
```

- [ ] 可以检测已存在的Deformable-DETR
- [ ] 可以克隆官方仓库
- [ ] 可以编译CUDA算子
- [ ] 可以运行测试
- [ ] 有清晰的输出信息

### verify_environment.sh

测试脚本:

```bash
bash scripts/verify_environment.sh
```

- [ ] 检查Deformable-DETR目录
- [ ] 检查CUDA算子
- [ ] 检查Python依赖
- [ ] 给出清晰的成功/失败信息

---

## 🎨 代码质量检查

### 代码风格

- [ ] Python代码有适当的注释
- [ ] 函数和类有docstring
- [ ] Shell脚本有使用说明

### 文档质量

- [ ] 没有明显的拼写错误
- [ ] 链接都可以正常工作
- [ ] Markdown格式正确
- [ ] 代码块语法高亮正确

---

## 🧪 功能测试

### 模拟用户操作

#### 测试1: Submodule方式

```bash
# 在另一个目录测试
cd /tmp
git clone --recursive <your-repo-url> test-exp9
cd test-exp9

# 编译CUDA算子
cd external/Deformable-DETR/models/ops
bash make.sh
python test.py

# 验证环境
cd ../../../../src/experiments/exp9_pseudo_query
bash scripts/verify_environment.sh
```

- [ ] 可以成功克隆
- [ ] Submodule自动拉取
- [ ] CUDA算子编译成功
- [ ] 环境验证通过

#### 测试2: 手动安装方式

```bash
# 在另一个目录测试
cd /tmp
git clone <your-repo-url> test-exp9-manual
cd test-exp9-manual

# 运行自动安装脚本
cd src/experiments/exp9_pseudo_query
bash scripts/setup_deformable_detr.sh

# 验证环境
bash scripts/verify_environment.sh
```

- [ ] 可以成功克隆
- [ ] 自动安装脚本正常工作
- [ ] CUDA算子编译成功
- [ ] 环境验证通过

#### 测试3: 导入测试

```bash
cd src/experiments/exp9_pseudo_query
python test_modules.py
```

- [ ] 可以成功导入Deformable DETR模块
- [ ] 可以成功导入实验9自定义模块
- [ ] 单元测试通过

---

## 📊 文档完整性检查

### 用户视角

假设你是第一次接触这个项目的用户:

- [ ] 从`README.md`能快速了解项目是什么
- [ ] 能看到清楚的依赖警告
- [ ] 知道如何安装依赖 (submodule或手动)
- [ ] 知道如何运行实验
- [ ] 遇到问题时知道去哪里找帮助

### 维护者视角

假设你是维护这个项目的人:

- [ ] 代码结构清晰
- [ ] 文档说明了设计决策 (为什么用submodule)
- [ ] 有完整的文件清单
- [ ] 有故障排查指南

---

## 🚀 提交前最终检查

### Git状态

```bash
cd /path/to/RemoteCLIP-main
git status
```

应该看到:

```
新文件:
  src/experiments/exp9_pseudo_query/DEPENDENCIES.md
  src/experiments/exp9_pseudo_query/GITHUB_UPLOAD_GUIDE.md
  src/experiments/exp9_pseudo_query/MISSING_FILES_SUMMARY.md
  src/experiments/exp9_pseudo_query/UPLOAD_CHECKLIST.md
  src/experiments/exp9_pseudo_query/scripts/setup_deformable_detr.sh

修改的文件:
  src/experiments/exp9_pseudo_query/README.md
  src/experiments/exp9_pseudo_query/FILES_INVENTORY.md

(如果使用submodule)
  .gitmodules
  external/Deformable-DETR (新submodule)
```

### 提交信息模板

```bash
git add src/experiments/exp9_pseudo_query/DEPENDENCIES.md
git add src/experiments/exp9_pseudo_query/GITHUB_UPLOAD_GUIDE.md
git add src/experiments/exp9_pseudo_query/MISSING_FILES_SUMMARY.md
git add src/experiments/exp9_pseudo_query/UPLOAD_CHECKLIST.md
git add src/experiments/exp9_pseudo_query/scripts/setup_deformable_detr.sh
git add src/experiments/exp9_pseudo_query/README.md
git add src/experiments/exp9_pseudo_query/FILES_INVENTORY.md

# 如果使用submodule
git add .gitmodules external/Deformable-DETR

git commit -m "Exp9: Add comprehensive dependency documentation

Add detailed documentation for Deformable DETR dependency:
- DEPENDENCIES.md: Detailed explanation of dependency architecture
- GITHUB_UPLOAD_GUIDE.md: Complete guide for uploading to GitHub
- MISSING_FILES_SUMMARY.md: Explain why Deformable DETR was missing
- UPLOAD_CHECKLIST.md: Pre-upload checklist
- setup_deformable_detr.sh: Automated installation script

Update existing documentation:
- README.md: Add dependency warning and setup instructions
- FILES_INVENTORY.md: Add external dependency section

(Optional) Add Deformable DETR as git submodule

Fixes: Missing standard Deformable DETR implementation
Closes: #[issue-number]
"
```

---

## ⚠️ 常见问题检查

### Q1: 如果用户没有看到`external/Deformable-DETR/`?

- [ ] `README.md`中有清楚的警告和说明
- [ ] `DEPENDENCIES.md`有详细的获取步骤
- [ ] `setup_deformable_detr.sh`可以自动安装
- [ ] `verify_environment.sh`会提示缺少依赖

### Q2: 如果CUDA算子编译失败?

- [ ] `DEPENDENCIES.md`有故障排查建议
- [ ] 说明了CUDA和PyTorch版本要求
- [ ] 提供了官方文档链接

### Q3: 如果用户不知道用submodule还是手动安装?

- [ ] `README.md`推荐使用submodule
- [ ] `GITHUB_UPLOAD_GUIDE.md`对比了两种方法
- [ ] 两种方法都有清楚的步骤说明

---

## 📋 最终检查清单

在推送到GitHub之前,最后确认:

### 文件完整性

- [ ] 所有核心代码文件都在 (14个Python + 7个shell)
- [ ] 所有文档文件都在 (10个markdown)
- [ ] requirements.txt存在

### 依赖说明

- [ ] README.md有依赖警告
- [ ] DEPENDENCIES.md详细且准确
- [ ] 选择了submodule或手动安装方式
- [ ] 提供了自动安装脚本

### 测试验证

- [ ] 在新环境中测试过克隆流程
- [ ] CUDA算子可以编译
- [ ] 导入测试通过
- [ ] verify_environment.sh通过

### 文档质量

- [ ] 没有明显错误
- [ ] 链接都正常工作
- [ ] 格式正确美观
- [ ] 用户可以轻松理解

### Git状态

- [ ] 所有新文件已添加
- [ ] 修改的文件已添加
- [ ] commit信息清楚
- [ ] 准备好推送

---

## ✅ 签字确认

**检查完成日期**: ______

**检查者签名**: ______

**状态**:
- [ ] ✅ 所有检查通过，可以上传
- [ ] ⚠️ 部分问题待解决
- [ ] ❌ 需要重大修改

**备注**:

_____________________________________________

_____________________________________________

_____________________________________________

---

## 📚 参考资料

- [Git Submodule文档](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
- [Deformable DETR官方仓库](https://github.com/fundamentalvision/Deformable-DETR)
- [Markdown语法指南](https://www.markdownguide.org/)

---

**版本**: v1.0  
**创建日期**: 2026-01-29  
**维护者**: Exp9 Team
