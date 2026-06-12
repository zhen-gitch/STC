# CODEX_PROMPT_TEMPLATES.md

本文件用于保存本项目常用 Codex 提示词模板，避免每次重复输入上下文。

项目根目录：

```bash
/usr/local/conda/zhen/stc
```

推荐启动方式：

```bash
cd /usr/local/conda/zhen/stc
git checkout dev
git pull
git status
codex
```

或：

```bash
codex --cd /usr/local/conda/zhen/stc
```

---

# 0. 通用使用原则

每次使用 Codex 前，先确认：

```bash
pwd
git branch --show-current
git status
```

默认要求 Codex：

1. 先阅读项目文档，不要直接修改文件。
2. 先给计划，再执行修改。
3. 每次只完成一个明确任务。
4. 不要同时做结构迁移、模型逻辑修改和实验配置修改。
5. 不要提交 `configs/local_paths.yaml`、`logs/`、`weights/`、`*.pth`、`*.pt`、`*.ckpt`。
6. 修改后必须说明改了哪些文件、为什么改、如何验证。
7. 大改前先建议 commit 或创建分支。

---

# 1. 新会话启动模板

适用于每次打开 Codex 后的第一条消息。

```text
请先阅读以下文件：

- AGENTS.md
- README.md
- docs/CODEX_CONTEXT.md
- docs/CURRENT_STATUS.md
- docs/TODO.md
- docs/BUG_LOG.md

当前项目是基于 AVEC2014 面部视频的抑郁程度 BDI 预测项目，使用 PyTorch Lightning、timm、DeiT/ViT backbone、时序建模、多任务学习和可视化诊断。

请不要修改任何文件。请先完成以下事情：

1. 总结当前项目结构。
2. 总结当前主要训练入口。
3. 总结当前配置组织方式。
4. 总结当前模型主流程。
5. 检查是否有明显的结构问题或风险。
6. 给出下一步建议。

输出时请分成：
- Project summary
- Important files
- Current risks
- Recommended next steps
```

---

# 2. 项目结构检查模板

适用于目录迁移后，让 Codex 检查结构是否合理。

```text
请检查当前仓库结构是否适合长期论文项目开发。

要求：

1. 不要修改文件。
2. 检查根目录、configs、src、scripts、docs、papers、tests 的组织是否合理。
3. 检查是否有应该移动的文件。
4. 检查是否有应该加入 .gitignore 的文件或目录。
5. 检查是否有空目录需要 README.md 或 .gitkeep。
6. 检查是否存在路径命名不一致的问题，例如 data/datasets、trainer/training、utils 滥用等。
7. 给出分级建议：
   - P0：必须立即修复
   - P1：强烈建议修复
   - P2：后续优化

不要进行任何实际修改。只输出检查报告和建议命令。
```

---

# 3. 安全重构模板

适用于让 Codex 修改目录、import、模块名称等工程结构。

```text
我要进行一次安全重构。请严格遵守：

1. 先阅读相关文件。
2. 先给出修改计划，不要立即修改。
3. 每次只处理一个重构目标。
4. 不要改变模型算法逻辑。
5. 不要改变训练超参数。
6. 不要修改 local_paths.yaml。
7. 不要删除任何实验结果、日志或权重文件。
8. 修改后运行或建议运行：
   - python -m compileall src scripts
   - 必要的 import check

本次重构目标：

【在这里填写目标，例如：把 src/models/adaptive_mask.py、decomposition.py、mtl_blocks.py 移动到 src/models/temporal_blocks/，并修复所有 import。】

请先输出：
1. 将修改哪些文件。
2. 为什么要这样改。
3. 是否可能影响训练。
4. 具体执行步骤。
5. 验证命令。

等我确认后再修改。
```

---

# 4. 配置系统整理模板

适用于整理 YAML 配置。

```text
请检查并整理当前配置系统。

项目希望采用三层配置：

1. configs/common/avec2014_base.yaml：公共基础配置。
2. configs/local_paths.yaml：本机或服务器私有路径，不提交 git。
3. papers/paper01_avec2014_depression_baseline/configs/*.yaml：当前论文的实验覆盖配置。

要求：

1. 不要把路径、权重、本地日志目录写入公共配置。
2. 不要提交 configs/local_paths.yaml。
3. 实验配置只写覆盖项，不复制完整大文件。
4. 保证最终训练时可以通过 OmegaConf.merge 合并配置。
5. 训练开始时应保存最终合并后的 model_config.yaml 到日志目录。
6. 不要修改模型代码，除非配置读取方式必须同步调整。

请先检查当前 configs 和 papers 下的配置文件，给出：
- 当前问题
- 推荐结构
- 需要移动或新建的文件
- 每个配置文件应该包含哪些字段
- 修改后的训练命令
```

---

# 5. 训练报错诊断模板

适用于 OOM、DDP、loss NaN、导入错误等。

```text
下面是训练报错日志。请帮我诊断。

要求：

1. 先判断是配置问题、数据问题、模型问题、DDP 问题、显存问题，还是 import/path 问题。
2. 不要立刻修改代码。
3. 优先给出最小可验证修复。
4. 如果需要修改代码，请明确指出文件、函数、替换片段。
5. 如果涉及 OOM，请优先检查：
   - BATCH_SIZE
   - CHUNK_SIZE
   - MAX_SEQ_LEN
   - 是否解冻 backbone
   - 是否启用多视图 contrastive
   - 是否重复 backbone forward
   - 是否启用 grad checkpointing
6. 如果涉及 DDP，请检查：
   - strategy
   - sync_dist
   - rank0 保存
   - dataloader sampler
   - 多进程下载权重风险

错误日志如下：

【粘贴日志】
```

---

# 6. 模型代码审查模板

适用于审查 `end_to_end.py`、loss、metrics、runner。

```text
请审查以下模型相关代码，但不要修改文件：

重点检查：

1. 是否存在重复创建 backbone。
2. TIMM_PRETRAINED 和 MODEL_WEIGHT_PATH 是否正确传入 build_feature_backbone。
3. 冻结和解冻 backbone 的逻辑是否正确。
4. 是否只解冻 DeiT/ViT 最后 N 个 blocks。
5. grad checkpointing 是否只在需要时启用。
6. 多视图 contrastive 是否导致 orig/v1/v2 三次 backbone forward。
7. loss 权重是否合理，尤其 CON_LOSS_WEIGHT、CLS_LOSS_WEIGHT。
8. DDP 下 self.log 是否需要 sync_dist。
9. best/latest segmented weights 是否只在 global_zero 保存。
10. validation/test 可视化是否可能只使用 rank0 子集。
11. 是否有 no_grad 使用错误导致 backbone 无法训练。
12. 是否有 detach 使用错误导致 loss 无法回传。

请输出：
- P0 bug
- P1 risk
- P2 cleanup
- 推荐修改顺序
- 每项对应的文件和函数
```

---

# 7. 实验设计模板

适用于规划 paper01 的实验和消融。

```text
请基于当前项目，为 paper01_avec2014_depression_baseline 设计一组稳健实验。

目标：
基于 AVEC2014 面部视频进行 BDI 抑郁严重程度预测。

当前模型大致为：
video -> visual backbone -> projection -> temporal filter/decomposition -> CGC -> regression/classification/contrastive heads。

请设计：

1. 主实验 baseline。
2. backbone 对比实验。
3. 是否微调 backbone 的实验。
4. contrastive loss 消融。
5. CORAL/classification 辅助任务消融。
6. temporal decomposition 消融。
7. 可视化诊断输出。
8. 每个实验对应的 YAML 覆盖配置。
9. 推荐运行顺序。
10. 如何记录到 EXPERIMENT_LOG.md。

要求：
- 优先保证第一篇论文可完成。
- 不要设计过多不必要实验。
- 明确每个实验回答什么研究问题。
- 给出实验表格草案。
```

---

# 8. 论文写作辅助模板

适用于写 method、experiment、related work 草稿。

```text
请帮助我为当前论文撰写或修改以下部分：

论文项目：
paper01_avec2014_depression_baseline

任务：
AVEC2014 面部视频抑郁程度 BDI 预测。

当前方法：
轻量视觉 backbone + 时序建模 + 多任务监督 + 可视化诊断。

请先阅读：
- papers/paper01_avec2014_depression_baseline/manuscript/outline.md
- papers/paper01_avec2014_depression_baseline/manuscript/method.md
- papers/paper01_avec2014_depression_baseline/experiments/EXPERIMENT_LOG.md
- docs/CODEX_CONTEXT.md

本次写作目标：

【填写：例如，重写 method 中 temporal decomposition 部分】

要求：
1. 不要夸大实验结论。
2. 不要编造结果。
3. 如果结果缺失，请用 TODO 标注。
4. 用学术但清晰的英文。
5. 保持与当前代码实现一致。
6. 输出前说明依据了哪些文件。
```

---

# 9. Git 提交前检查模板

适用于让 Codex 帮忙检查是否可以 commit/push。

```text
请帮我做提交前检查，不要修改文件。

请检查：

1. git status
2. git diff --stat
3. 是否误提交：
   - configs/local_paths.yaml
   - logs/
   - weights/
   - *.pth
   - *.pt
   - *.ckpt
   - __pycache__/
   - *.pyc
4. 是否有大文件。
5. 是否需要运行：
   - python -m compileall src scripts
   - import check
6. 根据变更内容建议 commit message。

请输出：
- 是否可以提交
- 风险文件列表
- 推荐 git add 命令
- 推荐 commit message
- 推送命令
```

---

# 10. Codex 修改后复查模板

适用于 Codex 完成修改后让它自查。

```text
请复查你刚才的修改。

要求：

1. 列出所有修改过的文件。
2. 说明每个文件修改了什么。
3. 说明是否改变了模型算法逻辑。
4. 说明是否改变了训练配置默认值。
5. 检查是否破坏 import。
6. 检查是否需要更新 README、AGENTS.md、docs/CURRENT_STATUS.md 或 TODO.md。
7. 给出建议运行的验证命令。
8. 如果还有风险，请明确列出。

不要继续修改文件，除非我确认。
```

---

# 11. 当前项目默认上下文块

当 Codex 对项目背景不清楚时，可复制下面这段。

```text
项目背景：

这是一个基于 AVEC2014 面部视频的抑郁程度评估项目，目标是预测 BDI 分数。项目使用 PyTorch Lightning，主要模型为 EndToEndDepressionModel。

当前核心流程：
video frames -> soft spatial face gate -> visual backbone from timm/iresnet -> projection -> temporal filtering/decomposition -> masked pooling -> CGC multi-task expert layer -> regression/classification/contrastive heads。

当前重点：
1. 第一篇论文 paper01_avec2014_depression_baseline 优先。
2. 代码结构要支持长期复用，但不要提前展开 paper02/paper03。
3. 配置采用 base + local_paths + paper-specific experiment overrides。
4. 正式训练优先使用本地预训练权重，避免 DDP 多进程联网下载。
5. local_paths.yaml、logs、weights、checkpoints 不允许提交。
6. 当前任务主目标是稳定训练、控制过拟合、规范实验记录。
```

---

# 12. 推荐组合用法

## 新会话

先用：

```text
模板 1：新会话启动模板
```

## 做结构整理

先用：

```text
模板 2：项目结构检查模板
```

然后用：

```text
模板 3：安全重构模板
```

## 调训练报错

使用：

```text
模板 5：训练报错诊断模板
```

## 做实验规划

使用：

```text
模板 7：实验设计模板
```

## 提交前

使用：

```text
模板 9：Git 提交前检查模板
```

---

# 13. 每次任务的最小提示词格式

如果不想复制长模板，可以用这个最小格式：

```text
请在当前仓库中完成以下任务：

任务：
【填写任务】

约束：
1. 先阅读 AGENTS.md、README.md、docs/CURRENT_STATUS.md。
2. 先给计划，不要直接修改。
3. 不要修改 local_paths.yaml、logs、weights、checkpoint。
4. 不要同时做无关重构。
5. 修改后说明改了哪些文件，并给出验证命令。

请开始前先总结你理解的任务和风险。
```
