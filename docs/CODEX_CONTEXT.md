# CODEX_CONTEXT.md

本文档是 Codex 参与本项目时必须优先阅读的长期上下文。后续所有项目文档默认使用中文撰写，除非文件名、命令、类名、函数名、配置键或论文术语需要保留英文。

## 当前架构决策

项目采用 **legacy full model 与 MTL-Lite 新主线硬边界隔离** 的架构。

核心原则：

```text
旧大模型：整体迁入 src/legacy/full_model/，只作为可运行历史快照
新主线模型：独立放在 src/models/mtl_lite.py，不继承旧模型
通用模块：保留在 src/models、src/metrics、src/datasets 等主线位置
诊断系统：独立规划为 src/diagnostics/，尽量离线运行
```

不要继续把 MTL-Lite 新逻辑加入旧的 `EndToEndDepressionModel`。旧模型复制到新路径后可按旧方式运行，因此暂时放弃对旧模型的额外修复、重构和 import 清理。legacy 只需要 README 说明边界、运行方式和维护策略。

## 当前研究主线

当前论文方向是 **MTL-Lite 轻量级多任务 BDI 预测模型**，面向 AVEC2014 风格的人脸视频抑郁程度预测。

目标流程：

```text
人脸视频帧 -> 视觉 backbone -> 时序编码器 -> 共享视频表征 -> BDI 回归头 + 有序严重程度分类头
```

论文主张：

有序严重程度预测为连续 BDI 回归提供结构化辅助监督；轻量、可解释、可消融的多任务结构比继续堆叠复杂模块更适合作为当前论文主线。

## 当前项目状态

- 项目已经在服务器环境通过 debug smoke，可以完整运行旧模型训练流程。
- 当前正在从旧大模型转向 MTL-Lite 新主线。
- 旧大模型相关代码应整体进入 `src/legacy/full_model/`，但不再作为主要维护对象。
- `scripts/train.py` 使用标准配置栈：
  `configs/avec2014_base.yaml` + 被 git 忽略的 `configs/local_paths.yaml` + 可选 override。
- `scripts/diagnose.py` 的 import 与 `--help` 检查已经通过。
- `configs/local_paths.yaml` 是私有路径文件，不允许修改或提交。
- `configs/pre/default_config.yaml` 仅作为历史兼容配置。

## 仓库位置

```text
C:\CodeXWorkSpace\PaperWork\STC
```

Codex 的默认 shell 可能打开在其他目录。运行命令前必须确认工作目录。

## 目标目录边界

### 新主线与通用模块

```text
src/
  config.py
  datasets/
  metrics/
  models/
    backbone_factory.py
    task_heads.py
    outputs.py
    mtl_lite.py
    temporal/
      encoders.py
      pooling.py
  diagnostics/
```

### 旧模型归档区

```text
src/legacy/full_model/
  README.md
  src/
    models/
      end_to_end.py
      mtl_blocks.py
      iresnet.py
    utils/
    losses/
    trainers/
  scripts/
  configs/
  tests/
```

## 模块边界

### 保留为通用模块

- `src/config.py`
- `src/datasets/dataset.py`
- `src/metrics/metrics.py`
- `src/models/backbone_factory.py`
- `src/models/task_heads.py`

### 放入 legacy 的旧模型专属能力

- `EndToEndDepressionModel`
- CGC / expert routing
- contrastive learning
- adaptive mask
- PCGrad
- uncertainty weighting
- LDS label weighting
- `loss_dist` 训练路径
- segmented weight save/load
- 旧版可视化 hooks
- 旧版 runner、脚本、配置和测试快照

legacy 只保留说明和历史快照。除非用户明确要求复现旧模型结果，否则不要继续修复 legacy 内部代码。

## MTL-Lite 主模型边界

MTL-Lite 只包含：

- 视觉 backbone
- 特征投影
- 时序编码器
- mask-aware 视频级池化
- BDI 回归头
- 有序严重程度分类头
- compact loss
- MAE/RMSE/CCC metrics

MTL-Lite 不包含：

- contrastive learning
- adaptive mask
- PCGrad
- CGC
- uncertainty weighting
- 旧版 segmented checkpoint 逻辑
- 训练过程内嵌复杂诊断绘图

## 诊断与可视化边界

主模型保持轻量，但论文项目需要丰富诊断工具。

诊断系统应进入：

```text
src/diagnostics/
```

推荐方向：

- 训练过程诊断：loss 曲线、metric 曲线、learning rate、prediction mean/std、梯度范数；
- 预测结果诊断：prediction-target scatter、residual histogram、BDI 区间误差、severity group 误差；
- 表征与归因诊断：embedding、t-SNE/UMAP、temporal weights、Grad-CAM、attention CAM、occlusion sensitivity；
- 论文报告导出：case study 图组、metrics 表格、LaTeX 表格。

诊断逻辑不得改变训练、验证或测试结果。

## Loss 与 Metric 约定

- 回归 loss 使用 normalized BDI：`BDI / max_score`。
- 报告 MAE/RMSE/CCC 时使用真实 BDI 尺度。
- CCC loss 与 CCC metric 在同一尺度输入时应满足：
  `ccc_loss = 1 - ccc_metric`。
- LDS 和 `loss_dist` 不属于 MTL-Lite 主贡献，只能作为消融项。

## 实施路线

1. 添加 `src/legacy/full_model/README.md`，说明旧模型是历史快照，不再主动修复。
2. 确保 legacy 下不提交 `local_paths.yaml`、日志、权重或 checkpoint。
3. 新增 `src/models/outputs.py`。
4. 新增 `src/models/temporal/pooling.py`。
5. 新增 `src/models/mtl_lite.py`。
6. 新增 MTL-Lite forward/backward/config 测试。
7. 设计新主线训练入口或调整现有 runner，使其面向 MTL-Lite。
8. 新增 regression-only 与 MTL-Lite baseline override。
9. 新增 `src/diagnostics/` 并逐步迁移可视化能力。
10. 在服务器运行 MTL-Lite debug smoke。
11. 对比 regression-only 与 MTL-Lite。
12. baseline 稳定后逐项加入 CCC、LDS、`loss_dist` 消融。

## 安全重构规则

- 修改前先阅读相关文件。
- 每次只处理一个重构目标。
- 不改变模型算法逻辑，除非用户明确确认。
- 不改变训练超参数，除非用户明确确认。
- 不修改 `configs/local_paths.yaml`。
- 不删除数据集、日志、checkpoint、权重或实验结果。
- legacy 只作为历史快照，不再主动修复 import 和 smoke，除非用户明确要求。
- 新模型不要继承旧模型。

## 验证命令

基础检查：

```bash
python -m compileall src scripts tests
```

legacy 归档检查：

```bash
git diff -- src/legacy/full_model/README.md
```

通用模块检查：

```bash
python -c "from src.models.backbone_factory import build_feature_backbone; print('backbone import ok')"
python -c "from src.models.task_heads import build_regression_task_head; print('task heads import ok')"
python -c "from src.metrics.metrics import ConcordanceCorrCoefMetric, concordance_ccc_loss; print('metrics import ok')"
```

新增 MTL-Lite 后：

```bash
python -c "from src.models.mtl_lite import MTLLiteDepressionModel; print('mtl lite import ok')"
python -m pytest tests/test_mtl_lite_forward.py tests/test_mtl_lite_loss_backward.py
```
