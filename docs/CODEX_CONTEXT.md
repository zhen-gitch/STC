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

### 研究路线修订：OpenFace 行为表征优先

当前视频帧序列已经由 OpenFace 裁剪和对齐，所用 OpenFace 版本可能不是最新版。后续分析不应简单描述为“背景过拟合”，而应关注 OpenFace aligned face 中仍然存在的非抑郁捷径：

- 身份纹理：脸型、肤色、皱纹、眼镜、胡须、发际线；
- 裁剪和对齐伪影：黑边、插值痕迹、边界位置、脸部尺度残留；
- 姿态和追踪质量：head pose、gaze、`confidence`、`success`、landmark 抖动；
- 视频质量：模糊、压缩、光照和分辨率；
- subject-level bias：模型可能记住身份或采集条件，而不是稳定面部行为。

因此，下一阶段优先级从继续搜索 `FINETUNE_LAST_N_BLOCKS` 转向：

```text
OpenFace aligned RGB baseline
-> OpenFace 质量与捷径诊断
-> landmark/AU/pose/gaze 行为 baseline
-> RGB + behavior late fusion
-> 面部行为辅助任务的 MTL-Lite
```

相关研究和实验路线归档在 `docs/RESEARCH_NOTES.md`。后续 Codex 在设计实验或修改模型前，应优先阅读该文档。

非抑郁捷径验证框架归档在 `docs/SHORTCUT_AUDIT_DESIGN.md`。后续若用户要求实现 OpenFace 质量诊断、输入消融、shortcut-only baseline 或行为表征 baseline，应先阅读该文档，并优先采用离线诊断方式，避免改动训练主流程。

当前不建议直接升级 OpenFace 并覆盖已有数据。若使用 OpenFace 3.0、LibreFace 或其他工具，应作为独立数据版本和消融实验，不与当前 OpenFace 版本混用。

## 当前项目状态

- 项目已经在服务器环境通过 debug smoke，可以完整运行旧模型训练流程。
- 当前正在从旧大模型转向 MTL-Lite 新主线。
- 旧大模型相关代码应整体进入 `src/legacy/full_model/`，但不再作为主要维护对象。
- `scripts/train.py` 是旧端到端训练入口。
- `scripts/train_mtl_lite.py` 是 MTL-Lite 新主线训练入口。
- `scripts/diagnose_mtl_lite.py` 是 MTL-Lite 离线诊断与模型表征绘图入口。
- 新旧训练入口均使用标准配置栈：
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
- 表征与归因诊断：embedding、t-SNE/UMAP、相关系数热力图、遮掩影响热力图、关键帧重要性热力图、模型自身关注区域热力图；
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
7. 新增 `scripts/train_mtl_lite.py` 和 `src/trainers/mtl_lite_runner.py`，使新训练入口面向 MTL-Lite。
8. 新增 regression-only 与 MTL-Lite baseline override。
9. 新增 MTL-Lite debug smoke override。
10. 新增 `src/diagnostics/` 与 `scripts/diagnose_mtl_lite.py`，以离线方式生成训练曲线、回归诊断、embedding、相关热力图、遮掩影响热力图、关键帧重要性热力图和模型关注区域热力图。
11. 在服务器运行 MTL-Lite debug smoke。
12. 在服务器运行 MTL-Lite 离线诊断脚本。
13. 对比 regression-only 与 MTL-Lite。
14. 先完成 OpenFace 质量相关性、输入消融、landmark/AU/pose/gaze 行为 baseline。
15. 在行为表征 baseline 稳定后，再重新设计 MTL 辅助任务，并逐项加入 CCC、LDS、`loss_dist` 或任务权重消融。
16. Shortcut Audit 的最小可行版本应先实现 OpenFace quality summary、预测残差相关性、相关性热力图和 markdown 报告，再考虑输入消融与 behavior-only baseline。

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
python scripts/train_mtl_lite.py --override configs/mtl_lite_debug_smoke.yaml
python scripts/diagnose_mtl_lite.py --run-dir /path/to/LOG_DIR/mtl_lite_csv/version_0 --ckpt best
```

## Shortcut Audit 当前结论

`*_video` 与 `*_video_aligned` 的 `video_id` 规范化问题已经修复。最新有效
Shortcut Audit 输出中，`Matched samples: 100`，且样本均通过完整 `video_id`
匹配，不再依赖可能歧义的短 `subject_id`。

当前诊断结论：

- Shortcut Audit 风险等级为 medium；
- 最大绝对相关约为 `0.418`，来自 `AU07_c_mean` 与 `true_bdi`；
- OpenFace 的 AU、gaze、pose、quality 统计与 `true_bdi`、`pred_bdi`、
  `residual`、`abs_error` 均存在中等强度信号；
- 当前 RGB/MTL-Lite 模型仍有明显预测范围压缩，severe 组被系统性低估，
  minimal 组被系统性高估；
- `shortcut_predictor_results.csv` 的 in-sample predictor 结果过拟合风险很高，不能当作泛化性能；
- Shortcut Audit 已支持按 `subject_id` 分组的 shortcut-only predictor 交叉验证，并会额外输出
  `shortcut_predictor_grouped_cv.csv`；
- 下一步应在服务器复跑 Shortcut Audit，读取正式 grouped-CV 结果，并建立 AU/pose/gaze/landmark-only behavior baseline。

后续 Codex 在解释 Shortcut Audit 时，仍必须先确认 `Matched samples` 达到预期样本数；
若匹配数为 0 或明显偏低，只能判定为对齐失败，不能解释风险等级。

## P0 后续执行上下文

grouped-CV shortcut-only predictor 已经把当前风险判断从“可能由 OpenFace 统计特征完全解释”修正为“存在中等 shortcut 风险，但仍需要进一步定位 RGB/MTL-Lite 的失败模式”。后续 Codex 不应只根据 in-sample linear/ridge predictor 的高分继续扩大 shortcut 结论，也不应在缺乏诊断证据时继续围绕 backbone 解冻层数反复试验。

当前 P0 的合理推进顺序：

1. 先建立 `case_study_manifest`：把 severe 低估、minimal 高估、Freeform/Northwind 高差异和 low-error reference 固定为可复查样本集合。（已实现）
2. 再设计输入消融协议：使用 `rgb`、`grayscale`、`blur`、`center_mask`、`boundary_erased`、`landmark_heatmap` 判断模型是否依赖 RGB 纹理、身份线索、裁剪边界、黑边或非行为区域。（RGB dataset 变体已实现；`landmark_heatmap` 保留给 OpenFace landmark/behavior 路径）
3. 再设计 behavior-only baseline 接口：使用 AU、pose、gaze、landmark、landmark motion、confidence/success mask 建立不依赖 RGB 纹理的行为表征对照。（已实现独立入口）

实现这些任务时应保持以下边界：

- case study manifest 和输入消融设计优先作为离线诊断，不嵌入训练 forward；
- behavior-only baseline 应作为独立训练入口和独立配置，不污染 MTL-Lite 主线；
- 所有实验必须保持相同 split、seed、checkpoint 选择策略和核心指标，避免把数据划分或评估策略变化误解释为模型改进；
- 任何涉及训练超参数、dataset 输入变体实际接入或新模型训练入口的修改，都需要用户确认后再实施。

P0-2 当前实现位置：

- `src/diagnostics/case_studies.py`：构建并写出 case study manifest；
- `src/diagnostics/regression.py`：回归诊断输出 `case_study_manifest.csv` 和 `case_study_manifest.md`；
- `src/diagnostics/shortcut_audit.py`：Shortcut Audit 输出 `tables/case_study_manifest.csv` 和 `reports/case_study_manifest.md`。

P0-3 当前实现位置：

- `src/datasets/input_variants.py`：定义 `DATASET.INPUT_VARIANT` 的 RGB 输入变体；
- `src/datasets/dataset.py`：在 raw frames 进入 resize/normalize 之前应用输入变体；
- `configs/avec2014_base.yaml`：默认 `DATASET.INPUT_VARIANT: "rgb"`；
- `tests/test_input_variants.py`：验证输入变体行为。

当前支持 `rgb`、`grayscale`、`blur`、`center_mask`、`boundary_erased`。`landmark_heatmap` 需要真实 OpenFace landmark 坐标，不能由 RGB 帧伪造，因此当前在 RGB dataset 中作为保留值报错，后续应在 behavior baseline 或 OpenFace landmark dataset 中实现。

P0-4 当前实现位置：

- `src/datasets/openface_features.py`：读取 OpenFace CSV 并构建 AU/pose/gaze/landmark/quality 时序特征；
- `src/models/behavior_baseline.py`：OpenFace behavior-only GRU baseline；
- `src/trainers/behavior_baseline_runner.py`：独立 Lightning runner；
- `scripts/train_behavior_baseline.py`：独立训练入口；
- `configs/behavior_baseline.yaml`：behavior-only baseline override；
- `tests/test_openface_features.py` 与 `tests/test_behavior_baseline.py`：接口测试。

Behavior baseline 运行时需要通过本地配置或 override 提供 `DATASET.OPENFACE_ROOT`。该路径不应写入公共配置中的私有绝对路径，也不应修改 `configs/local_paths.yaml`，除非用户明确要求在本机维护该私有路径。

## 2026-06-14 Behavior baseline 后的新上下文

最新 behavior-only baseline 已完成训练，但结果显示强烈训练集拟合和较差泛化：test MAE 约 `9.93`，RMSE 约 `12.86`，CCC 约 `0.151`；best validation RMSE 约 `12.38`，而同 epoch train RMSE 约 `2.74`。后续 Codex 不应把 behavior-only train MAE/RMSE 很低解释为路线成功，也不应立即推进 RGB + behavior late fusion。

当前更合理的判断是：OpenFace CSV 中既包含有价值的面部行为线索，也包含身份、静态 landmark 几何、追踪质量、视频采集条件等容易被模型记忆的非抑郁信号。下一步必须先做 feature-group ablation 和 prediction-level 对齐比较，确定哪些特征组在 subject-level 泛化上真正有用。

后续优先顺序：

```text
behavior prediction export
-> behavior feature-group ablation
-> RGB vs behavior prediction-level comparison
-> stable behavior subset selection
-> RGB + behavior late fusion
-> behavior auxiliary MTL-Lite
```

实现时继续保持边界：

- 不修改 `configs/local_paths.yaml`；
- 不删除或覆盖任何日志、权重、checkpoint 或实验结果；
- 不在 test 结果之后反向调训练超参数；
- behavior baseline 默认作为独立入口，不污染 `scripts/train_mtl_lite.py`；
- late fusion 和辅助任务必须等待 behavior 特征子集稳定后再做。
