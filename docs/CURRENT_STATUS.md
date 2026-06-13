# CURRENT_STATUS.md

## 状态日期

2026-06-13

## 当前项目状态

项目已经通过服务器 debug smoke，可以完整运行旧版端到端训练流程。当前工作重点已经从“修复可运行性”转向“重构架构边界”，为 MTL-Lite 轻量级多任务抑郁预测模型建立干净主线。

## 当前架构方向

项目采用以下新架构：

```text
src/legacy/full_model/  # 旧大模型整体归档
src/models/             # MTL-Lite 新模型与通用模型模块
src/diagnostics/        # 独立诊断与可视化系统
```

关键决策：

- 旧大模型整体进入 legacy；
- 旧模型只作为历史快照保留，暂时不再投入额外修复或重构；
- 新模型不继承旧模型；
- 通用模块保留在主线位置；
- 新主线训练入口后续直接围绕 MTL-Lite 设计；
- 诊断与可视化系统从模型训练逻辑中解耦。

## 当前研究主线

论文主线是 **MTL-Lite BDI 预测模型**：

```text
人脸视频帧 -> 视觉 backbone -> 时序编码器 -> 共享视频表征 -> BDI 回归头 + 有序严重程度分类头
```

主任务：

- BDI 连续回归。

辅助任务：

- 有序抑郁严重程度分类。

非主线模块：

- contrastive learning
- adaptive mask
- PCGrad
- CGC / complex expert routing
- LDS
- `loss_dist`

这些模块后续只作为 legacy 能力、消融项或扩展项。

## 当前研究判断

当前视频帧序列已经由 OpenFace 裁剪和对齐，所用 OpenFace 版本可能不是最新版。因此，近期实验中出现的泛化问题不应只解释为普通背景过拟合，而应重点检查 OpenFace aligned face 中仍然存在的非抑郁捷径，包括身份纹理、裁剪边界、对齐伪影、姿态残留、追踪质量、光照和视频质量。

backbone 冻结与高层微调实验提示：仅调整 `FREEZE_BACKBONE` / `FINETUNE_LAST_N_BLOCKS` 不能充分解决泛化问题。下一阶段优先方向应转为 OpenFace 行为表征和诊断：

```text
OpenFace aligned RGB baseline
-> OpenFace 质量与捷径诊断
-> landmark/AU/pose/gaze 行为 baseline
-> RGB + behavior late fusion
-> 面部行为辅助任务的 MTL-Lite
```

相关论文线索和实验路线记录在 `docs/RESEARCH_NOTES.md`。非抑郁捷径验证框架的具体实施方案记录在 `docs/SHORTCUT_AUDIT_DESIGN.md`。

## 重要文件

- `docs/MTL_LITE_DESIGN.md`：新架构、模块边界、接口定义和实施路线。
- `docs/CODEX_CONTEXT.md`：Codex 长期上下文。
- `docs/RESEARCH_NOTES.md`：OpenFace 行为表征、相关论文和后续实验路线。
- `docs/SHORTCUT_AUDIT_DESIGN.md`：非抑郁捷径验证框架和实施方案。
- `docs/TODO.md`：分阶段任务计划。
- `docs/BUG_LOG.md`：问题与风险记录。
- `docs/EXPERIMENT_LOG.md`：已完成验证记录。
- `scripts/train.py`：当前训练入口。
- `src/config.py`：配置加载与合并。
- `src/datasets/dataset.py`：数据集与 datamodule。
- `src/models/backbone_factory.py`：通用 backbone factory。
- `src/models/task_heads.py`：通用任务头。
- `src/metrics/metrics.py`：通用指标。

## 已完成验证

- debug smoke 已在服务器环境完整通过。
- `scripts.diagnose` import 和 `--help` 检查已通过。
- 已新增 forward smoke 测试。
- 已新增 regression head backward 非零梯度测试。
- 已新增 loss/metric 一致性测试。
- 已新增 legacy full model README，说明旧模型快照边界。
- 已新增 MTL-Lite 输出 dataclass。
- 已新增 MTL-Lite mask-aware pooling 工具。
- 已新增 `MTLLiteDepressionModel` 骨架。
- 已新增 MTL-Lite forward、backward、config 测试。
- 已新增 MTL-Lite 新训练入口 `scripts/train_mtl_lite.py`。
- 已新增 MTL-Lite runner `src/trainers/mtl_lite_runner.py`。
- 已新增 regression-only、MTL-Lite baseline 和 MTL-Lite debug smoke 配置。
- 已新增 MTL-Lite 离线诊断入口 `scripts/diagnose_mtl_lite.py`。
- 已新增 `src/diagnostics/`，支持训练曲线、回归诊断、embedding、相关热力图、遮掩影响热力图、关键帧重要性热力图和模型关注区域热力图。
- MTL-Lite 已支持 `EXTRACT_FEATURE.FREEZE_BACKBONE` 和 `EXTRACT_FEATURE.FINETUNE_LAST_N_BLOCKS`，可通过配置冻结 backbone 或只微调最后若干 transformer blocks。
- 本地 Codex Python 缺少 `torch`、`pytorch_lightning` 和 `pytest`，MTL-Lite import/pytest 需在服务器训练环境验证。

## 当前优先级

1. 确保 legacy 中的 `local_paths.yaml`、日志、权重、checkpoint 不进入提交。
2. 记录当前 OpenFace 数据版本、生成命令、输出字段、裁剪尺寸和帧采样方式。
3. 优先实现 Shortcut Audit 最小可行版本：OpenFace quality summary、预测残差相关性、相关性热力图和 markdown 报告。
4. 若保留 OpenFace CSV，统计 `confidence`、`success`、pose、gaze、AU、landmark 抖动与 BDI/预测误差的相关性。
5. 运行输入消融：aligned RGB、grayscale、masked face、landmark heatmap、landmark/AU/pose only。
6. 建立 landmark-only 与 AU/pose/gaze-only temporal baseline。
7. 在行为 baseline 稳定后，再设计 RGB + behavior late fusion 和行为辅助任务 MTL。
8. 在服务器训练环境继续验证 MTL-Lite import、pytest、debug smoke 和离线诊断脚本。

## 当前风险

- legacy 不再作为主要维护对象，除非明确要求复现旧模型结果，否则不修复其内部 import。
- 当前工作区可能包含 legacy 迁移中的文件移动或复制，需要避免误删。
- 根目录和 legacy 中的 `local_paths.yaml` 都不应进入 git。
- 旧模型 debug smoke 通过不代表 MTL-Lite 已可运行。
- 诊断逻辑必须避免污染 validation/test。
- OpenFace aligned face 仍可能包含身份、裁剪伪影、姿态残留、追踪质量和视频质量等非抑郁捷径。
- 不同 OpenFace 版本生成的数据不应混用；升级 OpenFace 应作为独立数据版本和消融实验。
- Shortcut Audit 只能使用 validation/test 已有预测结果和 OpenFace 元数据做离线诊断，不得用 validation/test 统计量反向影响训练配置。
- 当前 BDI ordinal 辅助任务可能不足以强迫模型学习面部行为，后续需要考虑 AU、landmark motion、pose/gaze 等辅助任务。
- 多 GPU DDP metric logging 和 best checkpoint 行为仍需专门验证。
- bf16/mixed precision 稳定性仍需专门验证。

## 推荐验证命令

基础语法检查：

```bash
python -m compileall src scripts tests
```

MTL-Lite import 检查：

```bash
python -c "from src.models.outputs import MTLLiteOutput, MTLLiteLosses; print('outputs import ok')"
python -c "from src.models.temporal.pooling import masked_mean_pool; print('pooling import ok')"
python -c "from src.models.mtl_lite import MTLLiteDepressionModel; print('mtl lite import ok')"
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

MTL-Lite 测试：

```bash
python -m pytest tests/test_mtl_lite_forward.py tests/test_mtl_lite_loss_backward.py tests/test_mtl_lite_config.py
```

MTL-Lite debug smoke：

```bash
python scripts/train_mtl_lite.py --override configs/mtl_lite_debug_smoke.yaml
```

MTL-Lite baseline：

```bash
python scripts/train_mtl_lite.py --override configs/mtl_lite_baseline.yaml
```

MTL-Lite 离线诊断：

```bash
python scripts/diagnose_mtl_lite.py \
  --run-dir /path/to/LOG_DIR/mtl_lite_csv/version_0 \
  --ckpt best
```

诊断输出目录：

```text
/path/to/LOG_DIR/mtl_lite_csv/version_0/diagnostics/
```

Regression-only baseline：

```bash
python scripts/train_mtl_lite.py --override configs/regression_only_baseline.yaml
```

旧主线回归测试仅在需要复现 legacy 行为时运行：

```bash
python -m pytest tests/test_model_forward.py tests/test_loss_backward.py tests/test_loss_metric_consistency.py
```

debug smoke：

```bash
python scripts/train.py --override configs/debug_smoke.yaml
```

## 2026-06-13 Shortcut Audit 状态更新

最新一批 Shortcut Audit 输出文件显示 `Matched samples: 0`。原因是
`test_predictions.csv` 中的 `video_id` 形如 `203_2_Freeform_video_aligned`，
而 `openface_quality_summary.csv` 中的 `video_id` 形如
`203_2_Freeform_video`。因此当前 `shortcut_audit_report.md`、`shortcut_merged.csv`、
`shortcut_correlation.csv` 和 `shortcut_predictor_results.csv` 不能用于判断
shortcut risk。

手动去除 `_aligned` 后缀后可以匹配 100/100 个预测样本，说明问题主要是
Shortcut Audit 的 `video_id` 规范化缺口。下一步应先修复该对齐逻辑并重新运行
Shortcut Audit，再解释相关性热力图、shortcut-only predictor 和风险等级。

当前 `test_predictions.csv` 仍显示明显预测范围压缩：整体 MAE 约 8.91、RMSE 约
10.95、Pearson 约 0.35、CCC 约 0.29。minimal 组存在平均高估，severe 组存在
严重平均低估；后续分析应优先关注 severe 低估、Freeform/Northwind 同一 subject
预测一致性，以及 OpenFace pose/gaze/AU/quality 特征是否构成非抑郁捷径。
