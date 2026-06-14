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

## 2026-06-13 Shortcut Audit 有效结果更新

`_aligned` 后缀导致的 `video_id` 未匹配问题已经修复并在最新表征数据中复跑确认：
`shortcut_audit_report.md` 显示 `Matched samples: 100`，且 `shortcut_merged.csv`
中的样本均通过完整 `video_id` 匹配。当前 Shortcut Audit 结果可以解释。

最新有效结果显示 shortcut 风险为 medium，最大绝对相关为
`AU07_c_mean` 与 `true_bdi` 的相关性，约 `r = 0.418`。OpenFace 的 AU、gaze、
pose 和质量统计特征与 `true_bdi`、`pred_bdi`、`residual`、`abs_error` 均存在
中等强度信号，说明 aligned face 中仍有非抑郁捷径风险。

当前 MTL-Lite/RGB 预测仍存在明显范围压缩：整体 MAE 约 8.91、RMSE 约 10.95、
Pearson 约 0.35、CCC 约 0.29，预测标准差约 6.13，明显低于真实 BDI 标准差
约 11.48。minimal 组平均高估约 +6.89，severe 组平均低估约 -16.50；后续分析
应优先关注 severe 低估、Freeform/Northwind 同一 subject 预测一致性，以及
behavior-only baseline 是否接近当前 RGB 模型。

`shortcut_predictor_results.csv` 中 in-sample linear/ridge 表现很强，但当前是
100 个样本、94 个特征的训练内诊断结果，不能视为泛化性能。Shortcut Audit 已支持
按 `subject_id` 分组的 shortcut-only predictor 交叉验证，并额外输出
`shortcut_predictor_grouped_cv.csv`。下一步应在服务器复跑 Shortcut Audit，用正式
grouped-CV 结果判断 shortcut-only 特征是否接近当前 RGB 模型。

## 2026-06-13 P0 剩余任务设计

最新 grouped-CV shortcut-only predictor 结果显示，OpenFace shortcut 特征虽然与标签、预测和误差存在中等相关，但不能单独接近当前 RGB/MTL-Lite 模型的测试表现。因此当前风险判断应保持为 medium：需要继续诊断捷径，但不应把后续工作简化为“OpenFace 统计特征已经解释了模型”。接下来的 P0 应围绕模型主要失败模式展开：预测范围压缩、severe 系统性低估、minimal 系统性高估，以及 Freeform/Northwind 同一 subject 预测不一致。

当前 P0 顺序：

1. P0-2：建立 high-error / task-inconsistency case study manifest。固定 severe 低估、minimal 高估、任务间高差异和 low-error reference 样本集合，作为后续 attention、occlusion、keyframe、aligned face 图组的统一入口。
2. P0-3：设计输入消融协议。比较 `rgb`、`grayscale`、`blur`、`center_mask`、`boundary_erased`、`landmark_heatmap` 等输入变体，用于判断模型是否依赖身份纹理、裁剪伪影、边界黑边或非行为区域。
3. P0-4：设计 AU/pose/gaze/landmark-only behavior baseline 接口。建立不依赖 RGB 纹理的行为表征对照，后续再决定是否进入 RGB + behavior late fusion 和行为辅助任务 MTL。

这些任务当前均属于诊断与接口设计，不应修改 `configs/local_paths.yaml`，不应删除或覆盖既有实验结果，也不应在没有明确确认前改变训练超参数。

## 2026-06-14 P0-2 Case Study Manifest 实现

P0-2 已作为离线诊断能力落地。新增 `src/diagnostics/case_studies.py`，用于从已有 prediction/merged rows 中选择以下样本：

- `severe_underestimate`：severe 真实 BDI 高但预测明显偏低；
- `minimal_overestimate`：minimal 真实 BDI 低但预测明显偏高；
- `task_inconsistency`：同一 subject 的 Freeform/Northwind 等任务视频预测差异大；
- `low_error_reference`：低误差对照样本。

该能力已接入两个离线诊断入口：

- `plot_regression_diagnostics()` 会在 regression 诊断目录输出 `case_study_manifest.csv` 和 `case_study_manifest.md`；
- `run_shortcut_audit()` 会在 Shortcut Audit 的 `tables/` 与 `reports/` 下输出同名 manifest 文件。

该实现不改变训练 forward、不改变模型结构、不改变训练超参数，也不依赖 `configs/local_paths.yaml`。后续 P0-3 输入消融和 P0-4 behavior-only baseline 应以该 manifest 固定的 case 集合作为优先复查对象。

## 2026-06-14 P0-3 Input Ablation Variant 接入

P0-3 已以可选 dataset input variant 的方式接入主线数据集，默认行为保持 `rgb`，不会改变既有训练结果。新增 `src/datasets/input_variants.py`，并在 `AVECDataset` 中通过 `DATASET.INPUT_VARIANT` 调用。

当前支持的 RGB 帧输入变体：

- `rgb`：当前 OpenFace aligned RGB baseline；
- `grayscale`：弱化颜色和肤色线索；
- `blur`：弱化身份纹理和细粒度皮肤细节；
- `center_mask`：保留面部中央区域，弱化边界和外围区域；
- `boundary_erased`：弱化裁剪边界、黑边、头发、衣物残留等外围伪影。

`landmark_heatmap` 当前被显式保留给 OpenFace landmark/behavior baseline 路径。由于它需要真实 landmark 坐标，RGB dataset 中不会伪造该输入；如果误配为 `landmark_heatmap`，会直接报错。

默认配置已在 `configs/avec2014_base.yaml` 中加入：

```yaml
DATASET:
  INPUT_VARIANT: "rgb"
```

后续运行输入消融时，应只在 override 中修改该字段，并保持 split、seed、训练入口、checkpoint 选择策略和指标一致。

## 2026-06-14 P0-4 Behavior-only Baseline 接口实现

P0-4 已作为独立 OpenFace 行为表征 baseline 落地。该路线不使用 RGB 帧，不调用 MTL-Lite visual backbone，目标是用结构化行为变量判断当前 RGB 模型是否真正捕捉到可泛化的面部行为动态。

新增模块：

- `src/datasets/openface_features.py`：读取 OpenFace CSV，按 split 匹配视频，构建 AU/pose/gaze/landmark/quality 时序特征，并只用训练集统计量做标准化；
- `src/models/behavior_baseline.py`：OpenFace 特征投影 + GRU + mask-aware pooling + BDI 回归头，可选 ordinal 辅助头；
- `src/trainers/behavior_baseline_runner.py`：独立 Lightning runner；
- `scripts/train_behavior_baseline.py`：独立训练入口；
- `configs/behavior_baseline.yaml`：behavior-only baseline override；
- `tests/test_openface_features.py` 与 `tests/test_behavior_baseline.py`：接口测试。

使用要求：

```bash
python scripts/train_behavior_baseline.py \
  --override configs/behavior_baseline.yaml \
  --override configs/your_openface_paths.yaml
```

其中 `configs/your_openface_paths.yaml` 至少需要提供：

```yaml
DATASET:
  OPENFACE_ROOT: "/path/to/openface_csv_root"
```

该实现不修改 `configs/local_paths.yaml`，不删除或覆盖任何实验结果，也不改变 `scripts/train_mtl_lite.py` 的行为。后续需要在服务器使用真实 OpenFace CSV 运行 debug smoke，并与 RGB regression-only baseline 保持相同 split、seed 和指标进行比较。
