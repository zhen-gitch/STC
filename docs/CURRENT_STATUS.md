# CURRENT_STATUS.md

## 状态日期

2026-06-15

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

最新 RGB 输入消融进一步说明输入侧非行为线索值得研究。`center_mask` 在测试集上明显优于原始 `rgb`，而 `grayscale` 和 `blur` 变差；样例帧也显示 OpenFace aligned face 中存在黑色填充、硬裁剪边界、遮挡物黑块和对齐残留。这些现象支持“input artifact 是风险入口”的判断，但不能把 RGB 过拟合单独归因于黑边或黑填充。后续应同时审计 split/subject、时序采样、训练曲线泛化缺口、OpenFace 对齐几何、身份静态外观、姿态/追踪质量、任务语境和 severity calibration。

backbone 冻结与高层微调实验提示：仅调整 `FREEZE_BACKBONE` / `FINETUNE_LAST_N_BLOCKS` 不能充分解决泛化问题。下一阶段优先方向应转为 RGB 过拟合多因素审计：

```text
split integrity audit
-> temporal sampling audit / ablation
-> training overfit curve summary
-> alignment geometry audit
-> embedding identity retrieval
-> severity calibration verification
-> task inconsistency mixed-factor audit
```

权威路线记录在 `docs/RGB_OVERFITTING_AUDIT_PLAN.md`。相关论文线索记录在 `docs/RESEARCH_NOTES.md`。非抑郁捷径验证框架的具体实施方案记录在 `docs/SHORTCUT_AUDIT_DESIGN.md`。
系统机制路线图已整理到 `docs/OVERFITTING_MECHANISM_ROADMAP.md`，用于把现有黑边、geometry、temporal、identity、局部遮挡、calibration 和 behavior baseline 结果组织成统一研究路线。

## 重要文件

- `docs/MTL_LITE_DESIGN.md`：新架构、模块边界、接口定义和实施路线。
- `docs/RGB_OVERFITTING_AUDIT_PLAN.md`：RGB 过拟合多因素审计权威路线。
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

## 2026-06-14 Behavior-only baseline 结果与优先级重评估

最新 behavior-only baseline 已能完整训练并输出 `behavior_metrics.csv`。该实验使用 OpenFace 结构化特征而非 RGB 帧，当前结果显示其训练集拟合很强，但泛化仍明显不足：

- test MAE 约 `9.93`，RMSE 约 `12.86`，CCC 约 `0.151`；
- best validation RMSE 出现在 epoch 65，val MAE 约 `9.94`，val RMSE 约 `12.38`，val CCC 约 `0.324`；
- 同一 epoch 的 train MAE 约 `2.17`，train RMSE 约 `2.74`，train CCC 约 `0.975`；
- 最后 epoch 的 train/val RMSE 差距扩大到约 `10.68`。

当前判读：

- behavior-only baseline 暂时不能替代 RGB/MTL-Lite，也不应直接进入 late fusion；
- 该结果更像是 OpenFace 行为特征中存在可记忆的 subject/static geometry 信号，但可泛化行为动态仍没有被稳定提取；
- 原始 landmark 坐标、静态几何、身份相关形状和视频级采集差异可能是主要过拟合来源；
- 当前阶段不应继续把工作重点放在 backbone 解冻层数搜索，也不应因为 train MAE 很低就认为行为表征路线已经成功。

重新评估后的当前优先级：

### P0：必须立即处理

1. 为 behavior baseline 导出 val/test prediction CSV，字段尽量与 RGB/MTL-Lite `test_predictions.csv` 对齐，至少包含 `video_id`、`subject_id`、`task_name`、`true_bdi`、`pred_bdi`、`residual`、`abs_error`。
2. 对 behavior baseline 做特征组消融：quality-only、AU-only、pose+gaze-only、raw-landmark-only、landmark-delta-only、AU+landmark-delta、all-without-raw-landmarks。
3. 在相同 split、seed、metric 和 checkpoint 策略下，对齐比较 RGB/MTL-Lite 与 behavior-only 的整体指标、severity group 误差、Freeform/Northwind 一致性和 case overlap。
4. 记录 OpenFace CSV 匹配数、可用字段、特征维度、缺失字段和标准化统计来源，确保 behavior baseline 不发生 split 泄漏。

### P1：强烈建议处理

1. 在完成特征组消融后，再决定是否默认移除 raw landmark 坐标，优先保留 AU、landmark motion、pose/gaze motion 和 quality mask 等更接近行为动态的特征。
2. 降低 behavior baseline 容量并增强正则化，例如更小 hidden dim、单向 GRU、dropout、weight decay、早停和更短最大 epoch。
3. 建立 behavior prediction 诊断报告，复用 regression diagnostics 与 case study manifest。

### P2：后续优化

1. 只有当某个 behavior 特征子集在验证/测试上稳定后，再尝试 RGB + behavior late fusion。
2. 只有当行为子任务本身稳定后，再将 AU、landmark motion、pose/gaze 等作为 MTL-Lite 辅助任务。
3. 动态任务权重、GradNorm、PCGrad、LDS、`loss_dist` 仍应保持为后续消融项，而不是当前主线。

### P0 执行进展

- 已实现 behavior baseline 的 val/test prediction CSV 导出。
- 导出目录为 `behavior_baseline_csv/version_*/diagnostics/behavior/`。
- `val_predictions.csv` 与 `test_predictions.csv` 已包含 `video_id`、`subject_id`、`task_name`、`true_bdi`、`pred_bdi`、`residual`、`abs_error`、`severity_group`。
- 该导出发生在 best-checkpoint test evaluation 之后，不改变训练 forward、loss、metric、训练超参数或 checkpoint 选择策略。
- 已新增 `BEHAVIOR_FEATURES.FEATURE_SET` 命名特征组入口，默认值为 `custom`，不改变既有 behavior baseline。
- 当前支持 `quality_only`、`au_only`、`pose_gaze_only`、`raw_landmark_only`、`landmark_delta_only`、`au_landmark_delta`、`all_without_raw_landmarks`，用于后续服务器端批量消融。
- 已新增 `scripts/compare_behavior_predictions.py`，可将 RGB/MTL-Lite prediction CSV 与 behavior-only prediction CSV 对齐比较。
- 比较工具输出 `rgb_behavior_prediction_comparison.csv` 和 `rgb_behavior_prediction_summary.csv`，用于检查整体指标、severity 分组和逐样本谁更好。

## 2026-06-15 RGB 黑填充伪迹方向更新

已完成第一轮 RGB 输入消融复盘。各变体训练配置保持同一 split、seed、backbone、冻结策略、时序长度和主要训练入口，测试结果的核心结论如下：

- `center_mask` 当前最好：MAE 约 `7.94`，RMSE 约 `10.16`，Pearson 约 `0.51`，CCC 约 `0.48`；
- 原始 `rgb`：MAE 约 `8.91`，RMSE 约 `10.95`，Pearson 约 `0.35`，CCC 约 `0.29`，存在明显 prediction compression；
- `boundary_erased` 接近或略优于 `rgb`，但不如 `center_mask` 稳定；
- `blur` 和 `grayscale` 明显变差，说明“颜色”或“细粒度身份纹理”不是当前唯一主因；
- severe 低估仍未根治，说明黑边/外围伪迹可能解释一部分泛化问题，但还不能解释全部失败模式。

当前研究判断从“继续叠加 behavior / late fusion 任务”调整为“优先解释 RGB 输入模型为什么过拟合”。更具体的假设是：

```text
OpenFace aligned RGB
-> 纯黑填充 / 黑色遮挡块 / 硬裁剪边界 / 对齐残留
-> ViT 学到不可泛化的边界与像素突变捷径
-> test prediction compression、case-level 错误和 task inconsistency
```

已为下一轮黑伪迹消融接入以下输入变体：

- `black_to_gray`：将近黑区域替换为中性灰；
- `black_to_mean`：将近黑区域替换为当前帧非黑区域均值；
- `black_to_blur`：将近黑区域替换为模糊背景估计；
- `soft_center_mask`：使用软椭圆 mask 平滑边界，而不是制造新的硬边界；
- `inner_crop_resize`：裁掉外围黑边后再 resize，用于验证边界区域是否是主要捷径。

已新增黑伪迹离线审计入口：

```bash
python scripts/audit_black_artifacts.py \
  --predictions logs/rgb/test_predictions.csv \
  --image-root /path/to/aligned/frame/root \
  --output-dir logs/rgb/diagnostics/black_artifacts \
  --sample-step 10
```

该脚本会输出每个视频的黑像素比例、边界黑像素比例、中心黑像素比例、黑边界边缘强度、帧间黑像素变化，并与 `true_bdi`、`pred_bdi`、`residual`、`abs_error` 做相关性分析。

下一阶段优先级：

1. 运行更精确的边界黑区消融，优先只处理与图像边界连通的黑色区域，而不是替换全部中心近黑像素。
2. 对 `rgb`、`center_mask`、`black_to_gray`、`soft_center_mask` 和新边界黑区变体生成统一 summary，比较整体指标、severity bias、prediction std 和 Freeform/Northwind 一致性。
3. 对高黑边高误差、高黑边低误差、低黑边高误差三类 case 生成 aligned frame、attention、occlusion、keyframe 图组。
4. 将 severe 低估继续作为独立问题保留，不能把它完全归因于黑边伪迹。
5. 在完成上述证据前，暂不优先推进 RGB + behavior late fusion 或新的复杂辅助任务。

## 2026-06-15 黑伪迹审计后的实验安排

黑伪迹审计已完整匹配 100 个测试视频，`Missing videos = 0`，结果可以解释。审计显示 aligned frame 中黑区非常普遍：整体黑像素均值约 `0.24`，边界黑区均值约 `0.44`。但最大绝对相关只有约 `0.207`，说明黑区不是单独决定 BDI 或预测误差的强变量。

当前更精确的判断：

- `black_border_ratio_mean` 比 `black_center_ratio_mean` 更适合作为 OpenFace 对齐伪迹指标；
- 中心黑像素语义混杂，可能来自鼻孔、自然阴影、胡须、嘴角、麦克风或其他真实遮挡，不能直接当作伪迹；
- 高边界黑区四分位的平均误差明显高于低边界黑区四分位，约 `12.29` vs `7.45`；
- 在 moderate/severe 组内，边界黑区越多，预测越容易偏低，但样本量较小，应作为风险线索而非定论；
- `black_to_gray` 优于原始 `rgb`，但不如 `center_mask`，说明黑像素是因素之一，外围非行为区域、裁剪形状、脸部轮廓和姿态/尺度残留也可能共同作用。

下一轮实验不应继续粗暴替换全部黑像素。推荐新增三类更精确输入变体：

- `border_black_to_gray`：只替换与图像边界连通的近黑区域；
- `border_black_feather`：对边界连通黑区做软过渡，降低硬边界；
- `center_mask_black_to_gray`：在当前最优 `center_mask` 基础上，仅对残留边界连通黑区做中性化，验证二者是否互补。

上述三个变体已接入 `src/datasets/input_variants.py`，对应配置已新增到 `configs/input_ablation/`。本地 compile 验证通过；由于本地 Python 缺少 `pytest` 和 `torch`，还需要在服务器运行聚焦 pytest 与三组训练消融。

case study 优先集合：

- 高黑边高误差：`359_1`、`315_2`、`245_1`；
- 高黑边低误差：`247_3`；
- 低黑边高误差：`237_1`；
- `black_to_gray` 改善明显：`250_1`、`344_2`、`242_1`；
- `black_to_gray` 恶化明显：`206_2`、`226_2`、`210_2`。

## 2026-06-15 RGB 过拟合多因素判断

当前不应把 RGB 过拟合单独归因于黑边。更合理的论文判断是：RGB 模型可能同时利用身份静态外观、OpenFace 对齐几何、边界填充、姿态/追踪质量、视频长度/采样、任务语境差异和标签分布造成的 prediction compression。

权威路线文档：`docs/RGB_OVERFITTING_AUDIT_PLAN.md`。后续关于 RGB 过拟合机制、论文实验顺序和 P0/P1 审计优先级的判断，以该文档为准。当前文档只记录状态摘要。

下一阶段推荐排查顺序已更新为：

```text
split integrity audit
-> temporal sampling audit / ablation
-> training overfit curve summary
-> alignment geometry audit
-> embedding identity retrieval
-> severity calibration verification
-> task inconsistency mixed-factor audit
```

黑边/黑填充方向保留为 input artifact 子证据，而不是总解释。其中 `frame_count` / `sampled_frame_count` 在黑伪迹审计中与 `pred_bdi` 的相关性最高，约 `r = -0.207`，因此时序采样、视频长度和 padding 需要优先验证。severe 低估和 minimal 高估继续作为独立校准问题跟踪，不能仅靠输入伪迹解释。

P0 temporal sampling audit 已落地：

```text
src/diagnostics/temporal_sampling.py
scripts/audit_temporal_sampling.py
tests/test_temporal_sampling_audit.py
```

服务器运行示例：

```bash
python scripts/audit_temporal_sampling.py \
  --predictions logs/rgb/rgb_test_predictions.csv \
  --image-root /path/to/AVEC2014/face_images \
  --output-dir logs/rgb/diagnostics/temporal_sampling \
  --sample-step 10 \
  --max-seq-len 2000
```

注意：`--sample-step` 和 `--max-seq-len` 应与对应实验的 `resolved_config.yaml` 保持一致。

P0 temporal sampling ablation 已接入：

```text
src/datasets/temporal_sampling.py
PROCESS_TEMPORAL.SAMPLING_STRATEGY
configs/temporal_sampling/
```

默认 `stride_head` 保持旧行为不变。服务器训练示例：

```bash
python scripts/train_mtl_lite.py \
  --override configs/regression_only_baseline.yaml \
  --override configs/input_ablation/rgb.yaml \
  --override configs/temporal_sampling/uniform_256.yaml

python scripts/train_mtl_lite.py \
  --override configs/regression_only_baseline.yaml \
  --override configs/input_ablation/rgb.yaml \
  --override configs/temporal_sampling/uniform_512.yaml

python scripts/train_mtl_lite.py \
  --override configs/regression_only_baseline.yaml \
  --override configs/input_ablation/rgb.yaml \
  --override configs/temporal_sampling/uniform_1024.yaml
```

temporal crop 对照：

```bash
python scripts/train_mtl_lite.py \
  --override configs/regression_only_baseline.yaml \
  --override configs/input_ablation/rgb.yaml \
  --override configs/temporal_sampling/first_crop.yaml

python scripts/train_mtl_lite.py \
  --override configs/regression_only_baseline.yaml \
  --override configs/input_ablation/rgb.yaml \
  --override configs/temporal_sampling/middle_crop.yaml

python scripts/train_mtl_lite.py \
  --override configs/regression_only_baseline.yaml \
  --override configs/input_ablation/rgb.yaml \
  --override configs/temporal_sampling/random_crop.yaml
```

若 `uniform_512` 或 `uniform_1024` 触发 OOM，应先降低 `EXTRACT_FEATURE.BATCH_SIZE` 或 `EXTRACT_FEATURE.CHUNK_SIZE`，不要同时改动其他训练超参数。

## 2026-06-15 边界连通黑区消融结果

三组更精确的边界连通黑区消融已经完成，配置与前序 RGB 消融可比，区别仅在 `DATASET.INPUT_VARIANT`。

核心结果：

- `center_mask_black_to_gray` 当前整体 MAE/RMSE/Pearson 最好：MAE 约 `7.73`，RMSE 约 `10.02`，Pearson 约 `0.52`，CCC 约 `0.45`；
- `center_mask` 仍保持最高 CCC，约 `0.48`，且对 moderate/severe 更均衡；
- `border_black_feather` 明显优于原始 `rgb` 和 `black_to_gray`，MAE 约 `8.04`，RMSE 约 `10.13`，说明边界软化有效；
- `border_black_to_gray` 表现较弱，说明硬替换边界黑区不如 feather，也不如 center mask。

重要判读：

- `center_mask_black_to_gray` 的总体提升主要来自 minimal/mild，minimal MAE 从 `center_mask` 的约 `7.68` 降至约 `6.13`；
- 但它加重 severe 低估，severe MAE 约 `17.46`，比 `rgb` 和 `center_mask` 更差；
- 因此它不是全面更优，而是更偏低预测、更保守的校准版本；
- `border_black_feather` 对 severe 比 `center_mask` 更好，但 task consistency 比 `rgb` / `center_mask` 差；
- 后续必须将输入伪迹处理和 severity calibration 分开研究。

P0 prediction run summary 工具已新增：

```text
src/diagnostics/prediction_runs.py
scripts/summarize_prediction_runs.py
tests/test_prediction_runs.py
```

该工具用于统一输出 `prediction_run_summary.csv`、`severity_bias_summary.csv`、`task_consistency_summary.csv`、`pairwise_baseline_improvement.csv` 和 `prediction_runs_report.md`，避免后续每轮消融手工计算 prediction std、severity bias 和 task consistency。

已使用该工具对当前可用的 RGB input ablation 预测文件完成统一汇总。包含 `rgb`、`gray_scale`、`blur`、`boundary_erased`、`center_mask`、`black_to_gray`、`black_to_mean`、`black_to_blur`、`soft_center_mask`、`inner_crop_resize`、`border_black_feather`、`border_black_to_gray` 和 `center_mask_black_to_gray`。

推荐服务器复现命令：

```bash
python scripts/summarize_prediction_runs.py \
  --baseline rgb \
  --output-dir analysis_outputs/rgb_input_ablation_summary \
  --run rgb=logs/rgb/rgb_test_predictions.csv \
  --run gray_scale=logs/gray_scale/test_predictions.csv \
  --run blur=logs/blur/test_predictions.csv \
  --run boundary_erased=logs/boundary_erased/test_predictions.csv \
  --run center_mask=logs/center_mask/test_predictions.csv \
  --run black_to_gray=logs/rgb_ablation_black_to_gray/test_predictions.csv \
  --run black_to_mean=logs/rgb_ablation_black_to_mean/test_predictions.csv \
  --run black_to_blur=logs/rgb_ablation_black_to_blur/test_predictions.csv \
  --run soft_center_mask=logs/rgb_ablation_soft_center_mask/test_predictions.csv \
  --run inner_crop_resize=logs/rgb_ablation_inner_crop_resize/test_predictions.csv \
  --run border_black_feather=logs/rgb_ablation_border_black_feather/test_predictions.csv \
  --run border_black_to_gray=logs/rgb_ablation_border_black_to_gray/test_predictions.csv \
  --run center_mask_black_to_gray=logs/rgb_ablation_center_mask_black_to_gray/test_predictions.csv
```

统一表格的当前排序进一步支持以下判断：

- 整体 MAE 排名前三为 `center_mask_black_to_gray`、`center_mask`、`border_black_feather`；
- `center_mask` 仍保持最高 CCC 和接近原始 `rgb` 的 task consistency；
- `gray_scale`、`blur`、`inner_crop_resize` 和 `black_to_mean` 未能改善整体结果，说明过拟合不能简单解释为颜色、纹理或外围区域单因素；
- 所有较好变体的 `pred_std` 仍低于 `true_std`，severe 低估仍需作为 calibration / severity imbalance 问题单独处理。

## 2026-06-15 高优先级过拟合验证审查

当前不建议继续把主要精力放在新增 RGB mask 变体上。黑边/黑填充已被证明是可见风险入口，但不是单一充分解释；继续增加局部 mask 容易变成经验试错，论文价值不如系统审计过拟合机制。

下一批更高优先级任务如下：

1. **Split / subject integrity audit**：确认 train/val/test subject-disjoint，同一 subject 的 Freeform/Northwind 不跨 split，不存在重复视频目录、重复标签或 video_id 规范化错配。该项优先级最高，因为 split 一旦有问题，所有泛化结论都会被污染。
2. **Temporal sampling audit / ablation**：已接入实现，待服务器运行真实审计和六组训练消融。当前 `frame_count` / `sampled_frame_count` 与预测的相关性提示时序长度、截断或采样覆盖可能是隐藏捷径。
3. **Training overfit curve summary**：跨 run 汇总 best epoch、train/val RMSE gap、train/val MAE gap、val 最优后是否继续过拟合。该项可解释 behavior baseline 的 train-test gap，也可比较各 RGB 输入变体是否只是改变预测偏置。
4. **OpenFace alignment geometry audit**：统计 landmark bbox area/width/height/aspect、face center offset、eye distance 和 face scale，与标签、预测、残差和绝对误差相关。该项应从 P1 提到 P0，因为 `center_mask` 有效而 `inner_crop_resize` 变差，说明几何和尺度因素可能比单纯外围裁剪更关键。
5. **Embedding identity retrieval audit**：优先做同 subject Freeform/Northwind embedding paired retrieval，而不是直接训练 subject classifier。若同 subject embedding 高度互为近邻，说明 RGB backbone 编码了强身份/静态外观信息。
6. **Severity calibration verification**：使用 val predictions 拟合 post-hoc linear calibration，再应用到 test，检查 severe 低估和 minimal 高估是否缓解。该项用于区分输入捷径与 loss/标签分布导致的 prediction compression。
7. **Task inconsistency mixed-factor audit**：将 Freeform/Northwind prediction diff 与 frame_count、black-border、confidence、pose/gaze、alignment geometry 关联，判断任务不一致是否由可观测混杂变量驱动。

推荐执行顺序：

```text
split integrity audit
-> temporal sampling audit / ablation
-> training overfit curve summary
-> alignment geometry audit
-> embedding identity retrieval
-> severity calibration verification
-> task inconsistency mixed-factor audit
```

当前实验主线应写作“RGB 过拟合的多因素机制审计”，而不是“黑边导致过拟合”。黑边方向已足以支撑一个子结论：边界硬伪迹会影响泛化，但输入伪迹处理无法单独解决 severe 低估和 prediction compression。

## 2026-06-16 P0-A Split Integrity Audit 实现

P0-A split / subject integrity audit 已作为离线诊断能力落地。该任务优先级最高，因为如果 train/val/test 存在 subject 泄漏、同一 subject 的 Freeform/Northwind 跨 split、重复 video_id、label 错配或 prediction 与 split 无法唯一对齐，后续关于 RGB 过拟合、黑边、时序采样和 identity shortcut 的解释都会被污染。

新增实现：

```text
src/diagnostics/split_integrity.py
scripts/audit_split_integrity.py
tests/test_split_integrity.py
```

输出：

```text
tables/split_video_manifest.csv
tables/split_subject_overlap.csv
tables/split_label_distribution.csv
tables/split_prediction_alignment.csv  # 仅在提供 --predictions 时生成
reports/split_integrity_report.md
```

服务器运行示例：

```bash
python scripts/audit_split_integrity.py \
  --split-file /path/to/dataset_split.json \
  --label-dir /path/to/labels \
  --image-root /path/to/aligned/frame/root \
  --predictions logs/rgb/test_predictions.csv \
  --output-dir logs/rgb/diagnostics/split_integrity
```

判读约束：

- `split_integrity_report.md` 中 `status: PASS` 才能继续把后续测试结果当作 subject-disjoint 泛化结果解释；
- 若存在 `subject_split_overlap`，必须先修复 split 或单独报告污染风险；
- 若 prediction alignment 出现 `missing_in_split` 或 `ambiguous`，不能继续解释对应 prediction-level 诊断；
- 该脚本只读 split、label、image root 和 prediction CSV，不改变训练 forward、loss、metric 或 checkpoint。

## 2026-06-16 P0-B Training Overfit Summary 实现

P0-B training overfit curve summary 已作为跨 run 离线审计能力落地。该任务用于判断输入变体或 behavior baseline 的表观提升是否伴随更大的 train/val gap，避免把测试预测偏置变化误解释为真正泛化提升。

新增实现：

```text
src/diagnostics/training_overfit.py
scripts/summarize_training_overfit.py
tests/test_training_overfit.py
```

输出：

```text
tables/training_overfit_summary.csv
tables/training_curve_gap_by_run.csv
reports/training_overfit_report.md
```

服务器运行示例：

```bash
python scripts/summarize_training_overfit.py \
  --output-dir analysis_outputs/training_overfit_summary \
  --run rgb=/path/to/rgb/metrics.csv \
  --run center_mask=/path/to/center_mask/metrics.csv \
  --run center_mask_black_to_gray=/path/to/center_mask_black_to_gray/metrics.csv \
  --run border_black_feather=/path/to/border_black_feather/metrics.csv \
  --run behavior=/path/to/behavior_baseline/metrics.csv
```

判读约束：

- best epoch 优先按 `val_RMSE_epoch` 选择；若缺失则回退到 `val_MAE_epoch`，再回退到 `val_loss`；
- gap 定义为 `validation - train`，正值越大表示泛化缺口越大；
- `overfit_after_best_val=True` 表示 best val epoch 之后 validation 变差而 train 指标继续改善；
- 该脚本只读取 Lightning `metrics.csv`，不读取 test prediction，不改变训练或 checkpoint。

## 2026-06-16 P0-C Alignment Geometry Audit 实现

P0-C OpenFace alignment geometry audit 已作为离线诊断能力落地。该任务用于把黑边之外的 OpenFace 对齐几何显式量化，检查 face scale、landmark bbox、face center offset、eye distance 和 landmark jitter 是否与 BDI、预测残差或绝对误差相关。

新增实现：

```text
src/diagnostics/alignment_geometry.py
scripts/audit_alignment_geometry.py
tests/test_alignment_geometry.py
```

输出：

```text
tables/alignment_geometry_summary.csv
tables/alignment_geometry_merged.csv
tables/alignment_geometry_correlation.csv
tables/alignment_geometry_group_summary.csv
reports/alignment_geometry_audit_report.md
```

服务器运行示例：

```bash
python scripts/audit_alignment_geometry.py \
  --predictions logs/rgb/test_predictions.csv \
  --openface-root /path/to/openface_csv_root \
  --output-dir logs/rgb/diagnostics/alignment_geometry \
  --frame-width 112 \
  --frame-height 112 \
  --sample-step 1
```

判读约束：

- `alignment_geometry_correlation.csv` 用于检查几何变量与 `true_bdi`、`pred_bdi`、`residual`、`abs_error` 的线性关系；
- `alignment_geometry_group_summary.csv` 用于检查 severe 低估是否集中在异常 face scale、center offset 或 jitter；
- 如果 OpenFace landmark 坐标不是 aligned 112x112 坐标，应改用对应坐标尺度运行 `--frame-width` 和 `--frame-height`；
- 该脚本只读 OpenFace CSV 和 prediction CSV，不改变训练数据、split、loss、metric 或 checkpoint。

## 2026-06-16 P0-C Alignment Geometry 结果与坐标尺度修正

P0-C 已在真实 OpenFace CSV root 和当前 RGB `test_predictions.csv` 上运行，匹配质量正常：

```text
OpenFace videos summarized: 300
Matched prediction rows: 100
Missing prediction videos: 0
Max absolute correlation: 0.3746
```

主要相关性：

```text
landmark_bbox_height_mean  vs true_bdi:  r = 0.3746
normalized_face_scale_mean vs true_bdi:  r = 0.3410
landmark_bbox_area_mean    vs true_bdi:  r = 0.3410
landmark_bbox_width_mean   vs true_bdi:  r = 0.3041
eye_distance_mean          vs true_bdi:  r = 0.2991
landmark_bbox_height_mean  vs residual:  r = -0.2605
```

这说明 OpenFace landmark 检测几何与 BDI/severity 存在中等相关，并且更大的 bbox/scale 与更负 residual 相关，和 severe 低估方向一致。

坐标尺度检查确认：当前 OpenFace CSV 的 landmark 坐标不是 112x112 aligned face 坐标。示例 CSV 中 `x` 范围可达约 `150-643`，`y` 范围可达约 `-11-582`；而实际模型输入 jpg 为 `112 x 112`。OpenFace 日志中的 camera parameters `500,500,320,240` 提示源坐标系约为 `640 x 480`。因此，C 任务应解释为 **pre-alignment detection geometry confound**，而不是模型直接看到的 112x112 landmark geometry。

在使用 `--frame-width 640 --frame-height 480` 重跑后，当前可解释指标：

```text
landmark_bbox_width_mean
landmark_bbox_height_mean
landmark_bbox_area_mean
landmark_bbox_aspect_mean
eye_distance_mean
landmark_jitter_mean
normalized_face_scale_mean
face_center_offset_x_mean
face_center_offset_y_mean
```

这些指标反映 OpenFace 原始检测坐标系下的脸部尺度、检测几何和 landmark 稳定性。它们可能通过后续裁剪、对齐、缩放、黑边填充和插值过程间接影响 112x112 RGB 输入。

640x480 重跑后的 severity 分组显示：

```text
minimal:  true=4.96,  pred=11.85, residual=+6.89, scale=0.319
mild:     true=16.20, pred=14.34, residual=-1.86, scale=0.337
moderate: true=25.00, pred=17.34, residual=-7.66, scale=0.399
severe:   true=34.14, pred=17.64, residual=-16.50, scale=0.364
```

moderate / severe 的检测尺度明显大于 minimal，而模型预测没有随真实 BDI 同步上升，支持 prediction compression 与检测几何混杂并存的解释。后续仍建议增强 geometry 审计，显式输出原始 `landmark_x_min/x_max/y_min/y_max` 和相对比例指标，如 `eye_distance_to_bbox_height_ratio`，减少对固定 frame size 的依赖。

## 2026-06-16 P0-B Temporal Sampling Audit 真实运行结果

P0-B temporal sampling audit 已在当前 RGB `test_predictions.csv` 上完成真实运行，匹配质量正常：

```text
Videos summarized: 100
Matched prediction rows: 100
Missing videos: 0
Max absolute correlation: 0.2197
```

主要相关性显示，视频长度、截断和模型可见帧数与预测值存在弱到中等关系：

```text
truncated_frame_count vs pred_bdi: r = -0.2197
truncated_ratio       vs pred_bdi: r = -0.2090
raw_to_selected_ratio vs pred_bdi: r =  0.2090
sampled_frame_count   vs pred_bdi: r = -0.2073
frame_count           vs pred_bdi: r = -0.2073
```

该结果说明 temporal sampling 是 RGB 过拟合和 prediction compression 的候选混杂因素，但不是单独充分解释。最长视频四分位没有 padding，但平均预测更低、误差更高；这更像是 head-only 截断、长视频中性片段稀释或关键片段覆盖不足，而不是简单 padding 问题。Freeform 视频整体更长、padding 更少，Northwind 更短、padding 更多，但两类任务的平均绝对误差接近，因此任务级差异不能仅由 padding 解释。

当前下一步仍应运行已接入的 temporal sampling 训练消融：`uniform_256`、`uniform_512`、`uniform_1024`、`first_crop`、`middle_crop` 和 `random_crop`。每组训练后必须接入 `scripts/summarize_prediction_runs.py`，统一比较整体指标、prediction std、severity bias 和 Freeform/Northwind task consistency。
## 2026-06-17 局部遮挡与静态外观捷径审计方向

眼镜、麦克风、胡须等因素应被纳入后续 RGB 过拟合机制审计，但它们不应替代当前多因素主线。更准确的定位是：它们位于 **identity/static appearance shortcut** 与 **local occlusion artifact** 的交叉处。

理论依据包括 shortcut learning、面部遮挡识别和遮挡鲁棒表情识别相关研究。对本项目而言，这些因素有三类风险：

- 作为 subject identity 线索：眼镜、胡须、发际线和局部纹理可能帮助模型识别 subject，而不是学习抑郁相关行为；
- 作为局部遮挡：麦克风或口鼻附近遮挡会破坏嘴部、下半脸和 AU/landmark 观测；
- 作为 patch-level artifact：眼镜反光、麦克风黑块和胡须边界可能形成高对比局部 patch，被 DeiT/ViT backbone 放大。

后续建议先做 case-study 与区域遮挡验证，而不是立即全局删除这些区域。优先检查 severe 低估、minimal 高估、高 task diff、以及 `middle_crop` 明显改善/恶化的样本，判断模型关注或遮挡敏感性是否集中在眼镜、麦克风、胡须、下半脸或边界残留上。

## 2026-06-18 OpenFace 边界硬突变和平滑过渡消融计划

人脸裁剪边界的硬像素突变值得作为 input artifact 子机制继续验证。当前 `border_black_feather` 已明显优于原始 `rgb` 和普通 `black_to_gray`，说明问题可能不只是黑色填充面积，而是黑区与脸部区域之间的高对比过渡被 DeiT/ViT patch backbone 放大。

该方向的定位是：**OpenFace aligned face hard-transition artifact**。它属于输入 artifact 与 ViT patch shortcut 的交叉机制，不应替代 identity、geometry、temporal 和 calibration 等主线。

下一轮不建议继续做全图 blur 或粗暴全黑替换，而应做精确边界消融：

```text
edge_soften_only              # 只降低边界连通黑区与脸部交界处的高梯度，不改变大面积黑区
border_blur_fill              # 用邻近非黑区域的模糊颜色填充边界连通黑区
border_reflect_fill           # 用边界邻近像素反射/延展填充黑区
border_feather_blur_fill      # feather mask + blur/inpaint-like fill，验证软过渡是否优于固定灰色
center_mask_soft_boundary_v2  # 在 center_mask 上使用更宽、更自然的软边界
```

优先只做两组：`edge_soften_only` 和 `border_blur_fill`。它们最直接回答：模型到底依赖黑色面积，还是依赖黑色-肤色之间的硬突变边缘。

对照组应固定为：

```text
rgb
center_mask
black_to_gray
border_black_feather
center_mask_black_to_gray
```

判读时必须同时报告 overall metrics、prediction std、severity bias、task consistency 和 pairwise improvement。若平滑边界改善 MAE 但继续加重 severe 低估，则应解释为 artifact mitigation，而不是完整泛化解决方案。

## 2026-06-18 Temporal Sampling 消融与过拟合结果

六组 temporal sampling 训练消融已经完成，并已通过 `scripts/summarize_prediction_runs.py` 与 `scripts/summarize_training_overfit.py` 汇总。结论是：temporal location 会影响预测，但简单替换采样策略不能解决 RGB 过拟合。

整体 test 指标：

```text
middle_crop   MAE=8.8014, RMSE=10.7291, Pearson=0.4192, CCC=0.3806, pred_std=7.3842
uniform_512   MAE=8.8661, RMSE=10.9208, Pearson=0.3606, CCC=0.2966, pred_std=6.1048
uniform_1024  MAE=8.8684, RMSE=10.9257, Pearson=0.3621, CCC=0.2981, pred_std=6.1294
uniform_256   MAE=8.8686, RMSE=10.9214, Pearson=0.3615, CCC=0.2974, pred_std=6.1142
first_crop    MAE=8.8746, RMSE=10.9844, Pearson=0.3272, CCC=0.2522, pred_std=5.4438
rgb           MAE=8.9145, RMSE=10.9530, Pearson=0.3526, CCC=0.2925, pred_std=6.1587
random_crop   MAE=9.0369, RMSE=11.2329, Pearson=0.3847, CCC=0.3631, pred_std=8.1834
```

关键判读：

- `middle_crop` 是本轮最有价值的策略，整体 MAE/RMSE/Pearson/CCC 均优于 `rgb`，并将 severe 组 bias 从 `-16.50` 缓解到 `-14.55`；
- `middle_crop` 同时显著恶化 task consistency，Freeform/Northwind 平均预测差异从 `2.89` 增至 `4.63`，说明中段片段可能引入任务语境混杂；
- `uniform_256/512/1024` 几乎等价，说明瓶颈不是简单的均匀采样帧数不足；
- `first_crop` 加重 prediction compression，severe bias 变为 `-17.42`，不应作为替代策略；
- `random_crop` 增大 `pred_std` 并缓解部分 severe 低估，但整体 MAE/RMSE 最差且 task consistency 最差，不稳定。

训练曲线过拟合汇总显示，所有 temporal run 都是 `overfit_after_best_val=True`：

```text
middle_crop   best_val_rmse=10.5861, best_epoch=11, last_gap=6.5087
rgb           best_val_rmse=10.7153, best_epoch=6,  last_gap=6.4181
uniform_256   best_val_rmse=10.8494, best_epoch=6,  last_gap=6.4019
uniform_512   best_val_rmse=10.8587, best_epoch=6,  last_gap=6.4607
uniform_1024  best_val_rmse=10.8717, best_epoch=6,  last_gap=6.4424
first_crop    best_val_rmse=11.1440, best_epoch=6,  last_gap=6.5589
random_crop   best_val_rmse=11.2404, best_epoch=9,  last_gap=6.5972
```

因此，temporal sampling 的论文结论应保持克制：**temporal location matters, but simple sampling replacement does not solve RGB overfitting**。后续不建议继续扩展 `uniform_2048` 或更多普通 crop；更高价值的是 task inconsistency mixed-factor audit、severity calibration 和 identity/static appearance retrieval。
