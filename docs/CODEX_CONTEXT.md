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

### 研究路线修订：RGB 过拟合多因素审计优先

当前视频帧序列已经由 OpenFace 裁剪和对齐，所用 OpenFace 版本可能不是最新版。后续分析不应简单描述为“背景过拟合”，而应关注 OpenFace aligned face 中仍然存在的非抑郁捷径：

- 身份纹理：脸型、肤色、皱纹、眼镜、胡须、发际线；
- 裁剪和对齐伪影：黑边、插值痕迹、边界位置、脸部尺度残留；
- 姿态和追踪质量：head pose、gaze、`confidence`、`success`、landmark 抖动；
- 视频质量：模糊、压缩、光照和分辨率；
- subject-level bias：模型可能记住身份或采集条件，而不是稳定面部行为。

2026-06-15 的 RGB 输入消融和样例帧检查说明：OpenFace aligned face 中的黑填充、硬边界和遮挡黑块是重要的 input artifact 子证据。`center_mask` 明显优于原始 `rgb`，而 `grayscale`、`blur` 变差，说明输入侧非行为线索确实值得研究。但最新黑伪迹审计与边界连通黑区消融也说明，黑边/黑填充不能单独解释 severe 低估、task inconsistency 和 prediction compression。

因此，下一阶段优先级从继续搜索 `FINETUNE_LAST_N_BLOCKS`、继续增加 RGB mask 或提前进入 late fusion，转向 RGB 过拟合多因素审计：

```text
split integrity audit
-> temporal sampling audit / ablation
-> training overfit curve summary
-> alignment geometry audit
-> embedding identity retrieval
-> severity calibration verification
-> task inconsistency mixed-factor audit
-> stable behavior subset
-> RGB + behavior late fusion
```

权威研究路线归档在 `docs/RGB_OVERFITTING_AUDIT_PLAN.md`。相关研究背景在 `docs/RESEARCH_NOTES.md`。后续 Codex 在设计实验或修改模型前，应优先阅读这两个文档。
更高层的系统机制路线图位于 `docs/OVERFITTING_MECHANISM_ROADMAP.md`。后续不要把单个 artifact 当作总解释，应按该文档的 Layer 0-6 逐层验证：数据有效性、prediction compression、input/local occlusion、identity/static appearance、OpenFace geometry/quality、temporal/task context、model optimization。

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
python scripts/diagnose_mtl_lite.py --run-dir experiment/default/mtl_lite/version_0 --ckpt best
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

当前支持 `rgb`、`grayscale`、`blur`、`center_mask`、`boundary_erased`、`black_to_gray`、`black_to_mean`、`black_to_blur`、`soft_center_mask`、`inner_crop_resize`。`landmark_heatmap` 需要真实 OpenFace landmark 坐标，不能由 RGB 帧伪造，因此当前在 RGB dataset 中作为保留值报错，后续应在 behavior baseline 或 OpenFace landmark dataset 中实现。

P0-3 的最新解释口径：

- `center_mask` 的改善不能简单解释为“模型只需要面部中心行为区域”；它也可能意味着该变体抑制了 OpenFace 黑填充、硬裁剪边界、麦克风黑块或外围对齐伪迹。
- `boundary_erased` 接近或略优于 `rgb`，但不如 `center_mask`，说明外围区域确实有风险，但硬擦除本身也可能产生新的边界。
- `grayscale` 和 `blur` 变差，说明当前不应把问题缩小为肤色/颜色捷径或高频纹理捷径。
- severe 低估仍然存在，后续需要把黑伪迹问题与标签分布、severity-aware loss/sampling 和 subject-level bias 分开分析。

P0-3 黑伪迹实现位置：

- `configs/input_ablation/black_to_gray.yaml`
- `configs/input_ablation/black_to_mean.yaml`
- `configs/input_ablation/black_to_blur.yaml`
- `configs/input_ablation/soft_center_mask.yaml`
- `configs/input_ablation/inner_crop_resize.yaml`
- `src/diagnostics/black_artifacts.py`：统计黑像素和硬边界伪迹；
- `scripts/audit_black_artifacts.py`：离线审计入口。

建议服务器运行顺序：

```bash
python scripts/train_mtl_lite.py \
  --override configs/regression_only_baseline.yaml \
  --override configs/input_ablation/black_to_gray.yaml

python scripts/train_mtl_lite.py \
  --override configs/regression_only_baseline.yaml \
  --override configs/input_ablation/black_to_mean.yaml

python scripts/train_mtl_lite.py \
  --override configs/regression_only_baseline.yaml \
  --override configs/input_ablation/black_to_blur.yaml

python scripts/train_mtl_lite.py \
  --override configs/regression_only_baseline.yaml \
  --override configs/input_ablation/soft_center_mask.yaml

python scripts/train_mtl_lite.py \
  --override configs/regression_only_baseline.yaml \
  --override configs/input_ablation/inner_crop_resize.yaml
```

黑伪迹审计示例：

```bash
python scripts/audit_black_artifacts.py \
  --predictions experiment/default/rgb/version_0/diagnostics/regression/test_predictions.csv \
  --image-root /path/to/aligned/frame/root \
  --output-dir experiment/default/rgb/version_0/diagnostics/black_artifacts \
  --sample-step 10
```

黑伪迹审计后的最新约束：

- 不要把 `black_center_ratio_mean` 直接解释为 OpenFace 伪迹。中心近黑像素可能是鼻孔、嘴角、胡须、自然阴影、麦克风或真实遮挡。
- `black_border_ratio_mean` 是当前更可靠的边界填充/裁剪伪迹指标。
- 第二轮 ablation 中 `black_to_gray` 优于 `rgb` 但弱于 `center_mask`，说明黑像素替换有帮助，但不是完整解释。
- `black_to_mean` 和 `inner_crop_resize` 当前不应作为主线继续扩展。
- 已实现 `border_black_to_gray`、`border_black_feather`、`center_mask_black_to_gray`，下一步是在服务器运行 pytest 和三组训练消融。
- 新的边界黑区 mask 应只处理与图像边界连通的近黑区域，默认保留中心近黑区域。
- case study 必须对照高黑边高误差、高黑边低误差、低黑边高误差三类样本，避免把黑边风险过度泛化。

RGB 过拟合的后续解释必须采用多因素框架。除黑边外，优先考虑：

- 身份与静态外观：脸型、年龄、肤色、胡须、发际线、眼镜、皮肤纹理；
- OpenFace 对齐几何：face scale、landmark bbox、face center offset、eye distance；
- 姿态和追踪质量：confidence、success、pose、gaze、landmark jitter；
- 视频长度与采样：frame_count、sampled_frame_count、valid_ratio、padding_ratio、temporal crop；
- 任务语境差异：Freeform/Northwind 同 subject prediction inconsistency；
- 标签分布和校准：prediction compression、minimal overestimate、severe underestimate。

后续 Codex 在设计新实验时，应优先把这些因素做成离线 audit 或单因素消融，不要直接进入 RGB + behavior late fusion。当前建议顺序以 `docs/RGB_OVERFITTING_AUDIT_PLAN.md` 为准：split integrity -> temporal sampling -> training overfit curves -> alignment geometry -> embedding identity retrieval -> severity calibration -> task inconsistency mixed-factor audit。
眼镜、麦克风、胡须等因素应归入 identity/static appearance 与 local occlusion artifact 审计，而不是作为新的单一主因。处理原则：

- 先做 case-study、attention、spatial occlusion 和 embedding retrieval 复核；
- 优先检查 severe 低估、minimal 高估、高 task diff、以及 temporal `middle_crop` 改善/恶化样本；
- 不要默认把中心近黑区域全部删除，因为鼻孔、嘴角阴影、麦克风、胡须和真实遮挡语义混杂；
- 若局部遮挡确实影响大，再考虑 `glasses_region_erased`、`mouth_occluder_erased`、`beard_lower_face_erased` 等区域消融。

P0 temporal sampling audit 已实现：

- `src/diagnostics/temporal_sampling.py`
- `scripts/audit_temporal_sampling.py`
- `tests/test_temporal_sampling_audit.py`

该审计按 `AVECDataset` 的真实规则计算 `sampled_frame_count`、`model_max_len`、`selected_frame_count`、`padding_ratio`、`truncated_ratio` 和 `valid_ratio`。使用时必须让 `--sample-step`、`--max-seq-len` 与对应实验的 `resolved_config.yaml` 保持一致。

P0 temporal sampling audit 已在当前 RGB test prediction 上真实运行并匹配成功：

```text
Videos summarized: 100
Matched prediction rows: 100
Missing videos: 0
Max absolute correlation: 0.2197
```

主要发现是 temporal/truncation 指标与 `pred_bdi` 存在弱到中等相关：`truncated_frame_count` vs `pred_bdi` 约 `r = -0.2197`，`truncated_ratio` vs `pred_bdi` 约 `r = -0.2090`，`frame_count` / `sampled_frame_count` vs `pred_bdi` 约 `r = -0.2073`。长视频四分位预测更低、误差更高且无 padding，因此后续应优先验证长视频截断、首段采样偏置和关键片段覆盖不足，而不是只解释为 padding。该结果弱于 alignment geometry，但足以支撑 temporal sampling 训练消融。

P0 temporal sampling ablation 也已实现：

- `src/datasets/temporal_sampling.py`
- `PROCESS_TEMPORAL.SAMPLING_STRATEGY`
- `configs/temporal_sampling/uniform_256.yaml`
- `configs/temporal_sampling/uniform_512.yaml`
- `configs/temporal_sampling/uniform_1024.yaml`
- `configs/temporal_sampling/first_crop.yaml`
- `configs/temporal_sampling/middle_crop.yaml`
- `configs/temporal_sampling/random_crop.yaml`

默认策略 `stride_head` 复现旧行为：`frames[::SAMPLE_STEP][:MAX_SEQ_LEN // SAMPLE_STEP]`。新策略仅在 override 中显式设置时生效。`uniform_512` 和 `uniform_1024` 可能增加显存压力，若 OOM，应优先降低 batch/chunk，不要同时改变其他训练因素。

边界连通黑区消融最新结论：

- `center_mask_black_to_gray` 当前 MAE/RMSE/Pearson 最好，但 severe 低估更严重；
- `center_mask` 仍是更均衡的输入变体，CCC 最高；
- `border_black_feather` 支持边界软化假设，明显优于 raw RGB 和 `black_to_gray`；
- `border_black_to_gray` 不应继续扩展为主线。

P0 prediction run summary 已实现：

- `src/diagnostics/prediction_runs.py`
- `scripts/summarize_prediction_runs.py`
- `tests/test_prediction_runs.py`

已对当前可用 RGB input ablation 预测文件完成一次统一汇总，覆盖 `rgb`、第一轮输入变体、第二轮黑伪迹变体和最新边界连通黑区变体。后续所有输入消融和 temporal sampling 消融都应使用该工具统一输出 overall metrics、prediction std、severity bias、task consistency 和 pairwise improvement，再进行解释。

当前 RGB 输入消融的整体排序仍应谨慎判读：`center_mask_black_to_gray` 的 MAE/RMSE/Pearson 最好，但 severe 低估更严重；`center_mask` 的 CCC 和任务一致性更稳；`border_black_feather` 支持边界软化假设，但不是完整解决方案。不要把任一输入变体直接当作最终模型，应先进入 temporal sampling、alignment geometry、identity/static appearance 和 calibration 审计。

Temporal sampling 最新结论：六组训练消融已经完成，`middle_crop` 整体指标最好并缓解 severe 低估，但明显恶化 Freeform/Northwind task consistency；`uniform_256/512/1024` 几乎等价；所有 temporal run 都是 `overfit_after_best_val=True`。因此不要继续扩展普通 temporal crop 或 `uniform_2048`，应转向 task inconsistency mixed-factor audit、severity calibration 和 identity retrieval。

Severity calibration 最新结论：RGB baseline 的 `pred_std=6.13` 远低于 `true_std=11.48`，severe residual 为 `-16.50`；validation-fit 线性校准后 CCC 从 `0.2925` 降到 `0.2692`，severe residual 仍为 `-16.10`。因此当前应把 prediction range compression 作为独立核心机制，后续优先做 multi-run calibration summary、identity-residual 联合表和 severity-aware training ablation，而不是把线性校准当作最终模型方案。

OpenFace 边界硬突变方向：`border_black_feather` 已支持边界软化假设，下一步如果继续 input artifact 子线，应优先做 `edge_soften_only` 与 `border_blur_fill`，用于区分黑色面积和黑色-肤色硬突变边缘。不要做全图 blur 或粗暴全黑替换。

Identity retrieval 最新结论：`rgb_test` same-subject top-1 为 `0.66`、top-5 为 `0.85`，paired-task median rank 为 `1`，说明 RGB embedding 强烈保留 subject/static appearance。`border_black_feather_test` 身份检索更强，top-1 达 `0.75`，所以边界软化不等于去身份化。`center_mask` 和 `center_mask_black_to_gray` 在 test 上也没有显著降低 identity retrieval。`middle_crop` 降低 top-1 到 `0.49`，但伴随 task consistency 恶化，应解释为 temporal/task-context confound，而不是稳定 severity representation。下一步应实现 identity retrieval multi-run summary，并与 prediction summary 合并。

P0-A split / subject integrity audit 已实现：

- `src/diagnostics/split_integrity.py`
- `scripts/audit_split_integrity.py`
- `tests/test_split_integrity.py`

该审计用于在解释任何 RGB 过拟合实验之前确认 train/val/test subject-disjoint、Freeform/Northwind 未跨 split、video_id 未重复、label 可唯一匹配、prediction row 可唯一回连到 split manifest。若输出报告不是 `status: PASS`，应先修复 split 或明确报告数据污染风险，再解释后续实验。

服务器运行示例：

```bash
python scripts/audit_split_integrity.py \
  --split-file /path/to/dataset_split.json \
  --label-dir /path/to/labels \
  --image-root /path/to/aligned/frame/root \
  --predictions experiment/default/rgb/version_0/diagnostics/regression/test_predictions.csv \
  --output-dir experiment/default/rgb/version_0/diagnostics/split_integrity
```

P0-B training overfit summary 已实现：

- `src/diagnostics/training_overfit.py`
- `scripts/summarize_training_overfit.py`
- `tests/test_training_overfit.py`

该审计读取 Lightning `metrics.csv`，跨 run 汇总 best validation epoch、train/val RMSE gap、train/val MAE gap、val 最优后是否继续过拟合。它用于区分“输入变体真正改善泛化”和“只改变预测偏置或 checkpoint 选择表象”。

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

P0-C alignment geometry audit 已实现：

- `src/diagnostics/alignment_geometry.py`
- `scripts/audit_alignment_geometry.py`
- `tests/test_alignment_geometry.py`

该审计读取 OpenFace CSV 的 `x_*` / `y_*` landmark 坐标，统计 landmark bbox、face center offset、eye distance、normalized face scale 和 landmark jitter，并与 prediction CSV 中的 `true_bdi`、`pred_bdi`、`residual`、`abs_error` 做相关分析。它用于验证 `center_mask` 有效是否可能来自弱化对齐几何、脸部尺度、中心偏移或静态轮廓线索。

服务器运行示例：

```bash
python scripts/audit_alignment_geometry.py \
  --predictions experiment/default/rgb/version_0/diagnostics/regression/test_predictions.csv \
  --openface-root /path/to/openface_csv_root \
  --output-dir experiment/default/rgb/version_0/diagnostics/alignment_geometry \
  --frame-width 112 \
  --frame-height 112 \
  --sample-step 1
```

若 OpenFace landmark 坐标不是 aligned 112x112 坐标，应使用实际坐标尺度设置 `--frame-width` 和 `--frame-height`。本项目已根据 OpenFace camera parameters 使用 `640 x 480` 重跑，因此 normalized offset 和 scale 可作为原始检测坐标系下的相对几何指标解释。

P0-C 真实运行后的坐标尺度修正：

- 当前 OpenFace CSV landmark 坐标已确认不是模型输入的 `112 x 112` aligned frame 坐标；
- 示例范围：`x` 约 `150-643`，`y` 约 `-11-582`，而 aligned jpg 为 `112 x 112`；
- OpenFace 日志中的 camera parameters `500,500,320,240` 提示源坐标系约为 `640 x 480`，因此 geometry audit 已按 `--frame-width 640 --frame-height 480` 重跑；
- 因此 C 任务结果应解释为 **pre-alignment detection geometry confound**，即 OpenFace 原始检测坐标系下的 face scale / bbox / eye distance / landmark jitter 与 BDI 和 residual 的关系；
- 当前可解释的是 `landmark_bbox_width/height/area/aspect`、`eye_distance`、`landmark_jitter` 的相关性，以及基于 `640 x 480` 源坐标尺度的 `normalized_face_scale_mean` 和 `face_center_offset_*` 相对值；
- 仍不应把这些变量解释为模型直接看到的 112x112 landmark 坐标。

当前 C 任务主要结果：

```text
Matched prediction rows: 100/100
Max absolute correlation: 0.3746
landmark_bbox_height_mean vs true_bdi: 0.3746
landmark_bbox_area_mean   vs true_bdi: 0.3410
landmark_bbox_height_mean vs residual: -0.2605
normalized_face_scale_mean vs true_bdi: 0.3410
```

最新高优先级过拟合审查结论：

- RGB 过拟合多因素审计的权威路线文档是 `docs/RGB_OVERFITTING_AUDIT_PLAN.md`；
- 不再优先继续增加新的 RGB mask 变体；
- 下一批 P0 是 split / subject integrity、temporal sampling 真实运行、training overfit curve summary、OpenFace alignment geometry、embedding identity paired-task retrieval、severity calibration verification 和 task inconsistency mixed-factor audit；
- alignment geometry 已从 P1 提升到 P0，因为 `center_mask` 有效但 `inner_crop_resize` 变差，提示 face scale、bbox、center offset、eye distance 等几何因素可能是重要捷径；
- embedding 身份审计优先做 Freeform/Northwind paired retrieval，不先做 subject classifier；
- severity calibration 只作为机制验证，不能和输入消融混为最终模型调参；
- RGB + behavior late fusion 和行为辅助 MTL 继续暂缓，必须等上述过拟合机制审计和稳定 behavior 特征子集完成后再进入。

新的主线顺序：

```text
split integrity audit
-> temporal sampling audit / ablation
-> training overfit curve summary
-> alignment geometry audit
-> embedding identity retrieval
-> severity calibration verification
-> task inconsistency mixed-factor audit
-> behavior stable subset
-> RGB + behavior late fusion
```

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
