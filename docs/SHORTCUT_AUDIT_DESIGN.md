# SHORTCUT_AUDIT_DESIGN.md

本文档定义“非抑郁捷径验证框架”（Shortcut Audit Framework）的研究目标、数据需求、诊断模块、实验矩阵和实施路线。该框架用于验证模型是否依赖身份、OpenFace 追踪质量、裁剪伪影、姿态、光照、视频质量等非抑郁线索，而不是稳定的面部行为动态。

当前 RGB 过拟合多因素审计的权威路线见 `docs/RGB_OVERFITTING_AUDIT_PLAN.md`。本文档负责定义各类 audit 的输入、输出、字段和判读标准；黑边/黑填充只作为 input artifact 子证据之一，不作为唯一或主要解释。

该框架只做诊断和消融，不应改变训练、验证或测试标签，也不应污染 validation/test 统计。

## 1. 核心问题

当前项目使用 OpenFace 裁剪对齐后的人脸帧序列。即使原始背景已经大幅减少，aligned face 中仍可能保留以下捷径：

1. 身份捷径：脸型、肤色、皱纹、眼镜、胡须、发际线、皮肤纹理。
2. OpenFace 质量捷径：`confidence`、`success`、失败帧比例、landmark 抖动、tracking drift。
3. 姿态与 gaze 捷径：`pose_Rx`、`pose_Ry`、`pose_Rz`、gaze direction、头部运动幅度。
4. 裁剪与对齐伪影：黑边、插值痕迹、裁剪边界、人脸尺度残留、头发/衣服残留。
5. 局部遮挡与饰物捷径：眼镜、胡须、麦克风、反光、口鼻周围遮挡、局部黑块。
6. 视频质量捷径：模糊、压缩、亮度、对比度、分辨率、帧间抖动。
7. 时序采样捷径：有效帧数、padding 比例、静止帧比例、特定 subject 的采样模式。

验证目标：

- 判断这些变量是否与 BDI 标签相关；
- 判断这些变量是否与模型预测、残差或绝对误差相关；
- 判断这些变量单独是否能预测 BDI；
- 判断模型注意力、遮挡敏感性是否集中在面部行为区域，而不是边界、头发、眼镜或裁剪伪影。

## 2. 输入数据

最小输入：

```text
predictions.csv          # video_id, subject_id, true_bdi, pred_bdi, residual, abs_error
OpenFace CSV root        # 每个 subject/video 的 OpenFace 原始输出
split file               # train/val/test subject 划分
label files              # BDI 标签，仅用于 subject-level 汇总
```

可选输入：

```text
aligned face frames      # 用于输入消融和可视化 case study
attention/occlusion figs # 已有 MTL-Lite 诊断图
metrics.csv              # 用于训练曲线与泛化状态解释
```

OpenFace CSV 推荐字段：

```text
frame, timestamp, confidence, success
pose_Tx, pose_Ty, pose_Tz
pose_Rx, pose_Ry, pose_Rz
gaze_*
x_*, y_*                 # landmarks
AU*_r, AU*_c             # AU intensity / presence
```

如果当前 OpenFace 版本字段不完全一致，脚本应采用“字段存在则统计，不存在则跳过”的策略。

对齐约定：

- `video_id` 是 Shortcut Audit 的首选合并键；
- OpenFace 文件名应推断为完整 `video_id`，例如 `203_1_Freeform_video.csv` -> `203_1_Freeform_video`；
- 诊断脚本应在合并前规范化 `video_id`，将 `_aligned` 等处理流程后缀视为派生数据标记，而不是语义视频身份的一部分；
- 例如 `203_2_Freeform_video_aligned` 与 `203_2_Freeform_video` 应匹配到同一个 Freeform 视频；
- 规范化不得丢失任务名，必须继续区分 Freeform、Northwind 等不同任务视频；
- `subject_id` 只保留短 ID，例如 `203_1`；
- 当同一 `subject_id` 同时存在 Freeform 和 Northwind 等多个 OpenFace CSV 时，不能只按 `subject_id` 合并；
- 只有在某个 `subject_id` 对应唯一 OpenFace 文件时，才允许退回 `subject_id` 合并。

质量门槛：

- 如果 `shortcut_audit_report.md` 中 `Matched samples` 为 0，或明显小于 `predictions.csv` 中的样本数，该报告只能说明对齐失败，不能用于判断 shortcut risk；
- `shortcut_merged.csv`、`shortcut_correlation.csv`、`shortcut_predictor_results.csv` 只有表头时，应视为无效输出；
- 只有在匹配样本数达到预期后，才解释相关性热力图、shortcut-only predictor 和风险等级。

## 3. 输出结构

建议输出到每次实验的诊断目录下：

```text
logs/.../diagnostics/shortcut_audit/
  tables/
    openface_quality_summary.csv
    shortcut_correlation.csv
    residual_dependency.csv
    shortcut_predictor_results.csv
    input_ablation_results.csv
    black_artifact_summary.csv
    black_artifact_merged.csv
    black_artifact_correlation.csv
  figures/
    shortcut_correlation_heatmap.png
    residual_vs_confidence.png
    residual_vs_pose.png
    bdi_vs_openface_quality.png
    attention_region_summary.png
  reports/
    shortcut_audit_report.md
    black_artifact_audit_report.md
```

## 4. 诊断模块设计

### 4.1 OpenFace 质量汇总

每个 subject/video 统计：

```text
confidence_mean
confidence_std
low_confidence_ratio
success_ratio
failed_frame_ratio
pose_rx_mean / pose_rx_std / pose_rx_abs_mean
pose_ry_mean / pose_ry_std / pose_ry_abs_mean
pose_rz_mean / pose_rz_std / pose_rz_abs_mean
gaze_mean / gaze_std
landmark_motion_mean
landmark_motion_std
valid_frame_count
padding_ratio
```

其中 `landmark_motion` 可先使用相邻帧 landmark 坐标差分的平均 L2 范数。

### 4.2 相关性诊断

将 OpenFace 质量汇总与预测结果合并，计算：

- OpenFace 变量与 `true_bdi` 的相关性；
- OpenFace 变量与 `pred_bdi` 的相关性；
- OpenFace 变量与 `residual` 的相关性；
- OpenFace 变量与 `abs_error` 的相关性。

输出：

- `shortcut_correlation.csv`
- `shortcut_correlation_heatmap.png`
- residual scatter plots

### 4.3 捷径特征单独预测

使用非抑郁变量单独预测 BDI：

```text
OpenFace quality + pose + gaze + video quality -> BDI
```

首批模型建议：

- mean predictor；
- linear regression；
- ridge regression；
- random forest。

评估约束：

- in-sample predictor 只能作为过拟合敏感的 shortcut-risk 信号，不能作为泛化性能；
- 当样本数较少、OpenFace 特征数较多时，例如约 100 个样本对应约 90 个以上特征，in-sample linear/ridge 结果很容易虚高；
- shortcut-only predictor 应优先提供按 `subject_id` 分组的交叉验证，避免同一 subject 的 Freeform/Northwind 同时出现在训练折和测试折；
- 正式报告中至少应同时列出 mean baseline、当前 RGB/MTL-Lite 模型、shortcut-only ridge 多个 alpha 的 MAE、RMSE 和 Pearson；
- 如果 grouped CV 下 shortcut-only 模型接近当前 RGB 模型，应视为中高优先级风险，优先建立 behavior-only baseline 和输入消融，而不是继续单纯调 backbone。

本次 P0 实施设计：

- 新增 `evaluate_shortcut_predictors_grouped_cv()`，默认以 `subject_id` 分组，使用固定 seed 构造 folds；
- 同一 `subject_id` 的 Freeform/Northwind 样本必须始终进入同一 fold；
- 每个 fold 的标准化参数只能由训练 fold 估计，不能使用全体样本；
- grouped CV 至少输出 `mean`、`rgb_mtl_lite`、`ridge_alpha_10`、`ridge_alpha_100`、`ridge_alpha_1000`、`ridge_alpha_10000`；
- grouped CV 结果写入 `shortcut_predictor_grouped_cv.csv`，并追加到 `shortcut_predictor_results.csv`；
- `shortcut_audit_report.md` 中应分开显示 in-sample predictor 和 grouped-CV predictor，避免误读；
- 如果可用 subject 数少于 2，grouped CV 应跳过并返回空结果。

判读：

- 如果 shortcut-only 模型接近 RGB 模型，说明数据中存在强捷径；
- 如果 shortcut-only 模型不能预测 BDI，但能预测误差，说明这些变量是泛化风险因子；
- 如果 shortcut-only 模型表现很弱，仍需结合 attention 和输入消融判断。

### 4.4 输入消融

在相同 split、seed、训练入口和指标下比较：

```text
rgb                # 当前 OpenFace aligned RGB
grayscale          # 去除颜色捷径
blur               # 弱化身份纹理
center_mask        # 保留面部中央区域
boundary_erased    # 弱化裁剪边界、头发、衣服残留
black_to_gray      # 将近黑填充/遮挡区域替换为中性灰
black_to_mean      # 将近黑区域替换为当前帧非黑像素均值
black_to_blur      # 将近黑区域替换为模糊估计
soft_center_mask   # 使用软边界 mask，避免制造新的硬边界
inner_crop_resize  # 裁掉外围黑边后 resize
landmark_heatmap   # 使用 landmark 空间结构替代 RGB 纹理
behavior_only      # landmark/AU/pose/gaze only
```

短期可先实现离线输入变体或 dataset variant；长期可纳入配置：

```yaml
DATASET:
  INPUT_VARIANT: "rgb"
```

### 4.5 注意力与遮挡区域统计

复用已有诊断模块：

- `src/diagnostics/occlusion.py`
- `src/diagnostics/keyframes.py`
- `src/diagnostics/model_attention.py`

新增或后续规划区域级统计：

```text
eye_region_attention_ratio
mouth_region_attention_ratio
brow_region_attention_ratio
face_center_attention_ratio
boundary_attention_ratio
non_face_attention_ratio
```

理想情况：

- 关注区域集中在眼、眉、嘴、鼻唇沟；
- 遮挡这些区域时预测变化明显；
- 遮挡边界、头发、黑边、眼镜区域不应造成异常大的预测变化。

## 5. 风险判定标准

### 高风险

- OpenFace quality / pose / gaze 与 BDI 或 `abs_error` 明显相关；
- shortcut-only 模型能较好预测 BDI；
- 遮挡脸部边界、头发、黑边、眼镜区域导致预测大幅变化；
- 模型关注热力图主要集中在非面部行为区域；
- RGB 模型明显强于 behavior-only，但归因图不合理。

### 中风险

- 捷径特征不能直接预测 BDI，但能预测模型误差；
- high-error subject 集中在低 confidence、大姿态、强裁剪异常或视频质量差的样本；
- attention 区域在不同 subject 间不稳定。

### 低风险

- 捷径变量与 BDI、预测误差均弱相关；
- shortcut-only 模型表现接近 mean predictor；
- 模型关注区域稳定集中在眼、眉、嘴、鼻唇沟；
- masked face 或 behavior-only baseline 与 RGB baseline 表现接近或更稳。

## 6. 实施路线

### 阶段 A：最小可行版本

不改训练、不改模型，只实现离线统计：

```text
scripts/audit_shortcuts.py
src/diagnostics/openface_quality.py
src/diagnostics/shortcut_audit.py
```

目标功能：

1. 读取 OpenFace CSV；
2. 生成 subject-level quality summary；
3. 合并 `predictions.csv`；
4. 输出相关性表格和热力图；
5. 输出 `shortcut_audit_report.md`。

建议命令：

```bash
python scripts/audit_shortcuts.py \
  --predictions logs/.../diagnostics/predictions.csv \
  --openface-root /path/to/openface_csv \
  --split-file /path/to/split.json \
  --output-dir logs/.../diagnostics/shortcut_audit
```

### 阶段 B：输入消融

新增 dataset input variant 或离线输入变体生成流程：

```text
rgb
grayscale
blur
center_mask
boundary_erased
black_to_gray
black_to_mean
black_to_blur
soft_center_mask
inner_crop_resize
landmark_heatmap
```

每个变体至少跑 regression-only baseline，记录 MAE、RMSE、CCC、best epoch 和 test checkpoint 策略。

### 阶段 C：行为表征 baseline

新增轻量行为数据集和模型：

```text
src/datasets/openface_features.py
src/models/behavior_baseline.py
scripts/train_behavior_baseline.py
```

输入：

- landmarks；
- landmark velocity / acceleration；
- AU intensity / AU presence；
- head pose；
- gaze；
- confidence/success mask。

输出：

- BDI regression；
- 可选 severity ordinal；
- 可选 AU/pose/gaze reconstruction 辅助任务。

### 阶段 D：RGB + behavior late fusion

在 behavior baseline 有稳定结果后，再设计：

```text
RGB branch: MTL-Lite image backbone + temporal encoder
Behavior branch: landmark/AU/pose/gaze temporal encoder
Fusion: concat + MLP
Heads: BDI regression + behavior-aware auxiliary tasks
```

## 7. 与现有模块的关系

可复用：

- `src/diagnostics/io.py`
- `src/diagnostics/correlation.py`
- `src/diagnostics/occlusion.py`
- `src/diagnostics/keyframes.py`
- `src/diagnostics/model_attention.py`
- `scripts/diagnose_mtl_lite.py`

应新增：

- `src/diagnostics/openface_quality.py`
- `src/diagnostics/shortcut_audit.py`
- `scripts/audit_shortcuts.py`

暂不应修改：

- `configs/local_paths.yaml`
- legacy 目录下的旧模型逻辑；
- 当前 MTL-Lite 训练超参数；
- 已有日志、权重、checkpoint 和实验结果。

## 8. 论文表达

该框架可作为论文中的诊断章节：

```text
Shortcut Diagnosis and Behavior-oriented Validation
```

建议论点：

> 即使输入为 OpenFace aligned face，端到端视觉模型仍可能利用身份、姿态、追踪质量和裁剪伪影等非抑郁捷径。为此，本研究在模型改进前引入 shortcut audit，先验证非抑郁因素与标签、预测和误差之间的关系，再逐步构建 landmark/AU/pose/gaze 行为表征 baseline。

## 9. 当前最小任务

优先实现顺序：

1. 确认 OpenFace CSV 是否存在及字段格式；
2. 编写 OpenFace quality summary；
3. 合并 `predictions.csv` 与 quality summary；
4. 生成相关性热力图；
5. 输出 markdown audit report；
6. 再考虑输入消融和 behavior-only baseline。

## 10. P0 剩余任务设计

当前 grouped-CV shortcut-only predictor 的判读结论是：OpenFace quality、pose、gaze、AU 等统计特征与 BDI、预测和误差存在中等相关，但这些特征在 subject-level grouped CV 中不能单独接近 RGB/MTL-Lite 模型。因此 shortcut 风险应继续保留为 medium，而不是直接判定模型完全依赖 OpenFace shortcut。下一阶段 P0 的核心不是继续解释 in-sample predictor，而是系统定位预测范围压缩、severe 低估、minimal 高估和任务间不一致的来源。

### 10.1 P0-2：Case Study Manifest

目的：

- 将高误差样本从零散诊断输出整理为固定清单；
- 保证后续 attention、occlusion、keyframe、aligned face 可视化都围绕同一批样本复查；
- 同时保留 low-error reference，避免只看失败样本造成解释偏差。

样本类型：

```text
severe_underestimate        # severe 真实 BDI 高，但预测明显偏低
minimal_overestimate        # minimal 真实 BDI 低，但预测明显偏高
task_inconsistency          # 同一 subject 的 Freeform/Northwind 预测差异大
low_error_reference         # 误差较低的对照样本
```

建议输出：

```text
tables/case_study_manifest.csv
reports/case_study_manifest.md
figures/case_studies/<video_id>/...
```

核心字段：

```text
case_type
rank
video_id
subject_id
task_name
true_bdi
pred_bdi
residual
abs_error
severity_group
paired_task_pred_bdi
task_pred_diff
recommended_diagnostics
```

判读重点：

- severe 低估优先检查 `246_1`、`359_1`、`237_1`、`315_2`；
- Freeform/Northwind 高差异优先检查 `237_1`、`247_1`、`247_3`、`224_1`、`212_1`；
- 每类样本都应对应 attention、occlusion、keyframe 和 aligned face 可视化，而不是只看表格。

实现状态：

- 已新增 `src/diagnostics/case_studies.py`；
- 已接入 `src/diagnostics/regression.py`，回归诊断会输出 `case_study_manifest.csv` 与 `case_study_manifest.md`；
- 已接入 `src/diagnostics/shortcut_audit.py`，Shortcut Audit 会在 `tables/` 和 `reports/` 下输出同名文件；
- 当前实现只读取已有预测结果或 Shortcut Audit merged rows，不参与训练 forward，也不改变任何标签、split 或训练配置。

### 10.2 P0-3：Input Ablation Protocol

目的：

- 验证模型是否依赖 RGB 纹理、身份线索、裁剪边界、黑边、头发、衣物残留或其他非行为线索；
- 区分“模型没有学到抑郁相关行为”与“模型学到了可泛化但不充分的视觉行为线索”；
- 为后续是否改 dataset、训练入口或模型结构提供证据。

输入变体：

```text
rgb                # 当前 OpenFace aligned RGB baseline
grayscale          # 弱化颜色和肤色捷径
blur               # 弱化身份纹理、皱纹、皮肤细节
center_mask        # 保留面部中心行为区域
boundary_erased    # 弱化裁剪边界、黑边、头发、衣物残留
black_to_gray      # 将近黑填充/遮挡区域替换为中性灰
black_to_mean      # 将近黑区域替换为当前帧非黑像素均值
black_to_blur      # 将近黑区域替换为模糊估计
soft_center_mask   # 使用软边界 mask，避免制造新的硬边界
inner_crop_resize  # 裁掉外围黑边后 resize
landmark_heatmap   # 用几何结构替代 RGB 纹理
```

实验约束：

- 所有变体必须使用相同 split、seed、训练入口、checkpoint 选择策略和指标；
- 不得在观察 test 结果后反向调整训练超参数；
- 每个变体至少报告 MAE、RMSE、Pearson、CCC、prediction mean/std、severity group error、Freeform/Northwind task consistency；
- 如果 `boundary_erased` 明显改善 severe 低估或任务间一致性，应优先检查裁剪伪影；
- 如果 `landmark_heatmap` 或 behavior-only 接近 RGB，应优先转向行为表征建模。

实现状态：

- 已新增 `src/datasets/input_variants.py`；
- 已在 `AVECDataset` 中接入 `DATASET.INPUT_VARIANT`，默认值为 `rgb`，因此不改变既有训练行为；
- 当前 RGB dataset 已支持 `rgb`、`grayscale`、`blur`、`center_mask`、`boundary_erased`、`black_to_gray`、`black_to_mean`、`black_to_blur`、`soft_center_mask`、`inner_crop_resize`；
- `landmark_heatmap` 被显式保留为 OpenFace landmark/behavior baseline 路径，当前如果在 RGB dataset 中配置该值会报错，避免伪造 landmark 输入；
- 已在 `configs/avec2014_base.yaml` 中加入 `DATASET.INPUT_VARIANT: "rgb"` 作为默认约定；
- 已新增 `tests/test_input_variants.py`，用于验证输入变体的形状、dtype、alias 和保留值行为。

### 10.2.1 黑填充与硬边界伪迹扩展

第一轮输入消融显示 `center_mask` 明显优于原始 `rgb`，而 `grayscale` 和 `blur` 变差。结合样例帧中 OpenFace aligned face 的纯黑填充和黑色麦克风遮挡，当前 P0-3 需要进一步区分以下可能机制：

1. 模型依赖外围黑边或裁剪边界；
2. 模型依赖黑色遮挡块和面部之间的硬像素突变；
3. `center_mask` 改善来自去除黑伪迹，而不一定来自保留面部中心行为；
4. 硬 mask 本身可能制造新的边界，因此需要软 mask 对照。

新增黑伪迹审计：

```text
src/diagnostics/black_artifacts.py
scripts/audit_black_artifacts.py
```

审计输入：

```text
prediction CSV          # video_id, true_bdi, pred_bdi, residual, abs_error
aligned frame root      # OpenFace aligned frame directories
```

审计输出：

```text
tables/black_artifact_summary.csv
tables/black_artifact_merged.csv
tables/black_artifact_correlation.csv
reports/black_artifact_audit_report.md
```

每个视频统计：

```text
black_ratio_mean
black_ratio_std
black_border_ratio_mean
black_center_ratio_mean
black_boundary_edge_ratio_mean
black_ratio_delta_mean
sampled_frame_count
total_jpg_frame_count
```

判读规则：

- 如果 `black_to_gray`、`black_to_mean` 或 `black_to_blur` 明显改善，黑填充本身应作为 RGB 过拟合的重要原因报告；
- 如果 `soft_center_mask` 优于 `center_mask`，说明边界平滑比单纯遮挡更关键；
- 如果 `inner_crop_resize` 改善，说明外围黑边和裁剪区域是高风险捷径；
- 如果黑伪迹统计与 `abs_error` 或 `residual` 相关，即使对应 ablation 改善有限，也应在论文中报告为 artifact risk factor；
- 如果 severe 低估不随黑伪迹变体改善，应将 severe bias 作为独立失败模式继续研究。

诊断后的修正规则：

- `black_border_ratio_mean` 优先解释为 OpenFace 对齐/裁剪填充风险；
- `black_center_ratio_mean` 不能直接解释为伪迹，因为中心近黑像素可能来自鼻孔、嘴角、自然阴影、胡须、麦克风或真实遮挡；
- 粗暴替换全部黑像素可能破坏真实面部语义，因此后续变体应优先使用 border-connected black mask；
- 如果边界连通黑区消融改善，而中心黑区保持不变，则更能支持 OpenFace 边界填充伪迹假设；
- 如果边界连通黑区消融无效，而 `center_mask` 仍有效，则说明收益更可能来自去除外围非行为区域、脸部轮廓/发际线/姿态残留等混合线索。

下一轮建议输入变体：

```text
border_black_to_gray       # 只替换与图像边界连通的近黑区域
border_black_feather       # 对边界连通近黑区域做软过渡
center_mask_black_to_gray  # 在 center_mask 基础上处理中边界连通黑区
```

实现要求：

- 使用连通域或 flood fill 从图像四边出发构建近黑区域 mask；
- 不默认替换与边界不连通的中心黑像素；
- 单元测试必须覆盖鼻孔/嘴部暗区等中心黑块不被替换的情况；
- 训练配置、split、seed、checkpoint 选择策略和指标必须与 `rgb`、`center_mask`、`black_to_gray` 保持一致。

### 10.3 P0-4：Behavior-only Baseline Interface

目的：

- 建立不依赖 RGB 纹理的结构化行为表征对照；
- 判断 AU、pose、gaze、landmark motion 是否足以解释当前 RGB 模型的有效信号；
- 为后续 RGB + behavior late fusion 和行为辅助任务 MTL 提供干净接口。

建议模块：

```text
src/datasets/openface_features.py
src/models/behavior_baseline.py
scripts/train_behavior_baseline.py
configs/behavior_baseline.yaml
```

建议输入：

```text
AU intensity / AU presence
head pose
gaze
landmark coordinates
landmark velocity / acceleration
confidence / success mask
```

建议输出：

```text
BDI regression
optional severity ordinal
optional behavior reconstruction auxiliary task
```

判读方式：

- 如果 behavior-only baseline 接近或超过 RGB/MTL-Lite，说明当前有效信号很可能主要来自可结构化的面部行为变量；
- 如果 behavior-only 明显弱于 RGB，但 RGB attribution 不集中于眼、眉、嘴、鼻唇沟等合理区域，应继续优先排查非行为捷径；
- 如果 behavior-only 和 RGB 都表现出 severe 低估，则需要进一步处理标签分布、损失尺度和 severity-aware sampling，而不是单纯改 backbone。

实现状态：

- 已新增 `src/datasets/openface_features.py`，从 OpenFace CSV 读取 `confidence`、`success`、pose、gaze、AU、landmark 坐标，并可追加 temporal delta / acceleration；
- 已新增 `src/models/behavior_baseline.py`，使用轻量 GRU 时序编码器、mask-aware pooling、BDI 回归头和可选 ordinal 辅助头；
- 已新增 `src/trainers/behavior_baseline_runner.py` 与 `scripts/train_behavior_baseline.py`，作为独立训练入口；
- 已新增 `configs/behavior_baseline.yaml`，默认 `MODE: "behavior_baseline"`，并要求通过本地配置或 override 提供 `DATASET.OPENFACE_ROOT`；
- 已新增 `tests/test_openface_features.py` 与 `tests/test_behavior_baseline.py`；
- 该 baseline 不读取 RGB 帧，不复用 MTL-Lite visual backbone，不修改 `scripts/train_mtl_lite.py`，也不改变现有训练超参数。

建议运行方式：

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

## 11. Behavior baseline 后续审计扩展

最新 behavior-only baseline 说明：OpenFace 结构化特征不应被整体视为“干净行为信号”。当前完整特征训练出现明显 train/val/test gap，提示 raw landmark、静态几何、质量变量或 subject-specific 采集条件可能被模型记忆。因此 Shortcut Audit 后续需要从“OpenFace 特征是否相关”推进到“哪些 OpenFace 特征组可泛化”。

新增必要输出：

```text
behavior_test_predictions.csv
behavior_val_predictions.csv
behavior_feature_ablation_results.csv
rgb_behavior_prediction_comparison.csv
behavior_case_study_manifest.csv
```

推荐 feature-group ablation：

```text
quality_only
au_only
pose_gaze_only
raw_landmark_only
landmark_delta_only
au_landmark_delta
all_without_raw_landmarks
```

判读规则：

- 如果 `raw_landmark_only` 训练好、测试差，说明静态几何或身份信号风险高；
- 如果 `landmark_delta_only` 或 `AU+landmark_delta` 泛化优于 raw landmark，说明动态行为特征更可靠；
- 如果 `quality_only` 能预测误差或 BDI，需要把 OpenFace 追踪质量作为混杂因素报告；
- 如果 behavior-only 与 RGB 错误样本高度重叠，说明二者可能受相同 subject 或采集条件影响；
- 如果 behavior-only 能修正 RGB 的 severe 低估或 task inconsistency，才有必要优先进入 late fusion。

在这些审计完成前，RGB + behavior late fusion 只应作为 P2 计划，不应提前作为主要模型改进。

## 12. RGB 过拟合多因素审计扩展

黑边和黑填充审计之后，Shortcut Audit 不应只围绕单一 artifact。当前 RGB 过拟合需要拆成输入图像、OpenFace 对齐几何、时序序列、身份/静态外观、姿态/追踪质量、任务语境和标签分布等因素逐项验证。完整路线以 `docs/RGB_OVERFITTING_AUDIT_PLAN.md` 为准。

### 12.1 身份与静态外观审计

目标：

- 判断 RGB/MTL-Lite embedding 是否主要编码 subject 身份、脸型或静态外观；
- 判断去除脸部轮廓、头发、眼镜、胡须等静态区域是否提升泛化。
- 判断眼镜、麦克风、胡须等局部遮挡或饰物是否与高误差、severity bias 或 task inconsistency 共现。

建议输出：

```text
identity_proxy_results.csv
embedding_subject_cluster_report.md
region_ablation_summary.csv
```

建议实验：

- `face_contour_erased`
- `eye_mouth_only`
- `upper_face_only`
- `lower_face_only`
- `glasses_region_erased`
- `mouth_occluder_erased`
- `beard_lower_face_erased`
- subject proxy classifier on frozen RGB embeddings

判读：

- 如果 subject proxy accuracy 高，说明 embedding 中身份信号强；
- 如果 `face_contour_erased` 改善，说明脸型/轮廓/发际线等静态外观可能是捷径；
- 如果只保留眼嘴区域仍能接近 `center_mask`，说明模型有效信号可能集中在行为区域。
- 如果遮挡眼镜、麦克风或胡须区域后预测大幅变化，且变化与 BDI 真实分数无稳定关系，应将其报告为局部 occlusion shortcut 风险，而不是行为 biomarker。
- 如果这类局部区域只在少数 subject 上影响极大，应优先做 case-study 证据，而不是直接设计全局数据处理规则。

### 12.2 OpenFace 对齐几何审计

目标：

- 检查 aligned face 的尺度、中心偏移、眼距和 landmark 外接框是否影响预测；
- 区分黑边风险和更一般的 crop/alignment geometry risk。

建议特征：

```text
landmark_bbox_area
landmark_bbox_width
landmark_bbox_height
landmark_bbox_aspect
face_center_x
face_center_y
face_center_offset
eye_distance
mouth_to_eye_distance
```

输出：

```text
alignment_geometry_summary.csv
alignment_geometry_correlation.csv
alignment_geometry_audit_report.md
```

判读：

- 如果 face scale 或 center offset 与 `abs_error` 相关，应将 OpenFace alignment geometry 作为独立风险报告；
- 如果 geometry predictor 能在 grouped-CV 中预测误差，说明它是泛化风险因子；
- 如果 geometry 变量与 black-border 变量共同解释高误差，应优先做 crop/scale normalization ablation。

### 12.3 姿态、gaze 与追踪质量审计

目标：

- 判断 `confidence`、`success`、pose、gaze、landmark jitter 是否与标签、预测或误差相关；
- 区分真实面部行为与追踪质量混杂。

建议特征：

```text
confidence_mean / std / low_confidence_ratio
success_ratio / failed_frame_ratio
pose_rx / pose_ry / pose_rz mean/std/abs_mean
gaze mean/std
landmark_jitter_mean/std
```

判读：

- 若 quality 与 `abs_error` 相关，应作为数据质量风险，而不是行为 biomarker；
- 若 pose/gaze 与 `pred_bdi` 相关但 grouped-CV 不稳定，应避免直接作为辅助任务；
- 若 severe 低估集中在低 confidence 或大姿态样本，应优先做质量分层评估。

### 12.4 视频长度、采样与 padding 审计

黑伪迹审计已经显示 `frame_count` / `sampled_frame_count` 与 `pred_bdi` 的相关性最高，约 `r = -0.207`。因此时序采样是下一批 P0/P1 诊断重点。

建议特征：

```text
frame_count
sampled_frame_count
valid_ratio
padding_ratio
effective_sequence_length
clip_count
```

建议实验：

- fixed 256-frame uniform sampling；
- fixed 512-frame uniform sampling；
- first / middle / uniform temporal crop 对照；
- temporal occlusion by segment；
- 比较 `MAX_SEQ_LEN` 截断前后的 prediction stability。

当前实现状态：

- 已新增 `src/diagnostics/temporal_sampling.py`；
- 已新增 `scripts/audit_temporal_sampling.py`；
- 已新增 `tests/test_temporal_sampling_audit.py`；
- 当前离线审计会输出 temporal summary、merged table、correlation table、group summary 和 markdown report；
- 已新增 `src/datasets/temporal_sampling.py` 并在 `AVECDataset` 中接入 `PROCESS_TEMPORAL.SAMPLING_STRATEGY`；
- 已新增 fixed uniform 与 temporal crop override 配置，训练结果仍待服务器运行。

当前 RGB prediction 真实审计结果：

- `100/100` 个 test prediction row 成功匹配，`Missing videos = 0`；
- 最大绝对相关约 `0.2197`；
- `truncated_frame_count` 与 `pred_bdi` 相关约 `r = -0.220`；
- `truncated_ratio` 与 `pred_bdi` 相关约 `r = -0.209`；
- `frame_count` / `sampled_frame_count` 与 `pred_bdi` 相关约 `r = -0.207`；
- high frame-count quartile 的预测更低、误差更高，但 padding 为 `0`，提示长视频截断和采样覆盖比 padding 本身更值得优先验证；
- Freeform 更长、Northwind 更短且 padding 更多，但任务平均绝对误差接近，说明任务语境差异需要和其他混杂因素联合分析。

判读：

- 如果固定帧数采样显著改变预测压缩，说明当前时序覆盖或截断策略影响泛化；
- 如果长视频更容易被预测偏低，应检查中性片段稀释和 keyframe pooling；
- 如果同 subject task inconsistency 与 frame_count 差异相关，应把任务长度作为混杂变量报告。

### 12.5 任务语境差异审计

目标：

- 量化 Freeform/Northwind 同 subject 预测不一致；
- 判断 task-specific 行为、说话内容、遮挡、姿态和长度是否影响预测。

建议输出：

```text
task_consistency_summary.csv
task_inconsistency_manifest.csv
task_artifact_correlation.csv
```

判读：

- task name 仅作为诊断变量，不应直接作为当前训练输入；
- 如果 task inconsistency 与 pose/gaze/frame_count/black-border 相关，应优先修正对应输入或采样因素；
- 如果同 subject 两个任务都错，说明更可能是 subject-level bias 或 severity compression。

### 12.6 Prediction compression 与 severity calibration

目标：

- 将输入捷径问题与标签分布/损失导致的向均值收缩区分开；
- 解释 minimal 高估和 severe 低估。

建议输出：

```text
prediction_distribution_summary.csv
severity_calibration.csv
severity_bias_report.md
```

每个实验必须报告：

```text
true_mean / true_std
pred_mean / pred_std
minimal_bias
mild_bias
moderate_bias
severe_bias
```

当前实现状态：

- 已新增 `src/diagnostics/prediction_runs.py`；
- 已新增 `scripts/summarize_prediction_runs.py`；
- 已新增 `tests/test_prediction_runs.py`；
- 该工具会输出整体指标、prediction distribution、severity bias、task consistency 和 pairwise baseline improvement；
- 已对当前可用 RGB input ablation 预测文件生成统一汇总表；
- 后续所有输入消融、黑边消融和 temporal sampling 消融都应先用该工具生成统一表格，再进入论文解释。

推荐输出目录不要放在原始 `logs/` 下，避免混淆训练产物和二次分析产物：

```bash
python scripts/summarize_prediction_runs.py \
  --baseline rgb \
  --output-dir analysis_outputs/rgb_input_ablation_summary \
  --run rgb=logs/rgb/rgb_test_predictions.csv \
  --run center_mask=logs/center_mask/test_predictions.csv \
  --run border_black_feather=logs/rgb_ablation_border_black_feather/test_predictions.csv \
  --run center_mask_black_to_gray=logs/rgb_ablation_center_mask_black_to_gray/test_predictions.csv
```

论文判读时应至少同时报告 `prediction_run_summary.csv`、`severity_bias_summary.csv`、`task_consistency_summary.csv` 和 `pairwise_baseline_improvement.csv`。单看整体 MAE 会高估 `center_mask_black_to_gray` 的稳定性，因为该变体改善 minimal/mild 的同时加重 severe 低估。

判读：

- 如果输入消融改善整体 MAE 但 prediction std 仍远低于 true std，则 severe 低估不能只靠输入处理解决；
- 如果某个变体通过整体抬高预测改善 severe 但伤害 minimal，应报告为 calibration trade-off，而不是泛化提升；
- severity-balanced sampler、weighted loss、Huber/CCC loss 应作为单独实验，不与输入消融混在一起。

### 12.7 Split 与 subject integrity audit

目标：

- 确认当前 train/val/test split 的 subject-level 独立性；
- 确认同一 subject 的 Freeform/Northwind 不跨 split；
- 检查 video_id、subject_id、label file 和 OpenFace CSV 之间是否存在规范化错配；
- 检查重复视频目录、重复标签和缺失标签。

建议输出：

```text
tables/split_video_manifest.csv
tables/split_subject_overlap.csv
tables/split_label_distribution.csv
tables/split_prediction_alignment.csv  # optional, when --predictions is provided
reports/split_integrity_report.md
```

实现状态：

```text
src/diagnostics/split_integrity.py
scripts/audit_split_integrity.py
tests/test_split_integrity.py
```

运行示例：

```bash
python scripts/audit_split_integrity.py \
  --split-file /path/to/dataset_split.json \
  --label-dir /path/to/labels \
  --image-root /path/to/aligned/frame/root \
  --predictions logs/rgb/test_predictions.csv \
  --output-dir logs/rgb/diagnostics/split_integrity
```

最低判读标准：

- 任意 subject 不得同时出现在多个 split；
- 同一 subject 的多个 task video 应进入同一 split；
- 每个 video 必须能匹配唯一 label；
- 每个 prediction row 必须能回连到唯一 video directory 和 label source。
- `split_integrity_report.md` 中 `status: PASS` 才能继续把后续测试结果当作 subject-disjoint 泛化结果解释。

该项应放在所有新实验解释之前。如果发现 split 或 label 问题，应暂停现有 test 结果解释，先修正数据划分或重新生成预测。

### 12.8 Training overfit curve summary

目标：

- 量化不同 run 的 train/val 泛化缺口；
- 判断某个输入变体是否真正改善泛化，而不是只改变 test prediction bias；
- 解释 behavior baseline 中 train RMSE 极低但 test 结果较差的问题。

输出：

```text
tables/training_overfit_summary.csv
tables/training_curve_gap_by_run.csv
reports/training_overfit_report.md
```

实现状态：

```text
src/diagnostics/training_overfit.py
scripts/summarize_training_overfit.py
tests/test_training_overfit.py
```

运行示例：

```bash
python scripts/summarize_training_overfit.py \
  --output-dir analysis_outputs/training_overfit_summary \
  --run rgb=/path/to/rgb/metrics.csv \
  --run center_mask=/path/to/center_mask/metrics.csv \
  --run center_mask_black_to_gray=/path/to/center_mask_black_to_gray/metrics.csv \
  --run border_black_feather=/path/to/border_black_feather/metrics.csv \
  --run behavior=/path/to/behavior_baseline/metrics.csv
```

核心字段：

```text
run
best_monitor
best_val_epoch
best_val_value
train_value_at_best_val
train_val_gap_at_best
best_val_rmse / train_rmse_at_best_val / train_val_rmse_gap
best_val_mae / train_mae_at_best_val / train_val_mae_gap
last_train_rmse
last_val_rmse
val_degradation_after_best
train_improvement_after_best
overfit_after_best_val
status
```

判读：

- 如果某个变体 test MAE 改善但 train/val gap 仍扩大，应谨慎解释为泛化提升；
- 如果 behavior feature ablation 的 train/val gap 远大于 RGB，说明结构化特征仍携带身份或静态几何捷径；
- 如果 `center_mask` 同时降低 test MAE 和 train/val gap，才更适合作为稳定输入处理证据。

### 12.9 Alignment geometry audit

目标：

- 将黑边之外的 OpenFace 对齐几何显式量化；
- 检查 face scale、bbox shape、face center offset 和 eye distance 是否与标签、预测或误差相关；
- 判断 `center_mask` 改善是否可能来自弱化对齐几何和轮廓线索。
- 当前数据中 OpenFace CSV landmark 坐标已确认不是 112x112 aligned frame 坐标，而是原始检测坐标系；因此本审计应优先解释为 pre-alignment detection geometry audit。

输入：

```text
OpenFace CSV root
prediction CSV
```

输出：

```text
tables/alignment_geometry_summary.csv
tables/alignment_geometry_merged.csv
tables/alignment_geometry_correlation.csv
tables/alignment_geometry_group_summary.csv
reports/alignment_geometry_audit_report.md
```

实现状态：

```text
src/diagnostics/alignment_geometry.py
scripts/audit_alignment_geometry.py
tests/test_alignment_geometry.py
```

运行示例：

```bash
python scripts/audit_alignment_geometry.py \
  --predictions logs/rgb/test_predictions.csv \
  --openface-root /path/to/openface_csv_root \
  --output-dir logs/rgb/diagnostics/alignment_geometry \
  --frame-width 112 \
  --frame-height 112 \
  --sample-step 1
```

核心特征：

```text
landmark_bbox_width_mean / height_mean / area_mean / aspect_mean
face_center_x_mean / face_center_y_mean
face_center_offset_x_mean / face_center_offset_y_mean
eye_distance
normalized_face_scale
landmark_jitter
confidence_mean
success_ratio
```

当前解释边界：

```text
可直接解释相关性：
landmark_bbox_width_mean / height_mean / area_mean / aspect_mean
eye_distance_mean
landmark_jitter_mean

在使用 640x480 源坐标尺度重跑后可解释相对尺度：
normalized_face_scale_mean
face_center_offset_x_mean
face_center_offset_y_mean
```

判读：

- 如果 geometry features 与 `abs_error` 或 `residual` 相关，应把 OpenFace alignment geometry 作为 shortcut risk；
- 如果 severe 低估集中在异常 face scale、偏移或大姿态样本，应优先进行质量分层评估；
- 如果 task inconsistency 与 face scale 或 center offset 差异相关，应把任务采集/对齐差异作为混杂因素报告。
- 如果 OpenFace landmark 坐标来自原始视频坐标而非 aligned 112x112 坐标，应通过 `--frame-width` 和 `--frame-height` 使用对应坐标尺度。

已运行结果摘要：

- `300` 个 OpenFace 视频已汇总，`100/100` 个 test prediction row 成功匹配；
- 最大绝对相关约 `0.3746`；
- `landmark_bbox_height_mean`、`landmark_bbox_area_mean`、`landmark_bbox_width_mean`、`eye_distance_mean` 与 `true_bdi` 呈中等相关；
- `landmark_bbox_height_mean` 与 `residual` 负相关，提示更大的检测 bbox 与更强低估有关；
- 该结论应表述为 OpenFace 原始检测几何/预处理阶段混杂风险，而不是模型直接看见了 112x112 landmark 坐标。
- 使用 OpenFace 日志推断的 `640 x 480` 源坐标尺度重跑后，`normalized_face_scale_mean` 约在 `0.30-0.40`，可以作为原始检测坐标系下的相对 face scale 指标解释。

后续脚本增强建议：

```text
landmark_x_min / x_max / y_min / y_max
eye_distance_to_bbox_height_ratio
bbox_width_to_height_ratio
relative_center_within_detected_bbox
```

### 12.10 Embedding identity retrieval audit

目标：

- 检查 RGB/MTL-Lite learned embedding 是否更接近身份/静态外观，而不是 BDI severity；
- 避免一开始训练 subject classifier，因为 subject-independent split 下 test subject 未在 train 出现。

建议输入：

```text
diagnostics/embeddings/test_features.npz
prediction CSV
```

建议输出：

```text
embedding_identity_retrieval.csv
embedding_identity_report.md
embedding_similarity_matrix.png
```

建议指标：

```text
same_subject_top1_rate
same_subject_top3_rate
paired_task_rank_mean
paired_task_rank_median
severity_neighbor_agreement
task_neighbor_agreement
```

判读：

- 如果 Freeform 的最近邻常常是同 subject 的 Northwind，说明 embedding 强身份化；
- 如果 same-subject retrieval 强而 severity neighbor agreement 弱，说明模型表征更像身份/外观空间；
- 如果 `center_mask` 降低 same-subject retrieval，同时保持或提升 BDI 指标，可作为去身份化输入处理证据。

Multi-run summary 扩展：

```text
scripts/summarize_identity_retrieval_runs.py
```

输入为多个 identity retrieval 输出目录，例如 `rgb_test=logs/rgb_test_identity_retrieval`。脚本应读取每个目录下的：

```text
tables/embedding_identity_summary.csv
tables/embedding_identity_retrieval.csv
```

输出：

```text
tables/identity_retrieval_run_summary.csv
tables/identity_retrieval_severity_summary.csv
reports/identity_retrieval_runs_report.md
```

核心字段：

```text
run
split
same_subject_top1_rate
same_subject_top3_rate
same_subject_top5_rate
paired_task_rank_mean
paired_task_rank_median
paired_task_in_top1_rate
paired_task_in_top5_rate
severity_neighbor_agreement_top5_mean
task_neighbor_agreement_top5_mean
```

severity summary 需要按 `query_severity_group` 汇总 top-1/top-5 same-subject rate、severity agreement 和 paired-task rank，用于发现 moderate/severe 等组别是否更不稳定。

### 12.11 Severity calibration verification

目标：

- 区分输入捷径缓解和 prediction compression；
- 验证 severe 低估 / minimal 高估是否能被简单线性校准缓解。

建议流程：

```text
fit calibration on validation predictions only
pred_calibrated = a * pred + b
apply fixed a,b to test predictions
compare original vs calibrated metrics and severity bias
```

建议输出：

```text
severity_calibration_fit.csv
severity_calibration_test_summary.csv
severity_calibration_report.md
```

判读：

- 如果 calibration 明显改善 severe/minimal bias 但 Pearson 基本不变，说明排序信息存在但尺度被压缩；
- 如果 calibration 伤害整体 MAE 或 mild/moderate，应报告 trade-off；
- calibration 只作为机制验证，不能与输入消融混为最终模型优化。
当前 RGB baseline 结果：

```text
pred_calibrated = 0.870402 * pred + 2.689282
original:   MAE=8.9145, RMSE=10.9530, Pearson=0.3526, CCC=0.2925, pred_std=6.13, true_std=11.48
calibrated: MAE=8.8460, RMSE=10.8278, Pearson=0.3526, CCC=0.2692, pred_std=5.33, true_std=11.48
minimal residual: +6.89 -> +8.04
severe residual:  -16.50 -> -16.10
```

该结果说明：

- RGB baseline 的主要失败模式包含 prediction range compression；
- validation-fit 线性校准不能解决 severe underestimation；
- 后续需要构建 multi-run severity calibration summary，而不是只人工阅读单个报告。

建议新增后续汇总脚本任务：

```text
scripts/summarize_severity_calibration_runs.py
src/diagnostics/severity_calibration_runs.py
```

建议输入为多个 severity calibration 输出目录，例如：

```text
--run rgb=logs/severity_calibration
--run middle_crop=logs/middle_crop_severity_calibration
--run border_black_feather=logs/border_black_feather_severity_calibration
```

建议输出：

```text
tables/severity_calibration_run_summary.csv
tables/severity_calibration_group_bias_summary.csv
reports/severity_calibration_runs_report.md
```

核心字段：

```text
run
original_mae / calibrated_mae
original_rmse / calibrated_rmse
original_pearson / calibrated_pearson
original_ccc / calibrated_ccc
true_std
original_pred_std / calibrated_pred_std
minimal_residual_original / minimal_residual_calibrated
severe_residual_original / severe_residual_calibrated
ccc_delta
severe_residual_delta
```
