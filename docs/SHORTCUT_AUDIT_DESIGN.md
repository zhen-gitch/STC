# SHORTCUT_AUDIT_DESIGN.md

本文档定义“非抑郁捷径验证框架”（Shortcut Audit Framework）的研究目标、数据需求、诊断模块、实验矩阵和实施路线。该框架用于验证模型是否依赖身份、OpenFace 追踪质量、裁剪伪影、姿态、光照、视频质量等非抑郁线索，而不是稳定的面部行为动态。

该框架只做诊断和消融，不应改变训练、验证或测试标签，也不应污染 validation/test 统计。

## 1. 核心问题

当前项目使用 OpenFace 裁剪对齐后的人脸帧序列。即使原始背景已经大幅减少，aligned face 中仍可能保留以下捷径：

1. 身份捷径：脸型、肤色、皱纹、眼镜、胡须、发际线、皮肤纹理。
2. OpenFace 质量捷径：`confidence`、`success`、失败帧比例、landmark 抖动、tracking drift。
3. 姿态与 gaze 捷径：`pose_Rx`、`pose_Ry`、`pose_Rz`、gaze direction、头部运动幅度。
4. 裁剪与对齐伪影：黑边、插值痕迹、裁剪边界、人脸尺度残留、头发/衣服残留。
5. 视频质量捷径：模糊、压缩、亮度、对比度、分辨率、帧间抖动。
6. 时序采样捷径：有效帧数、padding 比例、静止帧比例、特定 subject 的采样模式。

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
  figures/
    shortcut_correlation_heatmap.png
    residual_vs_confidence.png
    residual_vs_pose.png
    bdi_vs_openface_quality.png
    attention_region_summary.png
  reports/
    shortcut_audit_report.md
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
