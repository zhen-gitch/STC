# RGB_OVERFITTING_AUDIT_PLAN.md

本文档是当前 RGB 输入模型过拟合研究的主控文档。后续涉及 RGB 过拟合原因、实验优先级、论文叙事和审计结果解释时，优先参考本文档；`CURRENT_STATUS.md` 记录当前状态，`TODO.md` 记录执行清单，`SHORTCUT_AUDIT_DESIGN.md` 记录具体脚本/输出规格，`RESEARCH_NOTES.md` 记录论文背景和研究线索。

## 核心判断

当前不应把 RGB 过拟合解释为“黑边导致过拟合”。更合理、更有论文价值的表述是：

> OpenFace aligned face 上的 RGB 模型过拟合可能来自身份静态外观、OpenFace 对齐几何、边界填充与遮挡伪迹、姿态/追踪质量、视频长度与时序采样、任务语境差异、ViT patch 级敏感性和标签分布压缩的共同作用。黑边是可见且可操作的风险入口，但不是单一充分解释。

这个判断来自三类证据：

1. 第一轮 RGB input ablation 显示 `center_mask` 明显优于原始 `rgb`，说明输入侧非行为线索确实重要。
2. 黑伪迹审计显示黑区很普遍，且高边界黑区组误差更大，但单一黑像素指标与误差的线性相关并不强。
3. 最新边界连通黑区消融显示 `center_mask_black_to_gray` 整体 MAE 最好，但 severe 低估更严重，说明输入伪迹处理无法单独解决 prediction compression。

## 相关研究依据

- Shortcut learning 研究指出，深度模型可能学习在标准测试条件下有效、但不能迁移到更真实或更困难条件的捷径规则：[Shortcut Learning in Deep Neural Networks](https://arxiv.org/abs/2004.07780)。
- 视觉 Transformer 缺少 CNN 的局部归纳偏置，在小数据集上更依赖数据规模、正则化和辅助约束，因此更需要检查 patch-level shortcut：[Efficient Training of Visual Transformers with Small Datasets](https://arxiv.org/abs/2106.03746)。
- 面部抑郁识别研究逐渐强调 temporal facial landmarks 和行为动态，而不是直接依赖冗余 RGB 外观：[FacialPulse](https://arxiv.org/abs/2408.03499)。
- AU 时间序列和表情动态可作为抑郁相关 biomarker 线索，支持后续用 AU/landmark/pose/gaze 做 behavior baseline 和审计：[Exploring Facial Biomarkers for Depression through Temporal Analysis of Action Units](https://arxiv.org/abs/2407.13753)。
- OpenFace / LibreFace / OpenFace 3.0 这类工具说明 landmark、AU、pose、gaze、confidence、success 本身就是可量化的面部行为与质量变量，应被用作诊断变量，而不仅是 RGB 对齐预处理的副产品：[LibreFace](https://arxiv.org/abs/2308.10713)，[OpenFace 3.0](https://arxiv.org/abs/2506.02891)。

## 当前输入特点

当前模型输入不是原始场景视频，而是 OpenFace 裁剪对齐后的面部帧序列：

- 每帧包含 aligned face，同时可能保留黑色填充、麦克风遮挡、脸部轮廓、头发/衣领残留、插值模糊和裁剪几何。
- 模型使用 DeiT/ViT 类 patch-based backbone。若输入为 `112 x 112` 且 patch size 为 `16`，一帧约对应 `7 x 7` 个 patch；局部硬边界、黑块、眼镜反光或裁剪残留可能占据完整 patch。
- 视频序列经过 `SAMPLE_STEP`、`MAX_SEQ_LEN`、padding/mask 和 temporal pooling；不同视频长度、任务片段和采样覆盖可能影响预测。
- BDI 是 subject/video-level 标签，帧级没有精确监督；模型容易利用稳定静态线索替代稀疏行为动态。

## 多因素过拟合地图

| 因素 | 可能机制 | 当前证据 | 优先级 | 下一步 |
|---|---|---|---|---|
| Split / subject integrity | subject 泄漏、任务视频跨 split、标签错配会污染泛化结论 | 尚未系统审计 | P0 | 先做 split integrity audit |
| 时序采样与视频长度 | 长度、截断、padding、采样位置影响 temporal pooling | `frame_count` / `sampled_frame_count` 与 `pred_bdi` 相关最高 | P0 | 跑 temporal audit 和 6 组 sampling ablation |
| 训练曲线泛化缺口 | train 继续下降但 val/test 不改善，说明模型记忆训练集 | behavior baseline 已显示强 train-test gap | P0 | 跨 run 汇总 train/val gap |
| OpenFace 对齐几何 | face scale、bbox、center offset、eye distance 等静态几何成为捷径 | `center_mask` 有效但 `inner_crop_resize` 变差 | P0 | alignment geometry audit |
| 身份与静态外观 | 脸型、肤色、胡须、眼镜、发际线、皮肤纹理携带 subject 信息 | RGB 与 full behavior 特征均有过拟合风险 | P0/P1 | paired-task embedding retrieval |
| 黑边/边界伪迹 | 黑填充、硬边界、麦克风黑块形成高对比 patch | border feather 有效，但不能解决 severe 低估 | P1 | 作为子证据保留，进入 case study |
| 姿态/追踪质量 | confidence、success、pose、gaze、jitter 混合真实行为和质量混杂 | OpenFace shortcut audit 已发现中等相关线索 | P1 | quality/pose/gaze mixed audit |
| 任务语境差异 | Freeform/Northwind 的说话内容、gaze、动作、长度不同 | task consistency 已显示同 subject 可有较大预测差异 | P0/P1 | task inconsistency mixed-factor audit |
| Prediction compression | MSE/标签分布导致向均值收缩，minimal 高估、severe 低估 | 所有较好输入变体仍 `pred_std < true_std` | P0 | val-fit test-apply calibration verification |
| ViT patch shortcut | 局部高对比 patch 或边界 patch 被 backbone 放大 | 与黑边/遮挡样例一致，但缺少 attribution 证据 | P1 | patch occlusion / attention case study |

## P0 实验安排

### P0-A Split / Subject Integrity Audit

目的：保护所有泛化解释的有效性。

当前状态：已实现离线审计入口。

```text
src/diagnostics/split_integrity.py
scripts/audit_split_integrity.py
tests/test_split_integrity.py
```

输出：

```text
split_integrity_report.md
split_video_manifest.csv
split_subject_overlap.csv
split_label_distribution.csv
split_prediction_alignment.csv  # optional
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

判读：

- 任意 subject 不得同时出现在多个 split。
- 同一 subject 的 Freeform/Northwind 等任务视频不应跨 split。
- 每个 video 必须能匹配唯一 label。
- 每个 prediction row 必须能回连到唯一 video directory 和 label source。
- 若该审计未通过，应暂停解释后续 input ablation、temporal sampling、alignment geometry 和 embedding identity 结果，先处理 split 或匹配污染。

### P0-B Temporal Sampling Audit / Ablation

目的：验证视频长度、截断和采样覆盖是否导致 prediction compression 或 task inconsistency。

已接入：

```text
scripts/audit_temporal_sampling.py
configs/temporal_sampling/uniform_256.yaml
configs/temporal_sampling/uniform_512.yaml
configs/temporal_sampling/uniform_1024.yaml
configs/temporal_sampling/first_crop.yaml
configs/temporal_sampling/middle_crop.yaml
configs/temporal_sampling/random_crop.yaml
```

训练消融：

```text
uniform_256
uniform_512
uniform_1024
first_crop
middle_crop
random_crop
```

每组训练后必须接入 `scripts/summarize_prediction_runs.py`，报告 overall metrics、prediction std、severity bias、task consistency 和 pairwise improvement。

### P0-C Training Overfit Curve Summary

目的：区分真正泛化提升和测试预测偏置变化。

当前状态：已实现跨 run 训练曲线过拟合汇总入口。

```text
src/diagnostics/training_overfit.py
scripts/summarize_training_overfit.py
tests/test_training_overfit.py
```

输出：

```text
training_overfit_summary.csv
training_curve_gap_by_run.csv
training_overfit_report.md
```

字段：

```text
run_name
best_val_epoch
best_val_rmse
train_rmse_at_best_val
train_val_rmse_gap
best_val_mae
train_mae_at_best_val
train_val_mae_gap
last_train_rmse
last_val_rmse
overfit_after_best_val
```

运行示例：

```bash
python scripts/summarize_training_overfit.py \
  --output-dir analysis_outputs/training_overfit_summary \
  --run rgb=experiments/mtllite/baseline/logs/mtl_lite_csv/rgb/metrics.csv \
  --run center_mask=experiments/mtllite/baseline/logs/mtl_lite_csv/center_mask/metrics.csv \
  --run center_mask_black_to_gray=experiments/mtllite/baseline/logs/mtl_lite_csv/center_mask_black_to_gray/metrics.csv \
  --run border_black_feather=experiments/mtllite/baseline/logs/mtl_lite_csv/border_black_feather/metrics.csv
```

优先比较：

```text
rgb
center_mask
center_mask_black_to_gray
border_black_feather
behavior baseline
behavior feature-group ablations
```

判读：

- 若某个输入变体 test MAE 改善，但 train/val gap 更大或 `overfit_after_best_val=True`，应谨慎解释为 prediction bias 改变，而不直接当作泛化提升。
- 若 behavior baseline 或 raw-landmark 特征组 train/val gap 显著大于 AU/pose/gaze 动态特征组，支持其含有更强身份/静态几何记忆风险。

### P0-D Alignment Geometry Audit

目的：把黑边之外的 OpenFace 对齐几何显式量化。

输出：

```text
alignment_geometry_summary.csv
alignment_geometry_correlation.csv
alignment_geometry_group_summary.csv
alignment_geometry_audit_report.md
```

特征：

```text
landmark_bbox_width / height / area / aspect
face_center_x / face_center_y
face_center_offset_x / face_center_offset_y
eye_distance
normalized_face_scale
landmark_jitter
```

判读：

- 若 geometry features 与 `abs_error` 或 `residual` 相关，应作为 shortcut risk。
- 若 severe 低估集中在异常 face scale、偏移或大姿态样本，应进行质量分层评估。
- 若 task inconsistency 与 face scale / center offset 差异相关，应把任务采集/对齐差异作为混杂因素报告。

### P0-E Embedding Identity Retrieval

目的：检查 RGB embedding 是否主要编码身份/静态外观。

优先方法：paired-task retrieval，而不是 subject classifier。

输出：

```text
embedding_identity_retrieval.csv
embedding_identity_report.md
embedding_similarity_matrix.png
```

指标：

```text
same_subject_top1_rate
same_subject_top3_rate
paired_task_rank_mean
paired_task_rank_median
severity_neighbor_agreement
task_neighbor_agreement
```

判读：

- 若 Freeform 的最近邻常是同 subject 的 Northwind，说明 embedding 强身份化。
- 若 same-subject retrieval 强而 severity neighbor agreement 弱，说明表征更像身份/外观空间。
- 若 `center_mask` 降低 same-subject retrieval，同时保持或提升 BDI 指标，可作为去身份化输入处理证据。

### P0-F Severity Calibration Verification

目的：区分输入捷径和标签/损失导致的 prediction compression。

流程：

```text
fit calibration on validation predictions only
pred_calibrated = a * pred + b
apply fixed a,b to test predictions
compare original vs calibrated metrics and severity bias
```

输出：

```text
severity_calibration_fit.csv
severity_calibration_test_summary.csv
severity_calibration_report.md
```

注意：calibration 只用于机制验证，不能作为观察 test 后的最终模型调参。

### P0-G Task Inconsistency Mixed-Factor Audit

目的：解释同一 subject 在 Freeform/Northwind 中预测不一致的来源。

关联变量：

```text
frame_count
sampled_frame_count
black_border_ratio
confidence
success
pose / gaze
alignment geometry
prediction severity bias
```

输出：

```text
task_inconsistency_manifest.csv
task_artifact_correlation.csv
task_inconsistency_report.md
```

## P1 实验安排

P1 只在 P0 证据完成后启动，避免继续经验试错。

1. **Patch / attention case study**：针对 severe 低估、minimal 高估、高 task diff、黑边改善/恶化 case 生成 attention、occlusion、keyframe 图组。
2. **Region ablation**：`face_contour_erased`、`eye_mouth_only`、`upper_face_only`、`lower_face_only`，用于验证行为区域和身份区域。
3. **Quality / pose stratified evaluation**：按 confidence、success、pose、gaze、jitter、face scale 分层报告指标。
4. **Behavior feature subset selection**：在 feature-group ablation 后决定哪些 OpenFace 特征进入稳定 behavior subset。
5. **Targeted robustness augmentation**：只在诊断明确后测试 border fill randomization、small affine jitter、brightness/contrast jitter 和 temporal random crop。

## 暂缓任务

以下任务暂缓，直到 P0/P1 机制审计完成：

- RGB + behavior late fusion；
- AU / landmark / pose / gaze auxiliary MTL；
- 动态任务权重、PCGrad、GradNorm、LDS、`loss_dist`；
- 大范围数据增强组合；
- 继续堆叠相似 RGB mask 变体。

## 论文实验主线

推荐组织方式：

```text
1. Baseline failure:
   prediction compression, severe underestimate, minimal overestimate, task inconsistency

2. RGB input artifact evidence:
   center mask, black artifact audit, border-connected black ablation

3. Multi-factor shortcut audit:
   temporal sampling, alignment geometry, identity retrieval, quality/pose, task context

4. Calibration analysis:
   separate input shortcut mitigation from severity compression

5. Behavior feature baseline:
   evaluate whether structured facial dynamics generalize better than raw RGB appearance
```

论文表述边界：

- 可以说黑边/硬边界伪迹是重要可见风险入口。
- 不应说 RGB 过拟合由黑边单独导致。
- 应强调本文贡献是建立了一套针对 OpenFace aligned face 的多因素 shortcut audit protocol，并用消融、审计和校准把输入伪迹、身份/几何捷径、时序采样和标签压缩分开验证。
