# RGB_OVERFITTING_AUDIT_PLAN.md

> 文档职责：RGB 过拟合研究主控文档，记录机制判断、实验优先级和论文叙事边界。执行清单见 `TODO.md`；低层脚本规格见 `SHORTCUT_AUDIT_DESIGN.md`；文档导航见 `DOCS_GUIDE.md`。

本文档是当前 RGB 输入模型过拟合研究的主控文档。后续涉及 RGB 过拟合原因、实验优先级、论文叙事和审计结果解释时，优先参考本文档；`CURRENT_STATUS.md` 记录当前状态，`TODO.md` 记录执行清单，`SHORTCUT_AUDIT_DESIGN.md` 记录具体脚本/输出规格，`RESEARCH_NOTES.md` 记录论文背景和研究线索。
更高层的机制地图见 `docs/OVERFITTING_MECHANISM_ROADMAP.md`。该文档用于统一组织 split、calibration、input artifact、identity/static appearance、local occlusion、OpenFace geometry、temporal/task context 和 behavior validation，避免后续实验零散拼凑。

## 读者先看：当前研究路线与下一步

当前 RGB 过拟合研究已经从“输入 artifact 逐项排查”和 “Shortcut-Regularized MTL” 进一步转向 **RPDF-Net：风险感知递进式去身份因子分解网络**。旧的黑边、center mask、boundary smoothing、temporal sampling、identity retrieval 和 severity calibration 实验仍然重要，但它们现在的角色是 **支撑 RPDF-Net 的问题定义、因子设计、支线验证和对照基线**。

当前论文问题应表述为：

> 静态 RGB depression regression 在 subject-independent 小样本设置下，会同时利用 subject/static appearance shortcut、OpenFace artifact proxy 和 severity label imbalance。为避免简单删除身份导致抑郁线索损失，本文将共享表征分解为低风险抑郁因子、受控身份-抑郁交叠因子、身份因子、伪迹因子和残差因子，并通过递进式风险评估选择隐私-效用更优的表征。

当前主线：

| 阶段 | 作用 | 当前状态 | 输出 |
|---|---|---|---|
| Stage A | RPDF 证据收口 | 下一步立即做 | layer-wise identity probe；error-identity coupling；artifact weak labels |
| Stage B | RPDF-lite 单级分解 | Stage A 后执行 | `z_dep/z_m/z_id/z_art/z_res` 与 controlled `z_m` 消融 |
| Stage C | 两级递进 RPDF | RPDF-lite 稳定后执行 | identity/artifact risk 是否逐层下降，BDI 是否保持 |
| Stage D | 支线有效性验证 | 逐步执行 | z_art、z_m gate、severity-balanced loss、multi-attacker、dynamic feature |

当前最重要的下一步：

```text
A1 layer-wise identity probe
A2 prediction error x identity similarity coupling
A3 artifact weak-label audit for z_art
A4 severity imbalance summary for severity-balanced branch
B1 RPDF-lite single-level factorization
B2 controlled z_m transfer ablation: alpha in {0, 0.25, 0.5, 1.0}
```

旧实验的定位：

- `center_mask`、`center_mask_black_to_gray`、`border_black_feather`、`middle_crop` 和 2x2 input variants 用于证明输入侧处理会重新分配 bias，但不能单独解决身份记忆和 severe underestimation。
- `identity_retrieval_summary` 用于证明 RGB embedding 保留 subject/static appearance，是 `z_id` 和 identity risk evaluation 的前置证据。
- `severity_calibration_summary` 用于证明 post-hoc calibration 不能替代训练目标修正，是 severity-balanced branch 的前置证据。
- `alignment_geometry`、`black_artifact` 和 temporal/task audits 是 `z_art` 与 artifact weak-label 设计的前置证据。
- `identity-adversarial MTL` 和 `severity-balanced regression` 不再是最终主线本身，而是 RPDF-Net 的重要对照基线与支线验证。

## 核心判断

当前不应把 RGB 过拟合解释为“黑边导致过拟合”。更合理、更有论文价值的表述是：

> OpenFace aligned face 上的 RGB 模型过拟合可能来自身份静态外观、OpenFace 对齐几何、边界填充与遮挡伪迹、姿态/追踪质量、视频长度与时序采样、任务语境差异、ViT patch 级敏感性和标签分布压缩的共同作用。黑边是可见且可操作的风险入口，但不是单一充分解释。

这个判断来自三类证据：

1. 第一轮 RGB input ablation 显示 `center_mask` 明显优于原始 `rgb`，说明输入侧非行为线索确实重要。
2. 黑伪迹审计显示黑区很普遍，且高边界黑区组误差更大，但单一黑像素指标与误差的线性相关并不强。
3. 最新边界连通黑区消融显示 `center_mask_black_to_gray` 整体 MAE 最好，但 severe 低估更严重，说明输入伪迹处理无法单独解决 prediction compression。
4. RGB baseline severity calibration 显示 `pred_std=6.13` 远低于 `true_std=11.48`，val-fit 线性校准只轻微改善 MAE/RMSE，CCC 反而下降，severe 低估几乎不变。

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
| 局部遮挡与饰物 | 眼镜、麦克风、胡须、口鼻周围遮挡、反光和局部黑块可能与 subject、任务或采集条件共现 | 样例帧存在麦克风黑块；中心黑像素语义混杂；输入 artifact 处理只能部分改善 | P1 | artifact/occlusion case study + 区域遮挡消融 |
| 姿态/追踪质量 | confidence、success、pose、gaze、jitter 混合真实行为和质量混杂 | OpenFace shortcut audit 已发现中等相关线索 | P1 | quality/pose/gaze mixed audit |
| 任务语境差异 | Freeform/Northwind 的说话内容、gaze、动作、长度不同 | task consistency 已显示同 subject 可有较大预测差异 | P0/P1 | task inconsistency mixed-factor audit |
| Prediction compression | MSE/标签分布导致向均值收缩，minimal 高估、severe 低估 | RGB baseline `pred_std=6.13` vs `true_std=11.48`；severe residual `-16.50`；线性校准后 CCC 下降且 severe 仍 `-16.10` | P0 | multi-run calibration summary + severity-aware training ablation |
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
  --predictions <LOG_DIR>/default/rgb/version_0/diagnostics/regression/test_predictions.csv \
  --output-dir <LOG_DIR>/default/rgb/version_0/diagnostics/split_integrity
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

当前真实运行结果：

```text
Videos summarized: 100
Matched prediction rows: 100
Missing videos: 0
Max absolute correlation: 0.2197
```

主要发现：

```text
truncated_frame_count vs pred_bdi: r = -0.2197
truncated_ratio       vs pred_bdi: r = -0.2090
raw_to_selected_ratio vs pred_bdi: r =  0.2090
sampled_frame_count   vs pred_bdi: r = -0.2073
frame_count           vs pred_bdi: r = -0.2073
```

分组判读：

- longest / high frame-count quartile 的预测均值更低、绝对误差更高，且几乎没有 padding，提示风险更可能来自长视频截断、首段采样偏置或长视频中性片段稀释；
- shortest / low frame-count quartile padding 比例最高，但平均误差反而更低，说明 padding 不是当前唯一驱动因素；
- severe 组仍表现为强低估，temporal 指标只能解释一部分误差方向，不能替代 severity calibration 分析；
- Freeform 平均更长、valid ratio 更高，Northwind 更短、padding 更高，但两类任务平均绝对误差接近，因此任务语境差异仍需和 pose/gaze、geometry、black artifact 等共同审计。

结论：temporal sampling 是次级但真实的混杂因素。后续优先运行 fixed uniform 与 temporal crop 训练消融，判断是否能缓解 prediction compression、severe 低估或 task inconsistency；在训练消融完成前，不应把当前 `stride_head` 采样直接定性为主要过拟合原因。

当前 temporal sampling 训练消融结论：

- `middle_crop` 整体最优：MAE `8.8014`、RMSE `10.7291`、Pearson `0.4192`、CCC `0.3806`；
- `middle_crop` 缓解 severe 低估，但 task diff mean 从 `2.89` 增至 `4.63`，说明 temporal location 与 task context 混杂；
- `uniform_256/512/1024` 几乎等价，不支持“采样帧数不足是主因”；
- `first_crop` 和 `random_crop` 都不适合作为主线替代；
- temporal training overfit summary 显示所有 temporal run 都是 `overfit_after_best_val=True`，说明采样替换不能解决训练后期记忆问题。

后续停止扩展普通 sampling 变体，转向 task inconsistency mixed-factor audit、severity calibration 和 identity retrieval。

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
  --run rgb=<LOG_DIR>/default/rgb/version_0/metrics.csv \
  --run center_mask=<LOG_DIR>/default/center_mask/version_0/metrics.csv \
  --run center_mask_black_to_gray=<LOG_DIR>/default/center_mask_black_to_gray/version_0/metrics.csv \
  --run border_black_feather=<LOG_DIR>/default/border_black_feather/version_0/metrics.csv
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

当前状态：已实现离线审计入口。

```text
src/diagnostics/alignment_geometry.py
scripts/audit_alignment_geometry.py
tests/test_alignment_geometry.py
```

输出：

```text
alignment_geometry_summary.csv
alignment_geometry_merged.csv
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

运行示例：

```bash
python scripts/audit_alignment_geometry.py \
  --predictions <LOG_DIR>/default/rgb/version_0/diagnostics/regression/test_predictions.csv \
  --openface-root /path/to/openface_csv_root \
  --output-dir <LOG_DIR>/default/rgb/version_0/diagnostics/alignment_geometry \
  --frame-width 112 \
  --frame-height 112 \
  --sample-step 1
```

判读：

- 若 geometry features 与 `abs_error` 或 `residual` 相关，应作为 shortcut risk。
- 若 severe 低估集中在异常 face scale、偏移或大姿态样本，应进行质量分层评估。
- 若 task inconsistency 与 face scale / center offset 差异相关，应把任务采集/对齐差异作为混杂因素报告。
- 若 OpenFace landmark 坐标来自原始视频坐标而非 aligned 112x112 坐标，必须通过 `--frame-width` 和 `--frame-height` 使用对应坐标尺度；本项目已根据 OpenFace camera parameters 使用 `640 x 480` 重跑，因此 normalized scale / offset 可作为原始检测坐标系下的相对几何指标解释。

当前真实运行结果：

```text
OpenFace videos summarized: 300
Matched prediction rows: 100
Missing prediction videos: 0
Max absolute correlation: 0.3746
```

主要发现：

```text
landmark_bbox_height_mean  vs true_bdi:  r = 0.3746
normalized_face_scale_mean vs true_bdi:  r = 0.3410
landmark_bbox_area_mean    vs true_bdi:  r = 0.3410
landmark_bbox_width_mean   vs true_bdi:  r = 0.3041
eye_distance_mean          vs true_bdi:  r = 0.2991
landmark_bbox_height_mean  vs residual:  r = -0.2605
```

坐标尺度修正与 640x480 重跑：

当前 OpenFace CSV landmark 坐标不是 112x112 aligned frame 坐标。示例 CSV 中 `x` 范围约 `150-643`，`y` 范围约 `-11-582`；而模型实际输入 jpg 为 `112 x 112`。OpenFace 日志中的 camera parameters `500,500,320,240` 提示源坐标系约为 `640 x 480`。因此，本节更准确的命名是 **pre-alignment detection geometry audit**：它量化的是 OpenFace 原始检测坐标系中的脸部尺度、bbox、眼距和 landmark 稳定性。这些因素可能通过裁剪、对齐、缩放、黑边填充和插值过程间接影响 112x112 RGB 输入。

解释边界：

- 可解释：`landmark_bbox_width/height/area/aspect`、`eye_distance`、`landmark_jitter` 的相关性。
- 在使用 `--frame-width 640 --frame-height 480` 重跑后可解释：`normalized_face_scale` 与 `face_center_offset_*` 的相对尺度。
- 仍需避免的误读：这些变量不是模型直接看到的 112x112 landmark 坐标，而是 OpenFace 原始检测/预处理阶段的几何混杂。
- 后续增强：输出 `landmark_x_min/x_max/y_min/y_max`、`eye_distance_to_bbox_height_ratio` 等不依赖固定 frame size 的相对几何指标。

640x480 重跑后的分组均值：

```text
minimal:  scale=0.319, residual=+6.89
mild:     scale=0.337, residual=-1.86
moderate: scale=0.399, residual=-7.66
severe:   scale=0.364, residual=-16.50
```

moderate / severe 的检测尺度更大，但预测没有同步上升，支持“检测几何混杂 + prediction compression”共同作用的解释。

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
扩展判读：眼镜、胡须、发际线、麦克风遮挡和局部反光应被视为 identity/static appearance 与 local occlusion 的交叉因素。它们可能不是 BDI 因果线索，但在小样本 subject-independent 设置中可能与某些 subject、任务录制条件或严重程度分布偶然共现。若 embedding retrieval 显示强同人聚类，后续 case study 应检查最近邻是否共享眼镜、胡须、麦克风、发型或局部遮挡，而不仅仅检查人脸整体相似。

当前 multi-run 结果：

```text
rgb_test                       same_top1=0.66, top5=0.85, severity_agree=0.492
center_mask_test               same_top1=0.67, top5=0.86, severity_agree=0.498
center_mask_black_to_gray_test same_top1=0.68, top5=0.84, severity_agree=0.522
border_black_feather_test      same_top1=0.75, top5=0.90, severity_agree=0.550
middle_crop_test               same_top1=0.49, top5=0.65, severity_agree=0.454
```

判读：

- `rgb` embedding 已强烈身份化；
- `border_black_feather` 预测表现较好但身份检索更强，说明 artifact smoothing 不等于去身份化；
- `center_mask` / `center_mask_black_to_gray` 没有在 test 上显著降低身份检索，不能称为 de-identification；
- `middle_crop` 降低身份检索，但 task consistency 变差，因此更像 temporal/task-context confound，而不是稳定 severity representation；
- 下一步需要 multi-run identity summary 工具，并与 prediction summary 合并分析。

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

RGB baseline 已完成真实运行：

```text
calibration: pred_calibrated = 0.870402 * pred + 2.689282
original:   MAE=8.9145, RMSE=10.9530, Pearson=0.3526, CCC=0.2925, pred_std=6.13, true_std=11.48
calibrated: MAE=8.8460, RMSE=10.8278, Pearson=0.3526, CCC=0.2692, pred_std=5.33, true_std=11.48
minimal residual: +6.89 -> +8.04
severe residual:  -16.50 -> -16.10
```

判读：

- RGB baseline 有弱排序信号，但预测动态范围被压缩；
- 简单线性校准不能修复 severe 低估，且会降低 CCC；
- 这说明后续不能只做 post-hoc calibration，应进入 multi-run severity calibration summary 与 severity-aware training ablation。

下一步具体研究方向：

1. 对所有关键输入/时序变体复跑 calibration，并输出统一表：`run, split, MAE/RMSE/Pearson/CCC, pred_std/true_std, minimal_residual, severe_residual, calibration_delta`。
2. 将 calibration summary 与 identity retrieval summary 合并，检查高 identity retrieval 是否伴随更强 prediction compression 或 severe underestimation。
3. 训练侧单独测试 severity-balanced sampler、severity-weighted regression loss、ordinal severity auxiliary head、Huber/CCC/mixed loss；每个实验必须同时报告 identity retrieval 与 task consistency，避免只通过抬高预测改善 severe。
4. 若某个 severity-aware 训练方法提高 `pred_std` 并缓解 severe 低估，但 identity retrieval 也升高，应解释为 severity/identity 纠缠，而不是直接称为泛化提升。

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

### P1-A OpenFace Boundary Hard-transition Smoothing

目的：验证 OpenFace aligned face 中黑区与脸部区域之间的硬突变边缘是否被 DeiT/ViT patch backbone 学成捷径。

候选变体：

```text
edge_soften_only
border_blur_fill
border_reflect_fill
border_feather_blur_fill
center_mask_soft_boundary_v2
```

优先实现 `edge_soften_only` 与 `border_blur_fill`。二者分别验证“边界高梯度”与“自然填充过渡”是否比固定灰色或简单 feather 更有效。对照组固定为 `rgb`、`center_mask`、`black_to_gray`、`border_black_feather` 和 `center_mask_black_to_gray`。

P1 只在 P0 证据完成后启动，避免继续经验试错。

1. **Patch / attention case study**：针对 severe 低估、minimal 高估、高 task diff、黑边改善/恶化 case 生成 attention、occlusion、keyframe 图组。
2. **Region / occlusion ablation**：`face_contour_erased`、`eye_mouth_only`、`upper_face_only`、`lower_face_only`、`glasses_region_erased`、`mouth_occluder_erased`、`beard_lower_face_erased`，用于验证行为区域、身份区域和局部遮挡区域。
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

## 三表联合阶段性证据

当前已将 input ablation prediction summary、identity retrieval summary 和 severity calibration summary 对齐到同一组核心 run：`rgb`、`center_mask`、`border_black_feather`、`middle_crop`、`center_mask_black_to_gray`。联合判读结果如下：

1. `center_mask_black_to_gray` 的 MAE 最低，但 severe bias 最差，说明整体 MAE 会被 minimal/mild 多数样本改善主导。
2. `center_mask` 的 CCC 最高、prediction std 最大、task consistency 接近 baseline，是当前最健康的输入侧 artifact mitigation 证据。
3. `border_black_feather` 的 severe bias 最轻，但 same-subject top-1/top-5 最高，说明边界软化不等于去身份化，可能伴随更强 subject/static appearance 表征。
4. `middle_crop` 降低 same-subject retrieval，但同时降低 severity agreement 并恶化 task consistency，说明 temporal location / task context 是独立混杂因素。
5. 所有 post-hoc linear calibration 均降低 CCC 并压缩 prediction std，因此 calibration 只能作为机制诊断，不应作为最终模型改进。

由此，当前 RGB 过拟合不应被解释为单一黑边、单一边界或单一 temporal sampling 问题，而应解释为 input artifact mitigation、identity/static appearance shortcut、severity range compression 和 task-context confound 的组合。
## 历史阶段：Shortcut-Regularized MTL 方案（已被 RPDF-Net 主线吸收）

本节保留为历史阶段记录。该方案曾将下一阶段从输入变体扩展收束为 **Shortcut-Regularized MTL**；截至 2026-06-25，它已被 RPDF-Net 主线吸收，其中 identity-adversarial MTL 与 severity-balanced regression 作为 RPDF-Net 的对照基线和支线验证。历史问题表述为：

> 静态 RGB depression regression 是否因为 subject-level shortcut 与 severity-level shortcut 学到非因果表征，从而导致泛化不稳定、身份记忆和中间分数段塌缩？

该规划保留 backbone 对人脸图像的丰富特征提取能力，但要求上层 MTL 显式约束共享表征：抑郁预测分支不能依赖 subject identity，回归损失不能被多数分数段主导。

### Stage A：收口诊断

Stage A 只保留两个收口任务，不再继续扩展普通输入滤镜、黑边替换、灰度/模糊或 mask 变体。

#### A1 Layer-wise Identity Probe

目的：定位身份信息在模型层级中的出现位置，判断身份可分性主要来自 backbone 中下层表征，还是上层回归/MTL head 对这些表征的利用。

输出：

```text
layer_name
same_subject_top1 / same_subject_top5
paired_rank_mean
subject classification accuracy 或 proxy accuracy
severity_agree / severity correlation
```

判读：若中间层已经有强 identity retrieval，而高层预测误差也与 identity similarity 耦合，则 Stage B 的 identity-adversarial MTL 有充分必要性。

#### A2 Prediction Error x Identity Similarity Coupling

目的：避免只证明“embedding 能识别身份”，却不能证明“预测使用了身份”。该任务检查 prediction residual / abs_error 是否随 identity similarity、same-subject rank、severity agreement 或 high-identity neighborhood 系统变化。

输出：

```text
error_identity_correlation.csv
high_error_high_identity_cases.csv
severity_bin_identity_error_summary.csv
identity_error_coupling_report.md
```

Stage A 完成条件：能够回答身份信息是否稳定存在、身份相似性是否与预测错误/偏置相关、是否支持进入 identity-adversarial MTL。

### Stage B：正式干预实验

Stage B 是当前主实验计划，只包含两类目标明确的上层干预：

1. `severity-balanced regression`：缓解 score-bin / severity-bin 不均衡。
2. `identity-adversarial MTL`：通过 GRL 和 identity head 抑制共享表征中的 subject 可用性。

正式实验组：

| 组别 | 机制 | 目的 |
|---|---|---|
| E0 | RGB MTL-Lite baseline | 固定对照 |
| E1 | + severity-balanced regression | 检验 severity imbalance 是否导致中间分数段塌缩 |
| E2 | + identity-adversarial branch | 检验上层 MTL 能否降低 subject shortcut |
| E3 | + severity-balanced regression + identity-adversarial branch | 检验两类 shortcut 是否互补 |

#### B1 Severity-Balanced Regression

目标不是提高预测值方差，而是降低标签分数段不均对梯度的支配，缓解 minimal / severe 等少数分段的系统偏置。

推荐起始损失：

```text
L_reg = mean( w(y_i) * SmoothL1(pred_i, y_i) )

w(y_i) = 1 / (smooth_count(bin(y_i)) + eps)^p
```

初始 sweep：

```text
p = 0.5
p = 1.0
bin = minimal / mild / moderate / severe
```

可作为后续扩展但不第一批全部加入：

```text
weighted MAE
MAE + eta * (1 - CCC)
ordinal severity auxiliary head
balanced sampler
```

判读指标：overall MAE/RMSE/CCC、各 severity group MAE/bias、prediction mean/std、calibration 后 CCC 是否继续下降、identity retrieval 是否恶化。

#### B2 Identity-Adversarial MTL

结构：

```text
shared representation
  -> depression regression head
  -> existing auxiliary MTL heads
  -> Gradient Reversal Layer -> subject identity head
```

训练逻辑：identity head 学习预测 subject ID；GRL 反转 identity loss 对共享表征的梯度，使共享表征对 subject ID 更不敏感。普通 identity classification head 不等于去身份化，必须使用 adversarial / GRL 机制。

建议 lambda sweep：

```text
lambda_id = 0.02, 0.05, 0.10, 0.20
```

不建议一开始设置过大，因为身份、年龄、脸型、表情和行为线索可能纠缠，过强去身份可能损伤有效面部行为信号。

判读指标：identity retrieval top1/top5 是否下降、subject proxy accuracy 是否下降、MAE/RMSE/CCC 是否稳定、severity group bias 是否改善、Freeform/Northwind task consistency 是否不恶化、train-val gap 是否缩小。

### Stage C：后续待考虑项，不列入当前实验计划

动态面部变化特征暂时不进入当前 Stage B 正式实验计划。它作为 Stage C 候选，触发条件是：Stage B 后 identity shortcut 或 static appearance shortcut 仍明显存在，或者 severity-balanced 与 identity-adversarial 只能改善偏置但不能带来更稳定的 severity representation。

候选方向仅保留为规划：

```text
feature_delta: delta f_t = f_t - f_{t-1}
OpenFace AU delta
landmark / pose / gaze delta
static + dynamic fusion
```

当前文档和任务表中，动态特征不应作为立即执行项，也不应与 Stage B 共同打包成一个“多模块模型”。

### 统一评估与停止规则

每个 Stage B 实验必须统一输出：

```text
prediction_run_summary
severity_bias_summary
identity_retrieval_run_summary
severity_calibration_run_summary
training_overfit_summary
task_consistency_summary
```

成功标准：

- E1 成功：少数 severity 分段 MAE/bias 改善，overall CCC 不明显下降，identity retrieval 不恶化。
- E2 成功：identity retrieval / subject proxy accuracy 下降，prediction metrics 不崩坏，severity bias 不恶化。
- E3 成功：同时满足 E1/E2 的核心约束，并且 train-val gap 或 task consistency 有改善。

停止规则：

- 若 E1 只抬高 severe prediction 但 CCC、task consistency 或 identity retrieval 恶化，不视为成功。
- 若 E2 降低 identity retrieval 但 severity agreement 或 CCC 明显下降，不视为去身份成功。
- 若 E3 没有优于 E1/E2，则不要继续堆更多模块，应回到 Stage A case-level coupling 分析。
- 在完成 Stage B 之前，不启动动态特征分支、optical flow、two-stream fusion 或新的输入滤镜族。

## RPDF-Net 与 Identity Suppression 方法谱系

当前身份记忆抑制不再以单独的输入滤镜或单独的 GRL 作为最终主线，而是作为 RPDF-Net 的信息分流问题处理：模型允许 backbone 和底层视觉表征充分保留面部信息，但要求上层共享表征把与预测相关、身份相关、artifact/domain 相关和残余噪声分开审计。

方法谱系在本项目中的定位如下：

| 方法族 | 本项目角色 | 当前结论 |
|---|---|---|
| 输入级身份/边界弱化 | 机制证据与对照 | 已证明 artifact mitigation 有价值，但不能单独解决身份记忆、severe bias 和 task context confound |
| subject-adversarial GRL | RPDF-Net 对照基线/支线 | 可验证“上层抑制身份可用性”是否有效，但不是最终结构本身 |
| severity-balanced regression | RPDF-Net 支线 | 用于缓解 BDI score-bin / severity-bin 不均衡，目标是降低少数分段偏置，而不是人为提高预测方差 |
| factorized disentanglement | RPDF-Net 主线 | 将共享表示分解为 `z_dep/z_m/z_id/z_art/z_res`，用可审计损失和风险指标验证每个因子的作用 |
| dynamic facial behavior | Stage D 待考虑项 | 只有当 RPDF-lite 仍无法保留足够抑郁行为线索时，再作为行为支线启动 |

RPDF-lite 的第一版应保持克制：

```text
H0 -> z_dep, z_m, z_id, z_art, z_res
prediction = Head([z_dep, alpha * z_m])
alpha in {0, 0.25, 0.5, 1.0}
```

其中 `z_m` 表示身份-抑郁交叠因子。它不能被完全否定，也不能无约束传递；应通过 `alpha` 或门控机制受控进入预测和下一层。后续两级递进版本再验证：

```text
H_k = Phi([z_dep^k, alpha_k * z_m^k])
```

判定模型是否“回到正轨”不能只看 BDI MAE/RMSE，也不能只看 identity retrieval 是否下降。每个关键实验都需要同时报告：

```text
BDI: MAE / RMSE / CCC
identity risk: same-subject top1/top5, paired rank, subject proxy accuracy
artifact risk: confidence/success, black/border ratio, geometry, edge gradient, center offset
severity bias: minimal/mild/moderate/severe MAE and bias, pred_std, calibration effect
task consistency: Freeform vs Northwind prediction agreement and task_diff
training behavior: train-val gap, best epoch, overfit speed
```

停止规则：若某个方法降低 identity retrieval 但明显损伤 CCC、severity agreement 或 task consistency，不能称为有效去身份；若某个方法改善 MAE 但 severe bias 或 prediction compression 更严重，也不能作为主线推进。RPDF-Net 的研究价值在于把这些风险放入同一个可检验框架，而不是继续零散堆叠输入处理、GRL、重加权和动态特征。
