# RESEARCH_NOTES.md

本文档记录与当前论文主线直接相关的研究背景、论文线索和后续实验方向。项目后续默认以中文维护研究笔记，论文题名、模型名、数据集名和链接保留英文。

## 当前问题判断

当前项目使用的是经过 OpenFace 裁剪和对齐后的人脸视频帧序列。因此，模型失效风险不应简单理解为“原始背景过拟合”，而应更准确地表述为：

> 模型可能在 OpenFace aligned face 中学习了身份、纹理、裁剪边界、对齐伪影、姿态残留、追踪质量、光照和视频质量等非抑郁捷径，而不是跨 subject 稳定的面部行为动态。

已经观察到的现象包括：

- regression-only baseline 在训练集上可以持续拟合，但验证集和测试集泛化不稳定；
- 冻结 backbone 底层、只微调最后 1 或 2 个 transformer blocks 后，并未明显改善 test 表现；
- last1/last2 结果提示问题不只是 backbone 可训练层数，而更可能是输入表征和监督信号没有充分约束模型关注抑郁相关面部行为；
- 当前 ordinal BDI 辅助任务本质上仍来自同一个 BDI 标签，可能不足以强迫模型学习 AU、landmark motion、gaze、pose 等行为线索。

## OpenFace 版本与数据约定

当前主数据版本来自已有 OpenFace 裁剪对齐流程，所用 OpenFace 版本可能不是最新版。短期内不建议直接升级 OpenFace 并覆盖已有数据，因为这会改变 crop、landmark、AU、pose、confidence 等分布，相当于更换数据版本。

推荐约定：

- 保留当前 OpenFace 版本生成的数据作为主数据版本；
- 明确记录 OpenFace 版本、命令、输出字段、裁剪尺寸和帧采样方式；
- 后续如需使用 OpenFace 3.0、LibreFace 或其他工具，应作为独立数据版本和消融实验；
- 不混用不同 OpenFace 版本生成的帧、landmark 或 AU 特征；
- 若存在 OpenFace 原始 CSV，应优先将 `confidence`、`success`、pose、gaze、AU、landmark 等结构化输出纳入诊断。

## 相关研究线索

### AVEC2014 与面部视频抑郁预测

- AVEC2014 / Audio-Visual Emotion Challenge 是当前项目数据设定的重要参照，BDI-II 连续分数预测是典型任务形式。
- 需要在论文中明确说明 split、subject 独立性、评价指标和是否使用 validation/test 标签参与任何统计。

### Temporal facial landmarks

- FacialPulse, 2024: https://arxiv.org/abs/2408.03499
- 研究动机与当前项目高度相关：端到端图像特征可能包含大量冗余和身份信息，而 temporal facial landmarks 更接近面部行为动态。
- 对本项目启发：应建立 landmark-only temporal baseline，并比较其与 RGB aligned face baseline 的泛化差异。

### 多尺度时序与抑郁相关特征增强

- Two-stage Temporal Modelling Framework, 2021: https://arxiv.org/abs/2111.15266
- 该方向强调短时行为片段、多尺度时序建模和 Depression Feature Enhancement，用于增强抑郁相关线索并抑制非抑郁噪声。
- 对本项目启发：后续可从简单的 clip-level temporal pooling、keyframe weighting、temporal occlusion 开始，不必一开始复现复杂图结构。

### Action Units 与面部行为 biomarker

- Exploring Facial Biomarkers for Depression through Temporal Analysis of Action Units, 2024: https://arxiv.org/abs/2407.13753
- 该方向强调 AU、expression、temporal statistics 与抑郁状态之间的关系。
- 对本项目启发：MTL 辅助任务应优先考虑 AU intensity、AU presence、expression distribution、landmark motion、pose/gaze 等行为信号，而不是只使用 BDI ordinal 分箱。

### OpenFace / LibreFace / OpenFace 3.0

- OpenFace 3.0, 2025: https://arxiv.org/abs/2506.02891
- LibreFace, 2023: https://arxiv.org/abs/2308.10713
- OpenFace 相关工具链提供 landmark、AU、head pose、gaze、confidence、success 等结构化面部行为输出。
- 对本项目启发：当前 OpenFace aligned frames 不应只作为图像输入，也应尽量利用 OpenFace CSV 作为诊断特征、辅助监督或轻量行为分支输入。

### 去身份化与捷径学习

- OpticalDR, 2024: https://arxiv.org/abs/2402.18786
- Shortcut Learning in Deep Neural Networks, 2020: https://arxiv.org/abs/2004.07780
- 这些研究支持一个关键风险判断：深度视觉模型可能优先学习身份、采集条件、质量差异等容易但不可迁移的特征。
- 对本项目启发：需要设计 identity/quality/pose/crop artifact 相关的诊断和消融，而不是只比较 backbone 或训练 epoch。

### 多任务损失与负迁移

- GradNorm: https://arxiv.org/abs/1711.02257
- Uncertainty Weighting: https://arxiv.org/abs/1705.07115
- PCGrad: https://arxiv.org/abs/2001.06782
- 这些方法不是当前 MTL-Lite 主线的第一优先级，但在引入 AU、landmark、pose、gaze 等辅助任务后，可作为负迁移控制和任务权重消融。

## 下一阶段实验方向

### P0：诊断模型是否学习了非抑郁捷径

详细实施方案见 `docs/SHORTCUT_AUDIT_DESIGN.md`。该框架命名为 Shortcut Audit Framework，目标是在继续修改模型前，先验证 OpenFace aligned face 中的身份、追踪质量、姿态、裁剪伪影、视频质量等非抑郁变量是否与 BDI、预测值、残差或绝对误差存在关系。

1. OpenFace 质量统计
   - 统计每个视频的 `confidence` 均值、方差、低置信帧比例；
   - 统计 `success` 失败帧比例；
   - 统计 pose/gaze 分布和 landmark 抖动；
   - 分析这些变量与 BDI、预测误差、残差、subject 的相关性。

2. 输入消融
   - aligned RGB face；
   - grayscale aligned face；
   - masked face，弱化脸部边界、头发、衣服、裁剪边缘；
   - landmark heatmap；
   - landmark/AU/pose only；
   - 低频或模糊图像，用于判断模型是否依赖细粒度身份纹理。

3. 归因与遮挡分析
   - model attention / Grad-CAM / input-gradient；
   - occlusion sensitivity；
   - keyframe importance；
   - 高误差和低误差 subject case study；
   - 重点检查模型关注区域是否集中在眼、眉、嘴、鼻唇沟，而不是脸部边缘、头发、眼镜、黑边或裁剪伪影。

### P1：建立行为表征 baseline

1. Landmark-only temporal baseline
   - 输入 OpenFace landmark 坐标；
   - 派生速度、加速度、关键区域距离；
   - 使用 GRU、TCN 或轻量 Transformer 做 video-level BDI 回归。

2. AU / pose / gaze baseline
   - 输入 AU intensity / AU presence、head pose、gaze、confidence/success mask；
   - 建立轻量时序模型；
   - 与 RGB baseline 在相同 split、seed、metric 下比较。

3. RGB + behavior late fusion
   - RGB branch 使用现有 MTL-Lite 图像分支；
   - behavior branch 使用 landmark/AU/pose/gaze 序列；
   - video-level representation 后 concat + MLP；
   - 先做 regression-only，再考虑多任务。

### P2：重构多任务学习目标

当前 BDI ordinal 辅助任务可保留为 baseline，但后续更值得尝试的辅助任务包括：

- AU intensity reconstruction；
- AU presence classification；
- landmark motion prediction；
- pose/gaze prediction；
- expression distribution prediction；
- temporal smoothness 或 motion contrast；
- frame quality / OpenFace confidence prediction，仅作为诊断或辅助约束。

引入这些任务后，再考虑 uncertainty weighting、GradNorm、PCGrad 或简单任务权重网格搜索。

## 推荐实验顺序

1. `E0_openface_quality_correlation`：OpenFace 质量、姿态、gaze、AU 与 BDI/误差相关性。
2. `E1_input_ablation`：RGB aligned、grayscale、masked face、landmark heatmap、landmark/AU/pose only 对照。
3. `E2_landmark_temporal_baseline`：landmark-only 时序回归。
4. `E3_au_pose_gaze_baseline`：AU/pose/gaze-only 时序回归。
5. `E4_rgb_behavior_late_fusion`：图像分支与行为分支后融合。
6. `E5_behavior_auxiliary_mtl`：以 AU、landmark motion、pose/gaze 为辅助任务的多任务模型。
7. `E6_multiscale_temporal`：多尺度 clip/video temporal aggregation。
8. `E7_loss_balancing`：在行为辅助任务稳定后，再做 uncertainty weighting、GradNorm、PCGrad 或任务权重消融。

## 当前结论

下一阶段不应继续主要押注 `FINETUNE_LAST_N_BLOCKS` 的层数搜索。更高价值的路线是：

```text
OpenFace aligned RGB baseline
-> OpenFace 质量与捷径诊断
-> landmark/AU/pose/gaze 行为 baseline
-> RGB + behavior late fusion
-> 面部行为辅助任务的 MTL-Lite
```

这一路线更符合论文项目的可解释性、可消融性和长期扩展需求。

Shortcut Audit 的最小可行版本应优先落地：

```text
OpenFace CSV -> subject-level quality summary
quality summary + predictions.csv -> correlation heatmap
quality summary + residual -> residual dependency report
shortcut_audit_report.md
```

## 2026-06-14 Behavior-only baseline 结果后的研究路线修订

最新 behavior-only baseline 使用 OpenFace 结构化特征进行 BDI 回归，结果显示训练集拟合很强但泛化不足：test MAE 约 `9.93`，RMSE 约 `12.86`，CCC 约 `0.151`；best validation RMSE 约 `12.38`，但对应 train RMSE 只有约 `2.74`。这说明 OpenFace 行为表征路线仍然有研究价值，但不能直接把“所有 OpenFace 特征”视为可靠行为表征。

新的研究假设：

- AU、landmark motion、pose/gaze motion 可能比 raw landmark 坐标更接近可泛化的行为动态；
- raw landmark coordinates 和静态 facial geometry 可能携带较强身份线索；
- OpenFace quality、confidence、success 和 tracking stability 既可作为质量控制变量，也可能成为预测捷径；
- behavior-only baseline 弱于 RGB 不代表行为线索无效，可能是特征组混杂、模型容量过大或评价粒度不够导致。

下一阶段实验应优先回答三个问题：

1. 哪些 OpenFace 特征组可以在 subject-level 泛化中稳定降低误差？
2. RGB/MTL-Lite 与 behavior-only 的错误样本是否重叠，还是互补？
3. severe 低估、minimal 高估和 Freeform/Northwind 不一致是否能被某些行为特征组解释或缓解？

因此，行为路线的实验顺序调整为：

```text
behavior prediction export
-> feature-group ablation
-> RGB vs behavior case overlap
-> stable behavior subset
-> late fusion
-> behavior auxiliary MTL
```

论文表述上，当前 behavior-only baseline 可作为一个重要诊断结论：直接使用完整 OpenFace CSV 特征并不会自动获得可泛化抑郁表征，必须通过特征组消融、去身份化和行为动态约束来筛选可靠线索。
