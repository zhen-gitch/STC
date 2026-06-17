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
- Occlusion-Adaptive Deep Network for Robust Facial Expression Recognition, 2020: https://arxiv.org/abs/2005.06040
- A survey of face recognition techniques under occlusion, 2020: https://arxiv.org/abs/2006.11366
- When Face Recognition Meets Occlusion, 2021: https://arxiv.org/abs/2103.02805
- 这些研究支持一个关键风险判断：深度视觉模型可能优先学习身份、采集条件、质量差异等容易但不可迁移的特征。
- 对本项目启发：需要设计 identity/quality/pose/crop artifact 相关的诊断和消融，而不是只比较 backbone 或训练 epoch。
- 对遮挡因素的启发：眼镜、麦克风、胡须和口鼻周围遮挡会改变局部可见面部区域，既可能破坏表情/AU/landmark 观测，也可能形成稳定 subject 或采集条件标记；因此应作为 occlusion shortcut 审计对象，而不是简单当作噪声删除。

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

## 历史推荐实验顺序

以下顺序是项目早期在 behavior baseline 尚未完成、RGB 输入伪迹尚未充分诊断时的路线。当前已被 `docs/RGB_OVERFITTING_AUDIT_PLAN.md` 中的多因素过拟合审计路线取代，保留在此仅作为历史背景。

1. `E0_openface_quality_correlation`：OpenFace 质量、姿态、gaze、AU 与 BDI/误差相关性。
2. `E1_input_ablation`：RGB aligned、grayscale、masked face、landmark heatmap、landmark/AU/pose only 对照。
3. `E2_landmark_temporal_baseline`：landmark-only 时序回归。
4. `E3_au_pose_gaze_baseline`：AU/pose/gaze-only 时序回归。
5. `E4_rgb_behavior_late_fusion`：图像分支与行为分支后融合。
6. `E5_behavior_auxiliary_mtl`：以 AU、landmark motion、pose/gaze 为辅助任务的多任务模型。
7. `E6_multiscale_temporal`：多尺度 clip/video temporal aggregation。
8. `E7_loss_balancing`：在行为辅助任务稳定后，再做 uncertainty weighting、GradNorm、PCGrad 或任务权重消融。

## 当前结论

下一阶段不应继续主要押注 `FINETUNE_LAST_N_BLOCKS` 的层数搜索，也不应过早进入 RGB + behavior late fusion。更高价值的路线是先完成 RGB 过拟合多因素审计：

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

这一路线更符合论文项目的可解释性、可消融性和长期扩展需求，也能避免把黑边、身份外观、OpenFace 对齐几何、时序采样和标签压缩混成一个笼统原因。

Shortcut Audit 的最小可行版本应优先落地：

```text
OpenFace CSV -> subject-level quality summary
quality summary + predictions.csv -> correlation heatmap
quality summary + residual -> residual dependency report
shortcut_audit_report.md
```

## 2026-06-15 RGB 黑填充与硬边界伪迹假设

第一轮 RGB 输入消融后，当前研究重点从“继续堆叠 behavior / late fusion 任务”调整为“解释 RGB 输入模型为什么过拟合”。这一调整更有论文研究价值，因为它直接面向当前模型失败机制，而不是在未解释 RGB 捷径的情况下叠加更多模块。

### 观察

OpenFace aligned face 并不是完全干净的面部行为输入。样例帧显示：

- 脸部轮廓外由纯黑像素填充；
- 遮挡面部的麦克风等区域也可能呈纯黑块；
- 黑色填充与真实脸部区域之间存在硬像素突变；
- 对 DeiT/ViT 这类 patch-based 模型来说，这类高对比边界可能成为稳定但不可泛化的视觉捷径。

第一轮输入消融支持该怀疑：

- `center_mask` 当前测试表现最好，MAE 约 `7.94`，RMSE 约 `10.16`，CCC 约 `0.48`；
- 原始 `rgb` MAE 约 `8.91`，RMSE 约 `10.95`，CCC 约 `0.29`；
- `boundary_erased` 接近或略优于 `rgb`，但不如 `center_mask`；
- `grayscale` 和 `blur` 变差，说明颜色或高频身份纹理不是唯一主因。

### 新假设

```text
OpenFace aligned face 中的黑填充、硬裁剪边界和黑色遮挡块
  -> 被 RGB backbone 学成非行为捷径
  -> 导致 subject-level 泛化不稳、prediction compression 和 case-level 错误
```

这个假设比“背景过拟合”更准确，因为当前输入已经经过 OpenFace 对齐，真正残留的风险是 aligned crop 自身的处理伪迹。

### 下一轮验证

新增输入变体用于验证黑填充和硬边界机制：

- `black_to_gray`：近黑区域替换为中性灰；
- `black_to_mean`：近黑区域替换为当前帧非黑像素均值；
- `black_to_blur`：近黑区域替换为模糊估计；
- `soft_center_mask`：用软边界替代硬 mask；
- `inner_crop_resize`：裁掉外围黑边后 resize。

新增离线审计：

```text
aligned frames + prediction CSV
-> black_ratio / border_black_ratio / center_black_ratio / black_boundary_edge_ratio
-> correlation with true_bdi / pred_bdi / residual / abs_error
```

### 判读规则

- 如果 `black_to_gray`、`black_to_mean` 或 `black_to_blur` 明显优于 `rgb`，说明黑填充本身就是重要捷径。
- 如果 `soft_center_mask` 优于 `center_mask`，说明原先 mask 的硬边界仍在制造新伪迹，后续应使用软 mask 或更自然的图像修复。
- 如果 `inner_crop_resize` 明显改善，说明外围边界/黑边比中心行为区域更影响泛化。
- 如果所有黑伪迹变体都不能改善，但 `center_mask` 仍最好，应检查中心区域选择、面部行为区域和 attention/occlusion 之间的关系。
- 如果 severe 低估仍不改善，应把它作为独立问题继续分析，包括标签分布、loss/sampling、subject bias 和 depression severity 表达差异。

论文表述建议：

> OpenFace alignment removes much of the raw background, but it may introduce structured preprocessing artifacts such as black padding, hard crop boundaries, and black occlusion regions. These artifacts can become shortcuts for patch-based visual backbones. We therefore diagnose and ablate alignment artifacts before introducing additional behavior-fusion modules.

### 诊断后的修正

黑伪迹审计已完整匹配 100 个测试视频，结果显示黑区在 OpenFace aligned face 中非常普遍，但不是强线性解释变量。最大绝对相关约 `0.207`，说明不能把 RGB 过拟合简单归因于单一黑像素指标。

更合理的解释是：

- 边界黑区主要对应 OpenFace 对齐和裁剪填充，更适合作为 artifact 指标；
- 中心近黑像素语义混杂，可能是鼻孔、嘴角阴影、胡须、自然面部阴影、麦克风遮挡或其他真实遮挡，不能直接视作预处理伪迹；
- 高边界黑区四分位的测试误差明显高于低边界黑区四分位，约 `12.29` vs `7.45`，提示边界黑区是泛化风险因子；
- 在 moderate/severe 样本中，边界黑区越多，预测越容易偏低，但样本量较小，应作为 case study 和后续消融的线索；
- `black_to_gray` 优于 `rgb` 但弱于 `center_mask`，说明黑像素替换有帮助，但外围非行为区域、脸部轮廓、裁剪形状、尺度和姿态残留也可能共同造成过拟合。

下一步不再粗暴替换全部近黑区域，而应只处理与图像边界连通的黑区：

```text
border_black_to_gray
border_black_feather
center_mask_black_to_gray
```

这些变体的目标是验证：保留鼻孔、嘴部阴影、麦克风等中心黑区语义信息时，仅中和 OpenFace 边界填充是否还能改善泛化。

## 2026-06-15 RGB 过拟合因素地图

黑填充和硬边界伪迹只是 RGB 输入过拟合的一部分。结合当前实验、OpenFace aligned face 的数据形态、行为 baseline 结果和相关研究，后续应把 RGB 过拟合拆成以下可验证因素，而不是继续寻找单一原因。

### 1. 身份与静态外观捷径

人脸 RGB 图像天然包含强身份信息，包括脸型、年龄、性别、肤色、皱纹、眼袋、胡须、发际线、眼镜、皮肤纹理和面部胖瘦。小样本 subject-independent 任务中，视觉 backbone 很容易学习这些稳定外观，而不是学习跨 subject 的面部行为动态。
眼镜、麦克风、胡须等因素尤其值得作为子问题审计。它们一方面是身份/外观线索，另一方面也是局部遮挡或局部高对比 artifact：眼镜会带来镜框边缘和反光，麦克风可能形成口部附近黑色遮挡块，胡须会改变下半脸纹理和嘴部边界。这些线索不一定与抑郁有因果关系，但在 AVEC2014 这类小样本视频任务中，可能与 subject、任务录制条件、说话方式或 severe/minimal 分布偶然共现，从而成为 shortcut。

相关研究线索：

- Shortcut Learning in Deep Neural Networks: https://arxiv.org/abs/2004.07780
- IDEnNet / Identity-Enhanced Network for Facial Expression Recognition: https://arxiv.org/abs/1812.04207
- FacialPulse: https://arxiv.org/abs/2408.03499

本项目证据：

- `center_mask` 优于 `rgb`，说明去除外围区域和部分静态外观后泛化改善；
- `blur` 变差，说明不是所有细节都是坏信息，模型仍需要局部面部线索；
- behavior-only full OpenFace 特征强过拟合，说明 raw landmark 和 static geometry 也可能携带身份信息。

建议验证：

- 设计 `face_contour_erased`、`eye_mouth_only`、`upper_face_only`、`lower_face_only` 等区域消融；
- 增加 `glasses_region_erased`、`mouth_occluder_erased`、`beard_lower_face_erased` 或手工标注 case study，用于验证眼镜、麦克风和胡须区域是否驱动预测；
- 检查 embedding 是否按 subject、脸型或外观聚类，而不是按 BDI 聚类；
- 训练 subject/identity proxy classifier，测试当前 RGB embedding 是否容易预测 subject。

### 2. OpenFace 对齐几何与裁剪伪迹

除了黑边，OpenFace aligned face 还可能保留或引入脸部尺度、裁剪位置、眼距、脸部中心偏移、插值模糊、头发/衣领/麦克风残留和轮廓形状等几何线索。这些变量可能和数据采集条件、subject 或任务相关。

最新坐标尺度确认显示：当前 OpenFace CSV 中的 landmark `x_*` / `y_*` 不是模型输入的 112x112 aligned face 坐标，而是 OpenFace 原始检测坐标系。示例 CSV 的 `x` 范围约 `150-643`、`y` 范围约 `-11-582`，而实际输入 jpg 为 `112 x 112`。OpenFace 日志中的 camera parameters `500,500,320,240` 提示源坐标系约为 `640 x 480`，因此 geometry audit 已按该尺度重跑。该方向更准确的表述应是 **pre-alignment detection geometry confound**：原始检测阶段的 face scale、bbox、eye distance 和 landmark jitter 可能通过后续裁剪、对齐、缩放、黑边填充和插值过程间接影响 112x112 RGB 输入。

建议验证：

- 从 landmark 外接框统计 face scale、face center offset、eye distance、bbox aspect ratio；
- 统计 aligned 后脸部是否偏上、偏下、偏左或偏右；
- 将这些几何量与 `true_bdi`、`pred_bdi`、`residual`、`abs_error` 做相关性；
- 对 high-error case 检查是否同时存在异常尺度、偏移或裁剪残留。

当前 C 任务结果：

- `300` 个 OpenFace CSV 视频汇总，`100/100` 个 test prediction row 成功匹配；
- 最大绝对相关约 `0.3746`；
- `landmark_bbox_height_mean` 与 `true_bdi` 相关约 `r = 0.375`；
- `landmark_bbox_area_mean` 与 `true_bdi` 相关约 `r = 0.341`；
- `landmark_bbox_width_mean` 与 `true_bdi` 相关约 `r = 0.304`；
- `eye_distance_mean` 与 `true_bdi` 相关约 `r = 0.299`；
- `landmark_bbox_height_mean` 与 `residual` 相关约 `r = -0.260`，提示较大的检测 bbox 与高 BDI 样本低估有关。
- 使用 `640 x 480` 源坐标尺度重跑后，`normalized_face_scale_mean` 约在 `0.30-0.40`，并与 `true_bdi` 相关约 `r = 0.341`。

论文表述边界：

- 可以说 OpenFace 原始检测几何与 BDI/severity 和 residual 存在中等相关，是 RGB shortcut 的候选混杂因素；
- 不应说模型直接看到了更大的 112x112 landmark scale，因为模型输入帧已经 resize 到 112x112；
- `normalized_face_scale_mean` 和 `face_center_offset_*` 在使用 `640 x 480` 源坐标尺度后可作为相对检测几何指标解释，但仍不能表述为模型直接观察到的 112x112 坐标。

### 3. 姿态、gaze 与追踪质量捷径

OpenFace 的 `confidence`、`success`、pose、gaze、landmark jitter 既可能是真实行为线索，也可能是追踪质量或采集条件混杂变量。抑郁相关行为可能包括低头、凝视减少和面部动作减少，但模型也可能只是学到了低质量追踪或特定姿态。

相关研究线索：

- OpenFace / OpenFace 3.0: https://arxiv.org/abs/2506.02891
- LibreFace: https://arxiv.org/abs/2308.10713
- Action Unit depression biomarkers: https://arxiv.org/abs/2407.13753

建议验证：

- 扩展 Shortcut Audit，加入 pose/gaze mean/std、confidence mean/std、failed frame ratio 和 landmark jitter；
- 使用 grouped-CV shortcut-only predictor 判断质量/姿态变量能否预测误差；
- 对 severe 低估 case 检查是否伴随低 confidence、大姿态或高 jitter。

### 4. 视频长度、采样与 padding 捷径

黑伪迹审计中最高相关项是 `frame_count` / `sampled_frame_count` 与 `pred_bdi`，约 `r = -0.207`。这提示模型预测可能受视频长度、有效帧数、采样覆盖或 padding/mask 影响。

可能机制：

- 不同任务视频长度不同；
- 长视频包含更多中性片段，稀释抑郁相关行为；
- `MAX_SEQ_LEN` 截断导致长视频关键片段丢失；
- chunk sampling 或均匀采样方式影响关键表情/动作覆盖；
- 有效帧比例影响 temporal pooling。

建议验证：

- 统计 `frame_count`、`sampled_frame_count`、`valid_ratio` 与预测和误差的相关性；
- 对比固定帧数均匀采样，例如 256 / 512 / 1024；
- 对比 first / middle / uniform / random temporal crop；
- 用 temporal occlusion 检查模型是否依赖少数片段。

当前 RGB prediction 审计结果已经支持把该项保留为 P0 混杂因素：`100/100` 个 test prediction row 成功匹配，最大绝对相关约 `0.220`。其中 `truncated_frame_count` 与 `pred_bdi` 相关约 `r = -0.220`，`truncated_ratio` 与 `pred_bdi` 相关约 `r = -0.209`，`frame_count` / `sampled_frame_count` 与 `pred_bdi` 相关约 `r = -0.207`。长视频四分位预测更低、误差更高且无 padding，说明问题更可能来自首段采样、长视频截断或关键片段覆盖不足，而不是简单 padding。该证据弱于 alignment geometry，但足以支撑 fixed uniform 与 temporal crop 消融。

### 5. Freeform/Northwind 任务语境差异

同一 subject 在 Freeform 和 Northwind 中可能表现出不同的说话内容、眼神方向、头部运动、朗读节奏、表情强度、遮挡和视频长度。若模型对任务语境敏感，它学到的可能是 task-specific visual pattern，而不是稳定 depression trait。

建议验证：

- 分任务报告 MAE/RMSE/Pearson/CCC；
- 对同一 subject 的 Freeform/Northwind 预测差异排名；
- 检查 task inconsistency 是否与视频长度、pose、gaze、黑边或追踪质量相关；
- task name 只作为诊断变量，不应在当前阶段直接作为训练输入。

### 6. 标签分布与 prediction compression

当前 RGB 模型存在明显预测压缩：真实 BDI 标准差约 `11.5`，而 `rgb` 预测标准差约 `6.1`。这会导致 minimal 高估、severe 低估。该问题不一定由输入伪迹造成，也可能来自 MSE/MAE 在小样本长尾标签分布下向均值收缩。

建议验证：

- 每个实验都报告 prediction mean/std 与 true mean/std；
- 做 severity group calibration；
- 尝试 severity-balanced sampler 或 label-bin balanced sampler；
- 尝试加权 MSE / Huber / CCC loss，但必须和输入捷径诊断分开做。

### 7. ViT/DeiT patch 级捷径

DeiT/ViT 这类 patch-based backbone 可能对局部高对比 patch、边界 patch、麦克风黑块、眼镜反光和裁剪残留非常敏感。黑边的全局相关性弱，不代表 patch-level attention 中没有局部捷径。

建议验证：

- 使用 attention rollout、Grad-CAM 或 input-gradient 诊断；
- 做 patch occlusion sensitivity；
- 分区域统计 eye、brow、mouth、lower face、face contour、boundary patch 的遮挡影响；
- 将高误差和低误差 case 都纳入可视化，避免只看失败样本。

### 8. 数据增强与鲁棒性不足

如果训练增强不能覆盖 test 中的 crop、黑边、光照、姿态和尺度变化，模型会记住固定预处理风格。后续增强应面向 artifact robustness，而不是一次性加入强增强。

建议优先级：

1. border fill randomization；
2. small affine jitter；
3. mild brightness/contrast jitter；
4. temporal random crop 或 uniform sampling；
5. 只在完成诊断后，再考虑更强的数据增强组合。

### 推荐排查顺序

```text
border-connected black/crop artifact
-> frame_count / temporal sampling audit
-> face scale / alignment offset / landmark bbox audit
-> identity/static appearance audit
-> task inconsistency audit
-> severity calibration and prediction compression
-> targeted robustness augmentation
```

当前研究表述应避免说“RGB 过拟合由黑边导致”。更准确的表述是：

> RGB 过拟合可能来自身份静态外观、OpenFace 对齐几何、边界填充、姿态/追踪质量、视频长度/采样、任务语境差异和标签分布压缩的共同作用。黑边是可见且可操作的风险入口，但不是单一充分解释。

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

## 2026-06-15 高优先级过拟合验证路线

权威路线文档：`docs/RGB_OVERFITTING_AUDIT_PLAN.md`。本节记录研究动机与论文表述，具体执行清单以 `docs/TODO.md` 为准，具体输出规格以 `docs/SHORTCUT_AUDIT_DESIGN.md` 为准。

在完成第一轮 RGB 输入消融、黑伪迹审计、边界连通黑区消融和统一 prediction summary 后，当前研究重点应从“继续尝试更多输入 mask”转向“验证 RGB 过拟合的关键机制”。原因是现有结果已经说明：

- `center_mask_black_to_gray`、`center_mask`、`border_black_feather` 均能不同程度改善原始 `rgb`；
- 但这些变体仍未解决 prediction compression；
- `center_mask_black_to_gray` 虽然整体 MAE 最好，却加重 severe 低估；
- `gray_scale`、`blur`、`inner_crop_resize`、`black_to_mean` 不支持颜色、纹理或外围区域任一单因素解释；
- 因此继续做相似 mask 的边际论文价值下降。

更值得优先验证的高价值问题如下。

### 1. Split / subject integrity

所有 subject-independent 泛化结论都依赖 split 正确性。必须确认：

- train/val/test 是否 subject-disjoint；
- 同一 subject 的 Freeform/Northwind 是否被放入同一 split；
- video_id 规范化是否导致误匹配；
- 是否存在重复视频目录、重复标签、标签文件与视频目录前缀不一致。

该审计不直接证明过拟合来源，但它是所有后续结论的有效性前提。

### 2. Temporal sampling and sequence coverage

当前黑伪迹审计中，`frame_count` / `sampled_frame_count` 与 `pred_bdi` 的相关性最强，提示模型可能对视频长度、采样位置、截断和有效帧比例敏感。高优先级验证包括：

- 真实数据上运行 temporal sampling audit；
- 对比 `uniform_256`、`uniform_512`、`uniform_1024`；
- 对比 `first_crop`、`middle_crop`、`random_crop`；
- 将每组结果接入 prediction run summary，报告 prediction std、severity bias 和 task consistency。

如果固定帧数采样显著缓解 prediction compression 或 task inconsistency，则当前均值池化/截断策略可能是主要过拟合入口之一。

### 3. Training curve overfit gap

仅看 test MAE 不足以解释过拟合。应跨实验统计：

- best validation epoch；
- train RMSE / MAE 与 val RMSE / MAE 的 gap；
- val 最优后 train 是否继续下降但 val/test 不改善；
- behavior baseline、RGB baseline 和各输入消融的 gap 是否一致。

该分析可以区分两类情况：输入变体真正改善泛化，或只是改变预测分布和测试集偏置。

### 4. OpenFace alignment geometry

OpenFace aligned face 不只包含黑边，还包含 face scale、bbox shape、face center offset、eye distance、轮廓形状和插值痕迹。建议把 alignment geometry audit 提升到 P0：

- 从 OpenFace landmark 坐标计算 bbox area/width/height/aspect；
- 计算 face center offset、eye distance、normalized face scale；
- 与 `true_bdi`、`pred_bdi`、`residual`、`abs_error` 做相关；
- 与 severe 低估、minimal 高估和 task inconsistency 做分组比较。

如果几何变量能解释误差或任务不一致，则 RGB 模型可能利用了对齐几何和静态人脸尺度，而不是抑郁相关行为。

### 5. Embedding identity retrieval

身份信息不建议首先用 subject classifier 验证，因为 subject-independent split 中 test subject 未在 train 出现。更适合先做 paired-task retrieval：

- 提取同一模型的 test embedding；
- 对每个 Freeform 样本查找最近邻，看 Northwind paired sample 是否排在 top-k；
- 反向从 Northwind 查 Freeform；
- 同时比较 embedding 是否按 BDI severity 聚类。

如果同 subject 的两个任务 embedding 高度互为近邻，而 severity 聚类弱，说明 RGB backbone 更强地编码了身份/静态外观。

### 6. Severity calibration verification

当前所有较好输入变体仍存在 `pred_std < true_std`。应使用 val set 拟合简单 post-hoc calibration，并只在 test 上评估：

```text
pred_calibrated = a * pred + b
```

目的不是调出最终模型，而是验证 severe 低估和 minimal 高估是否主要来自 prediction compression。如果 calibration 明显改善 severe/minimal 但不改变排序相关性，则后续应单独研究 severity-aware sampler、weighted loss、Huber/CCC loss。

### 7. Task inconsistency mixed-factor audit

Freeform/Northwind 同 subject prediction diff 需要与多因素关联：

- frame_count / sampled_frame_count；
- black_border_ratio；
- confidence / success；
- pose / gaze；
- alignment geometry；
- prediction severity bias。

如果 task inconsistency 能被这些变量解释，论文中应把任务语境作为混杂因素，而不是把两个任务简单视作同分布重复样本。

### 更新后的论文主线

推荐将论文实验路线组织为：

```text
1. Baseline failure: prediction compression, severe underestimate, task inconsistency
2. RGB input artifact ablation: center mask and border artifact evidence
3. Multi-factor shortcut audit: temporal, geometry, identity, quality, task context
4. Calibration analysis: separate shortcut mitigation from severity compression
5. Behavior feature baseline: evaluate whether structured facial dynamics generalize
```

核心表述应保持克制：

> 黑边和硬边界伪迹是 RGB 过拟合的重要可见入口，但不是唯一原因。当前证据更支持一个多因素解释：RGB 模型同时受到 OpenFace 对齐几何、身份静态外观、时序采样、任务语境和标签分布压缩的影响。
