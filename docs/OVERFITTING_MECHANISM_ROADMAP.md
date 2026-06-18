# OVERFITTING_MECHANISM_ROADMAP.md

本文档用于系统梳理 RGB 输入模型过拟合的机制研究路线。它不是新的单点实验清单，而是当前所有审计、消融和论文叙事的上层框架。后续具体实验配置仍放在 `RGB_OVERFITTING_AUDIT_PLAN.md`、`SHORTCUT_AUDIT_DESIGN.md` 和 `TODO.md` 中。

## 1. 研究目标

当前目标不再是寻找“某一个 artifact 导致过拟合”，而是建立一套可解释、可验证、可复用的机制审计协议：

```text
OpenFace aligned face RGB model failure
-> split / label validity
-> prediction compression and calibration
-> input artifact and local occlusion
-> identity / static appearance shortcut
-> OpenFace geometry / quality confounds
-> temporal sampling and task context
-> model capacity / optimization overfit
-> behavior-oriented representation validation
```

论文价值来自两点：

1. 把 RGB 模型的失败模式从经验现象拆成可验证机制；
2. 证明后续模型改进不是盲目堆模块，而是由机制审计驱动。

## 2. 文献依据

### Shortcut learning

Shortcut learning 研究指出，深度模型可能学习在标准测试条件下表现良好、但不具有稳健迁移能力的捷径规则。当前项目中，黑边、眼镜、胡须、麦克风、OpenFace geometry、视频长度和任务语境都可能成为这种捷径。

参考：

- Shortcut Learning in Deep Neural Networks: https://arxiv.org/abs/2004.07780

### ViT / DeiT 小样本风险

当前 RGB backbone 属于 patch-based visual transformer 路线。ViT 类模型在小数据集上缺少 CNN 的局部归纳偏置，容易依赖训练集中可重复出现的局部 patch 模式。对 `112 x 112` 输入和 `16 x 16` patch 来说，一帧约只有 `7 x 7` 个 patch，黑边、麦克风黑块、眼镜反光、胡须边界和裁剪残留都可能占据完整 patch。

参考：

- Efficient Training of Visual Transformers with Small Datasets: https://arxiv.org/abs/2106.03746

### 面部行为表征

抑郁识别更合理的视觉信号应来自稳定面部行为动态，例如 AU、landmark motion、pose/gaze dynamics，而不是冗余 RGB 外观。FacialPulse 和 AU biomarker 相关研究都支持用 temporal facial landmarks / AUs 作为行为对照，而不是只依赖端到端 RGB。

参考：

- FacialPulse: https://arxiv.org/abs/2408.03499
- Exploring Facial Biomarkers for Depression through Temporal Analysis of Action Units: https://arxiv.org/abs/2407.13753
- LibreFace: https://arxiv.org/abs/2308.10713
- OpenFace 3.0: https://arxiv.org/abs/2506.02891

### 局部遮挡与饰物

面部遮挡研究说明，眼镜、口鼻遮挡、胡须、麦克风等会改变局部可见面部区域，影响表情识别、身份识别和特征鲁棒性。它们在本项目中既可能是身份/静态外观线索，也可能是局部遮挡 artifact。

参考：

- Occlusion-Adaptive Deep Network for Robust Facial Expression Recognition: https://arxiv.org/abs/2005.06040
- A survey of face recognition techniques under occlusion: https://arxiv.org/abs/2006.11366
- When Face Recognition Meets Occlusion: https://arxiv.org/abs/2103.02805

## 3. 当前证据状态

### 已排除或弱化的解释

1. **Split 泄漏不是当前主因**
   - split integrity audit 已显示 train/val/test subject-disjoint；prediction alignment 100/100 匹配。

2. **单纯黑边不是充分解释**
   - 黑区普遍存在，高边界黑区组误差更高；但单一黑像素指标最大相关约 `0.207`。
   - 边界黑区处理能改善部分结果，但 severe 低估和 prediction compression 仍存在。

3. **单纯采样帧数不足不是主因**
   - `uniform_256/512/1024` 表现几乎一致。
   - temporal sampling 影响预测，但不能解决训练后期过拟合。

4. **完整 OpenFace behavior features 不是天然干净行为信号**
   - behavior-only baseline 在训练集上强拟合，但泛化弱，说明 raw landmark / geometry / quality 等也可能携带 subject/static shortcut。

### 仍成立的核心风险

1. **Prediction compression**
   - 多个 RGB 变体仍 `pred_std < true_std`。
   - RGB baseline severity calibration 已确认：test `pred_std=6.13`，`true_std=11.48`；severe residual `-16.50`，线性校准后仍为 `-16.10`。
   - minimal 高估和 severe 低估持续存在，且不能被简单 validation-fit 线性校准修复。

2. **Identity / static appearance shortcut**
   - 人脸 RGB 天然包含 subject identity。
   - blur / grayscale 没有简单改善，说明有效线索与身份/纹理可能纠缠。

3. **OpenFace pre-alignment geometry confound**
   - landmark bbox、face scale、eye distance 与 BDI / residual 存在中等相关。
   - 当前坐标是原始检测坐标系，不是 112x112 输入坐标，但可能通过裁剪、对齐、缩放、黑边填充和插值影响最终 RGB。

4. **Temporal location and task context confound**
   - `middle_crop` 改善整体指标和 severe 低估，但 task consistency 明显变差。
   - Freeform/Northwind 的任务语境、视频长度、gaze、说话方式和遮挡条件不能视为完全同分布重复样本。

5. **Local occlusion / accessory shortcut**
   - 眼镜、麦克风、胡须、反光、口鼻周围遮挡可能同时具有 identity marker、occlusion artifact 和 high-contrast patch 三重属性。

## 4. 机制层级图

### Layer 0: 数据和评估有效性

问题：当前结论是否建立在干净 split 和正确预测对齐上？

已完成：

- split integrity audit；
- prediction alignment check；
- training overfit summary。

判读：若本层失败，所有后续机制解释都暂停。

### Layer 1: 标签分布和预测压缩

问题：模型是否只是向训练集/验证集均值收缩？

现象：

- true BDI 方差远大于 pred 方差；
- severe 系统性低估；
- minimal 系统性高估。

已完成：

- RGB baseline validation-fit / test-apply linear calibration；
- severity group calibration report；
- 结果显示简单线性校准只轻微改善 MAE/RMSE，Pearson 不变，CCC 下降，severe 低估几乎不变。

下一步：

- 将 calibration verification 扩展到所有关键 input / temporal ablation；
- 构建 multi-run calibration summary，并与 identity retrieval / prediction summary 合并；
- 在诊断闭环后测试 severity-balanced sampler、severity-weighted loss 和 ordinal severity auxiliary head；
- calibration 只在机制验证阶段使用，不作为观察 test 后的最终调参。

判读：

- 若 calibration 改善 severe/minimal bias 但 Pearson 基本不变，说明排序信息存在但尺度被压缩；
- 若 calibration 无法改善，说明输入/表征层面也缺少稳定 severity ranking 信息，或 severity 信号被 identity/task/artifact shortcut 淹没；
- RGB baseline 当前属于第二种偏强：存在弱排序信号，但简单线性校准不足以恢复 severe 区间。

### Layer 2: OpenFace 输入 artifact

问题：aligned face 是否包含非行为图像伪迹？

因素：

- 黑边 / 黑填充；
- 边界硬突变；
- 插值模糊；
- 裁剪形状；
- 麦克风黑块；
- 眼镜反光；
- 胡须边界。

已完成：

- black artifact audit；
- black_to_gray / black_to_mean / black_to_blur；
- border_black_to_gray / border_black_feather / center_mask_black_to_gray。

下一步：

- local occlusion case list；
- spatial occlusion on glasses / mouth occluder / beard lower face；
- patch-level attention / occlusion visualization。

判读：

- 若 artifact 处理改善整体但加重 severe 低估，应归为 bias trade-off；
- 若特定局部区域遮挡导致预测剧烈变化且与 BDI 无稳定关系，应报告为 local shortcut。

### Layer 3: Identity / static appearance shortcut

问题：RGB embedding 是否更像 subject / appearance representation，而不是 BDI severity representation？

因素：

- 脸型；
- 年龄和皮肤纹理；
- 眼镜；
- 胡须；
- 发际线；
- 下半脸轮廓；
- 静态 landmark geometry。

优先实验：

1. paired-task embedding retrieval；
2. same-subject top-k retrieval；
3. severity-neighbor agreement；
4. case-level visual inspection of nearest neighbors；
5. subject proxy classifier only after retrieval evidence。

判读：

- same-subject retrieval 高、severity neighbor agreement 低：强 identity shortcut；
- `center_mask` 降低 identity retrieval 且保持 BDI 表现：去身份化输入处理有证据；
- nearest neighbors 共享眼镜/胡须/发型/遮挡：进入 local occlusion / accessory case study。

最新 multi-run 结果确认：RGB baseline same-subject top-1 为 `0.66`、top-5 为 `0.85`，paired-task median rank 为 `1`，说明 RGB embedding 明显偏 subject/static appearance。输入 artifact 变体并不自动降低身份线索；`border_black_feather` 在 test 上 top-1 升至 `0.75`，提示边界软化可能让同 subject 的静态外观更稳定。`middle_crop` 将 top-1 降至 `0.49`，但伴随 task consistency 恶化。因此 identity shortcut 与 task-context confound 必须联合解释，不能单看一个 retrieval 指标。

下一步机制要求：只有当某个变体同时降低 same-subject retrieval、提高 severity neighbor agreement、保持或提升 BDI 指标、并不恶化 task consistency，才可被称为有效的去身份化/行为化表征。目前尚无变体满足这一条件。

### Layer 4: OpenFace geometry / quality confound

问题：OpenFace 检测和对齐过程是否把数据采集差异转成模型可见线索？

因素：

- landmark bbox width/height/area/aspect；
- face scale；
- eye distance；
- center offset；
- confidence / success；
- pose / gaze；
- landmark jitter。

已完成：

- alignment geometry audit；
- 坐标尺度修正为 pre-alignment detection geometry；
- 发现 bbox/scale/eye distance 与 true BDI / residual 存在中等相关。

下一步：

- 增加 `landmark_x_min/x_max/y_min/y_max`；
- 增加 `eye_distance_to_bbox_height_ratio`；
- task inconsistency mixed-factor audit 中加入 geometry diff。

判读：

- 若 geometry 与 residual 相关，说明模型失败可能受预处理几何混杂影响；
- 若 geometry 与 task diff 相关，说明 Freeform/Northwind 的采集/对齐差异不可忽略。

### Layer 5: Temporal sampling and task context

问题：视频中的哪个片段被模型看到，是否影响预测和过拟合？

已完成：

- temporal sampling audit；
- uniform_256/512/1024；
- first/middle/random crop；
- temporal training overfit summary。

当前结论：

- uniform 帧数变化几乎无影响；
- first crop 较弱；
- middle crop 改善整体指标和 severe 低估，但明显恶化 task consistency；
- random crop 增加预测方差但整体不稳。

下一步：

- task inconsistency mixed-factor audit；
- middle_crop 改善/恶化 case list；
- temporal occlusion by segment；
- 不再优先扩展 uniform_2048 或更多普通 crop。

判读：

- temporal location matters；
- simple sampling replacement is not enough；
- task context confound should be reported explicitly。

### Layer 6: Model capacity and optimization overfit

问题：模型是否在所有输入策略下都能记忆训练集？

证据：

- temporal runs 全部 `overfit_after_best_val=True`；
- behavior baseline train RMSE 极低但 test 弱；
- RGB-family runs best val 后 train 继续下降、val 变差。

下一步：

- early stopping 机制记录；
- smaller temporal encoder / stronger dropout only after shortcut audits；
- targeted augmentation only after identifying shortcut source。

判读：

- 如果模型容量降低改善 train/val gap 但 test 不变，说明表征信号不足；
- 如果局部 artifact augmentation 改善特定 case，说明 shortcut source 明确。

## 5. 实验决策树

### Step A: 先确认结论有效性

```text
split integrity PASS?
  no  -> 修复 split / label / prediction alignment
  yes -> 进入 Step B
```

### Step B: 判断是否主要是 prediction compression

```text
val-fit calibration improves severe/minimal bias?
  yes -> severity calibration / loss / sampler 单独成线
  no  -> 输入和表征层面缺失有效 severity 排序，进入 Step C
```

### Step C: 判断是否强 identity shortcut

```text
paired-task retrieval strong?
  yes -> identity/static appearance audit + region/occlusion case study
  no  -> 进入 geometry / task context / behavior subset
```

### Step D: 判断是否 geometry / quality confound

```text
geometry or quality variables correlate with residual/task diff?
  yes -> quality/geometry stratified evaluation and normalization ablation
  no  -> 进入 behavior representation validation
```

### Step E: 判断 temporal / task context 是否驱动错误

```text
task diff correlates with temporal, geometry, pose/gaze, black artifact?
  yes -> task context confound analysis
  no  -> 更可能是 subject-level representation or label compression
```

### Step F: 行为表征是否稳定

```text
AU / landmark-delta / pose-gaze subset generalizes?
  yes -> stable behavior subset -> late fusion / behavior auxiliary MTL
  no  -> 继续特征去身份化、容量约束和质量分层
```

## 6. 推荐执行顺序

### P0: 已实现能力的结果闭环

1. 将 temporal sampling 消融和 training overfit 结果写入文档；
2. 运行 severity calibration verification；
3. 实现 / 运行 embedding identity retrieval；
4. 实现 task inconsistency mixed-factor audit；
5. 对 `middle_crop` 改善/恶化样本生成 case list。

### P1: 局部遮挡和身份区域验证

1. 建立 glasses / microphone / beard case list；
2. 对 case list 运行 spatial occlusion / attention；
3. 设计 `glasses_region_erased`、`mouth_occluder_erased`、`beard_lower_face_erased`；
4. 与 `center_mask`、`border_black_feather` 对照，判断是否真有局部遮挡 shortcut。

### P2: 干预式改进

只在 P0/P1 明确机制后再做：

- severity-balanced sampler；
- weighted MSE / Huber / CCC loss；
- targeted artifact augmentation；
- RGB + stable behavior late fusion；
- behavior auxiliary MTL。

## 7. 论文叙事建议

推荐章节结构：

```text
1. Baseline Failure Analysis
   prediction compression, severe underestimate, train/val overfit

2. Input Artifact Evidence
   center mask, black artifact audit, border-connected black ablation

3. Multi-factor Shortcut Audit
   split validity, temporal sampling, geometry, identity, occlusion, task context

4. Calibration and Task Context
   severity calibration and Freeform/Northwind inconsistency

5. Behavior-oriented Validation
   OpenFace feature baseline, feature group ablation, stable behavior subset
```

核心表述：

> RGB overfitting on OpenFace aligned face should be understood as a multi-factor shortcut problem. Black padding is a visible entry point, but the broader mechanism includes static identity appearance, local occlusion/accessories, pre-alignment geometry, temporal/task context, and label-distribution compression. Therefore, the contribution is not a single mask trick, but a systematic shortcut audit protocol for depression prediction from aligned facial videos.

## 7.1 Latest Temporal And Boundary Updates

### Temporal sampling ablation result

最新 temporal sampling 消融进一步收窄了时序因素的解释边界：

```text
middle_crop   MAE=8.8014, RMSE=10.7291, Pearson=0.4192, CCC=0.3806
uniform_512   MAE=8.8661, RMSE=10.9208, Pearson=0.3606, CCC=0.2966
uniform_1024  MAE=8.8684, RMSE=10.9257, Pearson=0.3621, CCC=0.2981
uniform_256   MAE=8.8686, RMSE=10.9214, Pearson=0.3615, CCC=0.2974
first_crop    MAE=8.8746, RMSE=10.9844, Pearson=0.3272, CCC=0.2522
rgb           MAE=8.9145, RMSE=10.9530, Pearson=0.3526, CCC=0.2925
random_crop   MAE=9.0369, RMSE=11.2329, Pearson=0.3847, CCC=0.3631
```

训练曲线审计显示所有 temporal run 均为 `overfit_after_best_val=True`。因此，temporal location 确实影响预测，尤其 `middle_crop` 能改善 overall metrics 和 severe 低估；但简单采样替换不能解决训练后期过拟合。`middle_crop` 同时恶化 Freeform/Northwind task consistency，应作为 task-context confound 线索，而不是最终采样策略。

### Boundary hard-transition submechanism

OpenFace 人脸裁剪边界的硬突变应作为 input artifact 的精确子机制继续验证。已有 `border_black_feather` 结果说明，软化边界比硬替换黑区更有价值。下一步应优先区分：模型到底依赖黑色填充面积，还是依赖黑色到肤色之间的高梯度突变。

优先候选：

```text
edge_soften_only
border_blur_fill
```

暂缓候选：

```text
border_reflect_fill
border_feather_blur_fill
center_mask_soft_boundary_v2
```

如果只平滑边界即可改善预测，论文表述应从 black padding shortcut 更精确地推进为 OpenFace alignment introduces structured high-contrast transition artifacts。

## 8. 当前停止规则

为了提高研究效率，以下方向暂不继续扩展：

- 继续堆叠相似 RGB mask 变体；
- 继续扩展 uniform 帧数，例如 `uniform_2048`；
- 在未完成 identity / calibration / task context 前直接 late fusion；
- 在未明确 shortcut source 前加入大规模随机增强；
- 仅凭 test MAE 选择最终模型。

每个后续实验都必须回答一个机制问题，而不是只追求单次数值提升。