# OVERFITTING_MECHANISM_ROADMAP.md

> 文档职责：上层机制地图和决策树，负责把 input artifact、identity、geometry、temporal、calibration 等机制组织成统一框架。当前状态见 `CURRENT_STATUS.md`；执行任务见 `TODO.md`；文档导航见 `DOCS_GUIDE.md`。

本文档用于系统梳理 RGB 输入模型过拟合的机制研究路线。它不是新的单点实验清单，而是当前所有审计、消融和论文叙事的上层框架。后续具体实验配置仍放在 `RGB_OVERFITTING_AUDIT_PLAN.md`、`SHORTCUT_AUDIT_DESIGN.md` 和 `TODO.md` 中。

阅读方式：

- 只看当前路线：读“当前路线总览”和“当前推进路线”。
- 查机制证据：读“当前证据状态”和“机制层级图”。
- 写论文叙事：读“论文叙事建议”和“回到正轨的判据”。
- 执行任务：回到 `TODO.md`，不要从本文直接派生新任务。

## 当前路线总览

本文档的当前权威目标正式更新为 **Auditable and Falsifiable Coarse-Grained Task-Nuisance Information Separation：可审计、可证伪的粗粒度任务-干扰信息分流**。机制审计已经说明：单个 artifact 不能解释全部过拟合，简单输入处理也无法同时改善 identity retrieval、severity bias、CCC 和 task consistency。因此下一步不再尝试把所有潜在机制逐项显式建模，也不预设语义解耦已经成立，而是把过拟合机制转化为能够被外部审计和反证的粗粒度信息分流结构。

```text
已完成证据层：input artifact / temporal / identity / calibration / geometry audits
        ↓
Stage A: Shortcut 证据收口
        layer-wise identity probe + error-identity coupling + artifact/quality audits as evaluation
        ↓
Stage B: Identity-adversarial task representation
        H0 -> z_dep, with verifiable shortcut suppression
        ↓
Stage C: Coarse task-nuisance information separation
        H0 -> z_dep, z_nuisance
        z_id only after the base split passes C3 and is re-authorized
        ↓
Stage D: Falsification and robustness validation
        multi-attacker / leakage matrix / severity-balanced loss / group-wise robustness
        ↓
Current intervention: face-valid clips + three semantic local/global views + train-only AU/head
        P0E maximal eligible profile -> GLA-FULL first
        -> dependency-aware subtraction -> bounded additive confirmation -> paired multi-seed
```

当前机制分工：

| 机制问题 | 证据来源 | 新路线中的处理 |
|---|---|---|
| subject/static appearance shortcut | identity retrieval、input variants、case frames | identity-adversarial baseline；`z_dep` identity risk evaluation；`z_id` 仅作 post-C3 待审扩展 |
| 身份-抑郁交叠 | 表情基线、头动/眼动习惯、个体行为风格 | 不显式建 `z_m`；通过 `z_dep` utility、identity risk 和 `z_nuisance` leakage 联合评估 |
| severity label imbalance / middle-score collapse | severity bias、calibration summary、pred_std compression | severity-balanced regression 作为对照或支线 |
| input artifact / black boundary / OpenFace quality | black artifact、alignment geometry、confidence、bbox、valid ratio | 审计变量、post-hoc probes、case study 和 group-wise evaluation；不默认建 `z_art` |
| temporal/task context | temporal sampling、task consistency | 混杂监控和 task consistency evaluation；不显式建 `z_ctx` |
| dynamic facial behavior | literature / landmark dynamics / local crops | 眼眉/鼻颊/嘴部三语义区共享backbone；P0E合格的AU/head只作train-only辅助目标，val/test保持RGB-only target access |

读本文档时，应把前面 Layer 0-6 理解为证据地图，把“当前推进路线”理解为当前模型路线。若旧段落中的“下一步”与本总览冲突，以本总览和 `RGB_OVERFITTING_AUDIT_PLAN.md` 的最新 task-nuisance Stage A-D 为准。

### 2026-07-30 全模型优先混合消融转折

PB-P0A3权威full-rich v2和A2完整coverage已经通过，但P0B-P0E、扩展AU合同、三语义区crop manifest、模型和训练仍未授权。完整候选不再由global-only逐级前向堆叠，而是在P0E冻结最大依赖闭合eligible profile后先运行`GLA-FULL`与匹配reference；只要没有技术失败就完成`-HEAD -> -AU10/17 -> -AU4/6/7 -> -all AU -> -locals`的S1--S5阶梯。FULL科学失败时该阶梯只作诊断并停止扩展；科学通过时才做grid、no-cross、subject-deranged shuffled-aux和幸存组有限加回。

FULL优先只改变训练实验的验证顺序，不绕过任何数据、几何、来源、物理保真、风险或代码授权门禁。每个代码/运行包必须先按根`AGENTS.md`报告边界、架构/数据流、文件、命令、验证、风险和agent分工，结束该轮并等待明确授权。历史role-swap意味着最终只能报告locked post-selection benchmark，不能声称untouched test。

### 2026-07-14 机制路线转折（历史输入方案）

Stage C 结构、identity-gradient、显式参数正则和连续 severity weighting 的结果已经把当前失败进一步定位：过拟合不是单靠全局权重衰减、连续长尾重加权或一个可持续学习的身份 attacker 就能解除；表示维度变化在缺少有效语义梯度时只是在改变容量，不能证明信息被分离。split sensitivity 还说明 checkpoint 选择对小样本 subject-disjoint 验证集较敏感，后续论文级结论需要 repeated group folds，而不能继续把单次验证改进解释为稳定机制。

当时路线转向两层输入干预：face-valid连续片段和历史四区RGB-only局部增强。该四区、无AU监督和global-only推理假设现已由上方2026-07-30三语义区GLA方案取代，只保留为设计演进记录。

provenance-final与3帧raw-warp smoke已完成；下一门禁是用`FACE-S1`量化大姿态、遮挡、face coverage和landmark可信度。smoke不授权全量materialize。所有clip增加的是同一video的数据视图，不是独立subject；训练必须按source video归一或使用同模型video-bag loss。四个AU语义保持的landmark局部crop只有在匹配view数、面积和loss scale后稳定优于equal-area grid，才支持语义区域布局贡献。

当时登记的`LM-M0/LM-M1/LM-M2/LM-M3`前向链不再是当前执行入口；当前权威顺序见`GLOBAL_LOCAL_AU_EXPERIMENT_PLAN.md`和`TODO.md`。

## 1. 研究目标

当前目标不再是寻找“某一个 artifact 导致过拟合”，而是建立一套可解释、可审计、可证伪、可复用的机制审计与信息分流协议：

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

抑郁识别更合理的视觉信号应来自稳定面部行为动态，而不是冗余RGB外观。本项目用AU/FACS提供可复核的面部动作语义分区，用temporal landmark逐帧定位并保持区域完整；进入模型的仍是局部RGB。AU biomarker数值、AU序列和AU监督不进入本项目模型。

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

## 8. Temporal 与 Boundary 最新更新

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

## 9. 当前停止规则

为了提高研究效率，以下方向暂不继续扩展：

- 继续堆叠相似 RGB mask 变体；
- 继续扩展 uniform 帧数，例如 `uniform_2048`；
- 在未完成 identity / calibration / task context 前直接 late fusion；
- 在未明确 shortcut source 前加入大规模随机增强；
- 仅凭 test MAE 选择最终模型。

每个后续实验都必须回答一个机制问题，而不是只追求单次数值提升。

## 10. 三表联合后的机制更新

`rgb_input_ablation_summary`、`identity_retrieval_summary` 和 `severity_calibration_summary` 的联合结果已经支持更系统的机制解释：输入变体能改变 overall metrics 和 severity-group bias，但没有任何变体同时满足低 identity retrieval、高 severity agreement、低 task inconsistency、高 CCC 和低 severe bias。

关键证据：

- `center_mask` 是当前最健康的历史 input artifact mitigation：MAE `7.942`、CCC `0.477`、pred_std `8.132`，moderate bias 从 RGB 的 `-7.66` 改善到 `-2.08`。但 2026-07-03 真实输入帧审查显示它实际是鼻口小区域强遮挡，不能解释为完整中心脸行为证据；需用新增 `central_face_mask` 反证验证。
- `center_mask_black_to_gray` 的 MAE 最低 `7.726`，但 severe bias 达 `-17.46`，说明 overall MAE 不能单独作为模型选择依据。
- `border_black_feather` 的 severe bias 最轻 `-12.73`，但 same-subject top-1 达 `0.75`、top-5 达 `0.90`，提示 severe 端改善与 identity/static appearance shortcut 可能纠缠。
- `middle_crop` 将 same-subject top-1 降到 `0.49`，但 task diff 升至 `4.63` 且 severity agreement 最低 `0.454`，说明去身份化表象可能来自 temporal/task mismatch，而非更好的抑郁表征。
- 所有 run 的 post-hoc linear calibration 均降低 CCC，进一步支持 severe underestimation 是表征/优化/分布问题，而不是简单输出尺度偏移。

因此，当前机制地图应把 RGB failure 表述为多因素 shortcut 与 severity compression 的交叉，而不是单个 artifact 的因果链。

## 11. 当前推进路线：粗粒度 Task-Nuisance 主线

当前推进路线已经从细粒度 RPDF-Net 收敛为可审计、可证伪的粗粒度 task-nuisance 信息分流。该转移不是否定前一阶段，而是把 identity-adversarial MTL、severity-balanced regression、input artifact audit、identity retrieval 和 calibration summary 全部纳入证据层、对照基线和稳健性验证，同时避免给每个难以验证的潜在因素分配独立 latent，或把辅助损失收敛误写成语义解耦结论。

### Step 1: Shortcut 证据收口

目标：在实现信息分流模块前，证明可验证 shortcut 是否真的进入预测。

```text
A1 layer-wise identity probe
A2 prediction error x identity similarity coupling
A3 shortcut / artifact audit as evaluation
A4 severity imbalance / prediction compression summary
```

对应决策：

- A1/A2 支撑 identity-adversarial baseline 和身份风险审计；是否需要 `z_id` 必须等基础信息分流通过 C3 后重审；
- A3 支撑 artifact/quality/context probes、case study 和 group-wise evaluation；
- A4 支撑 severity-balanced baseline 或支线；
- existing identity retrieval / calibration / alignment / black artifact summaries 作为证据底座。

### Step 2: Identity-adversarial Task Representation

第一版先验证最小上层干预，不直接启用复杂因子分解：

```text
H0 -> z_dep
prediction = Head(z_dep)
subject_attacker = GRL(z_dep) -> subject_id
```

核心验证：

- `z_dep` 是否保持 BDI 预测能力；
- `z_dep` 的 identity retrieval / subject probe 是否下降；
- severity bias、task consistency 和 train-val gap 是否不恶化；
- 若 A1 成立但 A2 不成立，identity adversarial 只作为对照而非强 suppression 主线。

### Step 3: Coarse Task-Nuisance Information Separation

第二阶段只验证粗粒度单级分流：

```text
H0 -> z_dep, z_nuisance
prediction = Head(z_dep)
reconstruction = Recon([z_dep, z_nuisance]) -> H0
```

`z_id` 不进入第一版。只有 `C-REF/C-BN/C-REC/C-FULL` 通过 C3，且身份风险证据仍支持独立出口时，才允许重新评估：

```text
H0 -> z_dep, z_id, z_nuisance
prediction = Head(z_dep)
id_head(z_id) -> subject_id
subject_attacker = GRL(z_dep) -> subject_id
```

不显式建 `z_m`、`z_art`、`z_ctx`、`z_quality`。这些因素用 probes 和分组评估验证，不作为第一版 latent。

### Step 4: 反证与稳健性验证

最终模型不是一次性打开所有模块，而是逐项验证风险：

```text
multi-attacker identity evaluation
nuisance leakage evaluation
shortcut/artifact probes
severity-balanced branch
dynamic feature branch, deferred
```

每条支线必须报告对 BDI、identity risk、shortcut/artifact probe risk、severity bias、task consistency 和 train-val gap 的影响。若某支线只改善 MAE 但加重身份/伪迹风险，则不作为主模型组成。

## 12. 回到正轨的判据：从去身份化到行为化表征

身份消融的目标不是单纯降低 same-subject retrieval，而是避免模型用 subject/static appearance 替代 depression-relevant behavior。当前判据应同时包含：

```text
identity retrieval 不升高或下降
severity neighbor agreement 不下降
CCC 不下降
pred_std 不继续压缩
severe bias 改善
task consistency 不恶化
```

因此，`middle_crop` 虽然降低 identity retrieval，但不算回到正轨，因为它损害 severity agreement 和 task consistency；`border_black_feather` 虽然缓解 severe bias，也不算充分回到正轨，因为它增强 identity retrieval。真正可接受的方向必须在三表评估中同时通过 prediction、identity 和 calibration 约束。

## 13. 研究路线细化：必要性、可行性与中长期边界

当前 task-nuisance 路线的必要性来自三点：

1. 已有输入消融说明，单一黑边、单一边界或单一 temporal sampling 都不能解释全部失败模式；
2. identity retrieval、severity calibration 和 task consistency 结果说明，模型失败同时涉及身份记忆、prediction compression 和任务语境混杂；
3. OpenFace aligned face 不是干净行为输入，而是同时携带面部行为、身份纹理、对齐几何、质量/追踪状态和局部遮挡的混合输入，但这些因素无法被完整枚举和逐一验证。

可行性来自四个已有基础：

1. 现有 MTL-Lite / DeiT pipeline 已能训练和导出 prediction，为 layer-wise embedding audit 提供入口；
2. 现有 identity retrieval、severity calibration、alignment geometry 和 black artifact 脚本已经形成 Stage A 的大部分数据基础；
3. OpenFace CSV 和 aligned frames 可以产生 artifact/quality/context 审计变量；
4. GRL、deep imbalanced regression、multi-attacker evaluation 和 coarse factor separation loss 都可以作为低侵入支线逐步接入。

中长期边界：

```text
已完成：Stage A/Stage B、C0 规格冻结和 P0 seed/EarlyStopping
短期：C1 代码已本地实现；运行服务器 `C-REF/C-BN` smoke 与 100-step train-only calibration
中期：完成 C2 seed-42 screening 与 C3 三 seed validation gate
中长期：仅当信息分流在外部审计和多 seed 下稳定优于 paired `C-REF` 后，再考虑更复杂结构
长期：根据支线证据选择 severity-balanced、multi-attacker 或 dynamic feature
```

论文叙事的关键不是“提出更多模块”，而是“每个模块都有前置证据、进入条件、独立审计、反证条件和停止规则”。若某个支线只改善 MAE，却恶化 identity risk、severe bias、CCC 或 task consistency，则它应被作为机制反例记录，而不是进入最终主模型；若 Stage C 不优于 paired-seed `C-REF` 或多 seed 不稳定，则应明确报告当前信息分流假设未获支持。
