# GLOBAL_LOCAL_AU_EXPERIMENT_PLAN.md

> 文档职责：定义“完整对齐帧 + 三个语义局部帧 + 共享 backbone + 区域/跨区域 AU 辅助监督”的当前权威实验方案。输入质量与裁切几何门禁继续由 `VALIDITY_AWARE_TEMPORAL_SLICING_PLAN.md` 和 `SHORTCUT_AUDIT_DESIGN.md` 管理；历史 AU12/14/15 特权监督基础设施与来源合同见 `PRIVILEGED_BEHAVIOR_ALIGNMENT_PLAN.md`。

## 1. 状态、目标与主张边界

状态日期：2026-07-30。

本方案已经完成研究设计和文档冻结，尚未修改 dataset、model、runner 或训练配置，也未授权训练。PB-P0A3权威full-rich v2已经完成300/300视频、493,141/493,141行和单一182列schema；A2在完整physical train上给出`PASS_FULL_SOURCE_COVERAGE`、0 blocker、0 warning。该结果只完成来源/coverage门禁，不授权P0B、扩展AU、局部裁切、模型实现或训练。

实验路线现冻结为 **全模型优先的混合消融策略**：先筛选通过全部资格门禁后的最大完整候选，再执行依赖感知的反向减法消融，最后从强基线按固定顺序加回幸存因素复核。该策略用于尽快判断完整路线是否值得继续，同时避免仅凭逐项加法低估交互收益，也避免仅凭减法阶梯作出不受顺序影响的因果归因。

研究问题是：在 AVEC2014 风格的 subject-independent 小样本抑郁回归中，使用语义明确的局部 RGB 视图和区域相关 AU 动态作为训练期辅助监督，能否在不扩大 backbone 参数量的前提下，增强模型对面部行为线索的捕捉，并降低身份、静态外观、任务和采集条件捷径导致的过拟合。

本方案只提出可证伪假设，不预设以下结论成立：

- 局部裁切必然优于完整人脸；
- 共享 backbone 必然消除身份信息；
- 所有区域相关 AU 都具有相同强度的抑郁特异证据；
- AU 辅助损失下降等于抑郁泛化改善；
- OpenFace 输出等同于无误差 FACS 标注。

现有研究充分支持全局/局部互补和 AU 区域建模，但没有直接证明该结构在本项目 split 上一定缓解严重过拟合。核心研究依据包括：

- Wang et al., *Region Attention Networks for Pose and Occlusion Robust Facial Expression Recognition*, TIP 2019. [DOI 10.1109/TIP.2019.2956143](https://doi.org/10.1109/TIP.2019.2956143)
- Li et al., *EAC-Net: Deep Nets with Enhancing and Cropping for Facial Action Unit Detection*, TPAMI. [DOI 10.1109/TPAMI.2018.2791608](https://doi.org/10.1109/TPAMI.2018.2791608)
- Melo et al., *Combining Global and Local Convolutional 3D Networks for Detecting Depression from Facial Expressions*, FG 2019. [DOI 10.1109/FG.2019.8756568](https://doi.org/10.1109/FG.2019.8756568)
- Girard et al., *Social Risk and Depression: Evidence from Manual and Automatic Facial Expression Analysis*, FG 2013. [DOI 10.1109/FG.2013.6553748](https://doi.org/10.1109/FG.2013.6553748)
- Girard et al., *Nonverbal social withdrawal in depression: Evidence from manual and automatic analyses*, IVC 2014. [DOI 10.1016/j.imavis.2013.12.007](https://doi.org/10.1016/j.imavis.2013.12.007)
- Fu et al., *Facial action units guided graph representation learning for multimodal depression detection*, Neurocomputing 2024. [DOI 10.1016/j.neucom.2024.129106](https://doi.org/10.1016/j.neucom.2024.129106)

RAN 的消融是本方案的重要约束：完整人脸不能被局部区域替代；简单平均/拼接不足以保证收益；区域过多可能退化。因此首版只保留三个有明确解剖语义的局部区域和一个轻量融合模块。

## 2. 预处理、数据集与模型职责

职责边界冻结如下：

```text
预处理
  -> 输入读取、曝光/颜色处理、对齐、landmark坐标
  -> global/三个local crop生成
  -> frame/view validity、coverage、质量原因和provenance

Dataset / Sampler
  -> 只读取规范化RGB、mask、frame id和训练期特权目标
  -> 不把local crop当作独立样本
  -> 不因AU/OpenFace失败重新采样“更标准”的正脸

模型
  -> 只接收global/local RGB及valid mask
  -> 共享backbone编码、轻量区域融合、时序聚合和BDI预测
  -> 训练期可选AU/head小头；不负责检测、重新裁切或修复输入
```

原始视频、原始 aligned JPG、派生 RGB、OpenFace CSV 和 split 均保持不可变来源边界。所有派生输入必须写入独立 root，并记录源文件/派生文件 SHA-256、预处理参数、代码 commit、输入 manifest 和失败原因。

## 3. 四视图输入与裁切边界

每个被接受的时间帧固定产生以下四个视图：

```text
V0 global_face       完整对齐人脸
V1 eye_brow          双眼、双眉及少量眼下区域
V2 nose_cheek        鼻梁、鼻翼、双侧颧部及鼻唇沟上段
V3 mouth_lower_face  完整嘴唇、双侧嘴角、鼻唇沟下段及少量下巴
```

三个局部区域以主体语义互不重复为目标，但允许相邻区域保留约 5%--10% 的边界容错。轻微重叠只用于抵抗 landmark 抖动、姿态变化和肌肉作用跨边界，不得演变为多个高度重复的嘴角 crop。

裁切必须遵守以下合同：

- 坐标来自同一 aligned RGB 空间中经过 overlay 验证的 landmark/transform；禁止把原始 `640x480` 检测坐标直接独立缩放到 `112x112`；
- point 集合、polygon、margin、最小 coverage、越界和最小像素尺度只使用 physical train 的无标签几何审计冻结；
- validation/test 只应用冻结规则，不回调裁切边界；
- accepted crop 必须完整包含冻结的语义 polygon 和 margin；
- landmark 无效、语义区域被截断、有效面积不足或严重遮挡时，只将对应 local view 标为 invalid；若 global RGB 可见，仍可参加 BDI 主任务；
- 不镜像隐藏侧、不复制邻帧、不用插值伪造长缺失区域；
- 不直接拉伸不同比例的 crop，使用等比例 resize + padding；
- 同一帧的四个视图使用一致的 flip、颜色和曝光增强参数，禁止按区域独立随机改变颜色；
- 局部视图不计为新增 subject、video、clip 或独立样本。

## 4. AU 候选集和证据分层

AU 必须同时满足抑郁相关性、区域相关性、当前 extractor 可用性和训练集风险审计，不能因为 OpenFace 提供该列就自动纳入。

### 4.1 首轮候选

```text
depression-supported core:
  AU12, AU14, AU15

regional extension A:
  AU4, AU6, AU7

regional extension B:
  AU10, AU17
```

首轮完整候选白名单为：

```text
AU04_r, AU06_r, AU07_r, AU10_r,
AU12_r, AU14_r, AU15_r, AU17_r
```

解释边界：AU12/14/15具有相对直接的抑郁严重度、笑容和社会退缩研究依据；AU4/6/7/10/17是具有解剖和行为合理性的区域候选，但不能统一写成已经得到同等强度临床验证的“抑郁特异AU”。

### 4.2 延后或排除

```text
exploratory only:
  AU1, AU2, AU5, AU9, AU20, AU23, AU24, AU45

excluded from first round:
  AU25, AU26
```

AU1/2/5/9/20/23/24/45只有在首轮通过且重新预注册后才能进入独立扩展实验。AU25/26首轮排除，因为张口和下颌运动高度受语音、音素、说话量及 Freeform/Northwind 任务影响，容易形成任务捷径。gaze继续完全排除，不作为输入、目标、条件或消融因素。

## 5. 区域和跨区域 AU 处理

RGB 裁切可以保持主体互不重复，但面部肌肉作用范围不必服从硬裁切边界。AU6和AU14采用跨相邻区域特征预测：

```text
eye_brow token                         -> AU4, AU7
fuse(eye_brow, nose_cheek)             -> AU6
nose_cheek token                       -> AU10
fuse(nose_cheek, mouth_lower_face)     -> AU14
mouth_lower_face token                 -> AU12, AU15, AU17
```

跨区域 `fuse` 首版只允许 masked mean、gated sum 或单层线性投影，不新增区域专属 backbone、深层 MLP 或独立时序编码器。AU14不得强行只归属于鼻颊或嘴部；AU6不得强行只归属于眼部或脸颊。

OpenFace AU intensity通常不是左右侧独立标签，因此不允许用同一个帧级 AU 数值分别强监督左、右局部 crop。左右 landmark 只负责定位和可见性；AU监督作用于整体区域 token 或相邻区域融合 token。

## 6. 模型结构和复杂度边界

主结构冻结为：

```text
[B,T,4,C,H,W]
  -> reshape [B*T*4,C,H,W]
  -> one shared backbone
  -> [B,T,4,D]
  -> add small view/region embedding after backbone
  -> validity-aware gated/attention pooling per frame
  -> [B,T,D]
  -> existing temporal encoder
  -> video/clip representation
  -> BDI regression head

regional / cross-regional temporal summaries
  -> grouped AU auxiliary heads (train only)

global temporal summary
  -> optional head-motion auxiliary head (later stage only)
```

必须先在帧级把四个视图融合为一个 token，再进入现有时序模块。禁止首版把序列直接扩展为 `4T` 后送入 Transformer，因为自注意力计算可能从 `T^2` 增至约 `16T^2`。

复杂度目标：

- backbone 参数量不因视图数量增加；
- region embedding、门控和 AU 小头的新增参数原则上低于总模型的 1%；
- 像素级四视图的 backbone FLOPs 和训练激活显存可能接近 global-only 的 4 倍，必须在 smoke 中实测；
- AU/head小头只在训练期存在，不参与 checkpoint 选择；
- 首轮 train/validation/test 都使用相同的 global+有效 local 视图合同，但 validation/test 不读取 AU/head target；
- 若未来要求 global-only 部署，必须另建蒸馏/一致性实验，不能把首轮四视图模型在测试时静默切成 global-only。

ROIAlign 单次 backbone 方案只作为后续效率对照，不与首轮像素级局部帧同时改变。

## 7. 目标、mask与损失

### 7.1 BDI主任务

BDI 仍是唯一 checkpoint 选择目标。所有 clip 必须按 video 归一权重，或先聚合为唯一 video prediction 后计算一次 BDI loss。禁止因为一个帧有三个 local view 而使其 BDI 权重变为 global-only 的四倍。

### 7.2 AU辅助目标

首轮沿用保守的 clip/video级动态描述符，而不是拟合完整逐帧 OpenFace 向量：

- 每个 AU 的有效帧平均 intensity；
- 只跨连续有效帧的平均绝对变化；
- 每个 AU/描述符独立 mask 和 coverage；
- physical train 独立拟合 normalization；
- 不在首轮加入 bout、多个分位数或大规模共现搜索。

每个 AU 的静态与动态描述符先在 AU 内平均，再在区域组间平均，避免维度较多的嘴部组支配损失。连续目标使用 masked SmoothL1/Huber，target stop-gradient。

### 7.3 总损失

```text
L_total = L_BDI
        + ramp(e) * lambda_AU * mean_valid_region(L_AU_group)
        + optional ramp(e) * lambda_HEAD * L_HEAD
```

约束：

- 前10% epoch关闭辅助损失，随后约20% epoch线性ramp；
- `lambda_AU`只允许从`{0.01, 0.03, 0.1}`经100-step train-only梯度校准冻结，不用validation搜索；
- head-motion只有在P0E判为eligible且完成train-only梯度校准后才进入`GLA-FULL`；其贡献通过预注册的`-HEAD`阶梯验证；
- AU/head loss仅在train计算；validation/test隐藏OpenFace target root仍必须运行；
- checkpoint只按预注册BDI validation monitor/mode选择；
- 每组记录与BDI梯度的norm、cosine、conflict、cancellation和长期占比。

## 8. 数据和质量门禁

模型实现前必须通过以下门禁；几何与行为来源审计可独立推进，但必须在P0E资格清单处汇合：

1. `PB-P0A3`：权威full-rich v2完成300/300视频、493,141行、单一182列schema和全部hash/provenance闭环（已完成）；
2. `PB-P0B`：先审计/修复旧P0核心对mask-aware来源的兼容性，再对不可变v2来源执行core/schema/exact join/coverage；41个legacy strict失败不得误拒mask-aware来源；
3. `PB-EXP-AU`：为AU4/6/7/10/17另建versioned source/schema/coverage/descriptor合同；v2含列不等于这些列已获训练资格；
4. `PB-P0C`：matched raw/aligned物理保真分别裁决AU组和head轴；
5. `PB-P0D`：对实际训练描述符执行identity/task/exposure/quality风险审计；
6. `FACE-S1 + LM/REGION`：global可见性、三语义区point/polygon/margin/coverage overlay和crop manifest冻结；
7. `PB-P0E`：机器可读清单输出`REGION_ELIGIBLE`、按组AU/head资格、`CROSS_REGION_ELIGIBLE`、完整`eligible_component_profile`及其SHA；该profile必须是依赖闭合的最大合格集合，并保持模型代码授权为false。

pilot20当前结果为16 PASS / 4 FAIL，失败代理为：

```text
205_1_Freeform  0.990909
206_1_Freeform  0.920000
207_2_Freeform  0.698148
208_1_Freeform  0.979227
```

因此不得直接把strict `0.995`改成刚好通过的阈值，也不得删除失败视频。已冻结的mask-aware policy把100%视频/行/schema/hash/provenance作为硬完整性门禁，以`success==1 && confidence>=0.8`定义quality-valid，并把`success/confidence`只用于view/target validity和coverage。权威full判定为`PASS_FULL_SOURCE_COVERAGE`：quality/AU-valid 486,441帧、head-valid 485,863对；physical train总体/Freeform/Northwind分别为0.989273/0.983944/0.996984，task gap约0.01304，最低train视频仍有180个AU-valid帧和179个head pair。val/test仅报告，且该判定仍明确`P0B/training_authorized=false`。RGB可见但OpenFace失败的帧仍可参加global BDI；landmark局部裁切无效时只mask local view；AU无效时只mask对应AU loss。

A2 full决策当前存在一个仅影响报告分支的已知问题：状态为`PASS_FULL_SOURCE_COVERAGE`时，`next_action`仍错误写成`STOP_AND_REVIEW_POLICY_OR_SOURCE_ISSUES`。P0B前必须按代码实施前披露门禁单独授权修复并增加回归测试；不得用手工修改既有决策JSON掩盖该问题。

扩展AU后，现有只投影AU12/14/15的PB合同不再足够。未来必须版本化升级source manifest、白名单、schema校验、normalization、risk audit和测试，禁止手工给旧manifest补列或混用历史raw-video rich CSV。

## 9. 全模型优先的混合消融矩阵

所有条件保持同一physical split、seed、clip manifest、输入root、backbone、optimizer、precision、训练步数、severity loss、EarlyStopping和checkpoint policy。视图数量不同的对照按video保持相同BDI统计权重。

本文中的`GLA-FULL`严格等于同一个P0E eligibility manifest给出的最大依赖闭合`eligible_component_profile`，包括获批的global/local region、AU组、cross-region关系和head轴；它不是`src/legacy/full_model/`中的legacy full model，也不是Stage C历史运行`C-FULL`。任何未通过P0E或失去上游依赖的组件都必须在训练前从profile中删除；这属于门禁裁决，不得伪装成训练后的减法收益。所有run必须记录该profile及其manifest SHA。

### Stage 0：输入与实现验证，不比较test

```text
GLA-SMOKE-1B       one-batch forward/backward, mask和shape
GLA-CAL-100        seed42, 100-step梯度/显存/FLOPs校准
GLA-GRID-CAL-100   三个等面积grid对照的匹配校准
```

通过条件包括：loss/gradient有限、invalid view不贡献AU loss、四视图不改变video权重、AMP稳定、隐藏AU root的validation可运行、共享backbone确实只有一份参数。共享模块必须使用独立的确定性初始化随机流，保证新增/删除辅助head不会改变backbone初始化；各条件还必须配对sampler顺序和逐样本增强随机数流。

`GLA-CAL-100`只在最大eligible profile上进行一次。由train-only校准冻结的`lambda_AU/lambda_HEAD`、ramp和组权重必须跨全部消融复用；总辅助损失的分母固定为FULL白名单，被删除组贡献0，禁止因剩余组变少而自动放大其loss。正式训练前还必须冻结machine-readable strategy manifest，记录完整run ID、eligibility SHA、相邻比较、幸存阈值、最大run数和停止分支；不得依据validation结果临时补写矩阵。

### Stage 1：`GLA-FULL`优先筛选

| Run ID | 条件 | 作用 |
|---|---|---|
| `GLA-C-REF` | global-only强基线；与候选使用相同代码版本、split、seed和训练预算 | 提供匹配reference |
| `GLA-FULL` | 全部P0E-eligible三语义区、AU组、跨区域处理和head | 先回答完整路线是否具有继续价值 |
| `GLA-RGB-FULL` | global+三个语义local，无AU/head | Stage 2的S4阶梯；若`GLA-FULL`仅触发科学utility/risk失败，仍作为S1--S5诊断链的一环执行 |

seed42只读取validation；旧`C-REF`日志不能替代本次匹配`GLA-C-REF`。这里的“FULL优先”表示在任何组件筛选前先评估完整候选，匹配reference可以先运行或并行运行。若出现source/hash/join、val/test读取行为目标、NaN、mask/shape、AMP或无法解释的DDP技术失败，立即停止全部训练。若`GLA-FULL`只触发预注册的科学utility/risk失败，则仍完成S1-S5反向阶梯作故障定位，但跳过机制控制、加法复核、seeds43/44和benchmark。未触发科学停止条件时，Stage 2才作为正式组件证据继续。

### Stage 2：依赖感知的反向减法消融

下表是所有组均eligible时的标准阶梯。实际矩阵从P0E集合`E`生成：从`G+L+E`开始，按`H -> B -> A -> C -> L`固定顺序删除；不在`E`中的组记录`SKIPPED_INELIGIBLE`并保持编号，不得用validation决定跳过或重排。每个实际run ID必须附加剩余component profile和`S42/S43/S44`后缀。每一行都从同一预训练backbone独立初始化并完整训练，禁止继承上一行checkpoint：

| 顺序 | Run ID | 从上一层删除 | 剩余能力 |
|---|---|---|---|
| S0 | `GLA-FULL` | 无 | 最大eligible候选 |
| S1 | `GLA-S1-NO-HEAD` | head motion | 三语义区 + 全部eligible AU |
| S2 | `GLA-S2-NO-AU-B` | AU10/17 | 三语义区 + core + AU4/6/7 |
| S3 | `GLA-S3-CORE-AU` | AU4/6/7 | 三语义区 + AU12/14/15 |
| S4 | `GLA-RGB-FULL` | 全部AU辅助监督 | global + 三语义local |
| S5 | `GLA-C-REF` | 三个local及view fusion | global-only |

若某组未被P0E批准，相应阶梯必须在预注册run manifest中标为`SKIPPED_INELIGIBLE`并跳过，不能把“未进入FULL”写成消融结论。局部视图最后删除，因为区域AU依赖local/cross-region token；不得保留local-AU loss却移除其RGB来源。相邻阶梯只给出**顺序条件下的边际证据**，不等于因素的无条件独立因果效应。

同时冻结三个反证控制：

| Run ID | 匹配条件 | 反证问题 |
|---|---|---|
| `GLA-FULLAU-NO-CROSS` | 与不含head的最高eligible AU阶梯匹配；AU6固定以eye-brow为单区锚点，AU14固定以mouth-lower-face为单区锚点，均经过同维线性adapter并保持目标数/参数预算；只有对应AU与相邻视图均eligible时运行 | AU6/AU14跨区域融合是否真正必要 |
| `GLA-RGB-GRID` | global + 三个面积/长宽比/padding/FLOPs匹配的grid，共享同一backbone和融合器；逐view复用对应语义local的valid mask，即使grid像素可读也不额外放行 | 语义区域是否优于一般多裁切 |
| `GLA-AUX-SHUFFLED` | 匹配最终幸存的真实辅助条件；在physical train内对subject做固定无自映射置换，同一subject两任务使用同一donor且按task匹配；保留接收样本mask/coverage，val/test不读目标 | AU/head收益是否只是噪声正则化 |

### Stage 3：幸存因素的加法复核

`GLA-C-REF -> GLA-RGB-FULL`先复核局部RGB。core-only直接复用Stage 2的`GLA-S3-CORE-AU`；若AU4/6/7、AU10/17或head在减法阶梯中存活，则各自最多增加一个以`GLA-RGB-FULL`为共同基准的`GLA-ADD-A`、`GLA-ADD-B`或`GLA-ADD-HEAD`。只有“从FULL删除后变差”且“单组加回不触发失败”的因素，才能主张具有独立贡献；只在联合条件有效的因素必须标为interaction-dependent。

通过上述复核后最多构建一个`GLA-REDUCED`，按固定依赖顺序组合全部幸存组。失败组不重新搜索单AU组合、头容量或更多loss权重；跨区域处理只有匹配FULLAU阶梯优于`GLA-FULLAU-NO-CROSS`时保留。`GLA-REDUCED`必须在允许的随机波动范围内复现`GLA-FULL`方向，否则最终交互解释视为不稳定。

该阶段不是重新展开全组合搜索。精确加法run ID、保留组和比较对必须由Stage 2预注册阈值机械生成，并在执行任何加法run前写入只读matrix manifest。seed42最多允许6个阶梯、3个机制/反证控制、3个条件性单组加回和1个reduced rebuild，共13个full fit；未使用配额不得转为临时超参数或组合搜索。

### Stage 4：多seed与locked post-selection benchmark

1. seed42只做`GLA-FULL`筛选、减法阶梯和有限加法复核；
2. 只有同时得到减法与加法支持的最终简化候选才运行seeds43/44；`GLA-C-REF`保持paired seeds；
3. seeds43/44只扩展冻结winner、其直接matched ablation、`GLA-RGB-FULL`、`GLA-C-REF`和论文主张必需的一个机制控制，每个seed最多5个full fit；
4. paired-seed和subject-cluster bootstrap方向稳定后冻结全部超参数；
5. 由于本项目历史上已经执行过val/test role swap，original test曾参与checkpoint选择；新路线只能把保持关闭后的评估称为`locked post-selection benchmark`，不得声称untouched或unbiased final test；
6. 冻结后以一个具名evaluation package一次性读取该benchmark，之后不得回调结构、AU组、阈值或loss；role-swap只保留为历史split sensitivity证据，不再次参与路线选择。

因此，三seed加locked benchmark只能用于本项目路线决策和受限复核，不能单独支撑“无偏外部泛化”主张。若最终候选存活，论文级确认还需另行预注册repeated subject-disjoint group folds，并重新披露算力与运行授权；该确认不属于首轮FULL优先矩阵。

## 10. 指标、成功条件与停止条件

必须报告：

- BDI：MAE、RMSE、CCC、Pearson；
- 过拟合：train/validation gap、最佳epoch、后期退化斜率；
- 输出分布：prediction mean/std、compression ratio、常数预测检查；
- 严重度：minimal/mild/moderate/severe MAE、bias、worst-group；
- 身份：逐层identity probe、same-subject retrieval、paired-task verifier；
- 任务：Freeform/Northwind分组指标和同subject预测/残差差异；
- 行为：逐AU/逐组coverage、masked loss、gradient norm/conflict；
- 输入：逐区域valid ratio、曝光/眼镜/姿态/遮挡/OpenFace质量分组；
- 成本：参数量、FLOPs、峰值显存、step time和推理吞吐。

统计单位必须是subject，置信区间采用subject-cluster bootstrap，不能把帧、crop、clip或同subject两任务当成独立样本。

支持方案至少要求：

- 三个paired seed的BDI方向稳定，且train-validation gap不扩大；
- identity risk不增加超过预注册绝对门槛0.02；
- severity bias、prediction compression和task consistency不恶化；
- `GLA-FULL`seed42先通过匹配`GLA-C-REF`的utility/risk筛选；
- 每个最终保留因素同时获得反向删除与正向加回的方向一致证据；
- `GLA-RGB-FULL`相对`GLA-RGB-GRID`存在稳定优势，才能保留“语义区域”主张；
- 只有匹配FULLAU阶梯优于`GLA-FULLAU-NO-CROSS`，才能保留AU6/AU14跨区域收益主张；
- `GLA-AUX-SHUFFLED`不能复制最佳真实辅助目标收益；
- 辅助梯度不长期支配BDI梯度；
- 更复杂条件不优于更简单条件时保留简单模型。

技术失败与科学失败必须走不同分支：

- source/hash/join错误、validation/test读取行为目标、非有限loss/gradient、mask/shape错误、AMP不稳定或无法按设计解释的DDP unused parameter属于技术失败；立即停止全部训练，先披露并授权修复包，不得用后续消融解释；
- seed42相对匹配reference出现`delta_CCC < -0.05`或`delta_MAE > +0.50`，或identity/task/exposure、coverage、severity、compression、task robustness触发预注册风险门槛，属于科学失败；只允许完成S1--S5诊断阶梯，跳过controls、加法复核、multi-seed和benchmark；
- FULL科学通过后，若后续出现multi-seed方向不一致、只有auxiliary loss改善而BDI泛化不改善，或任何条件需要test反馈才能调整区域、AU或阈值，则停止扩展并保留负结果。

checkpoint仍只按预注册`val_RMSE_epoch`最小值选择；路线筛选使用冻结的MAE/CCC联合门槛以及identity/severity/task风险门槛，禁止在RMSE、MAE、CCC之间事后择优。除严重科学失败界外，non-inferiority与worst-group数值必须在strategy manifest中于首个full fit前冻结。

## 11. 复现合同

每次未来实验必须记录：

- git commit hash、branch、dirty status；
- base config、全部override和resolved config；
- random seed；
- physical split文件及其SHA-256；
- frame/clip/crop/coverage/target manifest及其SHA-256；
- 完整命令、GPU/device、precision、strategy；
- backbone预训练来源、参数量、FLOPs、输入尺寸；
- OpenFace版本、二进制/模型哈希、feature profile；
- AU列顺序、descriptor版本、valid规则和train-only normalization；
- optimizer、scheduler、训练预算、checkpoint monitor/mode；
- metrics、diagnostics和prediction输出路径；
- shuffled-target permutation seed；
- val/test是否保持关闭以及首次读取test的时间点。

## 12. 实施路线与当前未授权边界

后续工作顺序固定为：

```text
P0A3 full-rich v2 + full coverage decision（DONE）
 -> P0B前兼容性审计 + A2 next_action报告bug修复（需先披露并授权代码包）
 -> P0B core/coverage只读运行（单独运行授权）
 -> P0C物理保真 + P0D描述符风险 + 扩展AU versioned合同
 -> FACE-S1/LM三语义区overlay与crop manifest
 -> P0E最大eligible component profile
 -> 默认关闭的dataset/model/config实现（再次先披露并授权）
 -> import/config/one-batch/100-step smoke（单独运行授权）
 -> GLA-FULL seed42与匹配GLA-C-REF
 -> 技术失败：立即停止并重新披露修复包
 -> 无技术失败：完成S1--S5依赖感知减法阶梯
 -> FULL科学失败：阶梯仅作诊断并停止，不运行controls/additive/multi-seed/benchmark
 -> FULL科学通过：反证控制 -> 幸存因素有限加法复核 -> paired seeds 43/44
 -> 冻结后以一个具名package读取一次locked post-selection benchmark；role-swap仅保留为历史证据
```

当前只完成来源提取/A2 full coverage与文档方案。不得据此声明P0B、三语义区、跨区域AU、扩展AU白名单、共享backbone融合或训练入口已经实现。每个未来代码包必须先按仓库`AGENTS.md`向用户提交独立的边界/架构/数据流/接口/命令/验证/风险披露，结束该轮并等待明确授权；代码、数据生成、smoke、正式训练、commit和push互不自动授权。代码阶段还必须检查默认关闭兼容性、tensor shape、mask广播、sampling同步、梯度路径、AMP、DDP unused parameters、validation/test target隔离和checkpoint选择逻辑。

## 13. 决策记录

| 日期 | 决策 |
|---|---|
| 2026-07-29 | 局部视图由旧四区收敛为眼眉、鼻颊、嘴部三个主体语义区，保留完整global face。 |
| 2026-07-29 | 相邻区域允许5%--10%边界容错，不要求像素级绝对无重叠。 |
| 2026-07-29 | global/local使用同一个backbone；帧级融合后再进入原时序模块，禁止首版构造4T序列。 |
| 2026-07-29 | AU不再永久局限于AU12/14/15；首轮分级扩展至AU4/6/7/10/17，但必须经过来源、coverage和shortcut风险门禁。 |
| 2026-07-29 | 采纳跨区域AU：AU6由眼眉+鼻颊特征预测，AU14由鼻颊+嘴部特征预测。 |
| 2026-07-29 | gaze继续排除；AU25/26因语音/任务混杂不进入首轮。 |
| 2026-07-29 | AU/head只作为训练期辅助目标；首轮四视图模型的val/test使用同一RGB视图合同，但不读取OpenFace目标。 |
| 2026-07-30 | PB-P0A3权威full-rich v2与A2完整coverage门禁通过；legacy strict 0.995仍仅作诊断，P0B和训练未授权。 |
| 2026-07-30 | 实验顺序改为`GLA-FULL`优先筛选 -> 依赖感知减法 -> 幸存因素加法复核 -> multi-seed -> 单次locked post-selection benchmark package；保留grid、no-cross和shuffled反证控制。 |
| 2026-07-30 | `GLA-FULL`只含P0E批准组件，不等于legacy full model或Stage C `C-FULL`；资格失败不得伪装成消融收益。 |
