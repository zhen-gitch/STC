# GLOBAL_LOCAL_AU_EXPERIMENT_PLAN.md

> 文档职责：定义“AU/FACS语义分区 + landmark几何定位 + 完整对齐帧/三个局部RGB + 共享backbone”的当前权威实验方案。输入质量与时间切片门禁继续由 `VALIDITY_AWARE_TEMPORAL_SLICING_PLAN.md` 和 `SHORTCUT_AUDIT_DESIGN.md` 管理；历史AU数值监督与P0C证据见 `PRIVILEGED_BEHAVIOR_ALIGNMENT_PLAN.md`。

## 0. 2026-08-05权威路线：AU-guided landmark-localized RGB

计划ID为`PLAN-20260805-AU-GUIDED-LANDMARK-RGB-v1`。本节取代本文后续2026-07-30“AU数值辅助监督/GLA-FULL”方案的执行权；旧内容保留为设计演化和负结果背景，不得再作为代码、数据或训练入口。

### 0.1 研究问题和主张边界

AU/FACS在当前主路线中只负责回答“应关注哪些面部语义区域”，68点landmark负责回答“这些区域在当前aligned帧中的几何位置是否准确、完整和稳定”，RGB局部帧才是模型学习的数据。当前不读取`AUxx_r/AUxx_c`作为输入，不建立AU prediction head，不计算AU loss，也不因OpenFace AU数值失败而否决局部RGB。

P0C已经以技术`PASS`完成300视频、493,141帧审计，但`AU12_r/AU14_r/AU15_r`组为`INELIGIBLE_METRIC`。该结果冻结为“逐帧AU强度不适合作为当前版本辅助监督”的负证据；它不证明FACS区域语义无效，也不阻塞landmark裁切。extension AU数值审计、PB-P0D/P0E AU风险与PB-P1/P2训练分支从当前主路线撤下，只有未来重新注册AU数值监督时才恢复。

首轮只使用现有`112x112` aligned RGB。局部裁切只能重新分配模型注意力和尺度，不能恢复aligned输入已丢失的高频细节；高分辨率raw-warp另列为独立后续路线。确定性曝光/颜色校正继续不进入输入，photometric变化只作为physical-train在线增强。

### 0.2 职责和数据流

```text
AU/FACS研究
  -> 定义eye_brow / nose_cheek / mouth_lower_face语义范围

aligned RGB + aligned-space 68 landmarks + success/confidence
  -> 预处理器冻结逐帧crop box、validity、失败原因和provenance
  -> region manifest

Dataset / Sampler
  -> 读取global RGB和region manifest
  -> 确定性crop、等比例resize、neutral padding、有效段采样和train-only增强
  -> 不重新判断、修复或搜索区域

[B,T,4,3,112,112] + frame_mask[B,T] + view_mask[B,T,4]
  -> one shared backbone
  -> global-residual validity-aware fusion
  -> one-layer GRU + masked temporal mean
  -> BDI prediction[B]
```

预处理负责输入控制与质量裁决；模型只接受已冻结的RGB和mask。首版不立即物化约148万张局部JPEG，避免重复压缩和额外副本漂移；正式预处理只生成带SHA的crop manifest，Dataset机械执行裁切。只有smoke证明在线裁切成为I/O瓶颈时，才另行披露并授权缓存物化。

### 0.3 三语义区非重叠合同

固定视图顺序为：

```text
V0 global_face
V1 eye_brow
V2 nose_cheek
V3 mouth_lower_face
```

标准68点landmark的索引基准必须在未来policy中明确冻结。区域语义锚点为：眉眼使用眉毛与双眼点；鼻颊使用鼻梁/鼻翼并结合眼角和面廓锚点；嘴部使用内外唇并结合下颌/下巴锚点。三个主体RGB区域不复制相同像素，相邻边界只归属一个区域；安全margin通过移动公共边界实现，不同时向两个区域扩张。

accepted crop必须完整包含冻结的语义landmark/polygon，保持宽高比并采用resize+neutral padding。landmark缺失、非有限、严重越界、语义区域被截断、有效尺度不足、严重遮挡或时间跳变时，只将对应local view标为invalid；global仍有效时该帧继续参与BDI。所有local无效时，融合输出必须逐值退化为global。

### 0.4 Region资格门禁

现有T0A的300/300视频、零offset frame合同可复用。历史T0B的40个`FAIL`来自逐视频strict 0.995门槛，不直接适用于mask-aware local路线；AI辅助overlay的115 PASS/5 UNCERTAIN也不是人工批准。

新的门禁顺序固定为：

1. `REGION-P0A`：绑定aligned JPG、landmark CSV、frame、schema、split和SHA，只读取landmark/quality，不读取AU值；
2. `REGION-P0B`：physical-train pilot比较static canonical、raw dynamic、stabilized dynamic三种box；
3. `REGION-P0C`：仅用train无标签几何分布和人工overlay冻结margin、尺度、越界、coverage、stability和jump规则；
4. `REGION-P0D`：将冻结policy应用到300视频/493,141帧，val/test只报告，并输出region-only机器资格和带SHA的`eligible_region_profile`；该资格不依赖AU group eligibility。

候选门槛为：100%来源/hash/schema/frame绑定；accepted crop语义landmark containment为100%；physical-train每区域总体有效率不低于0.95；至少90/100个train视频的区域有效率不低于0.80；median adjacent-box IoU不低于0.75；jump rate不高于0.05；人工复核无系统性高姿态偏移或语义截断。低coverage视频不删除，只mask local view。若stabilized dynamic失败但static canonical通过，则退回static；两者均失败才判该区域不合格。

正式预处理至少输出`region_policy_v1.json`、`region_frame_manifest.csv`、`region_video_summary.csv`、`region_split_task_summary.csv`、`region_issues.csv`、overlay清单、`region_eligibility_decision.json`和`run_manifest.json`。这些未来产物尚未实现或授权。

### 0.5 Dataset、模型、损失与增强合同

输入合同为：

```text
views       [B,T,4,3,112,112]
frame_mask  [B,T]
view_mask   [B,T,4]
frame_ids   [B,T]
labels      [B]
```

模型只gather有效view，经一份共享DeiT-tiny backbone和共享projection后scatter回`[B,T,4,D]`；以global为不可丢失残差锚点进行masked local fusion，再进入现有单层GRU和masked mean。参数量不随view数成倍增加，但backbone FLOPs和激活显存最高接近global-only四倍，必须在smoke实测。

第一版只有train-only severity-balanced BDI regression：

```text
L_total = L_BDI_severity_balanced
```

不存在AU、gaze、head、pose、identity、ordinal、CCC或Stage C辅助梯度。validation/test/inference使用相同四视图合同但不增强。空间增强在四视图和整段时间上共享；普通颜色和方向感知曝光可按view独立，但每个view整段序列参数固定。global随机流不能因local删除而变化。

预处理先给出global有效连续段；Sampler不得删除内部无效帧后拼接时间。训练按video-first再采有效clip；validation/test确定性覆盖后先聚合成唯一video prediction再计算指标。REF、FULL、GRID和消融必须绑定同一clip manifest，局部view不能改变video/subject的BDI统计权重。

### 0.6 全模型优先混合消融

旧`GLA-FULL`名称停止使用，避免误解为包含AU loss。主候选固定为`GLA-RGB-FULL`：

| Run | 内容 | 目的 |
|---|---|---|
| `GLA-C-REF-S42` | global-only，匹配新采样/增强合同 | 强基线 |
| `GLA-RGB-FULL-S42` | global+三个语义local | 完整能力 |
| `GLA-RGB-NO-EYE-S42` | 删除眼眉 | 条件贡献 |
| `GLA-RGB-NO-NOSE-S42` | 删除鼻颊 | 条件贡献 |
| `GLA-RGB-NO-MOUTH-S42` | 删除嘴部 | 条件贡献 |
| `GLA-RGB-GRID-S42` | 面积、mask、FLOPs匹配的非语义区域 | 语义反证 |
| `GLA-RGB-NO-EXPOSURE-AUG-S42` | 仅关闭方向感知曝光增强 | 增强归因；FULL通过后才运行 |

FULL技术失败立即停止。FULL科学失败时只完成三个leave-one-region-out定位，不运行增强控制和multi-seed。只有`FULL > GRID`才能主张FACS语义区域有效。最终候选必须在paired seeds42/43/44方向一致，之后才允许一次locked post-selection benchmark；历史role-swap决定了该benchmark不能被表述为从未参与历史选择的纯test。

候选seed42门槛为：相对REF满足`delta_MAE<=+0.50`且`delta_CCC>=-0.05`；初步收益为MAE至少改善0.25或CCC至少改善0.03；identity top-1恶化不超过0.05；severe absolute bias恶化不超过1.0；同主体task prediction difference恶化不超过0.5。最终数值必须在首个训练run前写入独立strategy manifest，不能依据validation结果回填。

### 0.7 实现与授权顺序

当前仓库的`AVECDataset`和`MTLLiteDepressionModel`仍是单路`[B,T,3,H,W]`合同，没有三语义区Dataset、region manifest reader、四视图融合或对应runner。未来建议新增默认关闭的独立路径，保持旧配置、API和checkpoint兼容；不在首版直接重写旧baseline。

后续授权顺序固定为：region contract代码 -> train pilot -> full region audit -> dataset/model代码 -> one-batch和100-step smoke -> seed42 -> multi-seed -> locked benchmark。代码、运行、commit和push均须单独披露并授权。

## 1. 2026-07-30历史方案：AU数值辅助监督（已被第0节取代）

### 1.1 当时状态、目标与主张边界

> 本节及后续第2--12节仅保存2026-07-30方案原貌；其中`AU target`、AU辅助头、`GLA-FULL`、P0D/P0E和5%--10%区域重叠均不是2026-08-05当前合同。当前执行只以第0节为准。

状态日期：2026-07-30。

本方案已经完成headless GLA与train-only photometric augmentation的实现前设计冻结，尚未修改 dataset、model、runner 或训练配置，也未授权模型代码或训练。PB-P0A3权威full-rich v2已经完成300/300视频、493,141/493,141行和单一182列schema；A2在完整physical train上给出`PASS_FULL_SOURCE_COVERAGE`、0 blocker、0 warning。该结果只完成来源/coverage门禁，不授权P0B、扩展AU、局部裁切、模型实现或训练。

2026-07-30最终边界为：未来GLA不含head-motion辅助分支，gaze继续完全排除；pose/head历史提取与coverage只保留为质量、姿态分组和shortcut审计证据。确定性曝光校正、曝光派生mirror和颜色归一化也不进入当前GLA输入路线；模型始终读取原始对齐RGB及其语义crop，曝光与普通颜色变化只作为physical-train在线增强。

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
  -> 输入读取、对齐、landmark坐标和质量评估
  -> global/三个local crop生成
  -> frame/view validity、coverage、质量原因和provenance
  -> 不修改曝光或颜色，不生成当前训练所需的tone-normalized mirror

Dataset / Sampler
  -> 只读取冻结RGB、mask、frame id、pair mask和训练期AU目标
  -> physical train在线执行时序一致的曝光/普通颜色增强
  -> validation/test不增强且不打开AU target root
  -> 不把local crop当作独立样本
  -> 不因AU/OpenFace失败重新采样“更标准”的正脸

模型
  -> 只接收global/local RGB及valid mask
  -> 共享backbone编码、轻量区域融合、时序聚合和BDI预测
  -> 训练期可选AU小头；不负责检测、重新裁切、采样或修复输入
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
- flip/affine等空间增强在四视图之间以及各自时间序列内共享同一参数，避免破坏AU6/AU14跨区域对应；
- 曝光和普通颜色增强允许四视图独立采样，但同一view slot的完整帧序列只采样一次参数，禁止逐帧随机变化；
- 局部视图不计为新增 subject、video、clip 或独立样本。

### 3.1 Train-only photometric augmentation

GLA不使用确定性P1/P2 photometric normalization，也不读取曝光派生RGB root。所有photometric变换只在physical train在线执行；validation、test和inference始终读取未调整的原始对齐RGB与冻结local crop。AU target仍来自未增强输入且不随增强重算，这一安排只表示曝光/颜色不变性监督，不能声称恢复已剪切高光、压扁暗部或改善此前已经完成的OpenFace检测。

曝光增强按view slot分别使用label-blind physical-train统计冻结under/normal/over参考带。过曝view只允许向正常或略微欠曝移动，欠曝view只允许向正常或略微过曝移动，正常view保留identity路径并可弱双向扰动。首版预注册：整组clean概率`0.25`；其余样本每view曝光增强概率`0.50`；异常曝光越过正常带概率`0.25`；正常曝光向暗/向亮各`0.50`。曲线使用单调endpoint-preserving log/inverse-log族，不逐帧拟合；统计不足或曝光不稳定时no-op。

颜色增强使用普通图像`ColorJitter`，不拟合项目颜色参考：`brightness=0`、`contrast=0.2`、`saturation=0.2`、`hue=0.05`、`p_color=0.5`。`brightness=0`避免与方向感知曝光重复。顺序固定为：跨视图共享的空间增强 -> 每view独立且时间固定的ColorJitter -> 每view独立且时间固定的曝光增强 -> 恢复padding/invalid像素 -> 模型Normalize。随机流由`seed/epoch/video_id/clip_id/view_slot/augmentation_kind`稳定派生；view缺失、AU组删除或条件切换不得改变其他slot的随机流。

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
[B,T,4,C,H,W] + frame_mask/view_mask/frame_ids/pair_mask
  -> gather valid view slots only
  -> one shared backbone, chunked execution
  -> scatter valid features back to [B,T,4,F]
  -> shared Linear + LayerNorm + GELU projection
  -> [B,T,4,D]
  -> global-residual, validity-aware local fusion per frame
  -> [B,T,D]
  -> existing single-layer GRU
  -> video/clip representation
  -> BDI regression head

regional / cross-regional temporal summaries
  -> grouped AU auxiliary heads (train only)
```

固定view顺序为`global_face/eye_brow/nose_cheek/mouth_lower_face`；`view_mask[...,0]`必须等于`frame_mask`，local mask必须是其子集，每个序列至少包含一个有效global frame。首版GRU只接受有效前缀加尾部padding；内部global空洞必须由预处理切段或fail closed。`pair_mask[B,T-1]`显式声明可计算动态差分的相邻位置，禁止跨无效帧、source gap或采样断点求差。

融合使用global作为不可丢失的残差锚点：local score由`[global, local + view_embedding]`的共享线性层产生，valid local经masked softmax得到context，再由标量gate和共享线性adapter形成`fused = global + has_local * gate * adapter(context)`。view embedding只参与评分，不直接写入context；所有local无效时fused必须逐值等于global。gate bias初始化为`-2`，使训练初期以global路径为主。

必须先在帧级把四个视图融合为一个 token，再进入现有时序模块。禁止首版把序列直接扩展为 `4T` 后送入 Transformer，因为自注意力计算可能从 `T^2` 增至约 `16T^2`。

复杂度目标：

- backbone 参数量不因视图数量增加；
- region embedding、门控和 AU 小头的新增参数原则上低于总模型的 1%；
- 像素级四视图的 backbone FLOPs 和训练激活显存可能接近 global-only 的 4 倍，必须在 smoke 中实测；
- AU小头只在训练期计算loss，不参与 checkpoint 选择；
- 首轮 train/validation/test 都使用相同的 global+有效 local 视图合同，但 validation/test 不读取 AU target；
- 若未来要求 global-only 部署，必须另建蒸馏/一致性实验，不能把首轮四视图模型在测试时静默切成 global-only。

ROIAlign 单次 backbone 方案只作为后续效率对照，不与首轮像素级局部帧同时改变。

首版GLA继续使用匹配C-REF/E2的DeiT-tiny、`D=192`、单层GRU、normalized BDI regression和severity-balanced MSE。ordinal classification、identity adversarial、Stage C nuisance/reconstruction与CCC训练损失全部关闭；它们不是`GLA-FULL`的隐含组成，也不得与本轮区域/AU因素同时改变。

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
        + ramp(e) * lambda_AU * L_AU_fixed_denominator
```

约束：

- 前10% epoch关闭辅助损失，随后约20% epoch线性ramp；
- `lambda_AU`只允许从`{0.01, 0.03, 0.1}`经100-step train-only梯度校准冻结，不用validation搜索；
- AU loss仅在train计算；validation/test隐藏OpenFace target root仍必须运行；
- 先按descriptor、AU和实验组聚合；总分母固定为FULL eligibility profile，被删除或当前batch无有效target的组贡献0，禁止自动放大剩余组；
- checkpoint只按预注册BDI validation monitor/mode选择；
- 每组记录与BDI梯度的norm、cosine、conflict、cancellation和长期占比。

AU路径从融合前的区域投影特征分出。AU梯度只进入共享backbone、共享projection和AU head，不进入view fusion、时序编码器或BDI head；BDI梯度仍通过所有有效global/local视图进入完整主路径。因此本方案只能主张AU监督约束区域视觉表征，不能宣称其直接正则化GRU或完成时序特征对齐。

## 8. 数据和质量门禁

模型实现前必须通过以下门禁；几何与行为来源审计可独立推进，但必须在P0E资格清单处汇合：

1. `PB-P0A3`：权威full-rich v2完成300/300视频、493,141行、单一182列schema和全部hash/provenance闭环（已完成）；
2. `PB-P0B`：先审计/修复旧P0核心对mask-aware来源的兼容性，再对不可变v2来源执行core/schema/exact join/coverage；41个legacy strict失败不得误拒mask-aware来源；
3. `PB-EXP-AU`：为AU4/6/7/10/17另建versioned source/schema/coverage/descriptor合同；v2含列不等于这些列已获训练资格；
4. `PB-P0C`：matched raw/aligned物理保真只裁决候选AU组；pose/head结果继续作为历史诊断，不是训练资格轴；
5. `PB-P0D`：对实际训练描述符执行identity/task/exposure/quality风险审计；
6. `FACE-S1 + LM/REGION`：global可见性、三语义区point/polygon/margin/coverage overlay和crop manifest冻结；
7. `PB-P0E`：新增AU-only机器可读清单，输出`REGION_ELIGIBLE`、`AU_GROUP_ELIGIBLE(core/extension_A/extension_B)`、`CROSS_REGION_ELIGIBLE`、`AU_ELIGIBLE`、`GLA_PROFILE_ELIGIBLE`、完整`eligible_component_profile`及其SHA，并记录`excluded_by_design: [head_motion_auxiliary]`与`P1_CODE_AUTHORIZED=false`。该profile必须是依赖闭合的最大合格集合。

pilot20当前结果为16 PASS / 4 FAIL，失败代理为：

```text
205_1_Freeform  0.990909
206_1_Freeform  0.920000
207_2_Freeform  0.698148
208_1_Freeform  0.979227
```

因此不得直接把strict `0.995`改成刚好通过的阈值，也不得删除失败视频。已冻结的mask-aware policy把100%视频/行/schema/hash/provenance作为硬完整性门禁，以`success==1 && confidence>=0.8`定义quality-valid，并把`success/confidence`只用于view/target validity和coverage。权威full判定为`PASS_FULL_SOURCE_COVERAGE`：quality/AU-valid 486,441帧、历史head-valid 485,863对；physical train总体/Freeform/Northwind分别为0.989273/0.983944/0.996984，task gap约0.01304，最低train视频仍有180个AU-valid帧。head计数只保留为来源审计，不参与GLA资格。val/test仅报告，且该判定仍明确`P0B/training_authorized=false`。RGB可见但OpenFace失败的帧仍可参加global BDI；landmark局部裁切无效时只mask local view；AU无效时只mask对应AU loss。

A2 full决策当前存在一个仅影响报告分支的已知问题：状态为`PASS_FULL_SOURCE_COVERAGE`时，`next_action`仍错误写成`STOP_AND_REVIEW_POLICY_OR_SOURCE_ISSUES`。P0B前必须按代码实施前披露门禁单独授权修复并增加回归测试；不得用手工修改既有决策JSON掩盖该问题。

扩展AU后，现有只投影AU12/14/15的PB合同不再足够。未来必须版本化升级source manifest、白名单、schema校验、normalization、risk audit和测试，禁止手工给旧manifest补列或混用历史raw-video rich CSV。

## 9. 全模型优先的混合消融矩阵

所有条件保持同一physical split、seed、clip manifest、输入root、backbone、optimizer、precision、训练步数、severity loss、EarlyStopping和checkpoint policy。视图数量不同的对照按video保持相同BDI统计权重。

本文中的`GLA-FULL`严格等于同一个AU-only P0E eligibility manifest给出的最大依赖闭合`eligible_component_profile`，包括获批的global/local region、AU组和cross-region关系；head motion已按设计排除。它不是`src/legacy/full_model/`中的legacy full model，也不是Stage C历史运行`C-FULL`。任何未通过P0E或失去上游依赖的组件都必须在训练前从profile中删除；这属于门禁裁决，不得伪装成训练后的减法收益。所有run必须记录该profile及其manifest SHA。

### Stage 0：输入与实现验证，不比较test

```text
GLA-SMOKE-1B       one-batch forward/backward, mask和shape
GLA-CAL-100        seed42, 100-step梯度/显存/FLOPs校准
GLA-GRID-CAL-100   三个等面积grid对照的匹配校准
```

通过条件包括：loss/gradient有限、invalid view不贡献AU loss、四视图不改变video权重、AMP稳定、隐藏AU root的validation可运行、共享backbone确实只有一份参数。共享模块必须使用独立的确定性初始化随机流，保证新增/删除辅助head不会改变backbone初始化；各条件还必须配对sampler顺序和逐样本增强随机数流。

`GLA-CAL-100`只在最大eligible profile上进行一次。由train-only校准冻结的`lambda_AU`、ramp和组权重必须跨全部消融复用；总辅助损失的分母固定为FULL白名单，被删除组贡献0，禁止因剩余组变少而自动放大其loss。正式训练前还必须冻结machine-readable strategy manifest，记录完整run ID、eligibility SHA、相邻比较、幸存阈值、最大run数和停止分支；不得依据validation结果临时补写矩阵。

### Stage 1：`GLA-FULL`优先筛选

| Run ID | 条件 | 作用 |
|---|---|---|
| `GLA-C-REF` | global-only强基线；与候选使用相同代码版本、split、seed和训练预算 | 提供匹配reference |
| `GLA-FULL` | 全部P0E-eligible三语义区、AU组和跨区域处理 | 先回答完整路线是否具有继续价值 |
| `GLA-RGB-FULL` | global+三个语义local，无AU监督 | Stage 2的S3阶梯；若`GLA-FULL`仅触发科学utility/risk失败，仍作为S1--S4诊断链的一环执行 |

seed42只读取validation；旧`C-REF`日志不能替代本次匹配`GLA-C-REF`。这里的“FULL优先”表示在任何组件筛选前先评估完整候选，匹配reference可以先运行或并行运行。若出现source/hash/join、val/test读取行为目标、NaN、mask/shape、AMP或无法解释的DDP技术失败，立即停止全部训练。若`GLA-FULL`只触发预注册的科学utility/risk失败，则仍完成S1-S4反向阶梯作故障定位，但跳过机制控制、加法复核、seeds43/44和benchmark。未触发科学停止条件时，Stage 2才作为正式组件证据继续。

### Stage 2：依赖感知的反向减法消融

下表是所有组均eligible时的标准阶梯。实际矩阵从P0E集合`E`生成：从`G+L+E`开始，按`B -> A -> C -> L`固定顺序删除；不在`E`中的组记录`SKIPPED_INELIGIBLE`并保持编号，不得用validation决定跳过或重排。每个实际run ID必须附加剩余component profile和`S42/S43/S44`后缀。每一行都从同一预训练backbone独立初始化并完整训练，禁止继承上一行checkpoint：

| 顺序 | Run ID | 从上一层删除 | 剩余能力 |
|---|---|---|---|
| S0 | `GLA-FULL` | 无 | 最大eligible候选 |
| S1 | `GLA-S1-NO-AU-B` | AU10/17 | 三语义区 + core + AU4/6/7 |
| S2 | `GLA-S2-CORE-AU` | AU4/6/7 | 三语义区 + AU12/14/15 |
| S3 | `GLA-RGB-FULL` | 全部AU辅助监督 | global + 三语义local |
| S4 | `GLA-C-REF` | 三个local及view fusion | global-only |

若某组未被P0E批准，相应阶梯必须在预注册run manifest中标为`SKIPPED_INELIGIBLE`并跳过，不能把“未进入FULL”写成消融结论。局部视图最后删除，因为区域AU依赖local/cross-region token；不得保留local-AU loss却移除其RGB来源。相邻阶梯只给出**顺序条件下的边际证据**，不等于因素的无条件独立因果效应。

同时冻结四个反证/训练策略控制：

| Run ID | 匹配条件 | 反证问题 |
|---|---|---|
| `GLA-FULLAU-NO-CROSS` | 与最高eligible AU阶梯匹配；AU6固定以eye-brow为单区锚点，AU14固定以mouth-lower-face为单区锚点，均经过同维线性adapter并保持目标数/参数预算；只有对应AU与相邻视图均eligible时运行 | AU6/AU14跨区域融合是否真正必要 |
| `GLA-RGB-GRID` | global + 三个面积/长宽比/padding/FLOPs匹配的grid，共享同一backbone和融合器；逐view复用对应语义local的valid mask，即使grid像素可读也不额外放行 | 语义区域是否优于一般多裁切 |
| `GLA-AUX-SHUFFLED` | 匹配最终幸存的真实辅助条件；在physical train内对subject做固定无自映射置换，同一subject两任务使用同一donor且按task匹配；保留接收样本mask/coverage，val/test不读目标 | AU收益是否只是噪声正则化 |
| `GLA-FULL-NO-EXPOSURE-AUG` | 与`GLA-FULL`完全匹配，只关闭方向感知曝光增强；普通ColorJitter保持不变 | 曝光不变性增强是否产生可复现收益 |

### Stage 3：幸存因素的加法复核

`GLA-C-REF -> GLA-RGB-FULL`先复核局部RGB。core-only直接复用Stage 2的`GLA-S2-CORE-AU`；若AU4/6/7或AU10/17在减法阶梯中存活，则各自最多增加一个以`GLA-RGB-FULL`为共同基准的`GLA-ADD-A`或`GLA-ADD-B`。只有“从FULL删除后变差”且“单组加回不触发失败”的因素，才能主张具有独立贡献；只在联合条件有效的因素必须标为interaction-dependent。

通过上述复核后最多构建一个`GLA-REDUCED`，按固定依赖顺序组合全部幸存组。失败组不重新搜索单AU组合、更多loss权重或额外辅助分支；跨区域处理只有匹配FULLAU阶梯优于`GLA-FULLAU-NO-CROSS`时保留。`GLA-REDUCED`必须在允许的随机波动范围内复现`GLA-FULL`方向，否则最终交互解释视为不稳定。

该阶段不是重新展开全组合搜索。精确加法run ID、保留组和比较对必须由Stage 2预注册阈值机械生成，并在执行任何加法run前写入只读matrix manifest。seed42最多允许5个阶梯、4个机制/训练策略控制、2个条件性单组加回和1个reduced rebuild，共12个full fit；未使用配额不得转为临时超参数或组合搜索。

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
- seed42相对匹配reference出现`delta_CCC < -0.05`或`delta_MAE > +0.50`，或identity/task/exposure、coverage、severity、compression、task robustness触发预注册风险门槛，属于科学失败；只允许完成S1--S4诊断阶梯，跳过controls、加法复核、multi-seed和benchmark；
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
 -> 无技术失败：完成S1--S4依赖感知减法阶梯
 -> FULL科学失败：阶梯仅作诊断并停止，不运行controls/additive/multi-seed/benchmark
 -> FULL科学通过：反证控制 -> 幸存因素有限加法复核 -> paired seeds 43/44
 -> 冻结后以一个具名package读取一次locked post-selection benchmark；role-swap仅保留为历史证据
```

截至2026-07-30该历史方案，只完成来源提取/A2 full coverage与文档设计；当时不得据此声明P0B、三语义区、跨区域AU、扩展AU白名单、共享backbone融合或训练入口已经实现。后续实际进展及当前授权边界见本文第0节。

## 13. 决策记录

| 日期 | 决策 |
|---|---|
| 2026-07-29 | 局部视图由旧四区收敛为眼眉、鼻颊、嘴部三个主体语义区，保留完整global face。 |
| 2026-07-29 | 相邻区域允许5%--10%边界容错，不要求像素级绝对无重叠。 |
| 2026-07-29 | global/local使用同一个backbone；帧级融合后再进入原时序模块，禁止首版构造4T序列。 |
| 2026-07-29 | AU不再永久局限于AU12/14/15；首轮分级扩展至AU4/6/7/10/17，但必须经过来源、coverage和shortcut风险门禁。 |
| 2026-07-29 | 采纳跨区域AU：AU6由眼眉+鼻颊特征预测，AU14由鼻颊+嘴部特征预测。 |
| 2026-07-29 | gaze继续排除；AU25/26因语音/任务混杂不进入首轮。 |
| 2026-07-29 | 当时规划AU/head只作为训练期辅助目标；该head部分已被2026-07-30决定取代，val/test不读取OpenFace目标的边界继续保留。 |
| 2026-07-30 | PB-P0A3权威full-rich v2与A2完整coverage门禁通过；legacy strict 0.995仍仅作诊断，P0B和训练未授权。 |
| 2026-07-30 | 实验顺序改为`GLA-FULL`优先筛选 -> 依赖感知减法 -> 幸存因素加法复核 -> multi-seed -> 单次locked post-selection benchmark package；保留grid、no-cross和shuffled反证控制。 |
| 2026-07-30 | `GLA-FULL`只含P0E批准组件，不等于legacy full model或Stage C `C-FULL`；资格失败不得伪装成消融收益。 |
| 2026-07-30 | 取消未来head-motion训练分支；pose/head历史证据退为audit-only，P0E改为AU-only，减法顺序改为`B -> A -> core AU -> locals`。 |
| 2026-07-30 | 取消确定性曝光/颜色处理作为GLA输入；改用physical-train在线增强。空间参数跨视图/时间共享，photometric参数可按view独立但在各自序列内固定。 |
