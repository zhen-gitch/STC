# AVEC2014 Source and Derived Video Quality Record

> 文档职责：记录当前300个AVEC2014视频在原视频、aligned JPG、landmark/OpenFace派生层的已知问题、证据、处理边界与待补审计。本文是后续曝光调整、人脸片段选择、AU语义保持的landmark局部裁切、已登记行为伪标签来源合同和论文数据说明的权威问题档案。

## 核心区分

本项目必须区分三类问题：

1. **原视频问题**：人物离场、原始曝光剪切、真实大姿态、遮挡、出界、模糊、距离变化和视频时长不一致；
2. **aligned派生问题**：人脸检测/对齐失败后生成纯黑JPG、裁剪边界、插值、黑色padding和细节损失；
3. **landmark/OpenFace问题**：`success=0`、低confidence、landmark越界/跳变、坐标版本和工具版本不一致。

纯黑aligned JPG不等于原视频在该时刻为黑屏，也不等于人物缺席。已完成的source-presence复核证明，大部分纯黑占位对应“原视频仍有人，但检测/对齐失败”。论文、代码和报告不得把三层问题混写。

## 数据契约

当前审计覆盖：

```text
videos: 300
aligned JPG frames: 493,141
raw video geometry: 640 x 480 @ 30 FPS
split: train / val / test各100视频
tasks: Freeform / Northwind
```

300/300原视频帧数与aligned JPG数量完全一致，frame offset为0；这证明frame identity可回连，但不证明每个aligned帧包含有效人脸，也不证明landmark坐标有效。

原视频、原始aligned JPG和历史OpenFace输出必须只读保留。任何曝光或warp结果都写入独立版本目录，并记录源/派生SHA-256。

## 问题总表

| 问题 | 所在层 | 已知规模 | 当前处理 |
|---|---|---:|---|
| OpenFace/landmark失败 | landmark派生 | 6,501帧，1.3183% | 逐帧分类；不能静默当有效 |
| 纯黑aligned占位 | aligned派生 | 4,206帧，0.8529%；38视频/231 run | 回连原视频presence；不使用aligned邻帧合成 |
| 可见但检测失败 | landmark派生 | 2,295帧，0.4654% | global RGB不自动删除；landmark局部裁切不可用 |
| 人物仍在但aligned失败 | 原视频存在/派生失败 | 2,365帧 | 3帧真实raw-warp smoke已通过自动gate；全量策略/materialize未授权 |
| 人物缺席 | 原视频内容 | 1,841帧，1个视频 | 永久invalid，片段硬边界 |
| 欠曝候选 | 原视频/派生共同可见 | 18视频 | 17视频批准fixed log；`310_2_Freeform` keep_raw |
| 过曝候选 | 原视频/派生共同可见 | 4视频 | 2视频tone-only；`212_1/223_1 Freeform` keep_raw |
| raw可恢复而aligned已丢失的细节 | raw-vs-aligned | 0个通过候选 | 禁止声称恢复不存在细节 |
| 视频长度不一致 | 原视频 | 180–7,440帧 | 禁止固定head代表整视频 |
| 大幅遮挡/偏转/出界的无有效脸片段 | 原视频与检测层 | FACE-S1已完成全量几何/像素proxy分布；语义遮挡互斥计数尚未冻结 | train-only人工复核后冻结threshold manifest |

## 原视频本身存在的问题

### 人物离场

目前唯一确认的长人物缺席发生在：

```text
247_3_Freeform_video_aligned
person visible failure: frames 2078-2143
person absent: frames 2144-3984
person re-enters: frames 3985-4213
```

缺席段共1,841帧。此时原视频画面没有可用人脸，不能通过复制、光流、生成模型或前后帧插值补造。时间clip必须在2144和3984两侧断开，不能删除空房间后把前后行为连接成连续序列。

### 大姿态、遮挡、出界和距离变化

source-presence全量复核确认，人物仍在但aligned失败的片段包含：

- severe yaw/pitch或快速转头；
- 手、耳机、麦克风、头发等遮挡；
- 脸部部分或大部分离开画面；
- 人物位置过低、过近或尺度快速变化；
- 运动模糊和曝光变化。

这些现象在presence复核中作为原因描述，FACE-S1 phase-1又补齐了全量pose、coverage、in-frame、blur、residual、jump和confidence分布。`face_present`仍只证明人物在场，不证明存在适合global输入或landmark局部裁切的有效人脸；几何proxy也不能自动给出“手/麦克风/头发遮挡”的互斥语义计数，major occlusion仍需train-only人工标签。

### FACE-S1 phase-1全量分布

`face_usability_phase1_v2`覆盖全部300视频/493,141帧。486,640个landmark成功帧保留为`pending_threshold_review`；2,365个纯黑但人物在场帧标记`aligned_failure_pending_raw_warp`，2,295个可见检测失败帧标记`face_present_low_quality`，1,841个人物缺席帧标记`person_absent`。没有帧被批准为`face_usable`。

全量成功帧的主要无标签分位数为：landmark in-frame ratio q01/q50=`0.926/0.985`，face-hull visible ratio q01/q50=`0.905/0.992`，absolute pose的大多数样本位于约yaw 26度、pitch 21度以内，transform residual q95/q99=`0.132/0.205`，landmark jump q95/q99=`0.0119/0.0235`。这些值只用于组织train复核，不是已批准阈值。Freeform的pose/residual/jump尾部普遍高于Northwind，后续必须报告task分组排除率。

train-only 132个极端帧contact sheets表明：大yaw/pitch、低coverage和严重出界可由几何量定位；低blur与曝光/对比度混杂；高residual常与大姿态重合。12个最高jump的`t-1/t`复核没有显示系统性landmark teleport，高jump多为真实运动、表情或模糊变化。因此任何正式gate都必须联合多指标，jump和blur不能单独作为删除理由。

轻中度转头、低头、目光变化和自然遮挡可能属于抑郁相关行为，不能因为它们使检测更难就全部删除。严格片段选择只排除明确无有效脸的极端帧，并报告排除原因和分组分布。

### 原始曝光和剪切

初始全量审计识别18个欠曝、4个过曝候选、278个正常曝光视频。进一步使用92个train-normal视频全帧统计冻结：

```text
post-transform safety band q20/q80: [49.018280, 142.171405]
internal target q25: 53.740102
internal target q75: 135.769350
train-normal temporal luma-IQR q90: 10.243687
diagnostic q95: 12.024606
```

全22候选最终形成24个无间隙、无重叠reviewed exposure segment：

- 17个欠曝视频使用`stable_log`，共19个log segment；
- `211_1_Freeform`、`242_1_Northwind`使用`tone_only_overexposed`；
- `212_1_Freeform`、`223_1_Freeform`、`310_2_Freeform`使用`keep_raw`。

需要单独说明的边界：

- `219_1_Freeform`按无标签单变点最小二乘分为`1-1494`与`1495-3270`；
- `219_3_Northwind`按实际短暗阶段分为`1-65`与`66-1170`；
- 姿态、距离或遮挡引起的亮度变化不允许生成逐内容曝光参数。

`212_1_Freeform`和`223_1_Freeform`保留raw高光剪切，而不是使用不稳定增强；`310_2_Freeform`因连续自动曝光升降没有唯一粗粒度边界而keep_raw。

### 原视频也已丢失的细节

raw-vs-aligned审计把raw帧warp到相同aligned几何，在同一侵蚀face hull内比较剪切、gradient、Laplacian和entropy。22个曝光候选701个抽样帧中685个通过几何gate；结果为：

```text
raw_detail_recoverable: 0
raw_also_clipped: 499 frames
no raw recoverability evidence: 185 frames
raw less clipped but texture evidence insufficient: 1 frame
invalid comparisons: 16 frames
```

因此曝光处理只能称为tone normalization或把可见亮度移回train-normal包络。已经剪切、量化或编码丢失的纹理无法恢复，禁止通过继续增强后宣称恢复细节。

## Aligned JPG与检测派生问题

### 纯黑占位不是原始黑屏

4,206个纯黑aligned失败帧分布在38视频、231个run中。全量source review得到：

```text
person present but aligned detection failed: 2,365
person absent: 1,841
mixed/ambiguous after segmentation: 0
```

230个run全程人物存在，只有`247_3_Freeform`需要presence分段。由此冻结：

- 2,365帧只能进入`raw_frame_warp_only`；当前仅3帧smoke完成，不能视为全量可恢复；
- 1,841帧永久`keep_invalid`；
- aligned邻帧光流合成、复制上一张脸和跨离场段插值全部禁用。

早期报告中的59帧短gap光流和previous-frame copy只是被后续原视频证据推翻的历史方案，不能用于正式派生数据。

### 真实raw-frame warp smoke

`raw_frame_warp_smoke_v3`只在train split、完整source-presence gate和前后合法raw/aligned landmark锚点下运行。它在原视频相邻帧传播landmark并warp缺失时刻的真实raw像素，不混合前后aligned脸。自动通过3帧，combined tracking valid ratio为`0.980-1.000`、target transform RMSE为`0.981-1.245 px`、tracked-vs-interpolated transform分歧为`0.854-2.464 px`；另有1帧因前raw锚点无效而明确拒绝。contact sheet保留了目标时刻的身份、姿态、耳机/麦克风和表情。

该结果只证明最短单帧缺失在严格锚点条件下可行。输出仍是`AUTO_PASS_REVIEW_REQUIRED`，没有重跑landmark，也没有接入正式materialize；不得把3帧结果外推到全部2,365帧或用于训练。

### 可见失败帧的双重含义

2,295个OpenFace失败帧仍有可见RGB。它们对不同输入路径含义不同：

- 对global完整脸RGB：不应因`success=0`自动删除，必须进一步检查真实脸部可见性；
- 对landmark局部裁切：当前没有可信坐标，不能生成局部view；
- 对face-valid时间切片：大偏转/遮挡/出界片段应判低质量，普通可见检测失败可在统一landmark重跑后重新评估。

这也是为什么正式片段选择不能只读取OpenFace success，也不能把全部可见失败统一保留。

### 工具版本与坐标

当前全量aligned-landmark输出记录为OpenFace 2.2.0，493,141行与JPG数一致；但只有486,640帧`success=1`。历史最初aligned图片的生成版本仍不完全确认，后续会在派生输入方案冻结后统一重跑OpenFace，保证landmark版本一致。

当前已有坐标审计证明：原始OpenFace CSV约处于640x480检测坐标，不可直接按`112/640`、`112/480`独立缩放到aligned输入。landmark局部裁切只能使用：

1. aligned JPG上统一版本重新检测的aligned-space landmark；
2. 已保存并验证的逐帧alignment transform；
3. 通过overlay验证的canonical similarity映射。

没有合法映射时必须BLOCKED，不能猜测坐标。

### PB-P0已执行的AU/head训练期伪标签来源边界

`PRIVILEGED_BEHAVIOR_ALIGNMENT_PLAN.md`登记的独立研究分支目前只授权并完成PB-P0合同/CLI/no-gaze extractor基础设施、两视频debug与landmark-only阻断审计。其`AU12_r/AU14_r/AU15_r`与旋转head dynamics只能来自模型实际使用的aligned JPG完整序列、冻结OpenFace 2.2.0与冻结profile `-2Dfp -pose -aus`；历史raw-video `openface_features`不得补入当前aligned RGB监督。`debug2c`已对2/2视频、1,920帧/行`PASS`，但不构成全量提取或训练授权。当前audit-only rich manifest也不能直接改名或解释为training-target contract。

gaze已从该研究分支完全排除。正式landmark-only P0对300/300视频、493,141张JPG完成exact video/task/frame join，但因全量CSV缺少`AU12_r/AU14_r/AU15_r/pose_Rx/pose_Ry/pose_Rz`而`BLOCKED`，不得回退历史raw rich CSV。冻结source-video contract SHA-256为`12f50c2311b9d89dde81e27fa83891226ca5f447ed7d3d56fe38cf673a1c5a31`，300/300帧数匹配且统一30 FPS；行为合同只允许用它生成`t=(frame_id-1)/30`，不允许读取raw OpenFace路径或AU/pose数值。aligned JPG `-fdir`输出的`timestamp`为常数0，禁止用于速度；旋转差wrap后只在连续有效帧间计算。下一门禁是单独授权全量aligned rich提取后重跑coverage/source-provenance/train-only normalization；aligned pose物理保真与identity/task risk仍未运行通过，P1/P2和训练仍未授权。AU/head scalar目标不依赖landmark到local-crop的空间映射，因此既不能借T0b coordinate状态自动获批，也不能绕过当前face-usability和派生输入的数据披露要求。

## 视频时长和当前采样偏差

视频原始长度分布：

```text
min: 180 frames
q25: 1080
median: 1260
q75: 1860
q90: 3120
max: 7440
videos > 2000 frames: 66
videos > 3000 frames: 33
```

当前`stride_head(MAX_SEQ_LEN=2000,SAMPLE_STEP=10)`只采第1至1991帧范围。按批准raw-warp后的容量口径，83,840个可用帧（17.06%）位于head之后；部分长视频漏掉68%–73%的可用帧。

这说明当前训练不仅存在坏帧问题，也存在系统性前段偏置。新的face-valid切片必须覆盖所有合格连续run，而不是从整视频均匀随机取一个窗口，也不是继续只看开头。

## 后续曝光与片段选择如何使用本文

### 曝光

- 只使用`exposure_review_manifest.csv`中的固定video/segment曲线；
- pure-black和person-absent不参与亮度参数拟合；
- 不因pose/occlusion产生逐帧自适应参数；
- 每个派生帧记录curve、parameter、source hash和output hash；
- raw/repaired实验必须成对保留。

### 人脸片段选择

- presence gate先去除人物缺席；
- aligned failure gate区分pending raw-warp与可见失败；
- `FACE-S1`再判断大遮挡、大偏转、出界、模糊和landmark可信度；
- 只从连续`face_usable` run生成clip；
- 所有合格clip可用于训练，但按source video归一或video bag loss；
- validation/test先聚合为video prediction，clip不作为独立subject。

### AU语义保持的Landmark局部裁切

- AU/FACS定义眼眉、鼻颊、嘴部三个主体语义局部RGB视图；相邻区域只允许约5%--10%边界容错，详细polygon/margin合同见`GLOBAL_LOCAL_AU_EXPERIMENT_PLAN.md`；
- AU intensity/presence、AU序列、AU特征和landmark坐标都不作为推理输入；通过PB-P0E的AU动态只允许作为train-only辅助目标，validation/test不读取目标root；
- 使用验证后的aligned-space 68点landmark逐帧定位上述语义区域，并生成覆盖完整polygon和冻结margin的矩形RGB crop；
- local和global来自相同frame/clip，共享同一backbone，帧级validity-aware融合后进入同一temporal encoder和BDI head；
- landmark mapping、关键点集合、区域coverage或语义完整性无效时跳过local view，不从邻帧或隐藏侧伪造；
- 左右landmark只用于定位和可见性，不形成左右AU视图或重复监督；AU6与AU14的跨区域token处理由GLA合同控制；
- 三个等面积grid仅作为“语义区域是否优于任意局部裁切”的控制组；训练顺序采用FULL优先混合消融，不由本文定义。

## 数据版本与不可覆盖原则

建议后续固定以下版本：

```text
raw_video_v0                 # 原始AVEC视频，永久只读
aligned_jpg_v0               # 当前历史aligned图片，永久只读
tone_normalized_overlay_v1   # 仅reviewed曝光变化
raw_warp_overlay_v1          # 仅批准的人物存在aligned失败帧
aligned_repaired_mirror_v1   # 完整派生镜像，来源可追溯
landmark_aligned_v1          # 统一OpenFace版本的aligned-space landmark
face_usability_manifest_v1   # 冻结的人脸可用状态
face_clip_manifest_v1        # 冻结的训练/评估clip边界
```

每个版本必须记录git commit/branch、命令、Python/OpenFace版本、输入/输出SHA-256、split hash、阈值和人工review记录。数据集、split、标签和原始媒体永不原地覆盖。

## 论文中必须说明

论文数据与方法部分至少公开：

- 300视频、493,141帧和长度分布；
- 原视频与aligned/OpenFace派生问题的区别；
- 6,501个失败帧、4,206个纯黑占位、2,365个人物存在失败和1,841个人物缺席；
- 曝光候选、train-only安全带/目标、24个reviewed segment和3个keep_raw决定；
- `raw_detail_recoverable=0`及tone normalization主张边界；
- face-usability指标、阈值来源、排除原因和被排除比例；
- clip数不是独立subject数，video-level loss/metrics如何避免伪重复；
- raw与repaired配对消融及原始数据保留策略。

## 仍待补齐的证据

当前尚不能回答：

- 人工冻结阈值后，全300视频中extreme pose、major occlusion、严重出界分别有多少帧/run；
- 统一landmark重跑后2,295个可见失败帧中有多少可恢复；
- 严格face-valid gate会生成多少300/600/1200/2000帧clip；
- face质量筛选是否在subject、severity或Freeform/Northwind间分布不均；
- landmark local crop在大姿态下的有效率和与global crop的同步性。

这些问题由 `FACE-S1` train-only复核、独立threshold manifest和后续FACE-S2回答。在阈值冻结前，不运行正式片段训练，也不把phase-1的`pending_threshold_review`或OpenFace success当作最终人脸质量标签。

## 证据路径

```text
logs/au_region_tracking_audit/frame_failure_recovery/
logs/au_region_tracking_audit/source_video_presence/full_review/
logs/au_region_tracking_audit/exposure_temporal_stability_q90_v1/
logs/au_region_tracking_audit/exposure_review_full_q90_v1/
logs/au_region_tracking_audit/raw_detail_recoverability_v2/
logs/au_region_tracking_audit/validity_aware_slicing_v1/
logs/au_region_tracking_audit/face_usability_phase1_v2/
logs/au_region_tracking_audit/face_usability_temporal_review_v1/
```
