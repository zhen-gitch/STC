# Face-valid Temporal Segment Mining Plan

> 文档职责：冻结本项目“人脸可用片段挖掘、训练样本扩增、video-level聚合、AU语义保持的landmark局部裁切”的设计边界。原始/派生视频问题见 `AVEC2014_SOURCE_DATA_QUALITY.md`，脚本字段见 `SHORTCUT_AUDIT_DESIGN.md`，当前状态见 `CURRENT_STATUS.md`。

> **2026-07-30职责收敛**：本文继续作为face-valid时间切片与crop数据合同权威；局部视图已由历史四区改为眼眉、鼻颊、嘴部三语义区，train-only AU/head辅助监督和FULL优先混合消融统一由`GLOBAL_LOCAL_AU_EXPERIMENT_PLAN.md`管理。本文旧L0/L1/L2矩阵只保留历史背景，不再是训练执行入口。

## 2026-07-17 路线修订

本项目的时间切片目标不是保证窗口在视频内均匀随机，也不是把固定head替换为另一种随机采样。正式目标是：

1. 在完整帧率上识别真正存在可用人脸的连续片段；
2. 排除人物缺席、纯黑占位、不可解码、大幅遮挡、大幅偏转、严重出界和landmark不可信片段；
3. 将每个合格连续片段切成一个或多个可追溯clip，尽量利用全部高质量人脸数据；
4. 通过多个clip增加模型看到的数据视图和梯度多样性；
5. clip仍继承同一video/subject的弱标签，不能在论文中当作新增独立受试者。

AU在本项目中的边界必须分成“语义先验”和“模型数据”两层：

- **保留**AU/FACS作为局部RGB视图的语义分区依据，避免任意几何裁切破坏抑郁相关面部动作区域；
- **不使用**逐帧AU intensity/presence数值、AU序列或AU特征作为推理输入；
- **允许**通过PB-P0E资格门禁的聚合AU动态作为train-only辅助目标；validation/test隐藏目标root仍必须运行；
- 68点landmark只负责在每一帧动态定位完整语义区域，landmark坐标本身也不送入模型。

当前项目的数据流为：

```text
完整aligned face global view
+ 由68点landmark逐帧定位的眼眉/鼻颊/嘴部三个语义局部RGB crop
-> 同一个shared backbone -> 帧级validity-aware fusion -> temporal encoder / BDI head
-> 可选train-only区域/跨区域AU辅助head
-> validation/test使用同一冻结RGB视图合同，但不读取AU/head target
```

`FACE-*`命名用于人脸可用性和时间片段，`LM-*`命名用于landmark定位与局部RGB数据路径；它们不表示取消AU语义分区。历史 `AU-T* / AU-M*` 输出继续作为frame、coordinate、failure、exposure和区域跟踪审计证据。

## 现有只读审计的定位

`validity_aware_slicing_v1` 已证明当前数据存在两个基础问题：

- `stride_head(MAX_SEQ_LEN=2000,SAMPLE_STEP=10)` 最远只看到第1991帧；按批准raw-warp后的global容量口径，83,840个可用帧（17.06%）位于head范围之后；
- 直接把4,206个纯黑aligned占位作为切点，会把300视频碎成527个run，其中232个不足600帧。

该审计只使用纯黑、source presence和OpenFace success，尚未检查“虽有人脸但不适合作为有效输入”的大幅遮挡、大偏转、严重出界或极低可见面积。因此它只能作为容量下界和正式 `FACE-S1` 的输入，不能直接生成训练clip。

现有输出中的 `global_rgb_after_approved_raw_warp` 仍是hypothetical；raw-warp、统一landmark重跑和face-usability gate通过前不得用于训练。

## 人脸可用状态

正式审计必须给每帧分配且只分配一个状态：

| 状态 | 含义 | 是否可进入严格clip |
|---|---|---|
| `face_usable` | 人物存在、像素可见、landmark可信、姿态/遮挡/出界在允许范围 | 是 |
| `face_present_low_quality` | 人物存在，但大幅遮挡、偏转、模糊、过小、严重出界或landmark不可信 | 否 |
| `aligned_failure_pending_raw_warp` | 原视频人物存在，但aligned为纯黑占位 | raw-warp复核前否 |
| `person_absent` | 原视频人物离场或画面无人物 | 永久否，硬边界 |
| `unreadable_or_missing` | 文件缺失、不可解码或frame contract失败 | 永久否，硬边界 |

曝光状态是独立轴：`stable_log/tone_only_overexposed/keep_raw` 不直接决定face usability，也不能用逐帧亮度变化切掉低头、转头或表情片段。曝光变换只服从已冻结的segment manifest。

## FACE-S1 人脸可用性审计

### 允许使用的证据

`FACE-S1`不读取AU列。人脸可用性只由像素、presence和landmark几何决定；这项质量gate与后续AU语义裁切分区相互独立：

```text
decode_status
source_presence_status
global/nonblack/visible pixel ratio
OpenFace success/confidence（仅作检测质量）
68-point landmark in-frame ratio
face hull / bbox coverage and center offset
eye distance / face scale
landmark transform residual and temporal jump
landmark-derived yaw/pitch/roll or PnP pose
blur / gradient / clipping diagnostics
```

`pose`只作为“是否仍有可用脸部几何”的质量变量，不作为模型训练特征。阈值只由train split的无标签分布、人工overlay和明确几何约束冻结；不得使用BDI、prediction、validation/test utility调阈值。

### 大幅遮挡和大幅偏转

大姿态与遮挡不能只靠单个OpenFace `success` 判定。正式gate至少联合检查：

- landmark是否大量越出aligned画面；
- 左右眼、鼻、口关键点是否同时具备合理几何；
- face hull有效像素比例是否过低；
- 相邻帧similarity transform residual/速度是否异常；
- 原视频接触表中是否存在手、耳机、麦克风、头发或画面边缘遮挡主要面部；
- 大yaw/pitch下是否只剩过小侧脸或局部身体，而非可用面部。

真实轻中度转头、低头和自然遮挡可能携带抑郁行为，不能为了得到“漂亮正脸”全部删除。严格主协议只排除明确无有效脸的极端状态；同时保存 `quality_reason`，后续报告被排除帧与severity/task/subject分布，防止质量筛选变成隐式标签筛选。

### 先审计再冻结阈值

下一只读模块先输出分布和contact sheets，不预设最终数值。至少比较：

```text
confidence thresholds
landmark in-frame ratio thresholds
face hull coverage thresholds
absolute yaw/pitch bins
occlusion/blur bins
minimum contiguous run length
```

只有train-only人工复核通过后，才把阈值写入正式face-usability manifest。validation/test只使用冻结阈值并报告分布，不回调参数。

## FACE-S2 连续片段生成

### 边界规则

- `person_absent`、`unreadable_or_missing` 永久形成硬边界；
- `face_present_low_quality` 在严格主协议中形成质量边界；
- `aligned_failure_pending_raw_warp` 在修复验证前形成边界；
- 不删除无效帧后把前后片段伪装成时间连续；
- 不复制上一帧，不用aligned邻帧合成中间脸；
- 不跨任务、video或subject连接片段。

每个连续 `face_usable` run必须记录：

```text
split, subject_id, video_id, task_name
run_id, start_frame, end_frame, raw_frame_count, duration_seconds
usable/invalid reason counts around both boundaries
exposure_segment_id, source_presence_segment_id
landmark_version, threshold_version, input hashes
```

### 充分利用所有合格run

正式训练不再以“每视频每epoch随机一个窗口”为主协议。对每个合格run确定性生成全部候选clip：

1. run短于窗口时，完整保留并右侧padding；
2. run长于窗口时，按固定window/stride生成所有窗口；
3. tail达到冻结最短长度时保留为短clip，否则并入最后窗口或作为明确未用tail报告；
4. 所有clip必须落在一个 `face_usable` run内；
5. 同一manifest重复运行必须生成相同clip id和边界。

窗口长度与stride不能先凭validation性能选择。`FACE-S1`先只读统计以下候选：

```text
window raw frames: 300 / 600 / 1200 / 2000
overlap: 0% / 25% / 50%
minimum usable duration: 由train-only人脸run分布冻结
```

300/600帧可显著增加clip数，但BDI是video级标签，短clip的标签噪声更高；1200/2000帧样本更少但包含更完整的行为统计。正式选择必须同时报告clip数、每subject/video clip分布、任务分布、有效时长和重叠率。

## 样本扩增不等于独立样本增加

将300个视频切成更多clip会增加训练视图和优化步数，但不会增加独立subject数量。若直接把所有clip等权训练，长视频、Freeform视频或某些subject会主导loss，并产生伪重复（pseudo-replication）。

首版比较两种合法训练方式：

### Clip loss + source normalization

每个clip单独forward，但同一video的总权重固定：

```text
w_clip = 1 / N_clips(video)
L_video = sum_k(w_clip * L_clip_k)
```

这样所有合格clip均参与梯度，同时一个长视频不会因为切出更多窗口而获得更高总权重。可再报告subject级归一敏感性，但不默认把同subject的Freeform/Northwind合并为同一个训练bag。

### Video bag loss

同一个模型编码多个clip，先按有效时长聚合预测或特征，再计算一次video-level BDI loss：

```text
pred_video = sum_k(n_k * pred_clip_k) / sum_k(n_k)
L_video = regression(pred_video, BDI_video)
```

该方式更符合BDI标签语义，也是本项目优先的严谨方案。它不是多个模型；所有clip共享同一个backbone、时序编码器和head。若显存不足，可分clip累积特征/梯度，但必须验证与完整bag计算一致。

训练和论文必须同时报告：原始视频数、独立subject数、clip数、每video clip分布和effective loss weight，禁止只宣称“样本量从300增加到N”。

## Validation/Test聚合

validation/test必须使用冻结的face-valid manifest和确定性clip全集。每个video先聚合为一个prediction，再计算MAE/RMSE/CCC/Pearson：

```text
pred_video = sum_k(n_k * pred_clip_k) / sum_k(n_k)
```

clip prediction只用于：

- clip间离散度和不确定性；
- 按姿态/遮挡/曝光质量分组的失败分析；
- 检查某个video是否只有少量可用人脸；
- 检查局部与全局视图是否产生系统偏差。

clip不能作为独立subject进入主指标、bootstrap或显著性检验。

## AU语义保持的Landmark局部裁切 + 全脸增强

### 输入定义

首版用AU/FACS定义不可被任意切碎的粗粒度语义区域，再用aligned-space 68点landmark逐帧定位其完整外包络。模型推理输入只包含RGB像素和valid mask，不读取AU值或landmark坐标：

```text
global_face: 完整aligned face
eye_brow: 双眼、双眉和少量眼下区域
nose_cheek: 鼻梁、鼻翼、双侧颧部和鼻唇沟上段
mouth_lower_face: 完整嘴唇、双侧嘴角、鼻唇沟下段和少量下巴
```

每个语义组必须先冻结对应的68点landmark集合、polygon/外包络、相对margin、最小面积和越界规则，并通过aligned-space overlay确认。相邻区域只允许约5%--10%边界容错。局部view使用能覆盖完整语义polygon及margin的真实矩形RGB crop后resize；AU强度不决定裁切位置，也不生成AU heatmap。通过PB-P0E的AU12/14/15、AU4/6/7、AU10/17可按GLA方案作为train-only辅助目标，其中AU6和AU14允许跨相邻区域token预测。

### 语义完整性gate

每个frame-region只有同时满足下列条件才可进入local loss：

- frame join和aligned-space landmark mapping有效；
- 语义区域所需关键landmark集合完整，polygon顺序和几何关系合法；
- crop在裁边后仍覆盖冻结的完整语义polygon及最低margin，不把眉、眼周/颊、鼻-上唇或口-下颌区域截断；
- 有效像素面积、in-frame coverage和resize后最小有效尺度达到train-only冻结阈值；
- 大姿态或遮挡下只使用实际可见区域，不镜像、复制或合成隐藏侧。

任一条件失败时跳过该region的local loss并记录原因，global view仍保留。左右landmark可用于定位、可见性和coverage计算，但不产生左右AU输入、左右预测或左右loss。

global和local必须：

- 来自同一frame和同一时间clip；
- 复用同一随机flip/affine/color参数；
- 共享同一个backbone、temporal encoder和BDI head；
- local无效时跳过对应local loss，不伪造或镜像隐藏区域；
- validation/test使用冻结的global+有效local视图合同，但不读取AU/head target；global-only部署必须另立蒸馏/一致性实验。

### 控制组

为区分“任何裁切增强”与“AU语义保持的landmark定位裁切”，GLA矩阵保留以下语义对照：

```text
GLA-C-REF: global-only
GLA-RGB-GRID: global + three equal-area fixed/grid crops
GLA-RGB-FULL: global + three semantic landmark-guided RGB crops
```

`GLA-RGB-FULL`必须与`GLA-RGB-GRID`匹配局部view数量、面积/长宽比、padding、valid mask、loss scale、模型和训练预算。只有语义视图稳定优于grid，才能支持“语义区域布局优于任意局部裁切”；若二者近似，只能主张一般局部多裁切正则有效。完整运行顺序和AU/head消融以GLA文档为准。

## 实验矩阵

### 时间片段

| ID | 输入与监督 | 回答问题 |
|---|---|---|
| S0 | 当前整视频`stride_head` | 历史基线 |
| S1 | 严格face-valid clips，clip loss，video归一 | 全部高质量片段作为增强是否有效 |
| S2 | 同一clip全集，video bag loss | video级监督是否减少短clip标签噪声 |
| S3 | S2下比较300/600/1200/2000窗口 | 时间尺度敏感性；只在S2主协议下比较 |
| S4 | S2下比较0/25/50% overlap | 重叠增加是否提供信息还是只产生重复 |

### 历史局部增强矩阵（已由GLA矩阵取代）

| ID | 视图 | 推理 |
|---|---|---|
| L0 | global face | global |
| L1 | global + equal-area grid | 历史global-only部署假设 |
| L2 | global + four AU-semantic landmark-guided RGB crops | 历史global-only部署假设 |

L0/L1/L2只用于解释历史设计演进，不得继续运行。当前三语义区、统一train/val/test RGB合同和FULL优先混合消融见`GLOBAL_LOCAL_AU_EXPERIMENT_PLAN.md`；时间片段S0-S4是否恢复也必须先并入同一策略manifest，避免与视图/AU因素形成未注册笛卡尔积。

## 与曝光处理的关系

曝光调整和face segment选择共享同一原始问题档案，但决策互相独立：

- 曝光曲线按reviewed acquisition segment固定，不因姿态/遮挡逐帧变化；
- face usability按可见人脸几何决定，不因亮度被映射到安全带就自动通过；
- `keep_raw`视频仍可从其中选择有效脸片段；
- tone normalization不能把已剪切纹理称为恢复，也不能把无脸帧变为有效脸；
- 每个clip记录其包含的exposure segment和派生版本，支持raw/repaired配对实验。

## 通过条件与停止规则

### 数据gate

- 每个clip内 `face_usable` frame coverage满足冻结阈值；
- clip不跨hard/quality boundary；
- split/subject/video继承完全一致；
- 所有边界由标签无关证据生成；
- excluded reason、未用tail和短run均可追溯；
- validation/test不参与阈值冻结。

### 模型gate

- 相对固定S0，任一seed `delta_CCC < -0.05` 或 `delta_MAE > +0.50` 时停止；
- clip数增加但video-level utility、train-val gap和clip dispersion不改善时，不继续提高overlap；
- identity risk、severe bias或task consistency恶化时不判成功；
- `GLA-RGB-FULL`不优于`GLA-RGB-GRID`时停止“语义裁切特异性”主张；
- 不引入learned selector、区域专属backbone或多模型对齐来规避负结果；train-only AU/head小头只按GLA eligibility和消融合同存在。

## 下一实施顺序

1. [已完成] provenance-final frame audit和3帧raw-frame warp smoke；smoke不等于全量repair授权；
2. [已完成] `FACE-S1 phase-1 v2`全量只读分布审计，生成11类train-only contact sheets；
3. [待人工复签] 124个去重train帧的global/local-geometry/boundary标签已有AI辅助副本；local geometry通过只进入后续三语义区overlay审计，不等于local crop批准；
4. 生成全split确定性usable-run与clip候选统计，不训练；
5. 根据clip数量、时长、subject/task分布冻结主window/stride；
6. 实现S1 clip loss + video normalization；
7. 实现S2同模型video bag聚合；
8. 冻结三语义区landmark映射、完整性gate和train-only overlay；不得据此直接实现模型；
9. P0E和crop manifest汇合后，按`AGENTS.md`先披露并授权GLA默认关闭实现；
10. 训练严格采用`GLA-FULL`优先、依赖感知减法、有限加法复核和paired multi-seed；
11. 协议冻结前保持benchmark关闭；历史role-swap使其只能称locked post-selection benchmark。
