# PRIVILEGED_BEHAVIOR_ALIGNMENT_PLAN.md

> **2026-07-30方案演进**：本文档主体继续记录已经实现并冻结的AU12/14/15+head PB-P0来源、schema、pilot和门禁合同；不得回写为代码已经支持更多AU。未来训练设计已由`GLOBAL_LOCAL_AU_EXPERIMENT_PLAN.md`统一接管：完整脸+眼眉/鼻颊/嘴部三语义区共享backbone，最大P0E-eligible `GLA-FULL`优先筛选，再做依赖感知减法与幸存因素有限加法复核。扩展AU必须另行版本化升级source manifest、schema、coverage、normalization、risk audit和测试；当前PB-P0实现状态仍仅覆盖AU12/14/15。

> 文档职责：定义“抑郁相关 AU 与头部运动对 RGB 表征提供训练期特权辅助监督”的研究假设、数据合同、历史实验矩阵、实现边界和停止条件。当前已完成PB-P0基础设施、landmark-only v2阻断审计、PB-P0A3 full-rich v2和A2完整coverage判定；P0B-P0E、P1/P2和任何训练仍以`TODO.md`与再次授权为准。

## 0. 当前状态与决策

状态日期：2026-07-30。

- 用户已授权并完成PB-P0数据合同基础设施、no-gaze aligned-JPG提取入口、fresh debug2、pilot20、PB-P0A2 mask-aware policy、PB-P0A3权威full-rich v2和当前权威landmark-only v2阻断审计；既有授权不包含P0B-P0E、P1/P2、模型改动或训练。
- PB核心实现、extractor、测试与本方案已冻结在提交 `2685fa4`；本次文档同步作为其上的独立 docs-only follow-up 完成，未 amend 或改写核心提交。
- PB-P0A3权威v2已完成300/300视频、493,141行、单一182列schema和全部hash/provenance闭环；A2完整判定为`PASS_FULL_SOURCE_COVERAGE`、0 blocker/0 warning。legacy strict 0.995仍为259 PASS / 41 FAIL的诊断，不参与mask-aware资格判定。
- 当前未创建训练配置、未修改dataset/model/runner，也未启动训练。landmark-only P0仍是历史`BLOCKED`证据；rich v2已具备P0B输入来源，但P0B尚未运行或授权。
- 当前训练设计已由`GLOBAL_LOCAL_AU_EXPERIMENT_PLAN.md`收敛为三语义区共享backbone与FULL优先混合消融；本文件中的现有代码合同仍只覆盖`AU12/AU14/AU15`与旋转head dynamics。
- AU4/6/7/10/17属于待升级合同的计划候选，不得由现有PB-P0结果自动批准。
- gaze 已明确移除：不作为输入、辅助目标、关系对齐条件、超参数或消融因素。即使来源 CSV 含 gaze 列，也必须由白名单保证其不会进入目标张量。
- OpenFace 行为目标只在训练期使用；validation、test 和部署必须能够在没有 OpenFace root 的条件下仅使用 RGB 运行。

本方案中的“对齐”优先指 **分组辅助预测**，而不是把 RGB embedding 与完整 OpenFace 向量直接做强 MSE 同构对齐。

## 1. 研究问题与主张边界

### 1.1 研究问题

当前 aligned RGB 模型可能依赖身份纹理、静态外观、对齐几何、曝光条件和任务语境等容易学习但跨 subject 不稳定的捷径。拟检验的问题是：

> 使用具有抑郁行为研究依据的 AU 与头部运动动态作为训练期特权监督，能否引导 RGB 表征更多编码跨身份的行为动态，从而在不牺牲 BDI utility 的前提下减小 train-validation gap、身份泄漏和严重程度预测压缩？

### 1.2 可证伪假设

`H1`：相对于相同基座、split、seed、训练预算和 checkpoint 策略的 RGB reference，AU/head 辅助监督能够稳定改善或保持 BDI 指标，同时降低或不增加身份风险和 train-validation gap。

`H0`：AU/head 辅助监督不优于 reference，或收益可由打乱目标产生的普通噪声正则化解释，或身份风险、severity bias、task inconsistency 以不可接受幅度恶化。

### 1.3 不允许的过度主张

即使实验有效，也只能主张“抑郁相关行为特权监督与更健康的泛化指标相关，并通过预注册负对照支持其语义作用”。不能仅凭辅助损失下降声称：

- 已经学到因果抑郁生物标志物；
- 已经完全删除身份或全部非任务信息；
- AU/head 是 AVEC2014 过拟合的唯一原因；
- OpenFace 输出等同于无误差的真实 FACS/3D 行为标注。

## 2. 研究证据

### 2.1 抑郁相关行为证据

| 特征组 | 主要证据 | 本项目解释 | 证据限制 |
|---|---|---|---|
| AU12 | 较高抑郁严重度下笑容相关活动减少 | 使用 AU12 动态统计作为核心目标 | 自动编码显著性与研究条件有关 |
| AU14 | Girard 系列中活动和 AU12-AU14 共现增加，纵向被试内结果较稳定 | 纳入核心组，同时预注册去 AU14 敏感性实验 | 独立跨队列复现仍有限 |
| AU15 | 较高严重度下AU15发生/强度相关活动减少 | 与 AU12/14 共同构成小型白名单 | 证据略弱于 AU12/14，不能单独决定方案成败 |
| Head motion | 抑郁参与者头动幅度、速度和位置变化减少 | 使用去中心旋转动态，不使用绝对位置 | 摄像机、访谈任务和检测质量仍可能混杂 |

主要文献：

- Girard et al., 2013, *Social Risk and Depression: Evidence from Manual and Automatic Facial Expression Analysis*. [DOI 10.1109/FG.2013.6553748](https://doi.org/10.1109/FG.2013.6553748)
- Girard et al., 2014, *Nonverbal social withdrawal in depression: Evidence from manual and automatic analyses*. [DOI 10.1016/j.imavis.2013.12.007](https://doi.org/10.1016/j.imavis.2013.12.007)
- Alghowinem et al., 2013, *Head Pose and Movement Analysis as an Indicator of Depression*. [DOI 10.1109/ACII.2013.53](https://doi.org/10.1109/ACII.2013.53)
- Alghowinem et al., 2018, *Multimodal Depression Detection: Fusion Analysis of Paralinguistic, Head Pose and Eye Gaze Behaviors*. [DOI 10.1109/TAFFC.2016.2634527](https://doi.org/10.1109/TAFFC.2016.2634527)

### 2.2 训练期特权信息与辅助监督证据

现有研究支持“训练时使用额外行为或模态信息、测试时仅使用 RGB”的方法范式，但尚未直接证明本方案可解决 AVEC2014 RGB 回归过拟合：

- Vapnik and Vashist, LUPI. [DOI 10.1016/j.neunet.2009.06.042](https://doi.org/10.1016/j.neunet.2009.06.042)
- Hoffman et al., Modality Hallucination. [DOI 10.1109/CVPR.2016.96](https://doi.org/10.1109/CVPR.2016.96)
- Suresh and Ong, AU positive-matching for reducing spurious correlation in facial expression recognition. [DOI 10.1109/ACII55700.2022.9953865](https://doi.org/10.1109/ACII55700.2022.9953865)
- PAU-Net, AU as privileged/auxiliary facial information. [DOI 10.1109/TCDS.2022.3203822](https://doi.org/10.1109/TCDS.2022.3203822)
- *From the Lab to the Wild: Affect Modeling Via Privileged Information*. [DOI 10.1109/TAFFC.2023.3265072](https://doi.org/10.1109/TAFFC.2023.3265072)
- FAU-guided multimodal depression detection. [DOI 10.1016/j.neucom.2024.129106](https://doi.org/10.1016/j.neucom.2024.129106)
- Dis2DR depression representation distillation. [DOI 10.1145/3664647.3681227](https://doi.org/10.1145/3664647.3681227)

证据链应表述为：

```text
抑郁相关 AU/head 行为证据
+ 训练期特权信息/辅助监督方法先例
-> 有依据但尚未被本数据集直接验证的可证伪假设
```

## 3. 特征白名单与排除项

### 3.1 AU 核心组

仅允许连续 intensity 列：

```text
AU12_r
AU14_r
AU15_r
```

第一版描述符候选：

- 软激活率，或由 train-only 冻结阈值定义的激活率；
- 全有效帧和激活帧平均强度；
- bout 频率、持续时间和强度面积；
- 有效相邻帧上的强度变化速度；
- `AU12 × AU14` 软共现率；
- 每个描述符对应的有效覆盖率 mask。

`PB-AU-NO14` 必须只保留 AU12/AU15，用于检查结论是否过度依赖 AU14。

### 3.2 Head 核心组

仅允许：

```text
pose_Rx
pose_Ry
pose_Rz
```

`pose_R*`只作为派生源列，首版直接动态原语为`d_pose_Rx_dt/d_pose_Ry_dt/d_pose_Rz_dt`；raw `pose_R*`数值和绝对均值不作为辅助目标。可由这些原语聚合：

- 每轴去中心 RMS amplitude；
- angular velocity 的均值、RMS 或稳健分位数；
- rotation path length；
- stillness ratio；
- motion-bout 频率和持续时间；
- 有效覆盖率及最大合法时间间隔。

角速度必须在完整原帧率上根据外部 source-video frame-time contract 计算。既有全量 aligned-JPG `-fdir` 输出的 `timestamp` 为常数 0，严禁用于速度；当前冻结 `source_video_contract.csv` 的 30 FPS 契约，使用 `t=(frame_id-1)/30`。角度差采用 `atan2(sin(Δθ), cos(Δθ))` wrap 到最短有符号弧，只在前后帧 ID 连续、两端都有效且 `dt` 合法时差分，再应用与 RGB 相同的 temporal indices。禁止跨 invalid gap、先采样后 row-diff 或直接使用未 wrap 的角度差。

### 3.3 明确排除

- 全部 gaze 列和派生 gaze 特征；
- `pose_Tx/Ty/Tz`；
- 绝对头姿均值作为主目标；
- AU presence `_c`；
- AU1/4/6/7/10/17/24/25/26 等未进入预注册白名单的 AU；
- 全量 OpenFace embedding；
- landmark、face scale、crop geometry、PDM、HOG；
- `success/confidence` 作为语义目标或模型特征；
- 从 validation/test BDI 结果反向选择 AU、阈值或描述符。

`success/confidence` 只允许生成 mask、weight、coverage 和数据质量报告。

## 4. P0：数据合同与目标审计，不训练

P0 只验证目标是否同源、同步、可复现，并不创建模型分支或启动训练。

### 4.1 精确连接合同

连接键必须为：

```text
exact video_id
+ exact task_name
+ explicit frame_id
```

需要动态时间时，另由冻结source-video contract生成`t=(frame_id-1)/30`；aligned OpenFace `timestamp=0`既不是join key，也不是速度时钟。

规则：

- 禁止 subject-only fallback；
- Freeform/Northwind 不得因为同一 subject 而共享行为序列；
- 按 RGB 实际选中的 frame ID join，禁止按 CSV 行号 zip；
- RGB 与行为目标复用 `select_temporal_indices`，覆盖 `stride_head/uniform/first/middle/random`；
- 结构上的选中帧 join 必须为 100%；检测失败以显式 invalid mask 表示，不能静默补零；
- 缺列、重复 frame、重复 video/task、frame 歧义或外部 source clock 非法必须 fail closed；aligned `-fdir timestamp` 应验证为预期常数 0，但不得作为 join 或速度时钟。

### 4.2 OpenFace 来源与版本门禁

历史完整AU/pose CSV来自raw-video OpenFace；当前权威aligned-space行为来源已经是PB-P0A3 full-rich v2，landmark-only v2只保留为早期阻断审计。历史raw-video rich CSV不得给当前aligned RGB模型提供训练监督。自2026-07-28起已完成以下P0 source/provenance基础设施与提取门禁：

1. 冻结 OpenFace 2.2.0 二进制、主模型、AU predictor 依赖、完整命令和输入 manifest；
2. 冻结 `source_video_contract.csv` SHA-256 `12f50c2311b9d89dde81e27fa83891226ca5f447ed7d3d56fe38cf673a1c5a31`：300/300 视频 `PASS`、统一 30 FPS、aligned/raw 共 493,141 帧完全匹配。行为审计只读取 identity/frame-count/FPS 字段，并仅用它生成 `t=(frame_id-1)/30`；严禁读取其中的 raw OpenFace 路径或任何历史 raw AU/pose 数值；
3. 新增原始 aligned JPG 专用 no-gaze profile，OpenFace 参数固定为 `-2Dfp -pose -aus`，不请求 gaze/HOG/3D landmark/PDM/aligned 图或 tracked video；训练目标仍只白名单读取六列；
4. `debug2c`对2/2视频、1,920张JPG/1,920行CSV完成exact frame/schema审计并`PASS`；其后current-script fresh debug2和pilot20均已完成；
5. PB-P0A3权威full-rich v2已在clean Windows `dev@2538248`完成300/300视频、493,141行、单一182列schema和全部hash/provenance闭环；失败且不完整的v1禁止resume、复用或进入P0B；
6. 每个未来实验必须冻结一个 aligned-JPG 目标来源，禁止混合不同输入空间、版本或 feature profile。audit-only run manifest 不构成训练授权，仍需单独的 training-target contract；
7. OpenFace 3.0 只能作为独立 extractor-version 因素重新全量提取，不能与 2.2.0 输出混用。

full-rich provenance仍需由P0复核。现有`strict_rich`实现把legacy逐视频0.995写入来源PASS条件，因此在mask-aware v2上存在兼容性缺口：不得直接运行后把预期`BLOCKED`误写为来源失败，也不得通过删除41个legacy FAIL或弱化hash/schema/join/provenance解决。P0B前必须先披露并授权最小兼容修复；以下非coverage fail-closed门禁继续全部保留：

- 来源必须为 `run_scope=full_dataset`、`max_videos=0`，并精确声明 300 个输入/选中视频和 493,141 帧；debug 或 partial-rich 目录不能授权正式 P0；
- extractor 脚本 SHA 必须等于当前已审阅脚本，OpenFace 2.2.0 binary、model、README、package 和 no-gaze feature profile 必须全部匹配冻结值；
- source git commit 必须为 7–40 位十六进制，branch 必须合法非空，`git status --short` 必须可用且为空；status unavailable 不能解释为 clean。正式 `MaxVideos=0` 提取会在调用 OpenFace 前拒绝 dirty 或无法取得 status 的 checkout；
- legacy `min_success_ratio=0.995`、逐视频status和summary status必须保留且内部一致，但只作为诊断证据；P0B兼容修复后不得再要求每视频或summary为legacy `PASS`，实际有效性由冻结A2 mask-aware policy裁决；
- CSV header 必须为无重复、无缺失、无额外列的完整 182 列 schema；每一数据行必须同样具有 182 列，行数与图像数相同，`frame=1..N`，`timestamp` 有限且为 0，`success` 有限且属于 `{0,1}`，`confidence` 有限且位于 `[0,1]`，AU12/14/15 有限且位于 `[0,5]`，`pose_Rx/Ry/Rz` 有限且位于 `[-π,π]`；
- 相邻`extraction_summary.json`必须精确描述full-dataset计数并由最终`run_manifest.json`绑定SHA-256；允许其因legacy 0.995诊断写为`FAIL`，但A2独立coverage decision必须为`PASS_FULL_SOURCE_COVERAGE`；
- `_audit/csv_content_manifest.csv` 的 SHA-256 必须同时被 `extraction_summary.json` 和 `run_manifest.json` 绑定；P0 会重新核对每视频的 status、rows、schema SHA、文件大小和 CSV SHA-256；
- P0 还会读取既有 image-integrity candidate manifest，要求当前每张 aligned JPG 按精确相对路径、字节大小和 SHA-256 逐文件匹配，且 missing/extra/read-error 均为 0。跨 Windows、WSL 和 `/mnt/d` 的路径等价只用于定位文件，不豁免字节核验。

该rich provenance门禁只在300/300视频均具备完整rich schema时执行。landmark-only或partial-rich来源将其记录为`NOT_APPLICABLE`并继续schema-blocked，不把未执行解释为通过，也不对已知不合格来源执行全量JPG哈希。

若配对审计显示aligned-JPG旋转动态与可解释的原视频运动方向/相对变化不稳定，必须停止`PB-HEAD`和`PB-AU-HEAD`，不得回退到历史raw rich CSV补监督，也不得把alignment-warp proxy称为真实头部运动。

### 4.3 有效帧和动态计算

建议在查看 BDI validation/test 结果前冻结：

```text
valid = success == 1
        and confidence >= 0.8
        and required fields are finite
```

- `0.8` 是首版预注册候选；0.7/0.9 只能做无 BDI 标签的覆盖率敏感性审计，不根据下游结果择优。
- delta/velocity 只在 frame ID 连续、两端有效且外部 30 FPS 时钟给出的时间间隔合法时计算；旋转差使用 wrap 后的最短有符号角差。
- 不跨检测失败、缺帧或人物缺席边界差分。
- 每个 AU/head 描述符都有独立 valid mask；不能仅使用序列长度 mask。

### 4.4 统计与泄漏控制

- normalization 只使用 physical train split；
- 持久化列顺序、mean/std、阈值、源 manifest hash 和统计代码版本；
- val/test swapped 只改变逻辑评估角色，不改变 physical-train normalization；
- 分别报告 AU/head coverage，不使用总体覆盖率掩盖某一组缺失；
- 在训练前检查目标对 subject identity、task、曝光和 OpenFace quality 的可预测性；
- 当前亮度/颜色/曝光处理在本实验内保持冻结，不与行为监督同时改变。
- scalar AU/head pseudo-label 不依赖 local-crop landmark 空间映射，因此不能把现有 T0b coordinate gate 直接当作本方案授权；必须另立 behavior target coverage gate。现有 face-usability/coordinate 失败也不能被本方案静默绕过。

### 4.5 P0 基础设施产物与当前结果

```text
logs/privileged_behavior_alignment/p0_contract_landmark_only_blocked_v2/
  tables/behavior_frame_contract.csv
  tables/behavior_group_coverage.csv
  tables/behavior_target_stats.csv
  tables/behavior_contract_issues.csv
  tables/behavior_source_fidelity.csv
  tables/behavior_identity_task_risk.csv
  selected_target_manifest.json
  reports/behavior_contract_report.md
  run_manifest.json
```

P0 通过条件：

- 选中帧的结构 join 为 100%；
- video/task/frame 映射无歧义；
- OpenFace 版本、命令、模型和输入来源可追溯；
- AU/head coverage 分组报告完成；
- 没有任何 gaze 或非白名单列进入目标 schema；
- 数据来源差异足以解释时必须停止，不能带着未声明的 raw/aligned 混杂进入训练。

2026-07-28 当前权威 landmark-only v2 审计结论（v1 仅保留为历史证据）：

- physical split 中 300/300 视频、493,141 张 aligned JPG 完成 exact `video_id + task_name + frame_id` 结构连接，`exact_join_video_count=300`；
- `core_audited_video_count=0`、`rich_schema_video_count=0`，因为 300 个 CSV 均缺少 `AU12_r/AU14_r/AU15_r/pose_Rx/pose_Ry/pose_Rz`；
- 最终状态为 `BLOCKED`，共有 301 个 blocking issues：`missing_behavior_columns` 300 项和 `train_only_statistics_unavailable` 1 项；
- gaze target/value/normalizer/loss 访问和历史 raw-video rich OpenFace 文件访问均为 0；不得用历史 raw rich CSV 补列；
- `strict_rich_provenance_gate` 在 landmark-only 来源上为 `NOT_APPLICABLE` 而不是 `PASS`。只有 300/300 视频均具备完整 rich schema 时才执行昂贵的 CSV/JPG 逐字节门禁；
- 九项产物完整，`run_manifest.json` 中的三个 implementation SHA 与本次实现一致，并已记录其余八个输出文件的 SHA-256；
- `behavior_identity_task_risk.csv`仍为`NOT_RUN_P0_PLACEHOLDER`，aligned pose物理保真也未审计通过。该历史landmark-only状态不因新来源出现而改写。

2026-07-30新增权威来源/coverage结论：full-rich v2完成300/300视频、493,141行、单一182列schema和全部hash/provenance闭环；A2为`PASS_FULL_SOURCE_COVERAGE`、0 blocker/0 warning。quality/AU-valid为486,441帧，head-valid为485,863对；physical train总体/Freeform/Northwind分别为0.989273/0.983944/0.996984，task gap约0.01304。该结论不等于P0B、物理保真、风险或训练资格通过。

### 4.6 后续执行路线：A3已完成，先修复P0B兼容性

pilot20直接证实strict逐视频0.995不可行：20视频中16 PASS / 4 FAIL，失败为`205_1_Freeform=0.990909`、`206_1_Freeform=0.920000`、`207_2_Freeform=0.698148`、`208_1_Freeform=0.979227`。A3随后已按冻结mask-aware合同完成，完整train coverage通过；legacy strict结果继续保留为诊断。

因此当前权威执行链为：

```text
PB-R0 core/source sync（DONE）
-> PB-P0A0 current-script fresh debug2（DONE）
-> PB-P0A1 pilot20（DONE: 16 PASS / 4 FAIL）
-> PB-P0A2 mask-aware threshold/coverage policy decision（DONE: pilot feasibility PASS）
-> PB-P0A3 full-rich v2 + full coverage（DONE）
-> PB-P0B-COMPAT报告/来源兼容修复（CURRENT CANDIDATE；先披露并等待授权）
-> PB-P0B core contract（另行运行授权）
-> PB-P0C physical fidelity
-> PB-P0D final descriptor + nuisance risk
-> PB-P0E group eligibility
-> PB-P1-CODE（再次授权）
-> PB-P1-RUN（再次授权）
-> PB-P2（条件性再次授权）
```

| Gate | 输入与动作 | 出队产物 / PASS条件 | 停止或分支 | 授权边界 |
|---|---|---|---|---|
| `PB-R0` | 核心`2685fa4`、docs-only follow-up与pilot来源同步 | 当前脚本/source/provenance满足fresh debug和pilot合同 | 后续full仍必须使用clean、具名branch和匹配SHA | `DONE`；不等于full或训练授权 |
| `PB-P0A0` | 当前脚本、clean commit、全新目录、`MaxVideos=2` | 2/2 PASS、1,920行、182列、content manifest与summary/run-manifest hash闭环 | 任一provenance/schema/value-domain失败即停止 | `DONE`；不含full或训练 |
| `PB-P0A1` | 相同来源`MaxVideos=20`，25,980帧 | 25,980行目标与完整hash；实测OpenFace成功25,446帧，16 PASS / 4 FAIL | strict合同下full必然FAIL；禁止直接继续 | `DONE/FAIL_STRICT`；必须进入A2 |
| `PB-P0A2` | 在不读取BDI/prediction/checkpoint的前提下冻结machine-readable、versioned source/coverage policy及其校验器 | `DONE`：policy SHA `9a3fb7...1926`；pilot v5为可行性PASS/3 warning，full v2为`PASS_FULL_SOURCE_COVERAGE`、0 blocker/0 warning；val/test仅报告 | legacy strict仍只作诊断；full decision的`next_action`报告bug须在P0B前修复 | `DONE`；输出明确`P0B/training_authorized=false` |
| `PB-P0A3` | 300视频full-rich，新目录且不resume | `DONE`：权威v2为300/300视频、493,141行、182列、0禁用产物、全部provenance/hash闭合；A2 full coverage PASS | 失败且不完整的v1禁止复用或resume | `DONE`；不含P0B、模型或训练 |
| `PB-P0B-COMPAT` | 修复A2 full PASS时错误`next_action`，并使P0来源门禁接受legacy-summary FAIL但A2 mask-aware PASS | 回归测试覆盖PASS/FAIL分支；只替换legacy coverage状态的门禁语义 | 不得弱化hash/schema/join/provenance/value-domain；不得手改既有结果 | 当前代码候选；必须先披露package并等待明确授权 |
| `PB-P0B` | 对不可变v2 root运行兼容后的P0 CLI，再用A2 policy校验器读取coverage产物 | CLI机器状态`status=PASS`、300/300 core/schema/join、physical-train统计、0 blocker、九项产物，以及独立coverage decision PASS | blocker、常量列、统计不可用、来源失配或coverage policy失败即停止；当前CLI默认`minimum_std=0`，只拒绝`std==0`，非零实用方差阈值必须另行版本化 | P0B运行需在兼容修复验证后单独授权 |
| `PB-P0C` | 使用同版本/模型/profile新生成raw-space参考，与aligned rich按exact frame做AU/pose物理保真 | 每轴/每AU的方向、相关、幅度、lag和cross-talk报告 | Rx或Ry失败则删除HEAD；仅Rz失败只允许预登记Rx/Ry；AU失败则删除对应AU组 | 新raw参考与审计工具需单独授权 |
| `PB-P0D` | 冻结实际P1描述符和matched RGB reference；重新拟合描述符级physical-train normalization；执行identity/task/exposure/quality probes | training-target contract、descriptor manifest、coverage/risk/null报告 | 风险超限、coverage依赖nuisance或需按BDI结果选描述符即停止 | 只读风险审计；不含训练 |
| `PB-P0E` | 汇总P0B core/coverage、P0C fidelity、P0D risk和三语义区几何资格并生成机器可读资格 | `REGION_ELIGIBLE`、按组AU/head资格、`CROSS_REGION_ELIGIBLE`、`AU_ELIGIBLE/HEAD_ELIGIBLE/JOINT_ELIGIBLE`、完整`eligible_component_profile`及SHA、`P1_CODE_AUTHORIZED=false` | `GLA-FULL`严格等于最大依赖闭合集；某组失败时重新冻结精简矩阵，P0B PASS不能替代最终资格 | P1仍需明确授权 |
| `PB-P1-CODE` | 默认关闭的exact target loader、分组小头和train-only masked SmoothL1 | 默认兼容、sampling/mask/train-only/AMP/gradient测试与debug smoke | val/test读取OpenFace、梯度异常或默认行为变化即停止 | 模型代码需再次授权 |
| `PB-P1-RUN` | 不再运行本文件的历史六条件screen；PB-P1只向GLA提供默认关闭的训练基础设施 | smoke、校准和正式训练全部转交`GLOBAL_LOCAL_AU_EXPERIMENT_PLAN.md`中的具名`GLA-*`包 | 禁止从历史PB矩阵派生新run；历史role-swap只保留为既有证据 | 每个GLA运行包需分别授权 |
| `PB-P2` | 在GLA最终候选多seed通过后，先单独验证2--4秒短窗口，再按结果决定关系损失 | 独立P2规格与matched消融 | 联合profile不优于matched单组/减法对照、HEAD被阻断或batch关系不稳定即不进入 | 条件性再次授权 |

近期只按以下工作包出队，不跨级并行启动训练因素：

1. **WP0 / PB-R0-SYNC**：核心与PB-only文档已分别冻结且未amend；push后同步Windows clean checkout，并在同步后重新计算最终commit/branch。完成定义是Windows记录最终commit/branch、clean status、`2685fa4`祖先关系、extractor SHA和source-contract SHA；不产生任何新OpenFace数据。
2. **WP1 / PB-P0A0-DEBUG2**：从上述clean checkout在全新目录运行`MaxVideos=2`。只有2/2、1,920行、182列、完整content/summary/run-manifest哈希闭环全部通过才出队；失败直接修复来源或脚本，不进入pilot。
3. **WP2 / PB-P0A1-PILOT20**：单独授权后运行`MaxVideos=20`，冻结lexicographic前20视频、`selected_for_run`行集和`input_frame_contract.csv` SHA。无论20/20 PASS还是出现低于0.995的视频，都进入WP3；FAIL只表示strict full被阻断，不允许跳过正式决策。
4. **WP3 / PB-P0A2-POLICY（DONE）**：已生成`behavior_source_coverage_policy_v1.json`及SHA，实现最小只读校验器和测试；v5在不读取BDI、prediction或checkpoint的条件下通过pilot可行性门禁。
5. **WP4a / PB-P0A3-FULL（DONE）**：权威v2完成全量300视频并通过完整physical-train coverage；失败且不完整的v1只保留失败证据，禁止进入P0B、resume或事后改manifest。
6. **WP4b0 / PB-P0B-COMPAT（当前候选，未授权）**：先按`AGENTS.md`披露具名代码包并结束该轮；获明确授权后修复A2 full PASS的`next_action`分支和P0对mask-aware来源的legacy-status误拒，并增加回归测试。不得改动数据、split、标签或非coverage硬门禁。
7. **WP4b / PB-P0B-CORE+COVERAGE（未授权运行）**：兼容包验证后另行授权运行P0 CLI和WP3校验器。CLI权威状态名是`PASS/BLOCKED`而非`CORE_PASS`；实现只自动拒绝`std==0`常量列，coverage和非零实用方差必须由版本化policy另行裁决。
8. **WP4c / PB-P0C-FIDELITY**：先冻结规格，再新增matched raw/aligned参考生成、物理保真审计工具和测试，最后运行；这不是现成命令。
9. **WP4d / PB-P0D-TARGET-RISK**：先冻结`behavior_descriptor_v1`和matched RGB reference，再实现描述符、normalization与identity/task/exposure/quality probe工具并运行；这不是现成命令。
10. **WP4e / PB-P0E-ELIGIBILITY**：新增机器可读eligibility聚合器，汇总core、coverage、fidelity、risk和三语义区几何资格；输出region、分组AU/head、cross-region资格、完整`eligible_component_profile`及SHA，并固定`P1_CODE_AUTHORIZED=false`。
11. **WP5 / PB-P1至PB-P2**：仅在P0E生成机器可读资格且再次授权后，未来实现/训练服从`GLOBAL_LOCAL_AU_EXPERIMENT_PLAN.md`的FULL优先混合消融；本文件的旧PB矩阵不再是执行入口。

### 4.7 阈值策略与最终训练目标合同

`PB-P0A2`已完成。机器可读policy为`configs/behavior_alignment/behavior_source_coverage_policy_v1.json`，SHA-256为`9a3fb739078ab36e1fa4f8f67a8c8b167a76329941a591ad0ef9dd0c731a1926`；pilot输出为`logs/privileged_behavior_alignment/p0a2_mask_aware_policy_pilot20_2538248_v5/`，完整权威输出为`logs/privileged_behavior_alignment/p0a2_mask_aware_policy_full_a29ba49c_v2/`。它冻结`quality_valid = success==1 && confidence>=0.8`、各ratio分母、physical-train总体/task/逐视频阈值、val/test只报告、输入证据SHA和失败状态。当前P0 CLI只报告coverage，因此CLI的`status=PASS`仍不能替代独立coverage decision。

本次pilot有4个视频低于0.995，因此A2在以下两个互斥分支中选择了mask-aware分支：

- **Strict分支**：保留逐视频0.995，状态写为`STOPPED_DATA_INFEASIBLE`，不运行full-rich，不删视频、不改split，也不以曝光派生图静默替换当前RGB输入。
- **Mask-aware分支（已选择）**：新建versioned合同，将提取完整性硬门禁固定为100%视频/行/schema/hash/provenance，把`success/confidence`只用于valid mask和coverage；阈值仅用physical train冻结，val/test只报告。不得简单把`MinSuccessRatio`改成刚好通过的数值。

首版policy已冻结coverage门槛：physical-train总体`quality_valid_ratio>=0.98`，Freeform/Northwind各自`>=0.95`，每个train视频至少60个有效帧、head至少59个连续有效pair，不允许某训练视频整组完全无效，task间coverage gap不超过0.05。pilot只含8个train视频，其3项future-full warning现已由完整physical train正式裁决：总体0.989273、Freeform 0.983944、Northwind 0.996984、task gap约0.01304，最低train视频180个AU-valid帧和179个head pair，全部通过。partial目录仍不得进入P0B，也不得resume或事后改manifest。

完整A2 decision存在报告分支bug：`status=PASS_FULL_SOURCE_COVERAGE`时`next_action`错误写为`STOP_AND_REVIEW_POLICY_OR_SOURCE_ISSUES`。计数与门禁判定不受影响，但必须在P0B前通过具名代码包修复并补回归测试，不能手工改写既有JSON。

P0B的帧级六列统计不等于P1最终training target。P0D必须冻结小型`behavior_descriptor_v1`，首版限定为：

- AU12/14/15各自的有效帧平均强度与连续有效帧平均绝对变化；
- 通过P0C的head轴各自wrapped angular velocity RMS；
- 每组独立mask和coverage；
- 不加入bout、共现、多分位数或完整OpenFace向量；
- 对聚合描述符重新使用physical train拟合normalization，禁止直接复用帧级mean/std。

P0C候选硬门槛须在正式matched raw/aligned结果前冻结：subject-level median Spearman不低于0.70、subject-cluster bootstrap下界高于0.50、明显运动方向一致率不低于0.75、amplitude ratio中位数位于`[0.5,2.0]`、相关峰位于lag 0，且其它轴不能比对应轴相关更强。

P0D对exact `behavior_descriptor_v1`的保守风险门槛候选为：identity/session pair AUROC不高于0.70、cross-task top-1不高于10%、task AUROC不高于0.70、任一exposure/quality grouped-CV `R²`不高于0.25。阈值没有普适定律，必须在查看rich target的BDI效果前冻结并披露；超过门槛的组直接删除，不通过增加identity adversarial或继续搜索描述符补救。

### 4.8 成本预算与路线隔离

- fresh debug2和pilot20已完成；权威full-rich v2实际耗时10,064.6秒、占用587 MB。当前脚本不支持resume，失败v1不得继续使用。
- rich P0 audit建议预留45--90分钟；它还会读取CSV并逐文件核验493,141张JPG。
- 当前D盘约217GB可用，容量不是即时阻塞，但每次运行前仍须复核，并保证输出根与T0b、FACE、exposure完全分离。

| 路线 | 冻结来源 | 用途 | 不得授权 |
|---|---|---|---|
| T0b/local crop | 原始aligned JPG + landmark-only `-2Dfp -mloc` | 坐标与区域裁切门禁 | PB AU/head target |
| PB | 原始aligned JPG + `-2Dfp -pose -aus` no-gaze | 训练期特权目标候选 | exposure收益或local-crop坐标 |
| Exposure | 正式derived mirror + full diagnostic profile | raw-vs-exposure诊断 | 当前PB训练目标 |
| FACE | 人工复签后的usability/clip/local-crop manifests | RGB输入路线 | PB数据合同自动PASS |

FACE人工复签等只读工作可与PB数据审计并行，但训练因素必须串行隔离。首轮PB固定原始aligned RGB和同一E2/C-REF severity协议，不同时改变曝光、FACE clip/local-crop、identity adversarial、低信息熵、continuous weighting或新RGB增强。若未来采样帧、图像根或输入路线变化，P0D描述符、coverage、normalization和risk必须按实际帧集合全部重算。

## 5. P1：视频/clip 级分组辅助预测

### 5.1 推荐结构

```text
shared RGB representation
  ├─ existing BDI regression head
  ├─ AU auxiliary head
  └─ head-motion auxiliary head
```

首版使用与当前 RGB sample 对应的视频级或 clip 级聚合描述符，不做逐帧全向量强对齐。若后续 face-valid clip 路线成为冻结基座，行为描述符必须根据同一 clip 的实际帧集合重新计算。

### 5.2 损失

\[
L = L_{\mathrm{BDI}} + r(e)
\left(
\lambda_{\mathrm{AU}}L_{\mathrm{AU}} +
\lambda_{\mathrm{head}}L_{\mathrm{head}}
\right)
\]

其中：

- 每组 target 使用 train-only 标准化；
- 每组先在有效维度内独立求平均，避免维度较多的组支配损失；
- 连续描述符使用 masked SmoothL1/Huber；
- target 全部 stop-gradient；
- 前约 10% epoch 不加入辅助损失，随后约 20% epoch 线性 ramp；
- 初始总辅助权重候选限定为 `{0.01, 0.03, 0.1}`，只通过100-step train-only有限性、梯度比例和冲突审计冻结一个值，不用validation性能做系数扫描；
- auxiliary loss 只在 train stage 加入总损失；
- 标准validation/test运行不加载OpenFace target；如需行为预测诊断，只能在checkpoint冻结后通过独立只读离线审计连接目标，且不得回调checkpoint、阈值或超参数；
- checkpoint 仍只依据冻结的 BDI validation monitor/mode；
- 多视图训练首版只监督未增强 `orig`，避免翻转造成 head rotation 符号歧义；
- validation/test/inference 隐藏 OpenFace root 后必须仍可运行。

### 5.3 不采用的首版方案

- 不把 AU/head 拼成一个大向量后强制整个 RGB embedding 与其做 MSE；
- 不蒸馏当前可能已经严重过拟合的 behavior-only BDI teacher；
- 不同时加入低信息熵、identity adversarial、曝光变换或新的 RGB augmentation；
- 不以 auxiliary validation loss 选择模型；
- 不因单 seed 改善继续扩大 AU 或损失权重搜索。

直接最小化表示熵不保证删除身份信息，并可能造成表示塌缩；如未来研究，必须作为独立实验因素，而不是与本方案绑定。

## 6. 固定实验矩阵

> **历史矩阵，禁止作为当前执行入口。** 2026-07-30起，未来训练统一采用`GLOBAL_LOCAL_AU_EXPERIMENT_PLAN.md`中的`GLA-FULL`优先混合消融。下列`PB-*`条件只保留早期AU12/14/15+head假设和负对照设计证据，不授权运行，也不得与新的GLA run ID混用。

首轮以PB-P0D冻结的matched RGB reference和sampling manifest为基座；当前推荐是独立的原始aligned `C-REF/E2`协议，不等待或混入尚未冻结的FACE local-crop。必须保持backbone、split、采样、severity loss、optimizer、precision、训练预算和EarlyStopping不变，并记录不可变reference ID与manifest SHA，避免方案等待期间基座变化后仍沿用过时名称。

| Run ID | 条件 | 目的 |
|---|---|---|
| `PB-REF` | 行为监督关闭 | 匹配 reference |
| `PB-AU` | AU12/14/15 | 检验 AU 核心组 |
| `PB-HEAD` | 旋转头部运动动态 | 检验 head 核心组 |
| `PB-AU-HEAD` | AU + head | 首选联合候选 |
| `PB-AU-NO14` | AU12/15 | AU14 敏感性 |
| `PB-SHUFFLED` | 仅在physical train内按video固定seed打乱AU+head目标，val/test不参与置换 | 排除普通噪声正则化解释 |

运行顺序：

1. seed 42 debug smoke 和有限步梯度校准；
2. seed 42 validation screening；
3. 通过预注册 utility/risk gate 后运行 seeds 43/44；
4. 特征、权重和 checkpoint policy 冻结后才读取 test；
5. val/test swapped 仅作最后的 split sensitivity，不用于重新选择目标或权重；
6. P1 通过后，再在 continuous severity weighting `alpha=0.25` 上复核互补性，不在首轮同时改变两个因素。

## 7. P2：关系式或短窗口对齐

本节不是当前运行入口。仅在GLA最终候选至少3个paired seed通过，且联合辅助profile优于其matched单组/减法对照后，才可另立规格考虑：

- 2--4 秒短窗口 AU/head 动态预测；
- batch 内 pairwise cosine/distance preservation；
- 分组关系损失，不拼接完整 OpenFace embedding；
- 关系配对排除同一 subject，避免进一步强化身份；
- 小 batch 下关系估计不稳定时立即停止，不通过扩大 batch 或大量新超参数掩盖问题。

逐帧强对齐和完整向量 MSE 只允许作为负对照，不作为主方案。

P2获单独授权后只允许比较：

```text
冻结的GLA最终候选
冻结的GLA最终候选 + P2关系项
```

组合实验必须记录 BDI、AU、head 和 identity 损失的梯度范数、余弦冲突率和长期主导关系。

## 8. 评估与停止标准

### 8.1 必报指标

- BDI：MAE、RMSE、CCC、Pearson；
- 泛化：train-validation error gap、最佳 epoch、后期过拟合斜率；
- 压缩：prediction mean/std、`pred_std / true_std`；
- severity：minimal/mild/moderate/severe 的 MAE、bias、worst-group；
- identity：shared representation same-subject retrieval、paired-task verifier/probe；
- task：Freeform/Northwind 同 subject prediction/residual 差异；
- behavior：AU/head masked loss、coverage、每组梯度 norm；
- 稳健性：split sensitivity、quality/pose/coverage 分组结果。

统计单位必须为 subject；置信区间使用 subject-cluster bootstrap，不能把帧或同一 subject 的两个任务当作独立样本。

### 8.2 支持假设的联合门槛

只有同时满足才支持 `H1`：

- 至少 3 个 paired seed 的 BDI 指标稳定改善或不降；
- train-validation gap 不扩大；
- identity risk 不增加超过预注册门槛 `0.02`，理想下降约 `0.05`；
- severity bias、prediction compression 和 task consistency 不恶化；
- `GLA-AUX-SHUFFLED` 不能复现相同收益；
- auxiliary 梯度不能长期支配 BDI 梯度；
- 最终保留因素必须同时有反向删除和正向加回证据；联合profile若不优于matched简化对照，保留更简单模型。

### 8.3 停止条件

- seed42 `GLA-FULL`相对同seed匹配`GLA-C-REF`出现`ΔCCC < -0.05`或`ΔMAE > +0.50`时，视为科学utility failure；只完成S1--S5诊断，不运行controls、加法、43/44或benchmark，该边界必须在训练前写入resolved实验规格；
- gaze 不因联合模型表现不佳而重新加入；如未来重启，必须重新检索并单独授权；
- AU 或 head 单组持续恶化时移除该组，不用联合结果掩盖；
- identity risk 明显升高，即使 MAE 改善也不能视为缓解过拟合；
- shuffled target 获得同等收益时，只能解释为一般正则化；
- 多 seed 方向不一致时停止扩大目标、模型头或损失搜索；
- 数据源、frame join 或 train-only normalization 任一门禁失败时不得训练。

## 9. 复现合同

每个未来 run 必须记录：

- git commit hash、branch 和 dirty status；
- base config、全部 override 及 resolved config；
- random seed；
- physical split file 和逻辑 split 角色；
- 完整命令；
- GPU/device、precision、strategy；
- checkpoint monitor/mode；
- metrics 和 diagnostics 输出路径；
- OpenFace 版本、二进制/模型哈希、feature profile 和输入 manifest；
- 目标列顺序、描述符版本、valid 阈值和 train-only normalization statistics；
- frame join、coverage 和 shuffled-target permutation seed。
- final extraction summary、双重绑定的 CSV content manifest、逐 CSV rows/schema/size/SHA 和当前 JPG 对 candidate manifest 的逐字节绑定结果。
- matched RGB reference ID、实际sampling/clip manifest SHA和P0E final eligibility decision SHA。

## 10. P1/P2 未来最小实现边界

PB-P0 已实现独立的 schema/frame/valid-mask/pose-dynamics/train-only-statistics 合同、只读 CLI 和 aligned-JPG no-gaze 提取入口，且没有接入训练数据路径。只有`PB-P0E`明确给出对应组eligibility、最终training-target contract与风险门禁通过，并再次获得用户授权后，才允许实施下列 P1/P2 接入：

- 新增 `src/datasets/privileged_behavior.py`：exact manifest、frame join、白名单、mask、描述符和 train-only normalization；
- 修改 `src/datasets/dataset.py`：复用 RGB 选帧结果，可选返回嵌套 `privileged_behavior`，默认关闭时保持旧 batch contract；
- 修改 `src/models/outputs.py`：增加可选 AU/head prediction 和 loss 字段；
- 修改 `src/models/mtl_lite.py`：在 `shared_features` 后增加两个小头，并在 train-only loss 路径计算 masked loss；
- runner 仅在需要注入 train statistics 时做最小改动；
- 新增默认关闭的 `configs/behavior_alignment/`，不得修改现有 baseline 配置语义。

不直接复用当前 `src/datasets/openface_features.py` 的加载行为，因为现有 behavior-only loader 包含宽特征选择、subject fallback、缺失值补零、固定 `sample_step` 和跨失败边界 delta 等不适合本方案的语义。

建议配置命名仅作为未来接口草案：

```yaml
MODEL:
  PRIVILEGED_BEHAVIOR:
    ENABLED: false
    GROUPS: [au, head]
    TARGET_LEVEL: video
    CONFIDENCE_THRESHOLD: 0.8
    LOSS: smooth_l1
    TOTAL_WEIGHT: 0.03
    WARMUP_FRACTION: 0.10
    RAMP_FRACTION: 0.20
```

## 11. 测试与验证命令

当前 PB-P0 基础设施验证入口：

```bash
/home/zhen/miniconda3/envs/light/bin/python -m pytest \
  tests/test_privileged_behavior_contract.py \
  tests/test_privileged_behavior_p0.py \
  tests/test_openface_aligned_behavior_features_script.py
```

P1/P2 获得授权后计划新增的测试：

- 默认关闭时 config、batch、forward、loss 和 checkpoint 完全兼容；
- 白名单精确且 gaze 永不进入 schema；
- CSV 行重排后仍按 frame ID 同步；
- 五种 temporal sampling 策略全部同步；
- video/task/frame 歧义、缺列和重复 frame 硬失败；
- invalid frame 只改变 mask，不产生伪零目标；
- delta 不跨 invalid gap；
- normalization 只读取 physical train；
- target stop-gradient；
- auxiliary loss 只在 train 生效且能给 RGB 分支有限梯度；
- AMP 下 loss/gradient 有限；
- 隐藏 OpenFace root 后 val/test/inference 仍可运行。

以下为未来 P1/P2 预期命令；当前对应训练文件和配置尚不存在，不应执行：

```bash
python -m pytest \
  tests/test_privileged_behavior.py \
  tests/test_mtl_lite_behavior_alignment.py \
  tests/test_temporal_sampling.py \
  tests/test_mtl_lite_config.py \
  tests/test_mtl_lite_forward.py \
  tests/test_mtl_lite_loss_backward.py
```

```bash
python scripts/train_mtl_lite.py \
  --override configs/stage_c/common.yaml \
  --override configs/stage_c/c_ref_e2.yaml \
  --override configs/behavior_alignment/au_head.yaml \
  --override configs/stage_c/seeds/seed_42.yaml \
  --override configs/stage_c/debug_smoke.yaml
```

## 12. 剩余风险

- OpenFace AU/head 测量误差可能随肤色、光照、眼镜、遮挡和人脸形态变化；
- aligned warp 可能改变 head pose 的物理解释；
- 行为目标仍可能携带 subject、task、摄像机或检测质量信息；
- 小样本多任务训练可能产生负迁移或梯度支配；
- 从同一视频生成大量 clip 不增加独立 subject，必须按 video/subject 控制统计权重；
- 多个 AU 描述符和权重搜索会形成 validation 过拟合，必须冻结小型矩阵；
- 即使 identity probe 下降，也不代表所有身份信息已被删除；
- 该路线不能替代当前曝光、face usability、split integrity、severity bias 和 task consistency 审计。

## 13. 决策记录

| 日期 | 决策 |
|---|---|
| 2026-07-28 | 初始登记训练期 privileged behavior alignment 研究方案；随后仅授权 PB-P0 基础设施、两视频 debug 和 landmark-only 阻断审计，不授权全量 rich 提取、模型或训练。 |
| 2026-07-28 | 首版只保留 AU12/14/15 与旋转 head dynamics。 |
| 2026-07-28 | gaze 因缺少足够明确、可迁移的抑郁关联证据而完全移除，以降低混杂和工作量。 |
| 2026-07-28 | 首版采用分组辅助预测；关系式对齐延后；完整 embedding MSE 仅作负对照。 |
| 2026-07-28 | 完成 PB-P0 合同/CLI/no-gaze extractor 基础设施；冻结 300/300 PASS、30 FPS、493,141 frame-match 的 source-video contract，aligned `-fdir timestamp=0` 禁止用于速度，旋转差采用 wrap 且只在连续有效帧计算。 |
| 2026-07-28 | `debug2c` 以 `-2Dfp -pose -aus` 完成 2/2 视频、1,920 行 `PASS`；明确不得解释为全量提取授权。 |
| 2026-07-28 | 初始 landmark-only v1 P0 完成 300/300 exact join，但因缺少三列 AU intensity 和三列 rotation source 而 fail-closed 为 `BLOCKED`；禁止回退历史 raw rich CSV。 |
| 2026-07-28 | provenance 加固后的 `p0_contract_landmark_only_blocked_v2` 取代 v1 成为当前权威 landmark-only 阻断审计；v1 仅保留历史。strict-rich 门禁已实现，但因当前 rich schema 为 0/300 而明确为 `NOT_APPLICABLE`，未执行全量 rich 提取。 |
| 2026-07-28 | 早期将下一门禁登记为单独授权full-rich后重跑P0；尚未执行，随后由下一行的可行性证据进一步收紧。P1/P2和全部训练继续保持未授权。 |
| 2026-07-28 | 后续路线改为`PB-R0 -> fresh debug2 -> pilot20 -> threshold policy`，不再把300视频full-rich作为无条件下一步；原因是旧debug不满足最新hash合同，且41/300个同版landmark视频低于逐视频0.995。PB核心随后冻结在`2685fa4`，独立docs-only follow-up也已完成且未amend；当前R0只剩push后的Windows clean sync验收。 |
| 2026-07-29 | future训练设计由`GLOBAL_LOCAL_AU_EXPERIMENT_PLAN.md`统一接管：三语义区共享backbone；AU按12/14/15、4/6/7、10/17分级扩展；AU6与AU14采用跨区域特征。本文主体继续作为已实现AU12/14/15来源/P0历史合同。 |
| 2026-07-29 | pilot20完成：20视频中16个达到strict 0.995，4个失败；当时禁止直接full-rich，并将下一步固定为versioned mask-aware threshold/coverage policy。 |
| 2026-07-29 | PB-P0A2完成：冻结policy SHA `9a3fb7...1926`；权威v5为`PASS_PILOT_MASK_AWARE_FEASIBILITY`、0 blocker、3项future-full warning，且`full_rich/P0B/training_authorized=false`。下一候选动作是评审warning并单独授权A3。 |
| 2026-07-30 | PB-P0A3权威full-rich v2完成300/300视频、493,141行、单一182列schema；A2完整判定`PASS_FULL_SOURCE_COVERAGE`、0 blocker/0 warning。失败v1禁止复用；P0B/模型/训练仍未授权。 |
| 2026-07-30 | P0B前新增最小兼容代码门禁：修复A2 PASS的错误`next_action`，并避免P0把41个legacy strict失败误作mask-aware来源失败；非coverage硬门禁保持不变。 |
| 2026-07-30 | 本文PB固定矩阵降为历史；未来训练由GLA文档的FULL优先混合消融统一接管。 |
