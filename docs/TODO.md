# TODO.md

> 文档职责：只维护可执行任务和完成状态。不要在此重复长篇实验解释；机制解释见 `RGB_OVERFITTING_AUDIT_PLAN.md`，状态摘要见 `CURRENT_STATUS.md`，文档导航见 `DOCS_GUIDE.md`。

## 当前立即执行任务（权威入口）

本节是当前下一步的权威任务入口。下方较早的阶段任务和 P0/P1/P2 列表保留为历史记录或背景任务；若与本节冲突，以本节为准。

### 2026-08-05当前路线：region-first，AU数值监督退出主线

- [x] `PB-P0B-CORE`：300/300视频、493,141帧、core/schema/exact join/physical-train statistics和0 blocker已完成；不构成训练授权。
- [x] `PB-P0C-CORE-AU-FIDELITY`：mask-aware v2技术审计`PASS`，但AU12/14/15组为`INELIGIBLE_METRIC`。该结果冻结为AU数值监督负证据，不再阻塞landmark语义裁切。
- [x] `ROUTE-AU-GUIDED-LANDMARK-RGB`：批准`PLAN-20260805-AU-GUIDED-LANDMARK-RGB-v1`；AU只定义语义区域，landmark负责定位，模型只学习RGB。
- [ ] `REGION-P0A-CODE`：下一候选代码包。实现只读aligned-space landmark/region合同、三语义区非重叠box、mask-aware validity、overlay和manifest；不得读取AU值、BDI、prediction或checkpoint。
- [ ] `REGION-P0B-PILOT`：单独授权后在physical train比较static canonical/raw dynamic/stabilized dynamic，冻结人工复核包；不生成训练数据。
- [ ] `REGION-P0C-POLICY`：只根据train无标签几何分布和人工复核冻结margin、尺度、containment、coverage、IoU和jump门槛。
- [ ] `REGION-P0D-FULL`：单独授权后对300视频/493,141帧应用冻结policy，val/test只报告，输出region eligibility和逐帧manifest；默认不物化局部JPEG。
- [ ] `GLA-RGB-CODE`：region通过后另行授权默认关闭的四视图Dataset、共享backbone、global-residual融合和runner；不添加AU/head/gaze/identity/ordinal/Stage C路径。
- [ ] `GLA-RGB-SMOKE`：import/config/one-batch/backward/AMP/DDP/显存/FLOPs和100-step train-only smoke；不读取test。
- [ ] `GLA-RGB-S42`：先运行匹配`GLA-C-REF-S42`和`GLA-RGB-FULL-S42`，再按预注册规则运行NO-EYE/NO-NOSE/NO-MOUTH、GRID及条件性NO-EXPOSURE-AUG。
- [ ] `GLA-RGB-MULTI/LOCKED`：只有seed42通过后运行paired seeds43/44及一次locked post-selection benchmark。

当前禁止继续执行旧`PB-P0D/P0E -> AU head -> GLA-FULL`链。extension AU4/6/7/10/17虽然物理列存在，但不再需要数值资格审计；只有未来重新注册AU监督分支时才恢复。

### 2026-07-30历史任务地图（以下不再是当前执行入口）

### 历史目标

既有task-nuisance、identity和输入伪迹实验继续作为机制证据。2026-07-30当时的实施入口曾收敛为 **headless GLA全模型优先混合消融**，候选包含global/三个语义local和train-only AU辅助监督；该方案现已由上方2026-08-05 RGB-only region-first路线取代。

```text
Stage A Shortcut 证据收口
-> Stage B Identity-adversarial task representation
-> Stage C Coarse task-nuisance information separation
-> Stage D Falsification and robustness validation
```

### 当前任务地图

| 阶段 | 要回答的问题 | 下一步产出 |
|---|---|---|
| A. Shortcut 证据收口 | 身份、artifact/quality、severity imbalance 是否确实影响预测或误差？ | 已完成：A1+A2 成立，A3 仅作评估，A4 支持 severity-balanced |
| B. Identity-adversarial baseline | 只抑制可验证身份捷径是否足够？ | 已完成：E2 弱有效，E1/E3 无效，identity 泄漏全局未解，进入 Stage C（结论见 `CURRENT_STATUS.md` 2026-07-10 B5） |
| C. Coarse task-nuisance 信息分流 | `z_dep / z_nuisance` 是否在等参数条件下优于共享表征和 GRL baseline？ | seed-42 utility gate 已否证当前 96 维 bottleneck/split 家族，停止进入 C3 |
| D. Falsification / robustness | 信息分流失败来自 bottleneck、泄漏未降还是优化不稳定？ | 已完成 identity-gradient、正则化、连续加权和 split sensitivity 收口；不授权维度 sweep |
| E. 历史GLA组件资格门禁 | 三语义区、AU组和跨区域关系能否进入AU-only候选？ | P0B已通过，P0C core组`INELIGIBLE_METRIC`，因此该AU资格链终止并转入上方region-only门禁 |
| F. 历史全模型优先混合消融 | 最大合格`GLA-FULL`是否有能力改善泛化？ | AU版本未进入训练；当前改用上方`GLA-RGB-FULL`、区域减法与GRID反证 |

阅读规则：只执行新任务时读到“暂缓项”为止即可；后续章节主要是历史任务、已完成基础设施和旧阶段记录。

### 2026-07-30历史PB/GLA路线（冻结，不得执行）

> 下列未勾选项保存当时的任务设计，不再表示待执行授权。P0B/P0C的实际完成状态和P0D/P0E/AU训练链的退役结论见本文件顶部权威入口。

当前三语义区训练设计见`docs/GLOBAL_LOCAL_AU_EXPERIMENT_PLAN.md`；现有AU12/14/15来源与P0实现合同见`docs/PRIVILEGED_BEHAVIOR_ALIGNMENT_PLAN.md`。当前任务状态：

- [x] `PB-SPEC`：历史AU/head研究问题、来源合同和P0边界已冻结；当前GLA已在后续文档决策中移除head，gaze继续排除。
- [x] `PB-P0-INFRA`：exact `video_id + task_name + frame_id`合同、no-gaze extractor、strict provenance、coverage和physical-train normalization基础设施完成；65项测试通过。
- [x] `PB-P0-LANDMARK-V2`：当前权威landmark-only审计完成，300/300 exact join、493,141帧，因缺六个rich源列而预期`BLOCKED`。
- [x] `PB-R0-CORE`：PB核心实现、extractor、测试与方案冻结为提交`2685fa4`。
- [x] `PB-R0-DOCSYNC`：独立docs-only follow-up已完成，不amend核心且不夹带FACE/exposure/T0b。
- [x] `PB-R0-WINDOWS-SYNC`：已完成pilot20所需的当前脚本/来源同步与前置验收。
- [x] `PB-P0A0-DEBUG2`：当前脚本fresh debug2已完成，允许进入pilot20。
- [x] `PB-P0A1-PILOT20`：20视频、25,980帧完成；OpenFace成功25,446帧，16视频达到strict 0.995，4视频FAIL：`205_1_Freeform=0.990909`、`206_1_Freeform=0.920000`、`207_2_Freeform=0.698148`、`208_1_Freeform=0.979227`。禁止直接启动full。
- [x] `PB-P0A2-POLICY`：已冻结`behavior_source_coverage_policy_v1.json`（SHA-256 `9a3fb739078ab36e1fa4f8f67a8c8b167a76329941a591ad0ef9dd0c731a1926`）并实现只读校验器/CLI/测试。pilot20 v5为可行性PASS/3项future-full warning；随后full v2为`PASS_FULL_SOURCE_COVERAGE`、0 blocker/0 warning，且仍`P0B/training_authorized=false`。
- [x] `PB-P0A3-FULL-RICH`：权威v2完成300/300视频、493,141/493,141行、单一182列schema、587 MB、10,064.6秒和全部hash/provenance闭环；A2 full判定为`PASS_FULL_SOURCE_COVERAGE`、0 blocker/0 warning。失败且不完整的v1禁止resume、复用或进入P0B。
- [ ] `PB-P0B-CODE-COMPAT`：**当前下一代码候选，尚未授权**。先提交具名实施披露并等待用户确认；修复A2 full PASS却输出错误`next_action=STOP...`的报告分支，并审计/修复旧P0核心对mask-aware v2来源的兼容性。只放宽legacy coverage状态的误用，不得弱化hash/schema/join/provenance/数值域门禁。
- [ ] `PB-P0B-CORE`：上述代码兼容包验证并另行获得运行授权后，对不可变v2 root运行P0审计和A2 coverage校验器；要求CLI `status=PASS`、300/300 core/schema/join、physical-train统计、九项产物、0 blocker和coverage decision PASS。当前CLI默认只拒绝`std==0`常量列；任何非零实用方差阈值必须版本化。P0B PASS仍不是训练授权。
- [ ] `PB-P0C-FIDELITY`：新增独立matched-version raw/aligned AU物理保真审计，只裁决core/extension A/extension B候选AU。pose/head保真结果仅保留为历史质量、姿态分组和shortcut审计证据，禁止回退历史raw rich CSV作监督。
- [ ] `PB-P0D-TARGET-RISK`：冻结实际P1视频/clip描述符、重新拟合physical-train normalization，并对同一描述符执行identity/session、task、exposure和quality风险审计；不读取BDI/prediction/checkpoint。
- [ ] `PB-P0E-ELIGIBILITY`：新增AU-only机器可读聚合器，输出`REGION_ELIGIBLE`、`AU_GROUP_ELIGIBLE(core/extension_A/extension_B)`、`CROSS_REGION_ELIGIBLE`、`AU_ELIGIBLE`、`GLA_PROFILE_ELIGIBLE`、完整`eligible_component_profile`及其SHA，并记录`excluded_by_design: [head_motion_auxiliary]`和`P1_CODE_AUTHORIZED=false`。`GLA-FULL`严格等于该P0E最大依赖闭合集；某组失败时必须重新登记精简矩阵，不能静默继续联合方案。
- [ ] `PB-P1-CODE`：仅在P0E通过且再次授权后实现默认关闭的dataset AU target path、AU辅助头、train-only fixed-denominator masked loss和配置/测试；不得加入head或gaze分支。
- [ ] `PB-P1-RUN`：旧PB六条件screen不再作为运行入口；PB-P1只提供GLA所需的默认关闭训练基础设施，任何smoke、校准和正式训练均转交下方`GLA-*`矩阵并分别授权。历史role-swap只保留为既有split-sensitivity证据。
- [ ] `PB-P2-REL`：仅在GLA最终候选至少3个paired seed通过，且联合辅助profile优于其matched单组/减法对照后重新授权；先单独验证2--4秒短窗口预测，不与关系损失同时实现。

pilot20已经直接证实strict逐视频0.995会失败；A2因此采用冻结的mask-aware合同，而不是临时降阈值或删视频。pilot中的3项future-full风险已在授权full前评审，并由完整physical train结果正式裁决。旧PB来源/coverage实验没有加入gaze、曝光处理或新RGB增强；当前GLA则预注册一套跨所有条件固定的train-only photometric augmentation，不能把它反写成旧PB实验已经使用的因素。

完整full已经消除了pilot的3项future-full风险：physical train总体/Freeform/Northwind coverage均通过，task gap约0.01304。该PASS只消除来源coverage阻塞，不自动批准P0B、扩展AU、P0C/P0D/P0E、模型或训练。

### 全模型优先混合消融执行包（除已勾选文档包外均未授权）

- [x] `GLA-HEADLESS-PHOTOAUG-DOCS`：`DOC-20260730-GLA-HEADLESS-PHOTOAUG-v2`仅同步实现前文档；冻结head/gaze排除、AU-only损失、四视图张量/mask合同、按view独立且时间固定的train-only曝光/普通颜色增强、S0--S4阶梯和12-fit上限。未授权代码、数据生成、smoke、训练、commit或push。
- [ ] `GLA-DISCLOSE`：每个代码/数据/smoke/训练包先按根`AGENTS.md`报告package ID、边界、架构/数据流/tensor/mask/loss、文件、兼容性、split/test访问、命令、算力、输出、验证、风险和agent分工；结束该轮等待用户明确授权。
- [ ] `GLA-ELIGIBILITY`：完成PB-P0B/P0C/P0D、扩展AU合同、三语义区crop manifest和P0E；输出最大eligible profile及SHA，未通过组标为`SKIPPED_INELIGIBLE`。
- [ ] `GLA-STRATEGY-MANIFEST`：在任何full fit前冻结run ID、相邻比较、MAE/CCC与风险阈值、loss固定分母、最大run数、停止分支和locked benchmark包。
- [ ] `GLA-CODE`：单独披露并授权后，以独立路径实现默认关闭的四视图共享backbone、global-residual validity-aware融合、单层GRU、分组/跨区域AU小头和train-only photometric augmentation；不继承旧head/Stage C逻辑，并保持旧配置/API/checkpoint兼容。
- [ ] `GLA-SMOKE`：单独授权后执行import/config/one-batch、hidden-target val/test、AMP/DDP和100-step train-only校准；不读取benchmark。
- [ ] `GLA-FULL-S42`：先运行匹配`GLA-C-REF`与最大合格`GLA-FULL`。source/hash/join、行为目标泄漏、NaN、mask/shape、AMP或DDP等技术失败立即停止；若仅触发预注册的科学utility/risk失败，则只完成S1--S4反向阶梯作诊断，跳过机制控制、加法复核、multi-seed和benchmark。
- [ ] `GLA-SUB-S42`：只要没有技术失败，就按`AU-B -> AU-A -> all core AU -> locals`完成S0--S4独立训练阶梯：`GLA-FULL / GLA-S1-NO-AU-B / GLA-S2-CORE-AU / GLA-RGB-FULL / GLA-C-REF`。若FULL科学失败，该阶梯只作诊断且不运行controls；只有FULL科学通过时才运行grid、no-cross、subject-deranged shuffled-aux和`GLA-FULL-NO-EXPOSURE-AUG`控制并进入后续加法复核。
- [ ] `GLA-ADD-S42`：只对减法幸存组做至多3个单组加回和1个reduced rebuild；不得扩展全组合或临时weight sweep。
- [ ] `GLA-MULTI`：seed42最多12个full fit；最终简化候选、直接matched ablation和必要anchors/controls再运行paired seeds43/44，每seed最多5个full fit。
- [ ] `GLA-LOCKED-BENCHMARK`：冻结evaluation package后一次性读取原test；因历史role-swap，该结果只能称locked post-selection benchmark，不得触发重新选择或重跑同族配置。

### 历史编程实施控制（Stage A-C，非当前入口）

本节用于后续写代码时控制范围。任何新实现先对照这里判断是否允许进入下一阶段；若缺少前置证据，只能补诊断、补文档或补测试，不直接加模型复杂度。

| 控制点 | 允许做什么 | 禁止做什么 | 通过条件 |
|---|---|---|---|
| A0 证据收口 | 运行 A1-A4，补齐报告 join、表格校验和状态结论 | 改训练 forward、加入 GRL、加入 `TaskNuisanceBlock` | `CURRENT_STATUS.md` 和 `RGB_OVERFITTING_AUDIT_PLAN.md` 写明 A1-A4 四个结论 |
| B0 干预规格 | 只写 identity-adversarial baseline 的接口设计、配置键草案、loss 权重范围和评估表 | 同时实现 `z_nuisance`、`z_id`、reconstruction 或 decorrelation | A1+A2 结论决定 strong suppression / monitor-only 分支 |
| B1 最小代码 | 实现 `H0 -> z_dep -> BDI` + 可开关 GRL/id head，保持默认关闭或独立 override | 改默认 baseline 行为、改 split/label、引入细粒度 latent | import、config load、one-batch smoke、相关单测通过 |
| B2 对照运行 | 固定 RGB baseline、identity-adversarial、severity-balanced 三组对照 | 新增 RGB mask 族、late fusion、dynamic branch | 三组均输出 BDI、identity risk、severity bias、task consistency、train-val gap |
| C0 信息分流规格 | 设计 `TaskNuisanceBlock` 最小接口、等参数消融、审计矩阵和反证条件 | 将 recon/decorrelation 或单一 attacker 当作语义解耦证明 | 证明 B baseline 不足，且 C 的评价指标、multi-seed 规则和停止条件已写清 |
| C1 最小信息分流代码 | 实现 `H0 -> z_dep,z_nuisance`，预测只读 `z_dep`，recon/decorrelation 低权重可开关 | 显式划分 `z_art/z_ctx/z_pose/z_quality`，堆叠多级门控 | config/forward/loss/backward、默认兼容、多表征导出、seed-42 smoke 与 train-only calibration 通过 |

#### 立即可编程任务包

> 状态（2026-07-30历史）：**C2/C3 已停止；PB-P0A3/A2 full coverage已完成；当时入口改为GLA执行包**。任何旧四区、RGB-only AU或G0/G1/G2前向筛选只用于复现历史；当前不得覆盖`GLOBAL_LOCAL_AU_EXPERIMENT_PLAN.md`第0节的三语义区RGB-only、`GLA-RGB-FULL`优先混合消融。新region/dataset/model/config和任何运行仍须独立授权。

> 正则化审计已完成：四组 best-val RMSE 基本重合且全部继续过拟合，关闭系数 sweep，EarlyStopping 保留为训练策略。

> 连续 severity-density weighting 已完成：优于 plain MSE，但 CCC、预测压缩和 severe bias 均不及四档 E2，关闭 alpha/sigma sweep。后续若重新比较权重形状，必须先控制 E2 与 continuous 的样本平均 loss scale。

1. **A0-result-gate**：已完成 A1-A4 服务器输出收口。
   - 输入：A1 layerwise summary、A2 coupling report、A3 matched-only weaklabel report、A4 severity imbalance report。
   - 输出：四条结论已写入 `docs/CURRENT_STATUS.md` 和 `docs/RGB_OVERFITTING_AUDIT_PLAN.md`。
   - 结果：A1+A2 同时成立；A3 仅作为 probe/case/group-wise evaluation；A4 要求 severity-balanced regression 进入 Stage B。

2. **B0-stage-entry**：已确认 Stage B 可以推进。
   - 固定对照：`E0 RGB MTL-Lite baseline`、`E1 identity-adversarial MTL`、`E2 severity-balanced regression`、`E3 identity-adversarial + severity-balanced`。
   - 允许范围：GRL/subject attacker、severity-bin weighting 或 regression loss reweighting、二者组合。
   - 禁止范围：`TaskNuisanceBlock`、`z_nuisance`、显式 `z_art/z_ctx/z_pose/z_quality`、新 RGB 输入滤镜、dynamic branch、late fusion。
   - 成功条件：identity risk 或 severity bias 改善，且 CCC、task consistency、train-val gap 不明显恶化。

3. **B0-spec-first**：在写 GRL 代码前先写最小设计规格。
   - 特征入口：使用当前 MTL-Lite 的 `shared_features` 作为 Stage B `z_dep`，不新增 `TaskNuisanceBlock`。
   - 配置原则：`identity_adversarial` 与 `severity_balanced_regression` 默认关闭；E0 默认 baseline 行为必须完全不变。
   - id 类别来源：只从 train split subject 构建 `subject_id_to_index`；val/test subject 不进入类别表，不参与 identity loss。
   - GRL 范围：先固定 `lambda_id = 0.02, 0.05, 0.10, 0.20` 小范围 sweep，优先选择最低有效强度。
   - severity 权重：只从 train split 标签统计得到，按 minimal/mild/moderate/severe 四组计算，mean-normalize 并设置 min/max 截断。
   - loss 边界：第一版只加权 regression MSE；CCC loss 与 ordinal auxiliary loss 暂不加权。
   - 改动范围：设计文档、配置草案、测试计划；不先改 `src/models/mtl_lite.py`。

4. **B1-code-minimal**：只实现 identity-adversarial baseline 的最小闭环。
   - 允许文件：`src/models/`、`src/trainers/`、`configs/`、`tests/` 中与 MTL-Lite baseline 直接相关的文件。
   - 输出接口：`MTLLiteOutput.identity_logits` 可选；loss 结构中增加可选 identity loss，默认不影响既有调用。
   - 训练逻辑：`shared_features -> GRL -> subject_id head`；identity loss 仅在 train stage 且 subject 映射可用时启用。
   - 数据逻辑：runner 或 data module 从 train split 构建 subject map，不能扫描 val/test 扩类别。
   - 必备测试：import check、config loading check、dummy one-batch forward/loss、GRL 开关不会影响默认 baseline、val/test unknown subject 不触发 loss。
   - 必备报告：A1/A2 identity risk 降低、BDI utility 未崩、severity bias 和 task consistency 不恶化。

5. **B2-severity-balanced-minimal**：实现 severity-balanced regression 的最小闭环。
   - 允许先支持 minimal/mild/moderate/severe 四组权重，权重只能由训练集统计生成或配置显式给定。
   - 计算规则：推荐 `weight = (total / (num_bins * count_bin)) ** power`，先跑 `power=0.5`，再比较 `power=1.0`。
   - 稳定性规则：权重 mean-normalize，并设置 `min_weight/max_weight`；训练日志记录 bin count、raw weight、clipped weight、normalized weight。
   - 必备测试：权重计算、loss 缩放、默认关闭不改变 baseline、只改 regression MSE 不改 CCC/ordinal。
   - 必备报告：minimal 高估、severe 低估、pred_std/true_std、CCC、identity risk 和 task consistency。

6. **B3-fixed-experiments**：按固定矩阵跑完 Stage B，不临时加模块。
   - E0：`RGB MTL-Lite baseline`。
   - E1：`identity-adversarial MTL`。
   - E2：`severity-balanced regression`。
   - E3：`identity-adversarial + severity-balanced`。
   - 可选配置路径：`configs/stage_b/e0_rgb_mtl_lite.yaml`、`configs/stage_b/e1_identity_adversarial.yaml`、`configs/stage_b/e2_severity_balanced.yaml`、`configs/stage_b/e3_identity_adversarial_severity_balanced.yaml`。
   - 运行约束：四组必须使用同一 split、seed、input variant、optimizer、precision、checkpoint 和输出目录结构。
   - 执行脚本与命令清单：`scripts/stage_b/run_stage_b_matrix.sh`（训练+sweep+per-run 诊断）、`scripts/stage_b/aggregate_stage_b.sh`（横向对照表）；完整 runbook 见 `docs/STAGE_B_RUNBOOK.md`。

7. **B4-stage-b-report**：每组训练后统一跑诊断并产出横向对照表。
   - prediction：MAE/RMSE/Pearson/CCC、pred mean/std、train-val gap。
   - severity：minimal/mild/moderate/severe 的 count、MAE、bias、abs error。
   - identity：A1 layer/shared retrieval、A2 residual-identity coupling、subject attacker accuracy。
   - consistency：同 subject Freeform/Northwind prediction diff 和 residual diff。
   - artifact-risk：基于 matched-only A3 变量做分组评估，不把 A3 变量作为训练监督。

8. **B5-stage-b-gate**：写出是否进入 Stage C 的判定。
   - E1 有效：identity risk 下降且 BDI/severity/task consistency 未明显恶化。
   - E2 有效：minimal/severe bias 和 compression 改善，CCC 与 identity risk 未明显恶化。
   - E3 有效：同时优于或互补于 E1/E2，作为 Stage C 强 baseline。
   - 只有 E1/E2/E3 均不能在可接受代价内改善风险时，才进入 C0 粗粒度信息分流规格。

9. **C0-spec-before-code + P0-training-policy**（已完成）：B5 已确认 Stage B baseline 不足，Stage C 的接口、实验矩阵、seed、审计和停止条件已冻结，完整规格见 `docs/STAGE_C_RUNBOOK.md`。
   - 先写接口和损失：`z_dep` 预测、`z_nuisance` 候选互补出口、低权重 recon、低权重 cross-covariance/decorrelation；明确这些辅助损失不是语义解耦证明。
   - 固定等参数/等 bottleneck 对照，避免把增益误判为参数增加或降维收益。
   - 冻结第一版 leakage matrix：`H0/z_dep/z_nuisance x identity/task/pose/artifact/BDI`，并使用 fresh multi-attacker 报告最强风险。
   - 先写失败条件：若不优于 paired `C-REF`、multi-seed 不稳定、`z_nuisance -> BDI` 泄漏接近 `z_dep`，或风险下降以 CCC/severity/task consistency 恶化为代价，停止增加复杂度。
   - 禁止把 artifact/context/pose/quality 做成显式 latent；pose_rz / AU / black_border 第一版只作为审计轴，不进入训练监督。
   - P0 已完成：配置化 seed；EarlyStopping 与 best checkpoint 共用 monitor/mode；共享 base 默认关闭，Stage C common 开启；新增训练策略测试和同协议 `C-REF` 配置。
10. **C1-minimal-information-separation**（已完成）：在 C0 spec 冻结后实现最小闭环。
    - 已实现 `H0 -> z_dep, z_nuisance`，预测只读 `z_dep`，recon/decorrelation 低权重可开关。
    - 固定对照组：`C-REF`、`C-BN`、`C-REC`、`C-FULL`；Stage B 的 E0-E3 只作历史背景，不复用为 Stage C run ID。
    - 本地 config/forward/loss/backward、默认 state-dict 兼容和多表征导出测试已通过。
    - seed 42 smoke 与 100-step train-only calibration 已完成；冻结 `lambda_rec=0.001`、`lambda_xcorr=0.01`。
11. **C2-validation-screening**（utility gate 已完成并失败）：C-BN/C-REC/C-FULL 的 validation CCC 相对 C-REF 分别下降 `0.0990/0.1175/0.1147`；C-REC/C-FULL MAE 恶化超过 `0.50`。后续 leakage/group 只用于解释失败，不再决定是否进入 C3。
    - 已实现并运行 train→val BDI/task probe、identity retrieval/pair verifier、nuisance BDI gate、severity/task group robustness 和 subject bootstrap CI。
    - identity pair AUROC 未下降，severity worst-group gate 全部失败；pose_rz/black-border/face-offset-y 因缺 train weak-label 表保持 unavailable。
12. **C3-multi-seed-gate**（停止）：按预注册 severe utility failure 规则，不运行 seeds 43/44，不进入 Stage D final-test 路径。
13. **Postmortem capacity audit**（训练与 utility 审查已完成）：固定 C-BN `DEP_DIM=96/128/160/192`，seed 42，完整 40 epoch、validation-only。结果呈非单调，160 维仅部分恢复 utility，192 等维投影仍失败；外部 leakage/effective-rank 汇总仍待补，不恢复 C3。
14. **Identity-gradient audit**（已完成，未通过）：candidate 改善短期 validation utility 和 severe bias，但 external identity pair AUROC、same-subject retrieval 与末期过拟合恶化；冲突率约 `60.2%`。训练内 attacker collapse 不构成 identity invariance，维度实验不获授权。
15. **LM-T0a frame-contract inventory**（已完成并通过，原任务名AU-T0a）：300个视频全部PASS，最低join rate `1.0`，41,016个模型选中帧全部连接，最佳offset全部为`0`。该gate只证明frame identity，不证明空间坐标映射。
16. **LM-T0b coordinate-contract audit**（全量审计已完成但未通过）：冻结的 OpenFace 2.2.0 aligned-space 重检测输出 `t0b_coordinate_contract_of220_v1` 覆盖 300 视频、493,141 行，与 JPG 数一致；486,640 帧 `success=1`、6,501 帧失败。T0b 汇总为 `260 REVIEW_REQUIRED / 40 FAIL / 0 BLOCKED`；40 个 FAIL 按 split 为 train/val/test=`12/13/15`，原因是 `mapping_valid_ratio < 0.995`，不能由人工 overlay 复核覆盖。另有 364 个采样帧 mapping source detection failed、3 个采样帧 `in_bounds_ratio < 0.80`。权威报告和表格位于 `logs/au_region_tracking_audit/t0b_coordinate_contract_of220_v1/`；动态 mask、局部 crop 和训练仍被 T0b 阻断。
16a. **LM-T0b train-only overlay review**（AI辅助副本已验证，待人工复签）：权威PENDING包仍为`t0b_overlay_review_of220_v2`且保持不变；独立`Codex-AI-assisted`副本为`115 PASS / 5 UNCERTAIN / 0 FAIL`，validator给出`overlay=REVIEW_REQUIRED`、`coordinate_contract=FAIL`、`authorized=false`。先人工复签5个遮挡/裁切疑难项及其余样本；人工复核仍不能把40个自动FAIL视频改为PASS。
17. **FACE-T0c frame-failure / source-presence / exposure recovery**（provenance-final与3帧raw-warp smoke已完成，正式派生数据继续阻断）：`frame_failure_recovery_safe_v1`精确复现300视频/6,501失败/221块并记录实现与输出SHA-256。`raw_frame_warp_smoke_v3`在train短块中得到3个`AUTO_PASS_REVIEW_REQUIRED`、1个缺少合法raw锚点的fail-closed；未生成全量overlay，未重跑landmark。
17a. **FACE-S0 temporal capacity audit**（已完成，只作容量下界）：现有审计未量化大遮挡/偏转/出界，不能直接生成训练clip；“2000帧主窗口”和“每视频随机一个窗口”不再冻结。保留17.06% head遗漏和pure-black碎片化结论。
17b. **Exposure-derived OpenFace feature extraction infrastructure**（代码已完成，等待正式materialize）：独立Windows脚本要求`COMPLETE + mirror`、原始/派生全帧相对路径一致、修改帧双端SHA-256、repair/provenance manifest、统一CSV schema；显式提取quality/pose/gaze/2D/3D/PDM/AU，禁用HOG。输出只用于raw-vs-exposure配对审计，AU字段不进入当前模型。正式派生图像树生成后先跑2视频debug，再跑300视频。
17c. **Raw-vs-exposure OpenFace paired audit**（代码已完成，等待17b输出）：逐帧比较success迁移、confidence、2D landmark位移/motion；只有两侧相同rich-feature profile及AU/model哈希时才比较3D/pose/gaze/PDM/AU。当前landmark-only reference不允许借用历史raw-video特征，首轮输出固定为描述性`REVIEW_REQUIRED`。
18. **FACE-T0d raw-vs-repaired paired input ablation**（等待全量raw-warp策略、review gate和正式materialize）：固定同一split/seed/model/40 epoch/precision/checkpoint，对比raw、tone-only、warp-only和完整approved repair；3帧smoke不能作为训练输入。
19. **FACE-S1 phase-1 face-usability distribution audit**（已完成）：不读取AU列、BDI或prediction；300视频/493,141帧全部记账，6,501帧保持既有硬状态，486,640帧标为`pending_threshold_review`。输出split/task分位数、train-only规范脸、11类contact sheets和完整SHA-256 manifest；没有批准`face_usable`。
19a. **FACE-S1 train-only dual-lane threshold review**（AI辅助副本已验证，待人工复签）：124帧/372个决策单元均已在独立`Codex-AI-assisted`副本标为`REVIEWED`，结果为global `111 usable / 13 unusable`、local `92 candidate / 32 ineligible`、temporal `124 no_boundary`、0 uncertain；原始v2 `PENDING`模板保持不变。人工必须复签全部三栏，尤其是13个global reject和32个local ineligible；在此之前不得生成threshold manifest、`face_usable`或具体local crop。
20. **FACE-S2 deterministic segment/clip gate**：从全部连续`face_usable` run生成300/600/1200/2000窗口与0/25/50% overlap统计；冻结主window/stride前必须报告clip数、每video/subject/task分布、重叠和未用tail。禁止跨invalid边界或把clip当独立subject。
21. **FACE-M0 clip training contract**：实现全部合格clip训练，比较`clip loss + 1/N_clips(video)`与同模型video-bag聚合后单次BDI loss；默认关闭时旧dataset行为不变，metrics始终video-level。
22. **LM-M0 AU语义保持的local/global data path**：train-only返回global + brow/eye-cheek/nose-upper-lip/mouth-jaw四个landmark定位的完整RGB crop；所有视图来自同一frame/clip并共享backbone/temporal encoder/BDI head及空间增强。内部polygon只用于crop与coverage审计；禁止AU数值/序列输入、AU loss、区域专属head和隐藏侧伪造。
23. **LM-M1 100-step gradient calibration**：seed42比较L0 global、L1 global+four equal-area grids、L2 global+four AU-semantic landmark-guided RGB crops；匹配view数、总面积和loss scale，记录local/global梯度norm、cosine/conflict/cancellation和external identity risk。局部梯度主导或risk上升`>0.02`时停止。
24. **LM-M2 validation matrix**：先运行S0/S1/S2，再在冻结的最佳时间协议上比较L0/L1/L2。L2不优于L1时只能解释为一般crop augmentation，不继续增加AU语义裁切复杂度；L2获胜也不能声称使用了AU数值或学会逐AU识别。
25. **LM-M3 multi-seed gate**：seed42通过后才运行43/44。任一seed `delta_CCC<-0.05`或`delta_MAE>+0.50`立即停止；协议冻结前test关闭。

#### 实施验收命令

文档或配置改动后至少运行：

```bash
git diff --check -- README.md configs/README.md docs
```

代码改动后优先运行：

```bash
python -m compileall src scripts tests
python -m pytest tests/test_frame_recovery.py tests/test_au_region_tracking.py tests/test_audit_au_coordinate_contract.py
python -m pytest tests/test_mtl_lite_forward.py tests/test_mtl_lite_loss_backward.py
python scripts/train_mtl_lite.py --override configs/mtl_lite_debug_smoke.yaml
```

若改动涉及 Stage A 诊断脚本，额外运行对应单测：

```bash
python -m pytest tests/test_layerwise_identity.py tests/test_layerwise_identity_probe.py
python -m pytest tests/test_error_identity_coupling.py tests/test_artifact_weaklabels.py tests/test_severity_imbalance.py
```

### A. Shortcut 证据收口，优先执行

> 详细脚本/输出规格见 `docs/SHORTCUT_AUDIT_DESIGN.md`。核心约束：A1 只证 "embedding 有身份"，A2 才证 "prediction 可能用了身份"；只有 A1+A2 同时成立才进入强 identity suppression。artifact/context/pose/quality 等因素先作为审计变量，不作为第一版显式 latent。

#### A1 Layer-wise Identity Probe

- [x] A1-1 在 `src/models/mtl_lite.py` 增加诊断用 `register_layer_hooks(layer_names)` 与逐层特征导出 forward（不动训练 `forward`/`compute_losses`/checkpoint）。已新增 `LAYERWISE_PROBE_LAYERS`、`resolve_layer_module`、`register_layer_hooks`/`clear_layer_hooks`/`_collect_layer_features`，`forward` 增 `return_layer_features` 参数；`MTLLiteOutput` 增 `layer_features` 槽位。
- [x] A1-2 设计 layer list：`layer_stem`、`layer_block_0/3/6/9/11`、`layer_backbone_out`、`layer_temporal`、`layer_shared`（对照基线）。服务器输出已覆盖 9 层。
- [x] A1-3 实现 `scripts/audit_layerwise_identity_probe.py`：逐层导出 NPZ，复用 `compute_identity_retrieval_metrics` 检索。新增 `src/diagnostics/layerwise_identity.py`（多键 NPZ 读写 + 逐层检索聚合，纯 numpy 可单测），script 负责 load config/model/data + hook 注册 + batch 收集 + 调用聚合。本地 `pytest tests/test_layerwise_identity.py` 6/6 通过，`--help` 与 import smoke 通过。
- [x] A1-4 输出每层 same-subject top1/top3/top5、paired-task rank、severity/task neighbor agreement。当前未启用可选 subject proxy accuracy，后续如需 multi-attacker 再补。
- [x] A1-5 新增 `tests/test_layerwise_identity.py` 与 `tests/test_layerwise_identity_probe.py`，覆盖 hook 注册、多键 NPZ 读写、逐层检索聚合、layer 排序、训练路径零干扰、无 hook 回退、valid 帧切分、clear。本地 `pytest` 全部通过（6 + 6 = 12 个新测试），既有 36 个诊断测试 + 4 个 MTL-Lite 测试零回归。
- [x] A1-6 服务器对 RGB baseline 跑完 val/test split：每个 split 9 层、100 query、50 subject。

#### A2 Prediction Error x Identity Similarity Coupling

- [x] A2-1 实现 `src/diagnostics/error_identity_coupling.py`：合并 A1 per-query 检索结果 + prediction CSV（residual/abs_error）+ severity group。读 A1 多键 NPZ 重算 paired-task identity similarity（连续），可选 join A1 per_query CSV 的 rank/agreement（离散）。
- [x] A2-2 输出 `error_identity_correlation.csv`、`severity_bin_identity_error_summary.csv`、`high_error_high_identity_cases.csv`、报告。
- [x] A2-3 实现 `scripts/audit_error_identity_coupling.py` 入口。
- [x] A2-4 新增 `tests/test_error_identity_coupling.py`（7 个测试，本地全过）。
- [x] A2-5 明确写出"身份存在"与"身份参与预测"同时成立：A1 中 backbone 中层身份检索强，A2 中 identity/severity 邻域信号与 residual 耦合。

#### A3 Shortcut / Artifact Audit (evaluation only)

- [x] A3-1 实现 `src/diagnostics/artifact_weaklabels.py`：整合 `black_artifacts`/`alignment_geometry`/`openface_quality`/`temporal_sampling` 弱标签。按 normalized video_id join 4 个 summary CSV + prediction CSV，弱标签用 `source:field` 前缀防碰撞。
- [x] A3-2 输出 `artifact_weaklabel_summary.csv`、`artifact_weaklabel_correlation.csv`、`artifact_weaklabel_report.md`。相关性按 |corr(abs_error)| 降序，报告含 artifact/quality 变量定位（error-coupled / label-only confound / 仅审计）。
- [x] A3-3 实现 `scripts/audit_artifact_weaklabels.py` 入口。
- [x] A3-4 新增 `tests/test_artifact_weaklabels.py`（8 个测试，本地全过）。
- [x] A3-5 在 RGB baseline 上运行 A3，并补齐 matched-only correlation。结论：artifact/context/quality 变量只作为 evaluation/probe/case-study/group-wise robustness 维度；不转成 `z_art` 训练分支。

#### A4 Severity Imbalance / Prediction Compression Summary

- [x] A4-1 实现 `src/diagnostics/severity_imbalance.py`：复用 `prediction_run_summary`/`severity_bias_summary`/`severity_calibration_run_summary` join 成单表，计算 bin count/ratio、imbalance_ratio、pred_compression_ratio、minimal/severe residual、calibration delta_ccc。
- [x] A4-2 输出 `severity_imbalance_summary.csv`、`severity_imbalance_report.md`。报告含 stage_b_baseline / stage_d_side_branch / stage_d_optional 三档推荐及判据。
- [x] A4-3 实现 `scripts/summarize_severity_imbalance.py` 入口。
- [x] A4-4 新增 `tests/test_severity_imbalance.py`（7 个测试，本地全过）。
- [x] A4-5 决定 severity-balanced regression 是 Stage B 必跑基线：val/test 均存在 prediction compression、minimal 高估和 severe 低估，post-hoc calibration 不解决 CCC。

#### A0 Stage A 收口

- [x] A0-1 服务器对 RGB baseline 运行 A1-A4，并补齐 matched-only A3。
- [x] A0-2 在 `CURRENT_STATUS.md` 和 `RGB_OVERFITTING_AUDIT_PLAN.md` 写出四个结论：身份存在层级、身份是否参与预测、artifact/quality 是否只作为审计变量、severity-balanced 定位。
- [x] A0-3 正式关闭 Stage A，不再扩展普通输入滤镜、黑边替换、灰度/模糊/mask 族。

### B. Identity-adversarial Task Representation

> 已完成（2026-07-10 B5 收口）。E0-E3 + lambda/power sweep 共 12 run 跑完，结论：E2 弱有效（CCC +0.12、calibration 收益最大、过拟合最轻，作 Stage C 强 baseline 候选），E1/E3 无效（identity risk 未降、E3 utility 全面劣化），identity 泄漏全局未解（fresh attacker + A1 双信号收敛），A3 artifact-risk group 未被任何 defense 改善，12/12 普遍过拟合。详见 `CURRENT_STATUS.md` 2026-07-10 B5 结论。

- [x] B0 固定 Stage B 进入条件和实验边界：E0/E1/E2/E3 四组对照，禁止在 Stage B 混入 `TaskNuisanceBlock`、细粒度 latent、新输入滤镜或 dynamic branch。
- [x] B1 设计并实现最小 identity-adversarial MTL：`shared_features -> z_dep -> BDI`，并用 GRL/attacker 抑制 `z_dep -> subject_id`。
- [x] B1 只使用 train split subject map；identity loss 只在 train stage 启用，val/test 只做诊断。
- [x] B1 不引入 `z_art`、`z_m`、多级递进、learned gate、`TaskNuisanceBlock` 或 dynamic features。
- [x] B2 设计并实现 severity-balanced regression 最小接口：train-only severity-bin 权重、mean-normalize、min/max 截断、默认关闭配置。
- [x] B2 第一版只加权 regression MSE，不加权 CCC loss 或 ordinal auxiliary loss。
- [x] B3 固定对照：`E0 RGB MTL-Lite baseline`、`E1 identity-adversarial MTL`、`E2 severity-balanced regression`、`E3 identity-adversarial + severity-balanced`。
- [x] B4 统一报告 BDI metrics、identity retrieval/probe、severity bias、task consistency、train-val gap、artifact-risk group。
- [x] B5 Stage B 判读：E1/E2/E3 不能在不损伤 CCC/task consistency 的情况下充分缓解 identity risk 与 severity bias → 满足进入 C0 粗粒度信息分流条件。

### C. Coarse Task-Nuisance Information Separation

- [x] C0 冻结项目主张：`z_dep/z_nuisance` 是可审计信息分流假设，不以辅助损失收敛直接宣称语义解耦。
- [x] C0 冻结等参数/等 bottleneck 对照、leakage matrix、fresh multi-attacker、multi-seed 和失败条件（见 `STAGE_C_RUNBOOK.md`）。
- [x] P0 实现配置化 seed、EarlyStopping、runner 测试和同协议 `C-REF` 配置骨架。
- [x] C1 实现 `TaskNuisanceBlock` 最小接口：`H0 -> z_dep, z_nuisance`，默认关闭时无新增 state-dict key。
- [x] C1 预测、ordinal 和未来 identity head 只使用 `z_dep`。
- [x] C1 实现 stop-gradient reconstruction：`H0_recon = Recon([z_dep, z_nuisance])`。
- [x] C1 实现 float32 normalized cross-correlation；batch size 小于 2 返回可反传零值。
- [x] C1 同 checkpoint 导出 `H0/z_dep/z_nuisance`，旧 `features` 继续指向 prediction representation。
- [x] C1 服务器运行 seed 42 `C-REF/C-BN` debug smoke。
- [x] C1 服务器运行 100-step train-only calibration，冻结 `lambda_rec=0.001`、`lambda_xcorr=0.01` 并解除哨兵。
- [x] C0 固定 Stage C 对照：`C-REF/C-BN/C-REC/C-FULL`，第一版不加入 `z_id`。
- [x] C0 不显式划分 `z_art`、`z_ctx`、`z_pose`、`z_quality`；这些变量只用于 post-hoc probe、case study 和 group-wise evaluation。
- [x] C2 seed 42 utility screening：三组候选均触发 severe utility failure，当前结构假设在 utility gate 被否证。
- [x] C2 失败机制只读诊断：实现 `representation_leakage.py`、`group_robustness.py`、两个 CLI 和合成契约测试；已对四组 train/val 导出运行。
- [ ] 补齐 train weak-label summary 后，仅补跑 pose_rz/black-border/face-offset-y group 轴；禁止使用 val 阈值。
- [x] C3 seeds 42/43/44 gate：按停止规则取消，不运行 seed 43/44。
- [x] 运行 C-BN full-40 capacity audit：`DEP_DIM=96/128/160/192` 的训练与 best-val utility 审查已完成；只读 leakage/effective-rank 汇总保持待补。
- [x] 运行 identity-gradient audit 配对矩阵：短期 utility 改善，但 external identity leakage 与末期过拟合恶化；作为负机制消融收口，不授权维度实验。
- [ ] `AU-T0a/T0b` frame + coordinate contract：T0a 已通过；T0b 全量结果为 `260 REVIEW_REQUIRED / 40 FAIL / 0 BLOCKED`，总体 aligned OpenFace success `486640/493141=0.986817<0.995`。保留 `0.995` mapping-valid 和 `0.80` in-bounds 门槛；不得用人工 overlay 或曝光处理掩盖 mapping source detection failure。
- [ ] `LM-T0b` train overlay review：AI辅助签署副本和validator已完成，结果`115 PASS / 5 UNCERTAIN`；等待人工复签。5个UNCERTAIN及原始40个自动FAIL继续阻断当前三个语义区域的静态/动态crop audit。
- [x] `AU-T0c` 逐帧失败/曝光审计：正式清单、连续块、视频 summary、train-only 曝光目标、run manifest 和报告均已生成；未启动 FaceLandmark。
- [x] `AU-T0c` provenance-final：`frame_failure_recovery_safe_v1`完整运行，精确复现`300 videos / 6501 failures / 221 blocks`；run manifest包含core/CLI及全部输出SHA-256，q20/q80与q25/q75协议正确冻结。
- [x] `AU-T0c` 原视频 source contract：300/300 视频 PASS；纯黑 aligned 帧映射为 231 个原视频 run，并生成逐 run contact sheet 与 `source_presence_review_template.csv`；未重跑 FaceLandmark。
- [x] `AU-T0c` `247_3_Freeform` 精确分段：`2078-2143` 人物仍在、`2144-3984` 人物缺席、`3985-4213` 人物重新进入；全视频纯黑帧 gate 为 346 个 `raw_frame_warp_only`、1,841 个 `keep_invalid`。
- [x] `AU-T0c` 全量结构化人工 gate：38 视频/231 run/4,206 帧均被非重叠 REVIEWED segments 完整覆盖；2,365 帧 `raw_frame_warp_only`，1,841 帧 `keep_invalid`，无 mixed/ambiguous 残留。
- [x] `AU-T0c` raw-frame warp smoke：新增`raw-warp-smoke`子命令，只在train且source-presence全覆盖gate下选择最多3帧；v2对3帧自动通过并保留1个无合法raw锚点的拒绝案例。所有输出仍为review-required，不授权全量materialize或训练。
- [x] `AU-T0c` exposure curve family：欠曝冻结为 `log(1+a*x)/log(1+a)`，过曝冻结为 `(exp(b*x)-1)/(exp(b)-1)`；保留旧 gamma 函数仅作历史复现，正式物化不再调用。
- [x] `AU-T0c` exposure safety margin：否决 q10/q90 边缘目标；冻结 train-normal q20/q80=`49.018/142.171` 为处理后安全验收带，q25/q75=`53.740/135.769` 为内部拟合目标。
- [x] `AU-T0c` train-normal temporal threshold：92个train-normal视频全帧扫描完成；冻结luma-IQR q90=`10.243687`为整视频曲线资格上界，q95=`12.024606`仅作极端分层。18候选低于q90、2个q90-q95、2个超过q95；全局IQR不能替代短阶段边界复核。
- [x] `AU-T0c` raw-vs-aligned detail recoverability：92个 train-normal 视频/365个有效参考帧校准，22候选701帧中685帧有效；加入1个百分点最小实际剪切优势后 `raw_detail_recoverable=0`。16视频路由为 `tone_only_or_keep_raw`，6视频为 `inconclusive_keep_raw_until_review`，关闭 raw-space exposure correction 分支。现有 OpenFace 仅用于几何对应，派生输入冻结后统一版本重跑。
- [x] `AU-T0c` 6个 raw-detail 不确定视频复核：`218_1`、`219_1`、`225_2`、`328_1` 整视频 `stable_log`；`219_3` 按 `1-65/66-1170` 两段固定 log；`310_2` 因连续自动曝光变化且无唯一粗粒度切点而 `keep_raw`。拒绝为 `219_1` 的姿态阴影和 `225_2` 的手遮挡建立内容条件分段。
- [x] `AU-T0c` exposure gate：全22视频已由24个无间隙、无重叠REVIEWED segment覆盖。19个`stable_log` segment、2个`tone_only_overexposed`、3个`keep_raw`；`212_1`、`223_1`、`310_2`保持raw。完整manifest位于`logs/au_region_tracking_audit/exposure_review_full_q90_v1/`。
- [x] `AU-T0c` log contact sheets/full-frame checks：全部22候选已按q25/q75目标完成全帧只读评估；所有获批非identity segment中位亮度进入安全带，报告同时保留span retention与暗/亮剪切变化。过曝只声明tone normalization，不声明细节恢复。
- [ ] `AU-T0d` raw/repaired 配对输入消融：同协议比较，不允许把数据恢复与模型/损失改动混在同一实验。
- [ ] `[HISTORICAL-SUPERSEDED] AU-T1/T2`旧四区动态mask审计：只用于解释既有日志；未来三语义区几何任务由上方`GLA-ELIGIBILITY`接管。
- [ ] `[HISTORICAL-SUPERSEDED] AU-M0/M1/M2`旧四区、global-only推理和AU-vs-grid前向实现：不得继续执行。
- [ ] `[HISTORICAL-SUPERSEDED] AU-M3/M4`旧G0/G1/G2 seed42/43/44筛选：已由FULL优先混合消融取代。
- [ ] Stage D 后续扩展：只有基础两出口方案通过 C3 且证据仍支持时，才重新评估可选 `z_id`；不属于 C1-C3 第一版。

### D. Falsification and Robustness Validation

- [ ] D1 multi-attacker：使用 paired-task retrieval、kNN、linear/SVM 或 MLP attacker 报告最强 identity risk，避免单一 attacker 假安全。
- [x] D2 nuisance leakage：train→val BDI ridge gate 未触发，但 identity risk 未下降且 utility 已失败，不构成成功信息分流证据。
- [ ] D3 shortcut/artifact probes：用 artifact/quality/context 变量攻击 `z_dep`，只作为评估，不作为第一版训练监督。
- [ ] D4 severity-balanced 支线：验证 weighted SmoothL1 / MAE 是否改善 minimal/severe bias，并检查是否与粗粒度信息分流互补。
- [ ] D5 group-wise robustness：severity/task 已完成并显示 severity failure；artifact/pose/geometry 轴待 train weak-label 表。
- [ ] D6 learned dynamic-feature 支线仍暂缓：不新增 optical flow、AU-delta encoder、two-stream fusion 或独立动态模型。当前 AU 路线只用逐帧 landmark 动态跟踪输入区域，属于预处理/质量门禁，不等同于该支线。

### 路线细化后的执行闭环

- [x] A1/A2 完成后，明确写出“身份存在”和“身份参与预测”同时成立，允许进入 identity-adversarial baseline。
- [x] A3 完成后，只决定 artifact/quality/context 变量的审计和报告方式，不决定 `z_art` 进入第一版模型。
- [x] A4 完成后，决定 severity-balanced regression 是 Stage B 必跑基线。
- [x] B0 固定 Stage B 对照：`RGB baseline`、`identity-adversarial MTL`、`severity-balanced regression`、`identity-adversarial + severity-balanced`。
- [x] B5 Stage B 收口：12 run 跑完，E2 弱有效、E1/E3 无效、identity 泄漏全局未解，判定进入 Stage C（结论见 `CURRENT_STATUS.md` 2026-07-10）。
- [x] C0 固定 Stage C 对照：`C-REF/C-BN/C-REC/C-FULL`；`C-REF` 按新训练协议复现 E2 结构，第一版不加入 `z_id`。
- [x] P0 固定 seed 与 EarlyStopping，并建立 Stage C 配置骨架和防误跑哨兵。
- [x] C1 代码、服务器 smoke/calibration 和权重冻结均完成。
- [ ] D0 固定统一报告：BDI metrics、severity bias、identity risk、nuisance/BDI leakage、shortcut/artifact probe risk、task consistency、train-val gap。
- [x] 停止规则已冻结：若粗粒度信息分流不优于 paired-seed `C-REF`、multi-seed 不稳定，或风险下降以 utility/group robustness 恶化为代价，停止增加复杂度，转为诊断型贡献。

### 暂缓项

- [ ] 暂缓完整五因子多级门控全开版 RPDF-Net。
- [ ] 暂缓 `z_m` 受控传递、`z_art` 训练分支、两级递进 RPDF 和 learned mutual gate。
- [ ] 暂缓新增 RGB 输入滤镜、黑边替换、灰度、模糊、mask、boundary variants。
- [ ] 暂缓 optical flow、two-stream temporal model 和复杂 dynamic branch。

### 输入遮挡语义修正（2026-07-03 定点例外）

> 这是对历史 `center_mask` 结论的审查修正，不是重新打开普通输入滤镜搜索。

- [x] 审查真实 OpenFace aligned 输入帧，确认历史 `center_mask` 主要保留鼻梁、鼻子、鼻下/上唇附近小区域，不应解释为完整中心脸行为区域。
- [x] 新增 `central_face_mask` 输入变体和配置，用于保留眼、鼻、嘴和主要脸颊区域。
- [x] 新增测试，约束 `center_mask` 与 `central_face_mask` 的面积和关键区域覆盖差异。
- [ ] 服务器运行 `central_face_mask`，并与 `rgb`、历史 `center_mask`、`boundary_erased`、`center_mask_black_to_gray` 统一比较。
- [ ] 汇总报告必须同时包含 BDI metrics、severity bias、identity retrieval、task consistency 和 train-val gap；若 `central_face_mask` 明显弱于历史 `center_mask`，应将历史 `center_mask` 结论降级为极强局部遮挡证据。

## 历史与基础设施任务

以下章节保留架构、诊断工具、历史实验和旧任务队列。若与“当前立即执行任务（权威入口）”冲突，以当前入口为准。

## 架构目标（基础设施记录）

项目采用新架构：

```text
src/legacy/full_model/  # 旧大模型整体归档
src/models/             # MTL-Lite 模型基座与通用模型模块
src/diagnostics/        # 独立诊断与可视化系统
```

核心原则：

- 旧大模型整体迁入 legacy，保留为可运行历史快照；
- legacy 只补 README 和边界说明，暂时不投入额外修复或重构；
- MTL-Lite 独立实现，不继承旧模型；
- 通用模块保留在主线位置；
- 诊断系统独立，不强耦合训练 forward；
- 每次只推进一个重构目标。

## 阶段 1：legacy 归档说明

- [x] 添加 `src/legacy/full_model/README.md`，说明旧模型边界、运行方式和维护策略
- [x] 明确 legacy 是旧大模型快照，不再作为当前开发对象
- [x] 明确旧模型如需运行，应使用 legacy 快照自身的脚本和配置
- [x] 明确禁止提交 legacy 下的 `local_paths.yaml`、日志、权重和 checkpoint
- [x] 暂时放弃 legacy 内部 import 修复、runner 接入和额外重构，除非用户明确要求复现旧模型结果

## 阶段 2：MTL-Lite 模型基座基础接口

- [x] 新增 `src/models/outputs.py`
- [x] 定义 `MTLLiteOutput`
- [x] 定义 `MTLLiteLosses`
- [x] 新增 `src/models/temporal/__init__.py`
- [x] 新增 `src/models/temporal/pooling.py`
- [x] 实现 `masked_mean_pool`

## 阶段 3：MTL-Lite 新模型骨架

- [ ] 设计后续 `src/models/temporal/encoders.py`
- [x] 新增 `src/models/mtl_lite.py`
- [x] 实现 `MTLLiteDepressionModel` 骨架
- [x] MTL-Lite 不继承 legacy full model
- [x] MTL-Lite 只依赖通用 backbone、task heads、metrics 和 temporal utilities
- [x] 不引入 contrastive、CGC、adaptive mask、PCGrad、LDS 或 `loss_dist`

## 阶段 4：MTL-Lite 测试

- [x] 新增 `tests/test_mtl_lite_forward.py`
- [x] 新增 `tests/test_mtl_lite_loss_backward.py`
- [x] 新增 `tests/test_mtl_lite_config.py`
- [x] 使用 dummy backbone，避免下载权重
- [x] 不读取真实数据
- [x] 不依赖 `configs/local_paths.yaml`
- [x] 检查 forward shape
- [x] 检查 regression head 梯度非零
- [x] 检查 loss/metric 尺度一致

## 阶段 5：配置与 baseline

- [x] 新增 `LOSSES` 配置约定
- [x] 新增 regression-only baseline override
- [x] 新增 MTL-Lite baseline override
- [x] 新增 MTL-Lite debug smoke override
- [ ] 保持 `configs/local_paths.yaml` 私有且不提交
- [ ] 服务器运行 MTL-Lite debug smoke

## 阶段 6：诊断系统独立化

- [x] 梳理当前 `src/utils/visualize.py`
- [x] 新增 `src/diagnostics/`
- [x] 新增 `src/diagnostics/regression.py`
- [x] 新增 `src/diagnostics/embeddings.py`
- [ ] 新增 `src/diagnostics/temporal.py`
- [x] 新增 `src/diagnostics/model_attention.py`
- [x] 新增 `src/diagnostics/reports.py`
- [x] 新增 `src/diagnostics/correlation.py`
- [x] 新增 `src/diagnostics/occlusion.py`
- [x] 新增 `src/diagnostics/keyframes.py`
- [x] 支持 `predictions.csv` 导出
- [x] 支持 prediction-target scatter
- [x] 支持 residual histogram
- [x] 支持 BDI 区间误差分析
- [x] 支持 severity group 误差分析
- [x] 支持 high-error / low-error subject ranking
- [x] 支持 t-SNE / UMAP
- [x] 支持 metrics / predictions 相关系数热力图
- [x] 支持 occlusion sensitivity 遮掩影响热力图
- [x] 支持 temporal occlusion 关键帧重要性热力图
- [x] 支持模型自身关注区域热力图（Grad-CAM 可用时优先，否则回退到 input-gradient）
- [ ] 服务器运行 MTL-Lite 离线诊断脚本
- [ ] 保留旧诊断入口的向后兼容
- [ ] 新增 Shortcut Audit 离线诊断入口，不参与训练 forward

## 阶段 7：实验路线

- [ ] 运行 regression-only baseline
- [ ] 运行 MTL-Lite baseline
- [ ] 在相同 split、seed、backbone、时序编码器和指标下比较二者
- [ ] 建立 OpenFace 质量、姿态、gaze、AU 与 BDI/误差的相关性诊断
- [x] 运行第一轮输入消融：aligned RGB、grayscale、blur、center_mask、boundary_erased
- [x] 运行第二轮黑伪迹输入消融：black_to_gray、black_to_mean、black_to_blur、soft_center_mask、inner_crop_resize
- [ ] 建立 landmark-only temporal baseline
- [ ] 建立 AU / pose / gaze-only temporal baseline
- [ ] 建立 RGB + behavior late-fusion baseline
- [ ] 将辅助任务从单纯 BDI ordinal 扩展到 AU、landmark motion、pose/gaze 或 expression 相关行为任务
- [ ] 行为辅助任务稳定后，再考虑 MTL-Lite + CCC loss 消融
- [ ] 行为辅助任务稳定后，再考虑 MTL-Lite + LDS 消融
- [ ] 行为辅助任务稳定后，再考虑 MTL-Lite + `loss_dist` 消融
- [ ] 固定行为表征 baseline 稳定后，再考虑动态任务权重或梯度冲突处理

## 近期验证

- [ ] 添加 regression head 初始化测试，确保不使用训练标签统计量初始化预测参数
- [ ] 验证 `validation_step` 可以在 `no_grad` 下 forward，且不改变 metrics
- [ ] 确认 LDS label weighting 只使用训练集标签，不接触 val/test 标签
- [ ] 验证 `MODEL_WEIGHT_PATH` 能加载 raw `state_dict` 和 `{"state_dict": ...}` checkpoint
- [x] 让 `FREEZE_BACKBONE` / `FINETUNE_LAST_N_BLOCKS` 配置在 MTL-Lite 中真正控制 backbone 可训练范围
- [ ] 验证多 GPU 下 DDP metric logging 和 best-weight 保存行为
- [ ] 添加 bf16-mixed precision 下预测和 loss finite 测试

## 论文工具

- [ ] 标准化实验日志格式
- [ ] 添加将 `metrics.csv` 导出为 LaTeX 表格的脚本
- [ ] 将诊断图表输出组织为论文可用目录结构
- [ ] 维护 `docs/RESEARCH_NOTES.md`，记录 OpenFace、AVEC2014、面部行为建模、多任务学习和捷径学习相关论文

## 历史归档：OpenFace 行为表征研究路线（已由当前E/F路线取代）

以下早期待办仅保留历史，不再作为执行入口；其中gaze、behavior-only baseline、late fusion和宽特征辅助任务与当前no-gaze PB路线冲突，除非重新立项并授权，否则不得继续勾选实施。

- [x] 建立非抑郁捷径验证框架设计文档 `docs/SHORTCUT_AUDIT_DESIGN.md`
- [ ] 确认当前数据使用的 OpenFace 版本、命令、输出字段、裁剪尺寸和帧采样方式
- [ ] 确认是否保留 OpenFace 原始 CSV，并列出可用字段：`confidence`、`success`、pose、gaze、AU、landmark
- [ ] 统计每个视频的 OpenFace `confidence` 均值、方差和低置信帧比例
- [ ] 统计每个视频的 `success` 失败帧比例
- [ ] 统计 pose/gaze 分布和 landmark 抖动
- [ ] 分析 OpenFace 质量变量与 BDI、预测误差、残差的相关性
- [ ] 对高误差 subject 生成 aligned face、attention、occlusion、keyframe case study 图组
- [ ] 验证模型关注区域是否集中于眼、眉、嘴、鼻唇沟，而不是脸部边界、头发、眼镜、黑边或裁剪伪影
- [ ] 建立 `E0_openface_quality_correlation` 实验记录模板
- [ ] 建立 `E1_input_ablation` 实验记录模板
- [ ] 建立 `E2_landmark_temporal_baseline` 实验记录模板
- [ ] 建立 `E3_au_pose_gaze_baseline` 实验记录模板
- [ ] 建立 `E4_rgb_behavior_late_fusion` 实验记录模板

## Shortcut Audit 实施路线

- [ ] 新增 `src/diagnostics/openface_quality.py`，读取 OpenFace CSV 并生成 subject-level quality summary
- [ ] 新增 `src/diagnostics/shortcut_audit.py`，合并 `predictions.csv`、OpenFace quality summary 和 split 信息
- [ ] 新增 `scripts/audit_shortcuts.py`，作为非抑郁捷径验证的离线入口
- [x] 修正 Shortcut Audit 合并键：优先使用完整 `video_id`，避免 Freeform/Northwind 与短 `subject_id` 错配
- [x] 修正 Shortcut Audit 的 `video_id` 规范化：兼容 `*_video` 与 `*_video_aligned`，确保 OpenFace summary 与 prediction CSV 能够按同一视频匹配
- [x] 重新运行 Shortcut Audit，并确认 `shortcut_audit_report.md` 中 `Matched samples` 等于当前预测样本数；若为 0 或明显偏低，不得解释 shortcut risk
- [x] 输出 `openface_quality_summary.csv`
- [x] 输出 `shortcut_correlation.csv`
- [x] 输出 `shortcut_correlation_heatmap.png`
- [x] 输出 residual vs confidence / pose / quality 诊断图
- [x] 输出 `shortcut_audit_report.md`
- [x] 实现 shortcut-only BDI predictor baseline：mean、linear regression、ridge、random forest
- [x] 在 shortcut-only predictor 中优先加入按 `subject_id` 分组的交叉验证，避免同一 subject 的 Freeform/Northwind 泄漏到不同折中
- [x] 将 grouped CV shortcut-only predictor 写入正式诊断输出，至少报告 mean baseline、RGB 模型、ridge 多个 alpha 的 MAE/RMSE/Pearson
- [x] 设计并接入输入消融配置 `DATASET.INPUT_VARIANT`，支持 `rgb`、`grayscale`、`blur`、`center_mask`、`boundary_erased`
- [x] 扩展黑填充/硬边界伪迹输入消融，支持 `black_to_gray`、`black_to_mean`、`black_to_blur`、`soft_center_mask`、`inner_crop_resize`
- [x] 新增 `scripts/audit_black_artifacts.py`，离线统计 aligned frame 中黑像素、黑边界和硬边缘与预测误差的关系
- [ ] 将 `landmark_heatmap` 接入 OpenFace landmark CSV 或 behavior baseline 路径，不在 RGB dataset 中伪造 landmark 输入
- [ ] 设计区域级 attention/occlusion 统计：eye、brow、mouth、face center、boundary、non-face

## 历史实验诊断待办：预测压缩与捷径风险

- [x] 基于最新 `test_predictions.csv` 记录 regression 诊断：整体 MAE 约 8.91、RMSE 约 10.95、Pearson 约 0.35、CCC 约 0.29
- [x] 单独分析 severe 组系统性低估问题：severe 平均真实 BDI 约 34.14，平均预测约 17.64，平均残差约 -16.50
- [x] 单独分析 minimal 组系统性高估问题：minimal 平均真实 BDI 约 4.96，平均预测约 11.85，平均残差约 +6.89
- [x] 增加 Freeform/Northwind 同一 subject 预测一致性诊断，记录任务间预测差异和高差异 case
- [x] 在 Shortcut Audit 匹配修复后，重新判断 OpenFace pose/gaze/AU/quality 特征与 `true_bdi`、`pred_bdi`、`residual`、`abs_error` 的相关性
- [x] 暂不将 OpenFace shortcut 特征直接加入训练输入；应先作为离线审计变量和 behavior-only baseline 对照使用
- [x] 将 severe 组高误差样本整理为 case study 清单，优先检查 `246_1`、`359_1`、`237_1`、`315_2`
- [x] 将 Freeform/Northwind 高差异样本整理为 case study 清单，优先检查 `237_1`、`247_1`、`247_3`、`224_1`、`212_1`
- [x] 建立 AU/pose/gaze/landmark-only behavior baseline 接口，与当前 RGB 模型在相同 split/seed/test checkpoint 下比较

## P0 剩余任务设计：从 shortcut 诊断转向可解释改进

当前 grouped-CV shortcut-only predictor 结果显示，OpenFace shortcut 特征不能单独接近 RGB/MTL-Lite 模型的测试表现；因此后续 P0 任务不应继续只围绕 backbone 微调或 in-sample shortcut 分数，而应优先解释模型为何出现预测范围压缩、severe 系统性低估、minimal 系统性高估，以及同一 subject 在 Freeform/Northwind 之间预测不一致。

- [x] P0-2：建立 high-error / task-inconsistency case study 清单。目标是把 severe 低估、minimal 高估、Freeform/Northwind 高差异和 low-error reference 分成可复查样本集合，为 attention、occlusion、keyframe、aligned face 逐案检查提供固定入口。
- [x] P0-2 输出设计：生成 `case_study_manifest.csv` 与 `case_study_manifest.md`，字段至少包括 `case_type`、`rank`、`video_id`、`subject_id`、`task_name`、`true_bdi`、`pred_bdi`、`residual`、`abs_error`、`severity_group`、`paired_task_pred_bdi`、`task_pred_diff`、`recommended_diagnostics`。
- [x] P0-2 判读重点：优先检查 `246_1`、`359_1`、`237_1`、`315_2` 等 severe 低估 subject，以及 `237_1`、`247_1`、`247_3`、`224_1`、`212_1` 等任务间高差异 subject；同时加入若干 low-error 样本作为对照。
- [x] P0-3：设计输入消融协议，但暂不直接改训练超参数。目标是判断模型是否依赖 RGB 纹理、边界伪影、身份线索、裁剪黑边或局部区域，而不是稳定面部行为动态。
- [x] P0-3 输入变体设计：`rgb` 作为当前 baseline；`grayscale` 弱化颜色线索；`blur` 弱化身份纹理；`center_mask` 保留面部中心；`boundary_erased` 弱化裁剪边界、头发、衣物残留和黑边。
- [x] P0-3 第一轮结果判读：`center_mask` 当前优于 `rgb`，而 `grayscale` 和 `blur` 变差；下一步应优先验证 OpenFace aligned face 的纯黑填充、黑色遮挡块和硬裁剪边界，而不是继续堆叠 late fusion 或新辅助任务。
- [x] P0-3 黑伪迹变体设计：`black_to_gray`、`black_to_mean`、`black_to_blur` 用于替换近黑像素；`soft_center_mask` 用于验证软边界是否优于硬 mask；`inner_crop_resize` 用于验证外围黑边是否为主要捷径。
- [x] P0-3 服务器运行黑伪迹 ablation：保持与 `rgb`、`center_mask` 完全相同 split、seed、checkpoint 选择策略和指标。
- [x] P0-3 汇总 `rgb`、`center_mask`、`boundary_erased` 与五个黑伪迹变体的整体 MAE/RMSE/Pearson/CCC、prediction mean/std、severity group error 和 task consistency。
- [x] P0-3 运行黑伪迹审计，统计 `black_ratio`、`black_border_ratio`、`black_center_ratio`、`black_boundary_edge_ratio` 与 `true_bdi`、`pred_bdi`、`residual`、`abs_error` 的相关性。
- [ ] P0-3 对黑伪迹变体改善和恶化最明显的样本生成 case study 图组，重点检查麦克风黑块、脸部轮廓黑边、裁剪边界和模型关注区域。
- [ ] P0-3 后续补充：`landmark_heatmap` 应由 OpenFace landmark 坐标生成，归入 landmark/behavior baseline 路线，不能在只有 RGB 帧时伪造。
- [x] P0-3 评估约束：所有输入变体必须使用相同 split、seed、checkpoint 选择策略、训练入口和指标；优先记录 MAE、RMSE、Pearson、CCC、prediction mean/std、severe/minimal 分组误差和 Freeform/Northwind 一致性。
- [x] P0-4：设计 AU/pose/gaze/landmark-only behavior baseline 接口。目标是建立不依赖 RGB 纹理的行为表征对照，用来判断当前 RGB 模型是否真正捕捉到可泛化的行为动态。
- [x] P0-4 接口设计：新增 `src/datasets/openface_features.py`、`src/models/behavior_baseline.py`、`src/trainers/behavior_baseline_runner.py`、`scripts/train_behavior_baseline.py` 和 `configs/behavior_baseline.yaml`；输入包含 AU intensity/presence、pose、gaze、landmark、landmark temporal delta、confidence/success mask。
- [x] P0-4 判读方式：如果 behavior-only baseline 接近或超过 RGB/MTL-Lite，说明当前 RGB 输入中有大量可由结构化行为变量解释的有效信号；如果 behavior-only 显著弱于 RGB，但 RGB attribution 不集中在合理面部区域，则继续优先排查非行为捷径。
- [ ] 在服务器使用真实 OpenFace CSV 运行 behavior baseline debug smoke，并与 regression-only RGB baseline 使用相同 split/seed/metrics 对齐比较。

## Codex 任务队列

### Task 1

添加 `src/legacy/full_model/README.md`，完成旧模型归档说明。

### Task 2

新增 MTL-Lite 输出 dataclass 和 mask-aware pooling 工具。

### Task 3

新增 MTL-Lite 模型骨架。

### Task 4

新增 MTL-Lite forward/backward/config 测试。

### Task 5

新增 MTL-Lite baseline 配置和新训练入口。

### Task 6

新增 MTL-Lite 离线诊断与模型表征绘图系统。

### Task 7

整理 OpenFace 行为表征、相关论文和下一阶段实验路线，并将研究计划归档到文档。

### Task 8

构建 Shortcut Audit Framework 的最小可行实现：OpenFace quality summary、预测残差相关性、热力图和 markdown 报告。

## 2026-06-14 行为 baseline 后任务优先级重评估

最新 behavior-only baseline 训练结果显示：OpenFace 结构化特征路线可以在训练集上强拟合，但当前泛化不足。test MAE 约 `9.93`，RMSE 约 `12.86`，CCC 约 `0.151`；best validation RMSE 约 `12.38`，但同一 epoch 的 train RMSE 只有约 `2.74`。因此，下一阶段任务重点应从“直接融合行为特征”调整为“先判断哪些 OpenFace 特征真正可泛化，哪些只是身份或静态几何捷径”。

### P0：必须立即处理

- [x] 为 behavior baseline 导出 val/test prediction CSV，并与 RGB/MTL-Lite prediction schema 对齐。
- [x] 在 behavior prediction 中记录 `video_id`、`subject_id`、`task_name`、`true_bdi`、`pred_bdi`、`residual`、`abs_error`、`severity_group`。
- [x] 为 behavior baseline 增加 `BEHAVIOR_FEATURES.FEATURE_SET` 命名特征组入口，支持后续以最小 override 运行特征组消融。
- [ ] 为 behavior baseline 添加或运行 feature-group ablation：quality-only。
- [ ] 为 behavior baseline 添加或运行 feature-group ablation：AU-only。
- [ ] 为 behavior baseline 添加或运行 feature-group ablation：pose+gaze-only。
- [ ] 为 behavior baseline 添加或运行 feature-group ablation：raw-landmark-only。
- [ ] 为 behavior baseline 添加或运行 feature-group ablation：landmark-delta-only。
- [ ] 为 behavior baseline 添加或运行 feature-group ablation：AU+landmark-delta。
- [ ] 为 behavior baseline 添加或运行 feature-group ablation：all-without-raw-landmarks。
- [ ] 对齐比较 RGB/MTL-Lite 与 behavior-only 的整体 MAE/RMSE/Pearson/CCC。
- [ ] 对齐比较 RGB/MTL-Lite 与 behavior-only 的 severe 低估、minimal 高估和 Freeform/Northwind task consistency。
- [ ] 分析 RGB 错而 behavior 对、behavior 错而 RGB 对、二者同时错误、二者同时正确的 case overlap。
- [x] 新增 RGB/MTL-Lite 与 behavior-only prediction CSV 离线比较入口，输出逐样本对照和整体/severity summary。
- [ ] 在训练日志或诊断报告中记录 OpenFace CSV 匹配数、可用字段、特征维度、缺失字段和训练集标准化统计来源。

### P1：强烈建议处理

- [ ] 在 feature-group ablation 后重新决定 behavior baseline 默认特征组，暂不默认相信 raw landmark 坐标。
- [ ] 尝试更小 behavior baseline 容量，例如减小 hidden dim、使用单向 GRU、增加 dropout 或 weight decay。
- [ ] 为 behavior baseline 引入更严格 early stopping，避免 train RMSE 继续下降但 val/test 不改善。
- [ ] 将 behavior prediction CSV 接入现有 regression diagnostics 和 case study manifest。
- [ ] 将 behavior feature ablation 结果整理为 `behavior_feature_ablation_results.csv`，便于论文表格化。

### P2：后续优化

- [ ] 在存在稳定可泛化 behavior 特征子集后，再设计 RGB + behavior late fusion。
- [ ] 历史暂缓：AU、landmark motion、pose/gaze辅助任务MTL不属于当前路线；当前只使用AU语义保持的RGB crop。
- [ ] 在辅助任务稳定后，再考虑 GradNorm、PCGrad、uncertainty weighting、LDS 或 `loss_dist` 消融。

## 2026-06-15 RGB 黑填充伪迹任务队列

本节为已完成或基本完成的 input artifact 子证据队列。黑边/黑填充是 RGB 过拟合的可能原因之一，但不是唯一主因。该阶段曾将路线迁移到 `docs/RGB_OVERFITTING_AUDIT_PLAN.md` 中定义的多因素过拟合审计；当前这些结果作为 task-nuisance 主线的证据层保留，后续不再优先继续堆叠相似 RGB mask 变体。

### P0：立即执行

- [x] 将黑填充/硬边界伪迹假设写入项目文档。
- [x] 接入黑伪迹输入变体配置和实现。
- [x] 接入黑伪迹离线审计脚本。
- [ ] 在服务器运行 `python -m pytest tests/test_input_variants.py`。
- [x] 使用相同命令模板运行五个黑伪迹 ablation：
  - `configs/input_ablation/black_to_gray.yaml`
  - `configs/input_ablation/black_to_mean.yaml`
  - `configs/input_ablation/black_to_blur.yaml`
  - `configs/input_ablation/soft_center_mask.yaml`
  - `configs/input_ablation/inner_crop_resize.yaml`
- [x] 对五组新实验全部运行 `scripts/diagnose_mtl_lite.py --enable-regression`，生成 `test_predictions.csv`、case study manifest 和回归诊断图。
- [x] 运行 `scripts/audit_black_artifacts.py`，至少先对原始 `rgb` 预测进行审计。

### P1：完成第一轮证据闭环

- [x] 建立人工分析 summary，汇总 `rgb`、`center_mask`、`boundary_erased` 和五个新变体。
- [x] 统计黑伪迹审计指标与误差之间的相关性，并完成判读。
- [x] 修正中心黑像素判读：中心近黑区域可能来自鼻孔、自然阴影、胡须、嘴角或麦克风遮挡，不能直接视为 OpenFace 伪迹。
- [x] 初步确认：黑边是泛化风险因子之一，但不是单独强解释变量。
- [ ] 将本轮 ablation 和黑伪迹审计整理成论文表格草稿。
- [ ] 对 severe 低估仍不改善的情况，继续保留 severity-aware loss/sampling、标签分布和 subject-level bias 作为后续独立问题。

### P1.5：下一轮精确边界黑区实验

- [x] 实现 `border_black_to_gray`：只替换与图像边界连通的近黑区域，不处理中心近黑像素。
- [x] 实现 `border_black_feather`：只对边界连通黑区做软过渡或 feather，降低硬边界突变。
- [x] 实现 `center_mask_black_to_gray`：在 `center_mask` 基础上处理残留边界连通黑区，验证二者是否互补。
- [x] 为上述三个变体新增 `configs/input_ablation/*.yaml`。
- [x] 为边界连通黑区 mask 增加单元测试，确保鼻孔、嘴角和麦克风等中心黑块不会被默认替换。
- [x] 本地完成 compile 验证：`src/datasets/input_variants.py` 与 `tests/test_input_variants.py` 语法检查通过。
- [ ] 在服务器运行 `python -m pytest tests/test_input_variants.py`。
- [x] 在相同 split、seed、训练入口、checkpoint 策略下运行三组新 ablation。
- [x] 将三组新结果与 `rgb`、`center_mask`、`black_to_gray`、`soft_center_mask` 统一比较。

### P1.6：case study 复核

- [ ] 高黑边高误差 case：`359_1`、`315_2`、`245_1`。
- [ ] 高黑边低误差 case：`247_3`。
- [ ] 低黑边高误差 case：`237_1`。
- [ ] `black_to_gray` 改善明显 case：`250_1`、`344_2`、`242_1`。
- [ ] `black_to_gray` 恶化明显 case：`206_2`、`226_2`、`210_2`。
- [ ] 对上述 case 生成 aligned frame montage、attention、spatial occlusion、keyframe 图组。
- [ ] 比较模型关注区域是否落在边界黑区、麦克风遮挡、鼻孔/嘴部自然暗区或真实面部行为区域。

### P2：暂缓

- [ ] RGB + behavior late fusion。
- [ ] 历史暂缓：AU / landmark / pose / gaze辅助任务MTL，未获当前路线授权。
- [ ] 动态任务权重、PCGrad、GradNorm、LDS 或 `loss_dist`。

## 2026-06-15 RGB 过拟合多因素审计队列
系统机制路线图已新增：`docs/OVERFITTING_MECHANISM_ROADMAP.md`。后续 P0/P1/P2 任务应优先对齐该文档中的 Layer 0-6 和实验决策树，确保每个实验回答一个明确机制问题。

该历史阶段的权威路线见 `docs/RGB_OVERFITTING_AUDIT_PLAN.md`。核心结论：黑边/黑填充是 RGB 过拟合的可见风险入口，但不是单一充分解释。该阶段把 RGB 过拟合拆成多因素审计，逐项验证 split/subject integrity、时序采样、训练曲线泛化缺口、OpenFace 对齐几何、身份静态外观、姿态/追踪质量、任务语境和 severity prediction compression；当前 task-nuisance 路线继承这些审计作为证据层。

### P0：已接入的 temporal / prediction summary 能力

- [x] 视频长度与采样审计：统计 `frame_count`、`sampled_frame_count`、`valid_ratio`、`padding_ratio` 与 `true_bdi`、`pred_bdi`、`residual`、`abs_error` 的关系。
- [x] 新增 `src/diagnostics/temporal_sampling.py` 与 `scripts/audit_temporal_sampling.py`。
- [x] 新增 `tests/test_temporal_sampling_audit.py`。
- [x] 本地完成 compile 和 direct smoke 验证。
- [ ] 在服务器运行 `python -m pytest tests/test_temporal_sampling_audit.py`。
- [x] 在真实 RGB/MTL-Lite prediction CSV 和 aligned frame root 上运行 `scripts/audit_temporal_sampling.py`。
- [x] 实现固定帧数采样策略：256 / 512 / 1024 uniform frames。
- [x] 实现 temporal crop 策略：first / middle / random。
- [x] 新增 temporal sampling override 配置：
  `configs/temporal_sampling/uniform_256.yaml`、
  `uniform_512.yaml`、`uniform_1024.yaml`、
  `first_crop.yaml`、`middle_crop.yaml`、`random_crop.yaml`。
- [x] 新增 `src/datasets/temporal_sampling.py`，并在 `AVECDataset` 中接入 `PROCESS_TEMPORAL.SAMPLING_STRATEGY`。
- [x] 新增 `tests/test_temporal_sampling.py`。
- [ ] 在服务器运行 `python -m pytest tests/test_temporal_sampling.py tests/test_temporal_sampling_audit.py`。
- [ ] 运行 256 / 512 / 1024 uniform frame 三组训练消融。
- [ ] 运行 first / middle / random temporal crop 三组训练消融。
- [x] 输出 `temporal_sampling_audit_report.md`。
- [x] 新增 prediction run summary 工具，自动记录 `true_mean/std`、`pred_mean/std`、severity group bias、task consistency 和 pairwise baseline improvement。
- [x] 新增 `src/diagnostics/prediction_runs.py`、`scripts/summarize_prediction_runs.py`、`tests/test_prediction_runs.py`。
- [ ] 在服务器运行 `python -m pytest tests/test_prediction_runs.py`。
- [x] 对已有 RGB input ablation 全部运行 `scripts/summarize_prediction_runs.py`，生成统一论文表格底稿。
- [x] 当前统一汇总包含 `rgb`、`gray_scale`、`blur`、`boundary_erased`、`center_mask`、`black_to_gray`、`black_to_mean`、`black_to_blur`、`soft_center_mask`、`inner_crop_resize`、`border_black_feather`、`border_black_to_gray` 和 `center_mask_black_to_gray`。
- [ ] 将统一汇总表整理为论文正文/附录表格，明确区分 input artifact mitigation 与 severity calibration。

当前 temporal sampling audit 真实运行结论：

- [x] `100/100` 个 RGB test prediction row 成功匹配，`Missing videos = 0`。
- [x] 最大绝对相关约 `0.2197`，主要来自 temporal/truncation 指标与 `pred_bdi` 的关系。
- [x] `truncated_frame_count` 与 `pred_bdi` 约 `r = -0.2197`，`truncated_ratio` 与 `pred_bdi` 约 `r = -0.2090`，`frame_count` / `sampled_frame_count` 与 `pred_bdi` 约 `r = -0.2073`。
- [x] high frame-count quartile 预测更低、误差更高且无 padding，说明后续应优先验证截断和采样覆盖，而不是只处理 padding。
- [x] 将 temporal sampling 消融训练结果接入 `scripts/summarize_prediction_runs.py`，统一报告 prediction std、severity bias、task consistency 和 pairwise improvement。

- [x] 完成 temporal sampling 六组训练消融结果分析：`middle_crop` 整体最优但恶化 task consistency，`uniform_256/512/1024` 基本等价，`first_crop` 与 `random_crop` 不适合作为主线替代。
- [x] 完成 temporal training overfit summary：所有 temporal run 均为 `overfit_after_best_val=True`，采样替换不能解决训练后期记忆问题。

### P0：当前高优先级过拟合验证

当前审查结论：不建议继续优先堆叠新的 RGB mask 变体。黑边/黑填充方向已经形成阶段性证据闭环，后续更高价值的问题是验证 split/subject、时序采样、训练曲线过拟合缺口、OpenFace 对齐几何、embedding 身份信息、severity calibration 和 task inconsistency mixed factors。

- [x] P0-A split / subject 泄漏审计实现：确认 train/val/test subject-disjoint，同一 subject 的 Freeform/Northwind 不跨 split，且不存在重复视频目录、重复标签或 video_id 规范化错配。
- [x] P0-A 新增 `src/diagnostics/split_integrity.py`、`scripts/audit_split_integrity.py`、`tests/test_split_integrity.py`。
- [x] P0-A 输出 `split_integrity_report.md`、`split_video_manifest.csv`、`split_subject_overlap.csv`、`split_label_distribution.csv`，可选输出 `split_prediction_alignment.csv`。
- [x] 本地完成 P0-A compile 和 direct smoke 验证；本地 Python 缺少 `pytest`。
- [ ] 在服务器运行 `python -m pytest tests/test_split_integrity.py`。
- [ ] 在真实 split、label、aligned image root 和当前 `test_predictions.csv` 上运行 `scripts/audit_split_integrity.py`。
- [x] P0-B 训练曲线过拟合审计实现：跨 run 汇总 best epoch、train/val RMSE gap、train/val MAE gap、val 最优后是否继续过拟合。
- [x] P0-B 新增 `src/diagnostics/training_overfit.py`、`scripts/summarize_training_overfit.py`、`tests/test_training_overfit.py`。
- [x] P0-B 输出 `training_overfit_summary.csv`、`training_curve_gap_by_run.csv`、`training_overfit_report.md`。
- [x] 本地完成 P0-B compile 和 direct smoke 验证；本地 Python 缺少 `pytest`。
- [ ] 在服务器运行 `python -m pytest tests/test_training_overfit.py`。
- [ ] 对 RGB、center_mask、center_mask_black_to_gray、border_black_feather、behavior baseline 的真实 `metrics.csv` 运行 `scripts/summarize_training_overfit.py`。
- [x] P0-C OpenFace 对齐几何审计实现：统计 landmark bbox area/width/height/aspect、face center offset、eye distance、face scale、landmark jitter，并与 `true_bdi`、`pred_bdi`、`residual`、`abs_error` 相关。
- [x] P0-C 新增 `src/diagnostics/alignment_geometry.py`、`scripts/audit_alignment_geometry.py`、`tests/test_alignment_geometry.py`。
- [x] P0-C 输出 `alignment_geometry_summary.csv`、`alignment_geometry_merged.csv`、`alignment_geometry_correlation.csv`、`alignment_geometry_group_summary.csv`、`alignment_geometry_audit_report.md`。
- [x] 本地完成 P0-C compile 和 direct smoke 验证；本地 Python 缺少 `pytest`。
- [x] 在真实 OpenFace CSV root 和当前 `test_predictions.csv` 上运行 `scripts/audit_alignment_geometry.py`，匹配 100/100 test predictions。
- [x] 确认当前 OpenFace CSV landmark 坐标不是 112x112 aligned frame 坐标，而是原始 OpenFace 检测坐标系；示例 `x` 范围约 150-643，`y` 范围约 -11-582，而 aligned jpg 为 112x112。
- [x] 根据 OpenFace camera parameters `500,500,320,240` 推断源坐标系约 640x480，并使用 `--frame-width 640 --frame-height 480` 重跑 geometry audit。
- [ ] 在服务器运行 `python -m pytest tests/test_alignment_geometry.py`。
- [ ] 后续增强 geometry 审计：显式输出 `landmark_x_min/x_max/y_min/y_max`、`eye_distance_to_bbox_height_ratio` 等不依赖固定 frame size 的相对几何指标。
- [x] P0-D embedding 身份信息审计：已对 `rgb`、`center_mask`、`center_mask_black_to_gray`、`border_black_feather`、`middle_crop` 的 test/val 输出运行 paired-task retrieval。
- [x] P0-D 输出 `embedding_identity_retrieval.csv`、`embedding_identity_report.md`，报告 same-subject top-k retrieval、paired-task rank、severity neighbor agreement 与 task neighbor agreement。

- [x] 构建 `scripts/summarize_identity_retrieval_runs.py`，用于汇总多个 identity retrieval 输出目录。
- [x] 新增 `src/diagnostics/identity_retrieval_runs.py`，输出 `identity_retrieval_run_summary.csv`、`identity_retrieval_severity_summary.csv` 和 `identity_retrieval_runs_report.md`。
- [x] 为 identity retrieval multi-run summary 添加聚焦测试，覆盖 summary CSV、severity group summary 和 report 输出。
- [x] 将 identity retrieval summary 与 prediction summary 合并为论文核心表，至少包含 `MAE/RMSE/CCC/pred_std/severe_bias/task_diff/same_subject_top1/same_subject_top5/severity_agree/paired_rank_mean`。已由 `src/diagnostics/mechanism_summary.py`（P0-G.1 机制总表）落地，join prediction/severity_bias/task_consistency/identity/calibration 五表。
- [ ] 生成 high-identity / high-error case study，重点检查 `border_black_feather` severe 高身份样本、`middle_crop` 身份下降但 task diff 上升样本、moderate identity retrieval 失败样本。
- [x] P0-E severity calibration 验证：已对 RGB baseline 使用 val predictions 拟合 post-hoc linear calibration，并应用到 test，检查 severe 低估和 minimal 高估是否缓解。
- [x] P0-E 输出 `severity_calibration_report.md`、`severity_calibration_fit.csv`、`severity_calibration_test_summary.csv`，并明确该实验只用于验证 prediction compression，不作为最终模型调参结论。
- [x] 构建 `scripts/summarize_severity_calibration_runs.py`，汇总多个 severity calibration 输出目录。
- [x] 新增 `src/diagnostics/severity_calibration_runs.py`，输出 `severity_calibration_run_summary.csv`、`severity_calibration_group_bias_summary.csv` 和 `severity_calibration_runs_report.md`。
- [x] 对 `rgb`、`middle_crop`、`border_black_feather`、`center_mask`、`center_mask_black_to_gray` 运行统一 severity calibration summary。
- [x] 已联合审阅 severity calibration summary、prediction summary 和 identity retrieval summary，完成当前机制结论整理。
- [x] 正式机制总表已落地：新增 `src/diagnostics/mechanism_summary.py`、`scripts/summarize_mechanism.py`、`tests/test_mechanism_summary.py`（P0-G.1），join prediction/severity_bias/task_consistency/identity/calibration 五表为 `mechanism_summary.csv` 与 `mechanism_report.md`。
- [ ] 设计 severity-aware training ablation：severity-balanced sampler、severity-weighted regression loss、ordinal severity auxiliary head、Huber/CCC/mixed loss；所有结果必须同时报告 identity retrieval 与 task consistency。
- [ ] P0-F task inconsistency 混杂审计：将 Freeform/Northwind 差异与 frame_count、black-border、confidence、pose/gaze、alignment geometry 相关联。
- [ ] P0-F 输出 `task_artifact_correlation.csv` 与 `task_inconsistency_manifest.csv`。

执行顺序建议：

```text
split integrity audit
-> temporal sampling audit / ablation
-> training overfit curve summary
-> alignment geometry audit
-> embedding identity retrieval
-> severity calibration
-> task inconsistency mixed-factor audit
```

完成 P0 后，再考虑 P1 的区域输入变体、patch/attention case study 和 targeted robustness augmentation。

### P1：输入 artifact、几何与质量审计

#### OpenFace 边界硬突变和平滑过渡消融

- [x] 设计 `edge_soften_only`，只降低边界连通黑区与脸部交界处的高梯度，不改变大面积黑区。
- [x] 设计 `border_blur_fill`，用邻近非黑区域的模糊颜色填充边界连通黑区。
- [ ] 在服务器运行 `python -m pytest tests/test_input_variants.py`。
- [ ] 在相同 split、seed、训练入口、checkpoint 策略下运行 `configs/input_ablation/edge_soften_only.yaml` 和 `configs/input_ablation/border_blur_fill.yaml`。
- [ ] 将新结果与 `rgb`、`center_mask`、`black_to_gray`、`border_black_feather`、`center_mask_black_to_gray` 统一比较。
- [ ] 暂缓 `uniform_2048` 或更多普通 temporal crop，优先回答边界高对比突变是否是输入 artifact 子机制。
- [ ] 将新边界平滑变体与 `rgb`、`center_mask`、`black_to_gray`、`border_black_feather`、`center_mask_black_to_gray` 统一比较。
- [ ] 同时报告 overall metrics、prediction std、severity bias、task consistency 和 pairwise improvement，避免只按 MAE 选择。

- [ ] 若 P0-C 已完成，则本节转为扩展分析：增加插值模糊、亮度/对比度、边界 patch 统计和脸部轮廓残留。
- [ ] 姿态/追踪质量审计：统计 confidence、success、pose、gaze、landmark jitter。
- [ ] 检查 severe 低估是否集中在低 confidence、大姿态、高 jitter 或异常 face scale 样本。

### P1：身份与静态外观审计

- [ ] 设计 `face_contour_erased` 输入变体，弱化脸型、发际线和轮廓捷径。
- [ ] 设计 `eye_mouth_only` 或 `upper_lower_face` 区域变体，验证有效信号是否集中于行为区域。
- [ ] 建立眼镜、麦克风、胡须等局部遮挡/饰物 case list，优先从 severe 低估、minimal 高估、高 task diff 和 temporal middle_crop 改善/恶化样本中筛选。
- [ ] 设计 `glasses_region_erased`、`mouth_occluder_erased`、`beard_lower_face_erased` 等区域消融或手工 case-study 遮挡，用于判断局部 occlusion shortcut 风险。
- [ ] 对眼镜反光、麦克风黑块、胡须/下半脸纹理区域运行 spatial occlusion / attention 复核，比较预测变化是否符合真实 BDI。
- [ ] 在 P0-D paired-task retrieval 完成后，再决定是否训练 frozen RGB embedding subject proxy classifier。
- [ ] 对 embedding 做 subject-level 聚类或可视化，检查是否按 subject/外观而非 BDI 聚类。

### P1：任务语境审计

- [ ] 分 Freeform / Northwind 报告整体指标和 severity bias。
- [ ] 生成 task inconsistency manifest。
- [ ] 检查 task inconsistency 是否与 frame_count、pose/gaze、black-border、confidence 或 face scale 相关。

### P2：校准与损失实验

- [ ] 在输入捷径审计之后，再单独测试 severity-balanced sampler。
- [ ] 在输入捷径审计之后，再单独测试 weighted MSE / Huber / CCC loss。
- [ ] 输出 severity calibration report，避免把整体抬高预测误判为真正泛化提升。

### 历史阶段：Shortcut-Regularized MTL 实验规划（已被粗粒度主线继承）

本节保留为历史任务记录。当前正式路线已经调整为可审计、可证伪的粗粒度 task-nuisance 信息分流；本节中的 severity-balanced regression 与 identity-adversarial branch 应作为当前主线的对照基线和支线验证，而不是最终主线。

#### Stage A：收口诊断

- [ ] A1 实现 layer-wise identity probe：提取不同 backbone / temporal / shared representation 层的 embedding，输出 same-subject top1/top5、paired rank、subject proxy accuracy 和 severity agreement。
- [ ] A1 对 RGB baseline 至少运行 test split；若资源允许，补 val split 用于稳定性对照。
- [ ] A2 实现 prediction error x identity similarity coupling：检查 residual / abs_error 与 identity similarity、paired-task rank、severity neighbor agreement 的关系。
- [ ] A2 输出 high-error-high-identity case list，用于论文 case-study anchor。
- [ ] 完成 A1/A2 后关闭 Stage A，不继续扩展普通输入滤镜、黑边替换、灰度/模糊/mask 族。

#### Stage B：正式干预实验

- [ ] B1 实现 Gradient Reversal Layer 和 subject identity adversarial head，接入 MTL-Lite shared representation。
- [ ] B1 配置 `lambda_id=0.02,0.05,0.10,0.20` sweep。
- [ ] B1 运行 E1：`RGB baseline + identity-adversarial branch`。
- [ ] B2 实现 severity-balanced regression loss，支持 severity bin count / smoothed count 权重。
- [ ] B2 配置 `p=0.5` 与 `p=1.0` 两组起始实验；分箱先使用 minimal / mild / moderate / severe。
- [ ] B2 运行 E2：`RGB baseline + severity-balanced regression`。
- [ ] B3 选择最稳的 severity weight 与 identity lambda，运行 E3：`severity-balanced regression + identity-adversarial branch`。
- [ ] 对 E0/E1/E2/E3 统一运行 prediction summary、identity retrieval summary、severity calibration summary、training overfit summary 和 task consistency summary。
- [ ] 将 E0/E1/E2/E3 汇总为机制对照表：`MAE/RMSE/CCC/pred_std/severe_bias/task_diff/same_subject_top1/top5/severity_agree/train-val gap`。

#### Stage C：待考虑，暂不执行

- [ ] 暂缓 feature delta / RGB frame delta / AU delta / landmark-pose-gaze delta / static-dynamic fusion。
- [ ] 仅当 Stage B 后仍存在明显 static appearance shortcut，或 E1/E2/E3 无法改善 severity representation 时，再启动动态特征方案设计。
- [ ] 暂缓 optical flow、two-stream temporal model 和复杂动态辅助 MTL，避免当前阶段变成多模块堆叠。

#### 判读规则

- [ ] E1 只有在 identity retrieval / subject proxy accuracy 下降且 BDI 指标不崩坏时，才算 identity-adversarial 成功。
- [ ] E2 只有在少数 severity 分段 MAE/bias 改善且 CCC、task consistency、identity retrieval 不明显恶化时，才算 severity imbalance mitigation 成功。
- [ ] E3 只有在同时满足 E1/E2 的核心约束，并且 train-val gap 或 task consistency 有改善时，才作为该历史阶段的候选方案。

### 文献映射后的身份抑制任务

- [ ] 将输入级 `identity_texture_suppressed` / boundary smoothing 结果作为机制证据保留，不继续扩展为当前主线。
- [ ] 优先实现 subject-adversarial GRL，并明确普通 subject classification auxiliary head 会强化身份信息，不能作为去身份方案。
- [ ] 为 identity-adversarial 实验报告 same_subject_top1/top5、subject proxy accuracy、severity_agree、CCC、pred_std、severe bias、task_diff 和 train-val gap。
- [ ] 保留 local accessory / contour case-study occlusion 作为解释性分析，用于查看眼镜、麦克风、胡须、发际线等是否参与 high-error / high-identity case。
- [ ] 将 full generative de-identification / complete disentanglement 保留为论文讨论或远期方案，不作为当前 P0/P1 主线。
