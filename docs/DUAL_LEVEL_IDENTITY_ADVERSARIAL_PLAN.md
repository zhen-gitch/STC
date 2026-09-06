# DUAL_LEVEL_IDENTITY_ADVERSARIAL_PLAN.md

> 文档职责：定义"底层backbone/projector + 帧级身份对抗（T1）+ 抑郁时序主任务（T2）+ 时序层身份对抗（T3）"双层级身份控制新主线的设计契约、实验矩阵、预注册门禁与停止条件。本文档由用户于2026-09-04明确决策引入并取代`GLOBAL_LOCAL_AU_EXPERIMENT_PLAN.md`（PLAN-20260805，GLA-RGB/region-first）的执行权；GLA-RGB与OpenFace3源路线内容降级为历史/候选输入源背景，不再是执行入口。

## 0. 路线决策记录（PLAN-20260904-DUAL-LEVEL-IDENTITY-GRL-v1）

计划ID：`PLAN-20260904-DUAL-LEVEL-IDENTITY-GRL-v1`。本包为docs-only：未授权代码、配置、数据生成、smoke、训练、commit或push；训练授权继续为`false`。

### 0.1 用户决策（2026-09-04）

1. 身份头角色：**对抗抑制**（GRL梯度反转），不是监控探针，不是鼓励识别的辅助监督。
2. 抑制层级：主候选为**T1（帧级）+ T3（时序级）同时压制**；单层级只作为预注册的减法归因对照。
3. 路线关系：**替代GLA-RGB成为新主线**；`OF3-SOURCE-*`、`OF3-REGION-*`、`REGION-P0C/P0D`不再阻塞本路线，OpenFace3正式源降级为"未来输入源升级"独立候选。
4. 本包范围：仅本文档。入口文档（README/DOCS_GUIDE/TODO/CURRENT_STATUS）的权威指针同步是**下一个独立docs包**，未获授权前上述入口仍保留旧路线文字，存在已知入口滞后，以本文档为未来路线唯一权威。

### 0.2 数据不变量与输入源决策（rev.2，2026-09-04修订）

AVEC2014原视频、`dataset_split.json`、BDI标签文件与train/val/test角色不可变；不修改split、不删视频、不按val/test指标回调任何阈值；subject身份仅从`video_id`前缀派生且类别表只从physical train split构建。

输入源决策修订（用户2026-09-04，rev.2，替代原"输入源首版继续为legacy aligned JPG"条款）：本路线主候选的执行输入升级为OpenFace3.0正式产物——对齐RGB（`retinaface_5pt_arcface112_masked_v1`，112x112）经冻结`eva02_small_patch14_224.mim_in22k`（timm）提取帧级特征，与206维行为特征（schema `openface3_wflw98xy_gaze2_anonymous_mtl8_v1`）组成双流输入；帧预算512均匀降采样（策略`uniform512_v1`）。实施契约钉死于`EVA_DI_VALIDATION_PLAN.md`（`PLAN-EVA-DI-VALIDATE-v1`），代码落STC仓库`src/eva_di/`模块；trans仓库（RIB-Former）`AGENTS.md`§2明文将身份对抗方向排除出其研究范围，故对其资产仅限只读复用（权重/对齐图/`features.csv`/train-only归一化统计/参考实现）。legacy `112x112` aligned JPG输入（`DATASET.IMAGE_SOURCE`默认值不变）与既有MTL管线原样保留，仅继续服务`DI-REF`类历史锚点，不再作为新架构主候选的输入源。本文档§1–§7机制条款（H1/H2、张量契约、损失与warmup、实验矩阵、门禁与停止条件）不因本修订改变，由`eva_di`模块逐字继承。

### 0.3 工作区既有未提交内容处置

工作树中未提交的OF3合同代码与5份文档修改不删除、不提交、不继续扩展，冻结为"OpenFace3/region路线历史证据"；其提交或回滚策略另行披露。本路线任何代码包不得静默复用其未commit锚点。

## 1. 研究问题与可证伪主张

主线问题：**抑郁模型的身份捷径能否在空间表征源头被抑制，同时保持BDI效用？**

- `H1`：在帧级projected特征上加GRL身份对抗（T1），能降低**外部**度量（fresh LOVO Ridge attacker、identity pair AUROC、A1逐层same-subject检索）的帧级与时序级身份含量，且val BDI效用不劣于matched no-GRL参照。
- `H2`（先验负证据）：仅在pool后`shared_features`加GRL（即Stage B E1机制复现）不足以降低外部身份风险——已由E1/E3负结果与identity-gradient audit（外部pair AUROC 0.9308→0.9447、冲突率60.2%）确立；本路线把它降为T3归因对照，不重复其"单层压制即充分"的主张。
- 主张边界：GRL收敛与训练内attacker指标**不构成**身份去除证明；一切身份结论只认外部fresh attacker与检索审计。不主张"语义解耦"；只主张可测量的身份泄漏下降与其效用代价。
- 已知混杂（预注册声明）：AVEC中subject≈recording session，身份抑制可能等价于session特征抑制，其跨集泛化后果无法在本数据集内完全分离；论文限制声明必须列明。

## 2. 架构与张量契约

```text
video [B,T,3,112,112] + frame_mask [B,T] (True=valid, 前缀1后缀0) + subject_id [B]

backbone (chunked, valid帧only)          -> frame_raw  [B,T,D_backbone]
projector (Linear+LN+GELU)               -> frame_proj [B,T,192]   ← 出口P（帧级）
T1: frame_proj[valid] -> GRL(λ1) -> subject_head_frame -> [B*Tv, N_subj_train]
temporal GRU(1层, 前缀mask不变量断言)     -> frame_enc  [B,T,192]
masked temporal mean                      -> h0         [B,192]     ← 出口V（时序级）
T2: h0 -> reg head -> bdi_pred [B] (归一化尺度, 指标域clamp回[0,63])
T3: h0 -> GRL(λ3) -> subject_head_temporal -> [B, N_subj_train]
```

- 出口P/出口V是身份信息的两个**测量点**，与A1逐层审计的`layer_backbone_out`/`layer_shared`对应；A1的`layer_temporal`语义缺陷（与`layer_shared`同张量、非pool前GRU输出）必须在任何A1证据被引用前修复（见§5）。
- T1 loss只在`frame_mask=True`帧上计算；padding帧不进CE。T3沿用既有train-only subject表与stage门控，且**按样本mask**（batch内unmapped subject只剔除该样本，不整批置零）。
- identity head前向必须门控`self.training`；懒建head在注入后立即`.to(device)`。
- 新增配置键（默认全关、与旧checkpoint/state_dict兼容）：
  `MODEL.IDENTITY_ADVERSARIAL.FRAME_LEVEL.{ENABLE, WEIGHT, WARMUP_EPOCHS}`（T1），复用`MODEL.IDENTITY_ADVERSARIAL.{ENABLE, LAMBDA_ID}`作为T3；两λ共享同一warmup调度。

## 3. 损失与梯度路径

```text
L_total(train) = L_reg_sev(T2)                       # severity加权MSE（E2口径，train-only统计）
               + w1(e) * CE_frame(T1)                # 帧级subject CE；GRL仅反转进入backbone/projector的梯度
               + w3(e) * CE_video(T3)                # 视频级subject CE；GRL仅反转进入temporal+以下的梯度
w_k(e) = W_k * min(1, e / WARMUP_EPOCHS)             # 线性warmup，默认WARMUP_EPOCHS=5
```

- head自身参数始终按CE正常更新；backbone/projector/GRU只接收经GRL取反后的身份梯度分量。GRL在head输入处取反，既有`gradient_reversal.py`语义不变。
- 首版关闭ordinal与CCC辅助（`MODEL.AUXILIARY_TASKS.ORDINAL_CLASSIFICATION: False`、`CCC_WEIGHT: 0`），减少多任务混叠；此为相对`avec2014_base`的显式偏离，在路线配置中声明，不改base默认。
- λ候选集沿用Stage B口径`{0.02, 0.05, 0.10, 0.20}`；首轮`W1=W3=0.05`固定，只有主候选科学通过后将幸存层单独扫λ（每层至多2点）。
- severity权重表、CE权重与warmup系数固定float32构造（防bf16-mixed量化）；CE使用label smoothing 0.1以缓和对齐脸上"认人近乎满分"的梯度支配。

## 4. 实验矩阵与算力上限（seed 42，全为full fit）

| ID | 配置 | 角色 |
|---|---|---|
| `DI-REF` | T2 only + severity-balanced（E2结构锚点，无GRL） | matched参照 |
| `DI-T3` | REF + T3 GRL | 归因对照（E1机制复现） |
| `DI-T1` | REF + T1 GRL | 归因对照（新机制单变量） |
| `DI-FULL` | REF + T1 + T3 GRL | **主候选** |
| `DI-DERM` | DI-FULL但subject标签在train内deranged（保持类数/频次） | 负控：若与DI-FULL行为一致，效应非身份特异 |

seed42上限6个full fit（5 + λ至多1追加）。只有减法阶梯与负控一致指向身份特异效应时，幸存候选进入paired seeds 43/44（每seed至多3个full fit）。

## 5. 前置修复（代码包内、先于任何训练授权申请）

以下缺陷是审查（2026-09-04只读审查）确认的本路线正确性前提，未闭环前不得进入smoke：

1. `mtl_lite.py:928-929`：`layer_temporal`必须为pool前GRU输出且与`layer_shared`不同张量；同步修正`test_mtl_lite_forward.py:169-170`固化错误。
2. identity分支加`self.training`门控；unmapped subject改样本级mask（不再整批置None）。
3. 懒建subject head注入后`.to(self.device)`；记录checkpoint state_dict兼容边界。
4. 断言frame_mask前缀不变量（GRU无pack的先决条件）；空有效帧batch在collate层拒绝。
5. severity权重表与λ/warmup系数float32。
6. 复现性P0：新增依赖冻结文件（environment.yml或requirements锁定版本）；训练入口写`run_provenance.json`（git commit/branch/dirty、argv、seed、device、库版本）；诊断/决策型CLI记录`--split`元数据并在决策路径拒绝test（或要求显式`--allow-test-decision`）。

## 6. 预注册指标与停止条件（全部先看val，test只在locked阶段单次读取）

**技术门禁**（任一触发即停，不作科学结论）：NaN/Inf、shape/mask断言、默认关闭位bit-identical回归、bf16/AMP one-batch、DDP smoke、`DI-REF`与历史E2 val指标一致性核对。

**效用**（paired seed同协议 vs `DI-REF`）：
- 通过：`Δval_CCC >= -0.02` 且 `Δval_MAE <= +0.25`；
- severe失败（停止家族，转诊断）：`Δval_CCC < -0.05` 或 `Δval_MAE > +0.50`。

**身份（只认外部度量，全部val split；test侧仅locked阶段报告）**：
- fresh LOVO Ridge attacker top1（对出口P与出口V分别度量）；
- identity pair AUROC（A2口径）与A1 same-subject top1/top3（帧级出口按`layer_backbone_out`、时序级出口按`layer_shared`）；
- 机制成功判据：至少一项主外部指标相对`DI-REF`下降≥0.03且训练末期不回弹；训练内attacker accuracy只作诊断曲线。

**风险与鲁棒**：severity worst-group MAE/bias（口径[13,19,28]不变）、task consistency、train-val gap、identity/BDI梯度冲突率（`identity_gradient_audit`口径）。风险改善若伴随utility或group robustness恶化超阈值，按既有停止规则收口为负机制结论。

**multi-seed**：任一paired seed触发`ΔCCC<-0.05`或`ΔMAE>+0.50`即停止。

**locked benchmark**：历史role-swap在先，最终test评估只称locked post-selection benchmark，单次读取，不得回选配置。

## 7. 反证与后续分支

- `DI-FULL`科学失败（utility severe或身份指标不降）：保留减法阶梯与负控为诊断证据，不进入λ sweep、multi-seed与"更强对抗变体"堆叠（PCGrad/双层schedule/更多attacker等一律先重新注册）。
- 身份下降但utility小幅受损：进入效用-身份Pareto报告，λ降档一次复跑；仍severe则收口。
- 负控`DI-DERM`与`DI-FULL`同构（身份指标同样"下降"）：判定效应为非特异性正则/容量损失，主线降级为输入侧问题，重开region/输入路线讨论。
- 输入源升级（OpenFace3或region四视图）未来与本机制正交组合时，须以本路线幸存机制为锚另立计划ID。

## 8. 未来代码包预告（未授权，仅边界预览）

`DI-CODE-v1`预计触达：`src/models/mtl_lite.py`（T1帧级头+出口P hook+门控修复）、`src/trainers/mtl_lite_runner.py`（λ warmup调度、provenance、样本级subject mask）、`configs/dual_identity/*.yaml`（新目录，含debug_smoke与5个矩阵配置）、`tests/test_dual_identity_*.py`（前缀mask断言、GRL帧级梯度方向、默认位bit-identical、unmapped样本级mask、warmup调度）。不触碰dataset/split/label路径，不新增第三方依赖（除环境锁定文件），不改base配置默认值。

## 9. 验证（本docs包）

- 仅新增本文件，未修改任何既有文件；`git diff --check`干净；
- 未创建configs/脚本/notebook，未运行任何python/训练/审计；
- 入口文档同步项（下一docs包）：README当前路线段、DOCS_GUIDE权威路线段与职责表、TODO顶部权威入口（OF3/REGION任务标注PAUSED_BY-PLAN-20260904）、CURRENT_STATUS顶部新快照。
