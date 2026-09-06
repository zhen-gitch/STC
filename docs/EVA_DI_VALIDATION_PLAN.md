# EVA_DI_VALIDATION_PLAN.md

> 文档职责：将`DUAL_LEVEL_IDENTITY_ADVERSARIAL_PLAN.md`（`PLAN-20260904-DUAL-LEVEL-IDENTITY-GRL-v1`）的双层级身份抑制机制契约，落到"EVA02冻结视觉特征 + OpenFace3.0行为特征"新输入栈上的架构验证实施契约：钉死事实基线、架构与张量形状、效率契约、特征缓存契约、泄漏控制、实验阶梯与包边界。计划ID：`PLAN-EVA-DI-VALIDATE-v1`。本文档由用户2026-09-04决策引入；机制语义（H1/H2、T1/T2/T3、λ与warmup口径、门禁与停止条件）一律以`DUAL_LEVEL_IDENTITY_ADVERSARIAL_PLAN.md`为准，本文档不修改、不放宽任何机制条款。

> 设计附件：模块/接口/配置/路径约定钉死于 `EVA_DI_MODULE_DESIGN.md`（同一计划ID，docs-only）。

## 0. 决策记录与授权状态（2026-09-04）

用户决策（已冻结，非开放项）：

1. **代码落点**：STC仓库，新模块`src/eva_di/` + `configs/eva_di/` + `tests/eva_di/`。依据（实证）：trans仓库`AGENTS.md`§2明文规定identity-adversarial learning、subject/recording retrieval、identity shortcut归因不属于RIB-Former研究范围且禁止登记为其目标/门禁/指标；故trans仅作**只读资产复用**（权重、aligned图、`features.csv`、train-only归一化统计、参考实现），验证代码与一切身份方向产物不得落入trans。
2. **运行设备**：本机RTX 4060 Laptop（实测8188 MiB VRAM、12 CPU线程、约8 GB RAM）；conda环境`light`（实测timm 1.0.26、torch 2.8.0+cu129、CUDA可用）。不新增任何第三方依赖。RAM是最紧约束，直接决定缓存格式与驻留策略（§3、§4）。
3. **帧预算**：每视频均匀降采样至多512帧（策略版本`uniform512_v1`），**烘焙进抽取阶段**：只对选中帧解码JPEG；缓存键保留源`frame_index_zero`，未来升级全量策略不需作废既有帧级条目（重抽属独立未来包）。

授权状态：本包`EVA-DI-DOC-v2`为docs-only。`EVA-DI-CODE-v1`（代码/配置/测试）、`EVA-DI-EXTRACT-v1`（GPU抽取+数据盘写入）、`EVA-DI-SMOKE-v1`（smoke运行）均未授权；训练授权继续为`false`；test split继续锁定（决策只发生在val，test仅locked post-selection阶段单次读取）。

## 1. 钉死事实基线（全部2026-09-04只读核实）

### 1.1 视觉backbone与权重

- 精确型号：`eva02_small_patch14_224.mim_in22k`（timm），与trans现行视觉主干一致（`trans/src/ribformer/models/image.py:29-49`、`models/pretrained.py:85-97`、`training/relation_formal.py:37`三处一致登记）。
- 权重溯源：HF `timm/eva02_small_patch14_224.mim_in22k`，revision `79c7d4274f6dbf202549d8f976ae24eeaf97e5ad`，文件名`model.safetensors`，86,499,188字节，SHA256 `d3d632352efbd0a0a8269dce114ab8a214833f85ce6f738860221f11ab0f0c2f`，license MIT（证据URL见`pretrained.py:92-96`）。本地两份可用：`~/.cache/huggingface/hub/models--timm--eva02_small_patch14_224.mim_in22k`；`trans/weights/ribformer_backbone_weights_20260811_stage_a_v1.tar.gz`（tarball SHA `f0b111fd…864a`，内含`artifacts/eva02_small_patch14_224.mim_in22k/model.safetensors`）。装载方式：`pretrained_cfg_overlay={"file": …}`离线钉权，装载前逐字节校验上述SHA，失败即拒跑。
- 推理契约（逐行核实`image.py`）：输入float32∈[0,1]的`112x112` aligned face → bicubic对齐上采样至224（`_transform`，:300-307）→ 以timm `pretrained_cfg`运行时读取的mean/std归一化（不硬编码）→ `timm.create_model(…, num_classes=0)`的`forward_features`输出`[M,257,384]`（1 CLS + 16x16 patch，embed dim 384，:309-333结构断言）；冻结模式`no_grad`+有效帧紧凑批+`valid_frame_chunk_size`分块；编码前全黑帧guard（`apply_exact_zero_image_guard`，:65-72）。

### 1.2 OpenFace3.0正式产物（300视频全量，本机磁盘实测）

- 根：`/home/zhen/code/trans/outputs/openface3/20260727-avec2014-formal-features-masked-v2/samples/`，实测300个视频目录，每个含`features.csv`与`aligned/frame_%06d.jpg`（schema `retinaface_5pt_arcface112_masked_v1`，逐帧`aligned_image_sha256`列）。
- `features.csv`共42列：行为206维 = `landmarks_98_xy_aligned`(196) + `gaze_angle_x/y`(2) + `au_score_0..7`(8)，schema `openface3_wflw98xy_gaze2_anonymous_mtl8_v1`；逐帧标志`image_valid/behavior_valid/pair_valid`；含`split/task_id/frame_index_zero/alignment_status/issues`等列。逐runner完整性由`runner_logs/feature_runner_*.json`钉死（`accepted`旗标、`jobs_sha256`、逐视频`features_sha256`、设备与吞吐）。
- train-only行为归一化统计已存在：`behavior_normalization_train_v1/{technical,visibility,visual_quality}.json`（train帧计数160,494）。v1直接复用该钉死统计（记录其SHA），并以"仅train行重算→逐列容差断言"作一致性校验；两者不一致即停止并报告，不静默切换。

### 1.3 标签与split（本机磁盘实测）

- 根：`/home/zhen/dataset/depression/avec/2014/`：`dataset_split.json`（id带`_aligned`后缀，如`203_1_Freeform_video_aligned`）、`depression_labels/*_Depression.csv`（150个subject_session文件，AVEC原格式）。
- 与STC既有加载契约`src/datasets/dataset.py:276-307`（`<subject_session>_Depression.csv`）完全一致；`sample_id == video_id`，subject = `video_id[:5]`，subject类别表只从physical train split构建。
- split唯一真源为`dataset_split.json`；`features.csv`的`split`列仅作逐行一致性断言，不作来源。
- 同目录遗留`features/`与`openface_features/`（旧OpenFace 300个csv）为RISK-001行为特征过拟合教训（test CCC 0.151）的历史源，本验证不读取不覆盖。

### 1.4 环境

- 统一conda `light`，非交互入口显式`~/miniconda3/envs/light/bin/python`；不安装任何包；`HF_HUB_OFFLINE=1`。

## 2. 架构与张量契约（在`PLAN-20260904-DUAL-LEVEL-IDENTITY-GRL-v1`机制语义内实例化）

```text
Stage A（抽取，一次性）:
  aligned jpg（uniform512_v1选中帧） -> bicubic 112->224 -> timm cfg归一化
  -> eva02_small frozen forward_features [N,257,384] -> 每帧 CLS(384) + GAP over patches(384)
  有效性 = features.csv image_valid & ~exact-zero-black；无效帧不写特征、只留valid=False记录。

Stage B（装配，train-only统计）:
  v_t = CLS_t (+GAP_t 为配置开关，v1默认CLS only)      -> [B,T,384|768]
  b_t = 206d行为 -> train-only标准化 -> LayerNorm+MLP    -> [B,T,64]
  x_t = [v_t ; b_t] -> projector(Linear+LN+GELU)        -> p_t [B,T,192]   ← 出口P
  GRU(1层,192, mask-aware) -> h_t [B,T,192] -> masked-mean -> h0 [B,192]  ← 出口V

Stage C（任务头，机制条款逐字继承）:
  T2: h0 -> reg head -> bdi_pred（归一化尺度，指标域clamp[0,63]）
  T1: p_t[valid] -> GRL(λ1) -> subject_head_frame [B*Tv, N_subj_train]，label smoothing 0.1
  T3: h0 -> GRL(λ3) -> subject_head_temporal [B, N_subj_train]
  L = L_BDI + w1(e)·CE_frame + w3(e)·CE_video；w_k(e)=W_k·min(1, e/5)
  λ候选{0.02,0.05,0.10,0.20}，首轮W1=W3=0.05；severity权重/CE/warmup系数float32构造。
  v1关闭ordinal与CCC辅助；backbone冻结，可训练参数仅projector/GRU/heads/行为MLP（<1M）。

Stage D（身份判定，全部外部）:
  fresh LOVO Ridge attacker（对出口P与出口V分别）、pair-AUROC、A1 same-subject检索、
  DI-DERM乱序标签负控；判定只发生在val；训练内attacker只作诊断曲线。
```

- 512帧预算内的时序长度即模型最大T；padding在尾部，前缀不变量断言沿用机制文档§2；空有效帧样本在collate层拒绝。
- 行为NaN策略：任何非有限值 → 该帧`behavior_valid=False`，**禁止填零**（不继承`_safe_float` NaN→0缺陷）。

## 3. 效率契约（trans实证做法 → 本机4060落地；逐条有出处）

| # | trans做法（出处） | 本机落地 |
|---|---|---|
| 1 | 冻结编码器有效帧紧凑批`no_grad`分块 + 编码前全黑guard（`image.py:309-333,65-72`） | 同法；chunk=128；bf16 autocast；`uniform512_v1`使解码量降约69% |
| 2 | 缓存构建：离线钉权、`.tmp`原子写+rename、逐数组SHA256、manifest JSON、线程池IO、可续跑+独立校验脚本（`scripts/build_frozen_region_feature_cache.py:20,40-58`） | 逐条照搬为抽取写入契约 |
| 3 | 缓存读取：manifest指纹与编码器配置字段级比对；训练期全量驻留而非小LRU（`models/feature_cache.py:42-58`） | 同契约；驻留实测预算≈472 MB(CLs+GAP fp32, 153.6k帧)+127 MB(行为206d fp32)≈0.6 GB，8 GB RAM安全 |
| 4 | loader硬校验`num_workers=8,pin_memory=True,prefetch_factor=2`；解码领先编码一块（`relation_formal.py:827-829`、`data/clips.py:1486-1489`） | 驻留数组改为**num_workers=0、主进程向量化组批**（驻留态workers>0会按worker复制数组，8 GB RAM不允许）；解码线程池领先GPU块 |
| 5 | `precision: bf16-mixed`；`test_during_training:false`；runner日志含jobs_sha256/逐视频sha/accepted（d093配置:141、runner_logs） | 全部继承；`logs/eva_di_runner_*.json`同格式 |
| 6 | D093性能A/B方法论：配置字节同版、专用输出根、payload SHA校验、等价性证据不入研究集 | oracle等价性对照单独落`_equivalence`目录：同帧本实现与trans冻结路径cos≥0.999，证据不入指标集 |

预算估计：抽取≈15.4万帧，4060上bf16 no_grad约10–20 min（全量49万帧约40–90 min）；训练驻留+bf16-mixed，单epoch秒级~十秒级，8 GB VRAM富余。

## 4. 特征缓存契约（`eva_di_frame_feature_cache_v1`）

- 根：`/home/zhen/dataset/depression/avec/2014/eva02_features_v1/`（数据盘，仓库零入库）。
- 每视频：`cls.npy [Tv,384] fp32`、`gap.npy [Tv,384] fp32`、`frame_index_zero.npy int64`、`frame_valid.npy bool`、`meta.json`（源features.csv SHA、行号、选中策略）。dtype定fp32：与trans缓存惯例一致、oracle对照容差更干净、驻留预算允许。
- 全局`cache_manifest.json`：schema/策略版本`uniform512_v1`、权重SHA与revision、timm/torch版本、逐文件SHA256、逐视频计数、总帧数/有效帧数/exact-zero数。
- 写入纪律：`.tmp`原子写+rename、逐数组SHA、可续跑（跳过已校验条目）、装载时manifest指纹字段级比对、校验独立成脚本。
- 行为流不另立缓存：训练时直接从`features.csv`解析并驻留（300×csv，<0.5 GB）。

## 5. 泄漏与风险控制（钉桩）

1. split唯一真源`dataset_split.json`；`features.csv.split`列仅断言；不修改split、不删视频、不按val/test回调阈值。
2. 冻结骨干抽取零可训练参数（无监督），300视频全量抽取不构成标签泄漏；下游一切统计量train-only（行为标准化、BDI回归目标归一化、severity权重、subject类别表）。
3. RISK-001护栏：行为特征流已知可严重过拟合 → EVA-only / OF3-only / DUAL三向消融为必做；行为支路重正则+val早停；报告train-val gap。
4. test锁定：训练/评估入口无test路径；决策CLI拒绝test（locked阶段除外，单次读取）。
5. loader按`features.csv`行序断言`frame_index_zero`单调，绕开旧字典序glob缺陷；不读旧`openface_features/`。
6. 既有缺陷不继承清单：identity前向`self.training`门控、unmapped subject样本级mask、NaN禁填零、severity权重float32——在`eva_di`内按机制文档§5前置修复清单实现。

## 6. 实验阶梯与门禁

- 阶梯沿用机制文档§4：`DI-REF / DI-T3 / DI-T1 / DI-FULL / DI-DERM`，seed 42上限6个full fit；另加`EVA-only`、`OF3-only`两个输入消融（各1次，仅挂REF与DI-FULL两层，超出即停下披露）。
- 效用/身份/鲁棒门禁、severe停止线、multi-seed规则、locked benchmark规则逐字继承机制文档§6，不在此复述、不放宽。
- 身份判定只认外部fresh attacker/pair-AUROC/A1检索（val），出口P与出口V分别度量。

## 7. 包边界状态（授权链）

| 包 | 内容 | 状态 |
|---|---|---|
| `EVA-DI-DOC-v2` | 本文件 + `DUAL_LEVEL_IDENTITY_ADVERSARIAL_PLAN.md`§0.2修订 | **已授权，本包执行** |
| `EVA-DI-CODE-v1` | `src/eva_di/`、`configs/eva_di/`、`tests/test_eva_di_*.py`，CPU pytest验收，零GPU零数据盘写 | 未授权 |
| `EVA-DI-EXTRACT-v1` | 抽取脚本+oracle等价性对照+独立校验；写`eva02_features_v1/`与`logs/`；10视频smoke先行 | 未授权 |
| `EVA-DI-SMOKE-v1` | 2–4视频1-batch与overfit smoke | 未授权 |
| 训练/smoke以上运行、commit、push | — | 未授权（训练授权=false） |

边界互不传递。subagent仅承继只读侦察范围，不得越界写入或运行。

## 8. 验证（本docs包）

- 变更清单：新增本文件；`DUAL_LEVEL_IDENTITY_ADVERSARIAL_PLAN.md`仅§0.2一处修订（rev.2），其余章节零改动；`git diff --check`干净。
- 未创建configs/脚本/notebook/测试，未运行python/GPU/训练/审计，未写数据盘，未commit。
- 已知遗留（不在本包）：入口文档（README/DOCS_GUIDE/TODO/CURRENT_STATUS）仍描述旧路线且未登记本文档，权威入口以两份机制/验证文档为准；入口同步与DOCS_GUIDE注册属下一docs包。
