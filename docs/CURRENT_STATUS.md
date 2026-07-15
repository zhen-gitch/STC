# CURRENT_STATUS.md

> 文档职责：当前状态快照与最新阶段性结论。详细机制路线见 `RGB_OVERFITTING_AUDIT_PLAN.md` / `OVERFITTING_MECHANISM_ROADMAP.md`；具体任务见 `TODO.md`；文档导航见 `DOCS_GUIDE.md`。

## 状态日期

2026-07-15

## 阅读提示

本文档是状态快照，不是执行清单。当前路线只看本节到“推荐验证命令”；2026-06-13 之后的日期章节是历史记录和证据归档，若与当前快照冲突，以本文开头和 `TODO.md` 的权威入口为准。

## 当前权威快照：可审计、可证伪的粗粒度信息分流主线

当前项目目标正式设定为 **Auditable and Falsifiable Coarse-Grained Task-Nuisance Information Separation：可审计、可证伪的粗粒度任务-干扰信息分流**。这一调整继承已有 RGB input ablation、identity retrieval、severity calibration、temporal sampling、alignment geometry 和 black artifact 审计结果，但进一步收紧论文主张：`z_dep / z_nuisance` 是待检验的信息分流假设，不是预先成立的语义解耦结论；只对 subject identity 等可验证捷径变量进行弱监督或对抗约束。

### 当前摘要

| 维度 | 当前结论 |
|---|---|
| 项目目标 | 可审计、可证伪的粗粒度 task-nuisance 信息分流 |
| 第一版 latent | `z_dep`、`z_nuisance` |
| 可选 latent | `z_id` 不进入第一版；基础方案通过 C3 后才重新论证 |
| 不做 | 不显式建 `z_art/z_ctx/z_pose/z_quality`，不做多级 RPDF |
| 下一步 | aligned JPG 已通过跨机一致性门禁；先运行 OpenFace 2.2.0 两视频 debug 与全量 aligned-landmark 生成，再运行 `AU-T0b` 并人工审阅 train overlay |
| 审计判据 | BDI metrics、identity risk、nuisance/BDI leakage、shortcut probe risk、severity bias、task consistency、train-val gap |
| 主张边界 | reconstruction/decorrelation 收敛或单一 attacker 下降不等于语义解耦成功 |
| 反证条件 | 不优于 paired-seed `C-REF`、multi-seed 不稳定或风险改善伴随 utility/group robustness 恶化时停止增加复杂度 |

“可审计”要求每个表示出口和训练约束都对应可独立复现的外部测量；“可证伪”要求在编码和运行前冻结强 baseline、统一指标、multi-seed 规则和停止条件。负结果应作为机制结论保留，而不是通过继续堆叠 latent、门控或损失规避。

Stage C 的完整实施规格已写入 `docs/STAGE_C_RUNBOOK.md`。`C-REF / C-BN / C-REC / C-FULL` 的 seed-42 运行已完成，`H0=192`、`z_dep=96`、`z_nuisance=96`。相对 `C-REF`，三组候选的 validation CCC 分别下降 `0.0990 / 0.1175 / 0.1147`；C-REC/C-FULL 的 MAE 分别恶化 `0.5339 / 0.5254`，全部触发冻结的 severe utility failure。Stage C 家族继续停止进入 C3，不加入 `z_id`、pose/AU 训练监督或更多 latent；下述 GRL 实验作为独立梯度机制审计，不推翻该停止结论。

### 2026-07-14 近期机制实验收口与路线调整

近期四类实验已经收口，结论共同指向：继续增加全局参数正则、连续标签权重、身份对抗强度或表示维度，不能在当前证据下解释或解决过拟合。

1. **Identity-gradient audit**：candidate 的 best-val RMSE 从 `10.7153` 改善到 `10.5176`，CCC 从约 `0.362` 提升到 `0.477`，severe bias 从 `-15.763` 改善到 `-12.703`；但外部 identity pair AUROC 从 `0.9308` 升到 `0.9447`，same-subject top1 从 `0.52` 升到 `0.54`，说明身份泄漏没有下降。反向 identity/BDI 梯度范数比均值为 `0.3535`，冲突率 `60.2%`，末期 validation 退化从 `0.756` 扩大到 `2.183`。该实验仅作为“短期 utility 改善但去身份机制失败”的负机制消融，不授权维度 sweep。
2. **Explicit L1/L2 audit**：reference、L1、L2、elastic 的 best-val RMSE 均在 `10.708-10.720`，四组都在 best epoch 后继续过拟合。当前全局均值参数惩罚贡献较小，不能替代 EarlyStopping，关闭系数扩展。
3. **Continuous severity-density weighting**：`POWER=0.25/0.5` 均优于 plain MSE，但弱于四档 E2。alpha 0.25 的 val `MAE/RMSE/CCC=8.284/10.598/0.409`，alpha 0.5 为 `8.307/10.700/0.413`；E2 的 CCC 为 `0.457`，severe bias 也更好（`-12.030` vs `-14.320/-14.260`）。不继续 alpha/sigma sweep。连续模式按样本均值归一到 `1.0`，而 E2 的实际样本平均权重约 `0.861`，因此后续若比较权重形状，必须先控制整体 loss scale。
4. **Split sensitivity**：original 与 val/test-swapped 两组训练指标到 epoch 16 完全一致，说明训练过程没有改变，变化来自 checkpoint monitor 所见验证集合。epoch 10 相比 epoch 8 提高 CCC、扩大预测方差并缓解 severe bias，但略损伤 original-val MAE/RMSE 与 task consistency。swapped 条件使用原 test 选 checkpoint，只能作为探索性敏感性证据；论文级结论仍需重复 subject-disjoint folds/group CV。

因此，当前不再把“增大/缩小/先升后降表示维度”作为立即实验。维度只有在训练目标能提供可验证的语义梯度、且外部 leakage 风险确实下降后才有解释价值。下一条主动干预路线转为输入侧局部归纳偏置：保留全脸信息，同时用动态 AU/FACS 语义区域训练同一个共享模型。

### 2026-07-14 单模型 AU 语义区域路线

目标不是训练多个区域模型，也不是在推理时做多分支融合，而是让同一个 MTL-Lite backbone、时序编码器和 BDI head 在训练时同时处理全局脸与全部有效 AU 语义局部视图。各视图沿 batch 维展开并共享全部参数；validation/test/inference 只输入全脸。若 global-only inference 得到改善，才能说明局部训练改变了同一模型的表示偏好，而不是依赖额外推理模型。

首版只定义四个整体 AU 语义区域：brow（AU1/2/4）、eye-cheek（AU5/6/7/45）、nose-upper-lip（AU9/10）、mouth-jaw（AU12/14/15/17/20/23/24/25/26）。标准 OpenFace AU 输出不提供左右独立监督，因此左右 landmark 只能作为内部 tracking components，用于姿态可见性、质量权重和整体 mask 合成，不能形成独立语义视图、预测或损失。所有有效整体区域都参与训练，不随机只选一个；候选总损失为 `L_global + 0.5 * mean(L_brow,L_eye,L_nose,L_mouth)`，四区全部有效时每区最终权重为 `0.125`。区域外使用模糊背景和 feathered mask，禁止硬黑填充。

该路线的首要风险是帧/坐标契约与动态跟踪，而不是模型结构。现有 dataset 只返回排序和采样后的图像，没有暴露 JPG frame id 或 OpenFace 行号；现有 OpenFace CSV landmark 又位于约 `640x480` 检测坐标系，不能直接覆盖到实际 `112x112` aligned 输入。当前立即任务是只读 `AU-T0a/T0b`：先验证 JPG 文件名、CSV `frame/timestamp` 与当前采样索引的 join，再检查 OpenFace 对齐变换/元数据并确定坐标映射。之后才比较 static canonical、raw per-frame 和 temporally stabilized dynamic masks。训练实现必须等待 tracking gate 通过。

初始门槛冻结为：frame join rate `>=0.995`、median adjacent-mask IoU `>=0.75`、区域 valid-frame ratio `>=0.80`、landmark jump rate `<=0.05`，且 high-yaw 样本不能出现系统性错位。短缺失只允许插值，长失败保持 invalid；大 yaw 时左右 tracking components 分别估计可见性后合成为一个整体语义 mask，不镜像或补造隐藏侧。AU 语义区域还必须和四个 equal-area arbitrary grid 区域做面积匹配对照；若 AU 不优于 grid，只能主张局部正则有效，不能主张 FACS 语义有效。

### 2026-07-15 AU-T0a 通过与 AU-T0b coordinate-contract implementation

AU-T0a 已在 300 个视频上正确运行：`300 PASS / 0 FAIL / 0 BLOCKED`，最低 frame join rate 为 `1.0`，41,016 个当前模型选中帧全部精确连接，300 个视频的最佳 offset 均为 `0`。因此 frame-contract gate 已通过，但该结果只证明 JPG/OpenFace frame identity，不证明检测坐标到 `112x112` aligned input 的空间映射。

已新增只读 `src/diagnostics/au_coordinate_contract.py` 与 `scripts/audit_au_coordinate_contract.py`。T0b 不实现或允许 `112/640`、`112/480` 独立缩放，只接受三种可审计来源，`auto` 优先级固定为：逐帧显式 2x3 affine transform；在 aligned JPG 上重新运行 OpenFace 得到的 aligned-space landmark；调用方提供 canonical aligned template 后逐帧拟合 similarity transform。没有合法来源时输出 `BLOCKED`，不会静默猜测坐标。

固定输出包括 `coordinate_mapping_manifest.csv`、`coordinate_frame_summary.csv`、`coordinate_transforms.csv`、`overlay_manifest.csv`、`coordinate_contract_issues.csv`、`coordinate_contract_report.md` 和 `run_manifest.json`。`mapping_valid` 现在还强制要求实际映射来源的 OpenFace `success=1`；失败检测即使残留坐标仍在图像范围内也不能通过。自动检查通过只标记 `REVIEW_REQUIRED`，不能视为 T0b PASS；必须在 train split 人工审阅 high-pose、rapid-turn、low-confidence、high-residual 和 frontal-control overlays。validation/test 不参与阈值或映射调参。动态 mask 与训练实现继续被该人工 gate 阻断。

在本地 OpenFace aligned-landmark 重检测前，新增 `src/diagnostics/image_integrity.py` 与 `scripts/audit_image_integrity.py`，用于确认本地 aligned JPG 未损坏且与服务器逐文件一致。inventory 严格复现训练可见文件契约，只扫描一级视频目录中的直接 JPG；逐图记录文件 SHA-256、完整 Pillow 解码、`112x112` 尺寸和可选 decoded-RGB pixel SHA-256。服务器和本地分别生成排序 manifest 后由 compare 流式合并，只有路径集合、解码、尺寸和文件 SHA-256 全部一致才输出 `EXACT_PASS`。该工具不修改或重新编码图片。

完整性门禁现已在真实全量数据上通过：服务器与 Windows 均为 `300` 个视频目录、`493,141` 张 `112x112 RGB JPEG`，全部完整解码；`493,141/493,141` 为 `EXACT_MATCH`，两端 manifest SHA-256 均为 `07bc830c452a8faab1d58935d7f58d807313c218b3c4af8db053777c759daabb`，最终状态为 `EXACT_PASS`。因此本地 JPG 与服务器训练输入已建立逐字节同一性，可进入冻结版本重检测，不需要再做默认 pixel-hash 全量重算。

Windows OpenFace 已冻结为 `D:\Tools\OpenFace` 下的 `OpenFace 2.2.0`。`FeatureExtraction.exe` SHA-256 为 `5995ae5cce749c4969ac4dd7e62d3f740cc9f702961f9573be7e14c4ca5b7f86`，`model/main_ceclm_general.txt` SHA-256 为 `52f38548cffab1731f80e9e71f22a8b29373a2750eb6dc718069d56e82997543`。新增 `scripts/run_openface_aligned_landmarks.ps1`，只执行完整序列 `-fdir + -2Dfp + -mloc`，输出独立 aligned-space landmark CSV；脚本检查版本/哈希/图像门禁、帧行数与连续性、必要列和 `success>=0.995`，并写入二进制、模型、git、命令和逐视频审计清单。当前下一门禁是两视频 debug；通过后在新目录完成 300 视频全量生成，再把该目录作为 `--aligned-openface-root` 运行 T0b。原始 `face_images` 和历史 `openface_features` 继续冻结且不得覆盖。

### 2026-07-14 AU-T0a frame-contract implementation

已实现只读 `src/diagnostics/au_region_tracking.py` 与 `scripts/audit_au_region_tracking.py`。当前阶段只验证 aligned JPG 与 OpenFace CSV 的 frame contract，不生成区域裁剪、不恢复 landmark 坐标、不修改 dataset/model。诊断自动评估 `-2..2` frame offset，报告 0/1-based 差异、无法解析的文件名、重复/缺失帧、frame/timestamp 单调性，并用现有 `select_temporal_indices` 复现模型实际选中帧。

固定输出为 `frame_contract_summary.csv`、`selected_frame_mapping.csv`、`frame_contract_issues.csv` 和 `frame_contract_report.md`。实现使用 Python 标准库，避免 T0a 被训练环境的 numpy/torch 依赖阻塞。该实现状态已由上方 2026-07-15 的完整数据运行结论取代。

### 2026-07-13 Stage C C2 utility 否证结论

四组使用相同 seed、split、backbone、optimizer、precision、severity weighting、EarlyStopping 和 checkpoint policy。C-REF best-val `RMSE/MAE/CCC=10.7831/8.2539/0.4570`；C-BN 为 `11.2693/8.4809/0.3580`，C-REC 为 `10.9209/8.7878/0.3395`，C-FULL 为 `10.9047/8.7793/0.3423`。C-BN 与 C-REF 参数量只差 `288`，因此退化主要来自 96 维 prediction bottleneck，而不是参数规模。C-REC/C-FULL 参数量完全相同且指标几乎重合，reconstruction/decorrelation 没有恢复 utility，也没有显示 decorrelation 的增量证据。

这些服务器运行因旧 runner 无条件调用 `trainer.test()` 而提前产生 test 指标；C2 判定只使用 validation，已观察的 test 结果隔离为 exploratory，不用于选择结构或权重。runner 现增加 `RUN_TEST_AFTER_FIT`，旧配置缺省保持 `True`，Stage C common 固定为 `False`。

### 2026-07-13 Stage C 失败机制只读诊断

已实现并在四组 train/val 导出上运行 `representation_leakage.py` 与 `group_robustness.py`。所有 probe/阈值只在 train 拟合，validation 只评估；输入 NPZ、prediction CSV、checkpoint 和 split 均不修改。identity cosine pair-verifier AUROC 在 C-REF `z_dep` 为 `0.9290`，C-BN/C-REC/C-FULL `z_dep` 分别为 `0.9199/0.9344/0.9344`，没有达到相对 C-REF 绝对下降 `0.05` 的风险门槛。C-REC/C-FULL 的 `z_nuisance` AUROC 仍为 `0.9189/0.9209`，说明身份信息没有被路由出两个出口。

BDI ridge probe 显示 C-BN 的 `H0 -> z_dep` CCC 从 `0.3802` 降到 `0.2308`，直接支持 prediction bottleneck 是主要损伤点。C-REC/C-FULL 的 nuisance BDI gate 未触发（CCC gap `0.1152/0.1046`，MAE gap `1.2788/1.8915`），但这不构成成功证据，因为 utility 与 identity risk 已失败。validation group audit 中，三组候选的 severe MAE 相对 C-REF 增加 `0.8771/0.9960/1.0598`，severity worst-group gap 增加 `1.0352/2.0774/2.1278`，全部触发 group utility failure；task gap 缩小但不能抵消 severity failure。pose_rz、black-border、face-offset-y 尚无 train weak-label 表，保持 unavailable，禁止用 val 中位数补阈值。

### 2026-07-08 Stage A 证据收口结论

本轮对 `logs/analysis_outputs` 进行了全量只读审查：目录下共有 205 个文件，包括 87 个 CSV、32 个 Markdown 报告、82 个 PNG 和 4 个 NPZ；未发现 0 字节或异常小的 CSV/Markdown 文件。`val` 与 `test` 均为 100 个视频、50 个 subject、Freeform/Northwind 各 50，且两者 `video_id` 与 `subject_id` 重叠均为 0。Stage A 的 A1/A2/A3/A4 主要产物已经齐全。

RGB baseline 的预测表现显示稳定的 prediction compression：val MAE/RMSE/Pearson/CCC 为 8.4228/10.7286/0.4452/0.3599，test 为 8.9145/10.9530/0.3526/0.2925；预测标准差约为真实标准差的一半（val 6.1274 vs 11.9791，test 6.1587 vs 11.5387）。

Stage A 四个结论如下：

1. **A1 身份存在**：身份信息强且主要集中在 backbone 中层。val 中 `layer_block_6` same-subject top1=0.80、`layer_block_3` top1=0.79；test 中 `layer_block_3` 和 `layer_block_6` top1 均为 0.88。`layer_shared` 仍保留身份信息（val top1=0.52，test top1=0.66）。
2. **A2 身份参与预测**：身份/严重程度邻域信号与 residual 存在稳定耦合。val 中 `corr_residual_vs_severity_agree` 最高为 0.5124，test 中 `layer_shared/layer_temporal corr_residual_vs_severity_agree` 为 0.4589；test 中 `corr_residual_vs_paired_identity_sim` 在多层达到 0.40 左右。因此 A1+A2 同时成立，Stage B 可以进入 identity-adversarial baseline，而不是仅做身份风险监控。
3. **A3 artifact/quality 风险**：正式口径使用 `logs/analysis_outputs/artifact_weaklabels_matched/{val,test}/tables/artifact_weaklabel_correlation_matched.csv`。matched-only 表中 val/test 均为 100 行、122 个 weaklabel correlation，全部 `n=100`。val 中 `|corr(abs_error)| >= 0.2` 的 weaklabel 有 28 个，test 只有 5 个，交集仅 `openface_quality:AU10_r_mean`；val/test 相关向量稳定性弱（signed r=-0.1525，abs r=-0.1575）。因此 A3 只能作为 shortcut/artifact probe、case study 和 group-wise robustness 的依据，不作为显式 `z_art` 分支或训练监督损失。
4. **A4 severity imbalance / compression**：severity imbalance 是稳定主问题。val minimal bias=+6.6194、severe bias=-15.7630、imbalance_ratio=6.5、compression_ratio=0.5115；test minimal bias=+6.8856、severe bias=-16.5032、imbalance_ratio=3.5714、compression_ratio=0.5337。Post-hoc calibration 只轻微改善 MAE/RMSE，但 CCC 从 0.2925 降到 0.2692，并进一步压缩 pred_std（6.1278 -> 5.3337）。因此 severity-balanced regression 应进入 Stage B 必跑基线。

Stage A 已满足关闭条件。后续不再扩展普通 RGB mask、灰度、模糊、黑边替换或新的输入滤镜族；下一阶段固定比较 RGB baseline、identity-adversarial MTL、severity-balanced regression，以及二者组合。

### 2026-07-08 Stage B 进入判定

当前实验分析结果支持进入 Stage B。进入理由不是单一 MAE 或单一 shortcut 证据，而是 A1/A2/A4 同时给出可干预机制：

- A1+A2 支持 identity-adversarial MTL：身份信息在 backbone 中层强可分，且 identity/severity 邻域信号与 residual 耦合。
- A4 支持 severity-balanced regression：minimal 高估、severe 低估和 prediction compression 在 val/test 均稳定出现，post-hoc calibration 不能替代训练侧修正。
- A3 不支持显式 `z_art`：artifact/quality/context 变量仅进入 probe、case study 和 group-wise robustness。

Stage B 固定实验边界如下：

```text
E0 RGB MTL-Lite baseline
E1 identity-adversarial MTL
E2 severity-balanced regression
E3 identity-adversarial MTL + severity-balanced regression
```

Stage B 只允许上层、可开关、可复现实验干预：GRL/subject attacker、severity-bin weighting 或 regression loss reweighting、二者组合。禁止同时引入 `TaskNuisanceBlock`、`z_nuisance`、`z_art/z_ctx/z_pose/z_quality`、新 RGB 输入滤镜、dynamic branch 或 late fusion。

Stage B 成功不以 MAE 单点下降为准，最低判读指标固定为：MAE/RMSE/Pearson/CCC、pred_std/true_std、minimal/mild/moderate/severe bias 与 MAE、layer/shared identity retrieval 或 attacker risk、task_diff_mean、train-val gap、artifact-risk group 表现。若 E1/E2/E3 不能同时改善 identity risk 或 severity bias 且保持 CCC/task consistency 稳定，才进入 Stage C 的粗粒度 `z_dep/z_nuisance` 信息分流设计。

Stage B 实施路线已细化到 `docs/RGB_OVERFITTING_AUDIT_PLAN.md` 的“Stage B 实施路线（2026-07-08）”和 `docs/TODO.md` 的“立即可编程任务包”。后续编程按 B0 规格冻结 -> B1 identity-adversarial -> B2 severity-balanced -> B3/E0-E3 固定实验 -> B4 诊断汇总 -> B5 阶段判定推进。

### 2026-07-10 Stage B B5 证据收口结论

Stage B 固定实验矩阵已全部跑完并通过 version-aware 聚合：E0（base）+ E1/E3 的 lambda_id sweep（0.02/0.05/0.10/0.20）+ E2/E3 的 POWER sweep（0.5/1.0），共 12 个 run，每个 run 跑完整 B4 诊断链（prediction / A1 layerwise / A2 coupling / severity calibration / fresh-probe attacker / A3 artifact chain）。聚合表在 `logs/stage_b/aggregate/{prediction,identity_retrieval,severity_calibration,severity_imbalance,training_overfit,subject_attacker}/`，A3 matched 汇总在 `logs/stage_b/aggregate/a3_matched/`。下表为 12 run 的全信号对照（test split，100 视频/50 subject，true_std=11.54，chance=0.02）：

| run | MAE | CCC | atk_top1 | A1_top1 | cal ΔMAE | val退化↑ |
|-----|-----|-----|----------|---------|----------|----------|
| e0（base） | 8.915 | 0.293 | 0.54 | 0.66 | −0.07 | +0.76 |
| e1 | 8.903 | 0.389 | 0.56 | 0.65 | −0.26 | +2.18 |
| e1_lambda0.02 | 8.855 | 0.399 | 0.55 | 0.66 | −0.27 | +1.78 |
| e1_lambda0.1 | 8.977 | 0.377 | 0.55 | 0.65 | −0.25 | +2.18 |
| e1_lambda0.2 | 9.034 | 0.360 | 0.55 | 0.63 | −0.19 | +1.83 |
| **e2** | 8.914 | **0.410** | 0.55 | 0.67 | **−0.62** | +0.69 |
| **e2_power1.0** | 8.882 | **0.410** | 0.57 | 0.68 | **−0.66** | +0.64 |
| e3 | 9.100 | 0.333 | 0.55 | 0.65 | −0.19 | +2.21 |
| e3_lambda0.02 | 9.046 | 0.343 | 0.55 | 0.66 | −0.19 | +2.05 |
| e3_lambda0.1 | 9.165 | 0.315 | 0.55 | 0.64 | −0.16 | +2.66 |
| e3_lambda0.2 | 9.355 | 0.283 | 0.54 | 0.64 | −0.20 | +2.55 |
| e3_power1.0 | 9.247 | 0.297 | 0.55 | 0.66 | −0.21 | +1.90 |

说明：`atk_top1` 是 fresh linear-probe attacker（LOVO Ridge on `z_dep`，coverage=1.0），`A1_top1` 是 same-subject NN retrieval，两者都测 `z_dep` 的身份可分性（chance=0.02）。`cal ΔMAE` 是 val 拟合 a/b 后 test 校准的 MAE 变化。`val退化↑` = last_val_rmse − best_val_rmse（越大越过拟合）。

**B5 收口五条结论**：

1. **E2（severity-balanced regression）= 唯一接近有效的弱 baseline**：CCC 0.41（+0.12 vs e0）、calibration 收益最大（−0.62/−0.66）、过拟合最轻（+0.64-0.69）、修复 prediction compression（pred_std 6.16→8.72-9.18，true=11.54）。但 identity 仍泄漏（atk 0.55、A1 0.67，未降）——E2 只解了「预测压缩」与「severity 分布」，没解 identity shortcut。可作 Stage C 强 baseline 候选。

2. **E1/E3（identity-adversarial）= 无效**：atk_top1 与 A1 都没有下降（e1 atk=0.56 反而 > e0=0.54，e2_power1.0=0.57 最高），且 E3 utility 全面劣化（CCC 0.28-0.34 全线低于 e0 0.29，MAE 9.1-9.4）。lambda sweep 单调代价清晰（e1 CCC 0.399→0.360 随 0.02→0.2）。GRL 强度太弱，identity 对抗在当前 lambda 范围未能把 subject identity 从 `z_dep` 抹除。

3. **identity 泄漏全局未解（两条独立信号收敛）**：fresh attacker top1 0.54-0.57（chance 0.02）+ A1 NN 0.63-0.68，参数化线性 probe 与非参数 NN 两套独立口径一致确认 `z_dep` 强泄漏 subject identity。E1/E3 的对抗头未改善任一信号。

4. **A3 artifact-risk group 未被任何 defense 改善**：matched-only A3（12 run × 100 视频）显示最强 shortcut 轴是 OpenFace 头部姿态 `pose_rz_mean`（\|corr\|=0.319）+ 多个 AU。按 pose_rz 中位数切 high/low 组，e0 high-group MAE=10.38 vs low=7.45（gap 2.93），**所有 12 run 在 high-risk 组的 MAE 都不低于 e0**——E1/E3 普遍恶化 +0.4~0.7，E2_power1.0 仅在 AU26/face_offset_y 两轴改善，pose_rz/black_border 仍恶化。整体平均耦合（122 变量）E2 系最低（0.081-0.085 vs e0 0.095），但 high-risk 组未改善说明 E2 的整体收益是「假分散」而非真解耦。

5. **普遍过拟合（12/12）**：所有 run `overfit_after_best_val=True`，best-val→last 的 val 退化 0.64-2.66（E3 系最重 1.9-2.66）。best ckpt 被正确选用，但跑满 40 epoch 浪费算力且末态严重过拟合；该问题已由 P0 配置化 EarlyStopping 处理。

**subject attacker 说明**：AVEC2014 是 subject-disjoint split，联合训练的 `subject_id_head` 没见过任何 test subject（旧复用 train head 的 attacker 给 coverage=0、空指标，B5 该门槛原为死的）。已改为 fresh linear-probe attacker（LOVO Ridge on `z_dep`，在 test subject 自身上训，coverage=100%），让所有 run 含 E0/E2 baseline 都出真数字。该信号与 A1 NN **收敛不分化**（都测 z_dep 身份可分性），价值是让 B5 门槛 live 并提供参数化（线性）视角作为 A1 的收敛证据，不提供新的差异化结构。

**B5 阶段判定**：Stage B 的轻量级可开关 defense（identity 对抗 + severity 平衡）不足以解除 shortcut。E2 可作弱 baseline（整体 CCC 改善但 artifact/identity 组未改善），E1/E3 明确无效。**满足进入 Stage C 的条件**：验证显式粗粒度 `z_dep / z_nuisance` 信息分流假设；pose_rz / AU / black_border 第一版只作为审计轴。C0 与 P0 已完成，当前进入 C1；详细规格见 `docs/STAGE_C_RUNBOOK.md`。

### 2026-07-03 输入遮挡语义修正

对真实 OpenFace aligned 输入帧的人工审查显示，历史 `center_mask` 只保留约 13.5% 总像素，在 18 张样例帧上仅保留约 16.6%-18.2% 可见非黑区域，视觉上主要覆盖鼻梁、鼻子和鼻下/上唇附近。因此，历史 `center_mask` 结果不得继续解释为“保留面部中心行为区域”或“中心脸行为有效”；更准确的解释是 **tiny central nose/mouth patch ablation**，即通过强遮挡删除大部分身份外观、脸部轮廓、眼镜/胡须/头发和对齐边界线索。

为验证原有 input artifact / shortcut 结论是否仍成立，已新增 `central_face_mask` 作为新的对照消融。该变体覆盖眼、鼻、嘴和主要脸颊区域，保留约 45%-50% 总像素，用于区分“完整中心脸区域有效”与“旧 `center_mask` 的鼻口小块强遮挡偶然有效”。该修正是对历史输入消融语义的校准，不改变当前“先审计、再建模”的优先级。

当前路线的核心目标是：避免过细、不可穷尽的因子划分，建立可审计、可证伪的粗粒度信息分流机制：

```text
H0 -> z_dep, z_nuisance
```

其中：

- `z_dep`：候选抑郁预测主表征，最终 BDI 回归和 ordinal 辅助任务只依赖它；
- `z_nuisance`：候选互补出口；是否承载干扰信息、是否泄漏主要 BDI 信号，必须由独立 probe 验证；
- `z_id`：不进入 Stage C 第一版；只有基础两出口方案通过 C3 稳定性闸门且证据仍支持时，才重新评估；
- artifact、context、pose、quality 等因素不再作为第一版显式 latent，只作为离线审计变量、post-hoc probes、case study 和 group-wise evaluation 维度。

当前主线分为四层：

```text
Stage A 证据收口：证明 identity / shortcut / severity imbalance 是否确实参与错误模式
Stage B 对抗基线：验证 identity-adversarial task representation 是否降低身份捷径
Stage C 粗粒度信息分流：验证 z_dep / z_nuisance 是否在等参数条件下优于共享表征
Stage D 反证与稳健性验证：multi-attacker、leakage matrix、severity-balanced loss、group-wise robustness、task consistency
```

### 现在已经完成什么

- RGB input ablation、temporal sampling、identity retrieval、severity calibration、alignment geometry 等审计已经形成阶段性证据（Stage A 收口）。
- Stage B 固定实验矩阵（E0-E3 + lambda/power sweep，12 run）已全部跑完并 version-aware 聚合，B5 判定完成：E2 弱有效作 Stage C baseline，E1/E3 无效，identity 泄漏全局未解（atk+A1 双信号收敛），A3 artifact-risk group 未被任何 defense 改善，12/12 普遍过拟合。
- 现有输入变体能改变 overall MAE、severity bias 和 task consistency，但没有任何变体同时解决 identity retrieval、severe underestimation、CCC 和 task consistency。
- 输入级 `identity_texture_suppressed` / boundary smoothing 2x2 作为机制证据保留，不再作为下一阶段主线继续扩展。
- Post-hoc linear calibration 全部降低 CCC，因此只作为 prediction compression 诊断，不作为模型方案。
- `Shortcut-Regularized MTL` 保留为重要基线：`severity-balanced regression`（E2）作为 Stage C 强 baseline 候选，`identity-adversarial MTL`（E1，已证无效）作为支线消融对照。

### 当前正在推进什么

Stage A、Stage B 均已完成收口（见上方 2026-07-08 Stage A 结论与 2026-07-10 Stage B B5 结论）。Stage B 证明轻量级可开关 defense（identity 对抗 + severity 平衡）不足以解除 shortcut，下一步进入 Stage C 粗粒度信息分流验证：

1. ~~**B0 干预规格**~~：已完成（identity-adversarial MTL + severity-balanced regression 配置/损失/评估/默认关闭）。
2. ~~**B1 identity-adversarial baseline**~~：已完成（E1）——无效，identity risk 未降。
3. ~~**B2 severity-balanced regression**~~：已完成（E2）——弱有效，CCC +0.12、calibration 收益最大，但 identity/artifact 未解。
4. ~~**B3 组合对照**~~：已完成（E3）——无效，两约束叠加互相干扰，utility 劣化。
5. ~~**C0 spec-before-code**~~：已冻结信息分流接口、辅助损失的非证明性边界、等参数/等 bottleneck 对照、leakage matrix、multi-seed 和停止条件，详见 `STAGE_C_RUNBOOK.md`。
6. ~~**P0 training protocol**~~：已完成配置化 seed、与 best checkpoint 共用 monitor/mode 的 EarlyStopping、训练策略测试，以及 Stage C 配置骨架和同协议 `C-REF`。
7. ~~**C1 coarse task-nuisance information separation**~~：代码、smoke、train-only calibration 和冻结权重均已完成。
8. **C2 validation screening（utility 已否证）**：三组候选均触发 severe utility failure；停止进入 C3。后续只允许修复 validation-only 协议并运行失败机制诊断，不据此增加结构复杂度。

编程实施控制已细化到 `TODO.md` 的“编程实施控制（下一步）”。后续按 `P0（已完成） -> C1 -> C2 -> C3 -> Stage D` 推进；不在基础两出口方案通过闸门前扩展 `z_id`、细粒度 latent 或多级门控。

### 当前路线细化补充

最新研究路线进一步明确为四个闭环：证据闭环、对抗基线闭环、粗粒度信息分流闭环和反证/稳健性验证闭环。短期不实现完整 RPDF-Net，也不显式划分 `ctx/art/pose/quality` 等难以完全验证的潜在因子；Stage C 第一版只比较 `C-REF/C-BN/C-REC/C-FULL`，只有外部 multi-attacker、leakage matrix、group-wise evaluation 和多 seed 结果共同支持时，才允许声称风险得到降低。辅助损失仅定义结构偏置，不定义 latent 语义。

### 当前不做什么

- 不直接实现完整多级、多损失、多门控 RPDF-Net。
- 不把 `z_m`、`z_art`、`z_ctx`、`z_quality` 等细粒度因子作为当前主线。
- 不继续扩展普通 RGB mask、灰度、模糊、黑边替换或新的输入滤镜族。
- 不把动态特征、feature delta、AU delta、landmark/pose/gaze delta 列入第一版粗粒度信息分流模型。
- 不仅凭 MAE 选择模型，必须同时检查 CCC、severity bias、identity risk、artifact risk、task consistency 和 train-val gap。
- 不以 reconstruction、decorrelation、训练内 adversary 或单一 probe 的结果宣称语义解耦成功。

## 基础设施状态

项目已经通过服务器 debug smoke，可以完整运行旧版端到端训练流程。当前工作重点已经从“修复可运行性”转向“重构架构边界”，为 MTL-Lite 轻量级多任务抑郁预测模型建立干净主线。

## 架构边界

项目采用以下新架构：

```text
src/legacy/full_model/  # 旧大模型整体归档
src/models/             # MTL-Lite 新模型与通用模型模块
src/diagnostics/        # 独立诊断与可视化系统
```

关键决策：

- 旧大模型整体进入 legacy；
- 旧模型只作为历史快照保留，暂时不再投入额外修复或重构；
- 新模型不继承旧模型；
- 通用模块保留在主线位置；
- 当前训练入口围绕 MTL-Lite 轻量模型基座设计；
- 诊断与可视化系统从模型训练逻辑中解耦。

## 当前模型基座

MTL-Lite 是当前 BDI 预测训练和诊断的轻量模型基座：

```text
人脸视频帧 -> 视觉 backbone -> 时序编码器 -> 共享视频表征 -> BDI 回归头 + 有序严重程度分类头
```

主任务：

- BDI 连续回归。

辅助任务：

- 有序抑郁严重程度分类。

非基座模块：

- contrastive learning
- adaptive mask
- PCGrad
- CGC / complex expert routing
- LDS
- `loss_dist`

这些模块后续只作为 legacy 能力、消融项或扩展项。

## 机制证据摘要

当前视频帧序列已经由 OpenFace 裁剪和对齐，所用 OpenFace 版本可能不是最新版。因此，近期实验中出现的泛化问题不应只解释为普通背景过拟合，而应重点检查 OpenFace aligned face 中仍然存在的非抑郁捷径，包括身份纹理、裁剪边界、对齐伪影、姿态残留、追踪质量、光照和视频质量。

最新 RGB 输入消融进一步说明输入侧非行为线索值得研究。`center_mask` 在测试集上明显优于原始 `rgb`，而 `grayscale` 和 `blur` 变差；样例帧也显示 OpenFace aligned face 中存在黑色填充、硬裁剪边界、遮挡物黑块和对齐残留。这些现象支持“input artifact 是风险入口”的判断，但不能把 RGB 过拟合单独归因于黑边或黑填充。后续应同时审计 split/subject、时序采样、训练曲线泛化缺口、OpenFace 对齐几何、身份静态外观、姿态/追踪质量、任务语境和 severity calibration。

backbone 冻结与高层微调实验提示：仅调整 `FREEZE_BACKBONE` / `FINETUNE_LAST_N_BLOCKS` 不能充分解决泛化问题。下一阶段优先方向应转为 RGB 过拟合多因素审计：

```text
split integrity audit
-> temporal sampling audit / ablation
-> training overfit curve summary
-> alignment geometry audit
-> embedding identity retrieval
-> severity calibration verification
-> task inconsistency mixed-factor audit
```

机制路线详见 `docs/RGB_OVERFITTING_AUDIT_PLAN.md`。相关论文线索记录在 `docs/RESEARCH_NOTES.md`。非抑郁捷径验证框架的具体实施方案记录在 `docs/SHORTCUT_AUDIT_DESIGN.md`。当前执行入口仍以本文开头的权威快照和 `docs/TODO.md` 为准。
系统机制路线图已整理到 `docs/OVERFITTING_MECHANISM_ROADMAP.md`，用于把现有黑边、geometry、temporal、identity、局部遮挡、calibration 和 behavior baseline 结果组织成统一研究路线。

## 重要文件

- `docs/DOCS_GUIDE.md`：文档导航、读取顺序和职责边界，优先阅读。
- `docs/MTL_LITE_DESIGN.md`：新架构、模块边界、接口定义和实施路线。
- `docs/RGB_OVERFITTING_AUDIT_PLAN.md`：RGB 过拟合多因素审计与机制路线。
- `docs/CODEX_CONTEXT.md`：Codex 长期上下文。
- `docs/RESEARCH_NOTES.md`：OpenFace 行为表征、相关论文和后续实验路线。
- `docs/SHORTCUT_AUDIT_DESIGN.md`：非抑郁捷径验证框架和实施方案。
- `docs/TODO.md`：分阶段任务计划。
- `docs/BUG_LOG.md`：问题与风险记录。
- `docs/EXPERIMENT_LOG.md`：已完成验证记录。
- `scripts/train.py`：当前训练入口。
- `src/config.py`：配置加载与合并。
- `src/datasets/dataset.py`：数据集与 datamodule。
- `src/models/backbone_factory.py`：通用 backbone factory。
- `src/models/task_heads.py`：通用任务头。
- `src/metrics/metrics.py`：通用指标。

## 已完成基础验证

- debug smoke 已在服务器环境完整通过。
- `scripts.diagnose` import 和 `--help` 检查已通过。
- 已新增 forward smoke 测试。
- 已新增 regression head backward 非零梯度测试。
- 已新增 loss/metric 一致性测试。
- 已新增 legacy full model README，说明旧模型快照边界。
- 已新增 MTL-Lite 输出 dataclass。
- 已新增 MTL-Lite mask-aware pooling 工具。
- 已新增 `MTLLiteDepressionModel` 骨架。
- 已新增 MTL-Lite forward、backward、config 测试。
- 已新增 MTL-Lite 新训练入口 `scripts/train_mtl_lite.py`。
- 已新增 MTL-Lite runner `src/trainers/mtl_lite_runner.py`。
- 已新增 regression-only、MTL-Lite baseline 和 MTL-Lite debug smoke 配置。
- 已新增 MTL-Lite 离线诊断入口 `scripts/diagnose_mtl_lite.py`。
- 已新增 `src/diagnostics/`，支持训练曲线、回归诊断、embedding、相关热力图、遮掩影响热力图、关键帧重要性热力图和模型关注区域热力图。
- MTL-Lite 已支持 `EXTRACT_FEATURE.FREEZE_BACKBONE` 和 `EXTRACT_FEATURE.FINETUNE_LAST_N_BLOCKS`，可通过配置冻结 backbone 或只微调最后若干 transformer blocks。
- 本地 Codex Python 缺少 `torch`、`pytorch_lightning` 和 `pytest`，MTL-Lite import/pytest 需在服务器训练环境验证。

## 历史优先级记录（已被当前入口取代）

以下任务反映早期 MTL-Lite 和 Shortcut Audit 基础设施阶段的优先级。保留它们用于追溯工作来源；新的执行顺序以本文开头的 `当前权威快照` 和 `TODO.md` 的 `当前立即执行任务（权威入口）` 为准。

1. 确保 legacy 中的 `local_paths.yaml`、日志、权重、checkpoint 不进入提交。
2. 记录当前 OpenFace 数据版本、生成命令、输出字段、裁剪尺寸和帧采样方式。
3. 优先实现 Shortcut Audit 最小可行版本：OpenFace quality summary、预测残差相关性、相关性热力图和 markdown 报告。
4. 若保留 OpenFace CSV，统计 `confidence`、`success`、pose、gaze、AU、landmark 抖动与 BDI/预测误差的相关性。
5. 运行输入消融：aligned RGB、grayscale、masked face、landmark heatmap、landmark/AU/pose only。
6. 建立 landmark-only 与 AU/pose/gaze-only temporal baseline。
7. 在行为 baseline 稳定后，再设计 RGB + behavior late fusion 和行为辅助任务 MTL。
8. 在服务器训练环境继续验证 MTL-Lite import、pytest、debug smoke 和离线诊断脚本。

## 持续风险

- legacy 不再作为主要维护对象，除非明确要求复现旧模型结果，否则不修复其内部 import。
- 当前工作区可能包含 legacy 迁移中的文件移动或复制，需要避免误删。
- 根目录和 legacy 中的 `local_paths.yaml` 都不应进入 git。
- 旧模型 debug smoke 通过不代表 MTL-Lite 已可运行。
- 诊断逻辑必须避免污染 validation/test。
- OpenFace aligned face 仍可能包含身份、裁剪伪影、姿态残留、追踪质量和视频质量等非抑郁捷径。
- 不同 OpenFace 版本生成的数据不应混用；升级 OpenFace 应作为独立数据版本和消融实验。
- Shortcut Audit 只能使用 validation/test 已有预测结果和 OpenFace 元数据做离线诊断，不得用 validation/test 统计量反向影响训练配置。
- 当前 BDI ordinal 辅助任务可能不足以强迫模型学习面部行为，后续需要考虑 AU、landmark motion、pose/gaze 等辅助任务。
- 多 GPU DDP metric logging 和 best checkpoint 行为仍需专门验证。
- bf16/mixed precision 稳定性仍需专门验证。

## 推荐验证命令

基础语法检查：

```bash
python -m compileall src scripts tests
```

MTL-Lite import 检查：

```bash
python -c "from src.models.outputs import MTLLiteOutput, MTLLiteLosses; print('outputs import ok')"
python -c "from src.models.temporal.pooling import masked_mean_pool; print('pooling import ok')"
python -c "from src.models.mtl_lite import MTLLiteDepressionModel; print('mtl lite import ok')"
```

legacy 归档检查：

```bash
git diff -- src/legacy/full_model/README.md
```

通用模块检查：

```bash
python -c "from src.models.backbone_factory import build_feature_backbone; print('backbone import ok')"
python -c "from src.models.task_heads import build_regression_task_head; print('task heads import ok')"
python -c "from src.metrics.metrics import ConcordanceCorrCoefMetric, concordance_ccc_loss; print('metrics import ok')"
```

MTL-Lite 测试：

```bash
python -m pytest tests/test_mtl_lite_forward.py tests/test_mtl_lite_loss_backward.py tests/test_mtl_lite_config.py
```

MTL-Lite debug smoke：

```bash
python scripts/train_mtl_lite.py --override configs/mtl_lite_debug_smoke.yaml
```

MTL-Lite baseline：

```bash
python scripts/train_mtl_lite.py --override configs/mtl_lite_baseline.yaml
```

MTL-Lite 离线诊断：

```bash
python scripts/diagnose_mtl_lite.py \
  --run-dir <LOG_DIR>/default/mtl_lite/version_0 \
  --ckpt best
```

诊断输出目录：

```text
<LOG_DIR>/default/mtl_lite/version_0/diagnostics/
```

Regression-only baseline：

```bash
python scripts/train_mtl_lite.py --override configs/regression_only_baseline.yaml
```

旧主线回归测试仅在需要复现 legacy 行为时运行：

```bash
python -m pytest tests/test_model_forward.py tests/test_loss_backward.py tests/test_loss_metric_consistency.py
```

debug smoke：

```bash
python scripts/train.py --override configs/debug_smoke.yaml
```

## 历史记录与证据归档

以下章节按时间保留实现记录、实验结果和历史路线。它们用于追溯证据来源，不覆盖本文开头的当前权威快照。

## 2026-06-13 Shortcut Audit 有效结果更新

`_aligned` 后缀导致的 `video_id` 未匹配问题已经修复并在最新表征数据中复跑确认：
`shortcut_audit_report.md` 显示 `Matched samples: 100`，且 `shortcut_merged.csv`
中的样本均通过完整 `video_id` 匹配。当前 Shortcut Audit 结果可以解释。

最新有效结果显示 shortcut 风险为 medium，最大绝对相关为
`AU07_c_mean` 与 `true_bdi` 的相关性，约 `r = 0.418`。OpenFace 的 AU、gaze、
pose 和质量统计特征与 `true_bdi`、`pred_bdi`、`residual`、`abs_error` 均存在
中等强度信号，说明 aligned face 中仍有非抑郁捷径风险。

当前 MTL-Lite/RGB 预测仍存在明显范围压缩：整体 MAE 约 8.91、RMSE 约 10.95、
Pearson 约 0.35、CCC 约 0.29，预测标准差约 6.13，明显低于真实 BDI 标准差
约 11.48。minimal 组平均高估约 +6.89，severe 组平均低估约 -16.50；后续分析
应优先关注 severe 低估、Freeform/Northwind 同一 subject 预测一致性，以及
behavior-only baseline 是否接近当前 RGB 模型。

`shortcut_predictor_results.csv` 中 in-sample linear/ridge 表现很强，但当前是
100 个样本、94 个特征的训练内诊断结果，不能视为泛化性能。Shortcut Audit 已支持
按 `subject_id` 分组的 shortcut-only predictor 交叉验证，并额外输出
`shortcut_predictor_grouped_cv.csv`。下一步应在服务器复跑 Shortcut Audit，用正式
grouped-CV 结果判断 shortcut-only 特征是否接近当前 RGB 模型。

## 2026-06-13 P0 剩余任务设计

最新 grouped-CV shortcut-only predictor 结果显示，OpenFace shortcut 特征虽然与标签、预测和误差存在中等相关，但不能单独接近当前 RGB/MTL-Lite 模型的测试表现。因此当前风险判断应保持为 medium：需要继续诊断捷径，但不应把后续工作简化为“OpenFace 统计特征已经解释了模型”。接下来的 P0 应围绕模型主要失败模式展开：预测范围压缩、severe 系统性低估、minimal 系统性高估，以及 Freeform/Northwind 同一 subject 预测不一致。

当前 P0 顺序：

1. P0-2：建立 high-error / task-inconsistency case study manifest。固定 severe 低估、minimal 高估、任务间高差异和 low-error reference 样本集合，作为后续 attention、occlusion、keyframe、aligned face 图组的统一入口。
2. P0-3：设计输入消融协议。比较 `rgb`、`grayscale`、`blur`、`center_mask`、`boundary_erased`、`landmark_heatmap` 等输入变体，用于判断模型是否依赖身份纹理、裁剪伪影、边界黑边或非行为区域。
3. P0-4：设计 AU/pose/gaze/landmark-only behavior baseline 接口。建立不依赖 RGB 纹理的行为表征对照，后续再决定是否进入 RGB + behavior late fusion 和行为辅助任务 MTL。

这些任务当前均属于诊断与接口设计，不应修改 `configs/local_paths.yaml`，不应删除或覆盖既有实验结果，也不应在没有明确确认前改变训练超参数。

## 2026-06-14 P0-2 Case Study Manifest 实现

P0-2 已作为离线诊断能力落地。新增 `src/diagnostics/case_studies.py`，用于从已有 prediction/merged rows 中选择以下样本：

- `severe_underestimate`：severe 真实 BDI 高但预测明显偏低；
- `minimal_overestimate`：minimal 真实 BDI 低但预测明显偏高；
- `task_inconsistency`：同一 subject 的 Freeform/Northwind 等任务视频预测差异大；
- `low_error_reference`：低误差对照样本。

该能力已接入两个离线诊断入口：

- `plot_regression_diagnostics()` 会在 regression 诊断目录输出 `case_study_manifest.csv` 和 `case_study_manifest.md`；
- `run_shortcut_audit()` 会在 Shortcut Audit 的 `tables/` 与 `reports/` 下输出同名 manifest 文件。

该实现不改变训练 forward、不改变模型结构、不改变训练超参数，也不依赖 `configs/local_paths.yaml`。后续 P0-3 输入消融和 P0-4 behavior-only baseline 应以该 manifest 固定的 case 集合作为优先复查对象。

## 2026-06-14 P0-3 Input Ablation Variant 接入

P0-3 已以可选 dataset input variant 的方式接入主线数据集，默认行为保持 `rgb`，不会改变既有训练结果。新增 `src/datasets/input_variants.py`，并在 `AVECDataset` 中通过 `DATASET.INPUT_VARIANT` 调用。

当前支持的 RGB 帧输入变体：

- `rgb`：当前 OpenFace aligned RGB baseline；
- `grayscale`：弱化颜色和肤色线索；
- `blur`：弱化身份纹理和细粒度皮肤细节；
- `center_mask`：保留面部中央区域，弱化边界和外围区域；
- `boundary_erased`：弱化裁剪边界、黑边、头发、衣物残留等外围伪影。

`landmark_heatmap` 当前被显式保留给 OpenFace landmark/behavior baseline 路径。由于它需要真实 landmark 坐标，RGB dataset 中不会伪造该输入；如果误配为 `landmark_heatmap`，会直接报错。

默认配置已在 `configs/avec2014_base.yaml` 中加入：

```yaml
DATASET:
  INPUT_VARIANT: "rgb"
```

后续运行输入消融时，应只在 override 中修改该字段，并保持 split、seed、训练入口、checkpoint 选择策略和指标一致。

## 2026-06-14 P0-4 Behavior-only Baseline 接口实现

P0-4 已作为独立 OpenFace 行为表征 baseline 落地。该路线不使用 RGB 帧，不调用 MTL-Lite visual backbone，目标是用结构化行为变量判断当前 RGB 模型是否真正捕捉到可泛化的面部行为动态。

新增模块：

- `src/datasets/openface_features.py`：读取 OpenFace CSV，按 split 匹配视频，构建 AU/pose/gaze/landmark/quality 时序特征，并只用训练集统计量做标准化；
- `src/models/behavior_baseline.py`：OpenFace 特征投影 + GRU + mask-aware pooling + BDI 回归头，可选 ordinal 辅助头；
- `src/trainers/behavior_baseline_runner.py`：独立 Lightning runner；
- `scripts/train_behavior_baseline.py`：独立训练入口；
- `configs/behavior_baseline.yaml`：behavior-only baseline override；
- `tests/test_openface_features.py` 与 `tests/test_behavior_baseline.py`：接口测试。

使用要求：

```bash
python scripts/train_behavior_baseline.py \
  --override configs/behavior_baseline.yaml \
  --override configs/your_openface_paths.yaml
```

其中 `configs/your_openface_paths.yaml` 至少需要提供：

```yaml
DATASET:
  OPENFACE_ROOT: "/path/to/openface_csv_root"
```

该实现不修改 `configs/local_paths.yaml`，不删除或覆盖任何实验结果，也不改变 `scripts/train_mtl_lite.py` 的行为。后续需要在服务器使用真实 OpenFace CSV 运行 debug smoke，并与 RGB regression-only baseline 保持相同 split、seed 和指标进行比较。

## 2026-06-14 Behavior-only baseline 结果与优先级重评估

最新 behavior-only baseline 已能完整训练并输出 `behavior_metrics.csv`。该实验使用 OpenFace 结构化特征而非 RGB 帧，当前结果显示其训练集拟合很强，但泛化仍明显不足：

- test MAE 约 `9.93`，RMSE 约 `12.86`，CCC 约 `0.151`；
- best validation RMSE 出现在 epoch 65，val MAE 约 `9.94`，val RMSE 约 `12.38`，val CCC 约 `0.324`；
- 同一 epoch 的 train MAE 约 `2.17`，train RMSE 约 `2.74`，train CCC 约 `0.975`；
- 最后 epoch 的 train/val RMSE 差距扩大到约 `10.68`。

当前判读：

- behavior-only baseline 暂时不能替代 RGB/MTL-Lite，也不应直接进入 late fusion；
- 该结果更像是 OpenFace 行为特征中存在可记忆的 subject/static geometry 信号，但可泛化行为动态仍没有被稳定提取；
- 原始 landmark 坐标、静态几何、身份相关形状和视频级采集差异可能是主要过拟合来源；
- 当前阶段不应继续把工作重点放在 backbone 解冻层数搜索，也不应因为 train MAE 很低就认为行为表征路线已经成功。

重新评估后的历史优先级：

### P0：必须立即处理

1. 为 behavior baseline 导出 val/test prediction CSV，字段尽量与 RGB/MTL-Lite `test_predictions.csv` 对齐，至少包含 `video_id`、`subject_id`、`task_name`、`true_bdi`、`pred_bdi`、`residual`、`abs_error`。
2. 对 behavior baseline 做特征组消融：quality-only、AU-only、pose+gaze-only、raw-landmark-only、landmark-delta-only、AU+landmark-delta、all-without-raw-landmarks。
3. 在相同 split、seed、metric 和 checkpoint 策略下，对齐比较 RGB/MTL-Lite 与 behavior-only 的整体指标、severity group 误差、Freeform/Northwind 一致性和 case overlap。
4. 记录 OpenFace CSV 匹配数、可用字段、特征维度、缺失字段和标准化统计来源，确保 behavior baseline 不发生 split 泄漏。

### P1：强烈建议处理

1. 在完成特征组消融后，再决定是否默认移除 raw landmark 坐标，优先保留 AU、landmark motion、pose/gaze motion 和 quality mask 等更接近行为动态的特征。
2. 降低 behavior baseline 容量并增强正则化，例如更小 hidden dim、单向 GRU、dropout、weight decay、早停和更短最大 epoch。
3. 建立 behavior prediction 诊断报告，复用 regression diagnostics 与 case study manifest。

### P2：后续优化

1. 只有当某个 behavior 特征子集在验证/测试上稳定后，再尝试 RGB + behavior late fusion。
2. 只有当行为子任务本身稳定后，再将 AU、landmark motion、pose/gaze 等作为 MTL-Lite 辅助任务。
3. 动态任务权重、GradNorm、PCGrad、LDS、`loss_dist` 仍应保持为后续消融项，而不是当前主线。

### P0 执行进展

- 已实现 behavior baseline 的 val/test prediction CSV 导出。
- 导出目录为 `<LOG_DIR>/default/behavior_baseline/version_0/diagnostics/behavior/`。
- `val_predictions.csv` 与 `test_predictions.csv` 已包含 `video_id`、`subject_id`、`task_name`、`true_bdi`、`pred_bdi`、`residual`、`abs_error`、`severity_group`。
- 该导出发生在 best-checkpoint test evaluation 之后，不改变训练 forward、loss、metric、训练超参数或 checkpoint 选择策略。
- 已新增 `BEHAVIOR_FEATURES.FEATURE_SET` 命名特征组入口，默认值为 `custom`，不改变既有 behavior baseline。
- 当前支持 `quality_only`、`au_only`、`pose_gaze_only`、`raw_landmark_only`、`landmark_delta_only`、`au_landmark_delta`、`all_without_raw_landmarks`，用于后续服务器端批量消融。
- 已新增 `scripts/compare_behavior_predictions.py`，可将 RGB/MTL-Lite prediction CSV 与 behavior-only prediction CSV 对齐比较。
- 比较工具输出 `rgb_behavior_prediction_comparison.csv` 和 `rgb_behavior_prediction_summary.csv`，用于检查整体指标、severity 分组和逐样本谁更好。

## 2026-06-25 RGB 黑填充伪迹方向更新

已完成第一轮 RGB 输入消融复盘。各变体训练配置保持同一 split、seed、backbone、冻结策略、时序长度和主要训练入口，测试结果的核心结论如下：

- `center_mask` 当前最好：MAE 约 `7.94`，RMSE 约 `10.16`，Pearson 约 `0.51`，CCC 约 `0.48`；
- 原始 `rgb`：MAE 约 `8.91`，RMSE 约 `10.95`，Pearson 约 `0.35`，CCC 约 `0.29`，存在明显 prediction compression；
- `boundary_erased` 接近或略优于 `rgb`，但不如 `center_mask` 稳定；
- `blur` 和 `grayscale` 明显变差，说明“颜色”或“细粒度身份纹理”不是当前唯一主因；
- severe 低估仍未根治，说明黑边/外围伪迹可能解释一部分泛化问题，但还不能解释全部失败模式。

当时研究判断从“继续叠加 behavior / late fusion 任务”调整为“优先解释 RGB 输入模型为什么过拟合”。更具体的假设是：

```text
OpenFace aligned RGB
-> 纯黑填充 / 黑色遮挡块 / 硬裁剪边界 / 对齐残留
-> ViT 学到不可泛化的边界与像素突变捷径
-> test prediction compression、case-level 错误和 task inconsistency
```

已为下一轮黑伪迹消融接入以下输入变体：

- `black_to_gray`：将近黑区域替换为中性灰；
- `black_to_mean`：将近黑区域替换为当前帧非黑区域均值；
- `black_to_blur`：将近黑区域替换为模糊背景估计；
- `soft_center_mask`：使用软椭圆 mask 平滑边界，而不是制造新的硬边界；
- `inner_crop_resize`：裁掉外围黑边后再 resize，用于验证边界区域是否是主要捷径。

已新增黑伪迹离线审计入口：

```bash
python scripts/audit_black_artifacts.py \
  --predictions <LOG_DIR>/default/rgb/version_0/diagnostics/regression/test_predictions.csv \
  --image-root /path/to/aligned/frame/root \
  --output-dir <LOG_DIR>/default/rgb/version_0/diagnostics/black_artifacts \
  --sample-step 10
```

该脚本会输出每个视频的黑像素比例、边界黑像素比例、中心黑像素比例、黑边界边缘强度、帧间黑像素变化，并与 `true_bdi`、`pred_bdi`、`residual`、`abs_error` 做相关性分析。

下一阶段优先级：

1. 运行更精确的边界黑区消融，优先只处理与图像边界连通的黑色区域，而不是替换全部中心近黑像素。
2. 对 `rgb`、`center_mask`、`black_to_gray`、`soft_center_mask` 和新边界黑区变体生成统一 summary，比较整体指标、severity bias、prediction std 和 Freeform/Northwind 一致性。
3. 对高黑边高误差、高黑边低误差、低黑边高误差三类 case 生成 aligned frame、attention、occlusion、keyframe 图组。
4. 将 severe 低估继续作为独立问题保留，不能把它完全归因于黑边伪迹。
5. 在完成上述证据前，暂不优先推进 RGB + behavior late fusion 或新的复杂辅助任务。

## 2026-06-25 黑伪迹审计后的实验安排

黑伪迹审计已完整匹配 100 个测试视频，`Missing videos = 0`，结果可以解释。审计显示 aligned frame 中黑区非常普遍：整体黑像素均值约 `0.24`，边界黑区均值约 `0.44`。但最大绝对相关只有约 `0.207`，说明黑区不是单独决定 BDI 或预测误差的强变量。

当前更精确的判断：

- `black_border_ratio_mean` 比 `black_center_ratio_mean` 更适合作为 OpenFace 对齐伪迹指标；
- 中心黑像素语义混杂，可能来自鼻孔、自然阴影、胡须、嘴角、麦克风或其他真实遮挡，不能直接当作伪迹；
- 高边界黑区四分位的平均误差明显高于低边界黑区四分位，约 `12.29` vs `7.45`；
- 在 moderate/severe 组内，边界黑区越多，预测越容易偏低，但样本量较小，应作为风险线索而非定论；
- `black_to_gray` 优于原始 `rgb`，但不如 `center_mask`，说明黑像素是因素之一，外围非行为区域、裁剪形状、脸部轮廓和姿态/尺度残留也可能共同作用。

下一轮实验不应继续粗暴替换全部黑像素。推荐新增三类更精确输入变体：

- `border_black_to_gray`：只替换与图像边界连通的近黑区域；
- `border_black_feather`：对边界连通黑区做软过渡，降低硬边界；
- `center_mask_black_to_gray`：在当前最优 `center_mask` 基础上，仅对残留边界连通黑区做中性化，验证二者是否互补。

上述三个变体已接入 `src/datasets/input_variants.py`，对应配置已新增到 `configs/input_ablation/`。本地 compile 验证通过；由于本地 Python 缺少 `pytest` 和 `torch`，还需要在服务器运行聚焦 pytest 与三组训练消融。

case study 优先集合：

- 高黑边高误差：`359_1`、`315_2`、`245_1`；
- 高黑边低误差：`247_3`；
- 低黑边高误差：`237_1`；
- `black_to_gray` 改善明显：`250_1`、`344_2`、`242_1`；
- `black_to_gray` 恶化明显：`206_2`、`226_2`、`210_2`。

## 2026-06-25 RGB 过拟合多因素判断

当前不应把 RGB 过拟合单独归因于黑边。更合理的论文判断是：RGB 模型可能同时利用身份静态外观、OpenFace 对齐几何、边界填充、姿态/追踪质量、视频长度/采样、任务语境差异和标签分布造成的 prediction compression。

权威路线文档：`docs/RGB_OVERFITTING_AUDIT_PLAN.md`。后续关于 RGB 过拟合机制、论文实验顺序和 P0/P1 审计优先级的判断，以该文档为准。当前文档只记录状态摘要。

下一阶段推荐排查顺序已更新为：

```text
split integrity audit
-> temporal sampling audit / ablation
-> training overfit curve summary
-> alignment geometry audit
-> embedding identity retrieval
-> severity calibration verification
-> task inconsistency mixed-factor audit
```

黑边/黑填充方向保留为 input artifact 子证据，而不是总解释。其中 `frame_count` / `sampled_frame_count` 在黑伪迹审计中与 `pred_bdi` 的相关性最高，约 `r = -0.207`，因此时序采样、视频长度和 padding 需要优先验证。severe 低估和 minimal 高估继续作为独立校准问题跟踪，不能仅靠输入伪迹解释。

P0 temporal sampling audit 已落地：

```text
src/diagnostics/temporal_sampling.py
scripts/audit_temporal_sampling.py
tests/test_temporal_sampling_audit.py
```

服务器运行示例：

```bash
python scripts/audit_temporal_sampling.py \
  --predictions <LOG_DIR>/default/rgb/version_0/diagnostics/regression/test_predictions.csv \
  --image-root /path/to/AVEC2014/face_images \
  --output-dir <LOG_DIR>/default/rgb/version_0/diagnostics/temporal_sampling \
  --sample-step 10 \
  --max-seq-len 2000
```

注意：`--sample-step` 和 `--max-seq-len` 应与对应实验的 `resolved_config.yaml` 保持一致。

P0 temporal sampling ablation 已接入：

```text
src/datasets/temporal_sampling.py
PROCESS_TEMPORAL.SAMPLING_STRATEGY
configs/temporal_sampling/
```

默认 `stride_head` 保持旧行为不变。服务器训练示例：

```bash
python scripts/train_mtl_lite.py \
  --override configs/regression_only_baseline.yaml \
  --override configs/input_ablation/rgb.yaml \
  --override configs/temporal_sampling/uniform_256.yaml

python scripts/train_mtl_lite.py \
  --override configs/regression_only_baseline.yaml \
  --override configs/input_ablation/rgb.yaml \
  --override configs/temporal_sampling/uniform_512.yaml

python scripts/train_mtl_lite.py \
  --override configs/regression_only_baseline.yaml \
  --override configs/input_ablation/rgb.yaml \
  --override configs/temporal_sampling/uniform_1024.yaml
```

temporal crop 对照：

```bash
python scripts/train_mtl_lite.py \
  --override configs/regression_only_baseline.yaml \
  --override configs/input_ablation/rgb.yaml \
  --override configs/temporal_sampling/first_crop.yaml

python scripts/train_mtl_lite.py \
  --override configs/regression_only_baseline.yaml \
  --override configs/input_ablation/rgb.yaml \
  --override configs/temporal_sampling/middle_crop.yaml

python scripts/train_mtl_lite.py \
  --override configs/regression_only_baseline.yaml \
  --override configs/input_ablation/rgb.yaml \
  --override configs/temporal_sampling/random_crop.yaml
```

若 `uniform_512` 或 `uniform_1024` 触发 OOM，应先降低 `EXTRACT_FEATURE.BATCH_SIZE` 或 `EXTRACT_FEATURE.CHUNK_SIZE`，不要同时改动其他训练超参数。

## 2026-06-25 边界连通黑区消融结果

三组更精确的边界连通黑区消融已经完成，配置与前序 RGB 消融可比，区别仅在 `DATASET.INPUT_VARIANT`。

核心结果：

- `center_mask_black_to_gray` 当前整体 MAE/RMSE/Pearson 最好：MAE 约 `7.73`，RMSE 约 `10.02`，Pearson 约 `0.52`，CCC 约 `0.45`；
- `center_mask` 仍保持最高 CCC，约 `0.48`，且对 moderate/severe 更均衡；
- `border_black_feather` 明显优于原始 `rgb` 和 `black_to_gray`，MAE 约 `8.04`，RMSE 约 `10.13`，说明边界软化有效；
- `border_black_to_gray` 表现较弱，说明硬替换边界黑区不如 feather，也不如 center mask。

重要判读：

- `center_mask_black_to_gray` 的总体提升主要来自 minimal/mild，minimal MAE 从 `center_mask` 的约 `7.68` 降至约 `6.13`；
- 但它加重 severe 低估，severe MAE 约 `17.46`，比 `rgb` 和 `center_mask` 更差；
- 因此它不是全面更优，而是更偏低预测、更保守的校准版本；
- `border_black_feather` 对 severe 比 `center_mask` 更好，但 task consistency 比 `rgb` / `center_mask` 差；
- 后续必须将输入伪迹处理和 severity calibration 分开研究。

P0 prediction run summary 工具已新增：

```text
src/diagnostics/prediction_runs.py
scripts/summarize_prediction_runs.py
tests/test_prediction_runs.py
```

该工具用于统一输出 `prediction_run_summary.csv`、`severity_bias_summary.csv`、`task_consistency_summary.csv`、`pairwise_baseline_improvement.csv` 和 `prediction_runs_report.md`，避免后续每轮消融手工计算 prediction std、severity bias 和 task consistency。

已使用该工具对当前可用的 RGB input ablation 预测文件完成统一汇总。包含 `rgb`、`gray_scale`、`blur`、`boundary_erased`、`center_mask`、`black_to_gray`、`black_to_mean`、`black_to_blur`、`soft_center_mask`、`inner_crop_resize`、`border_black_feather`、`border_black_to_gray` 和 `center_mask_black_to_gray`。

推荐服务器复现命令：

```bash
python scripts/summarize_prediction_runs.py \
  --baseline rgb \
  --output-dir analysis_outputs/rgb_input_ablation_summary \
  --run rgb=<LOG_DIR>/default/rgb/version_0/diagnostics/regression/test_predictions.csv \
  --run gray_scale=<LOG_DIR>/default/gray_scale/version_0/diagnostics/regression/test_predictions.csv \
  --run blur=<LOG_DIR>/default/blur/version_0/diagnostics/regression/test_predictions.csv \
  --run boundary_erased=<LOG_DIR>/default/boundary_erased/version_0/diagnostics/regression/test_predictions.csv \
  --run center_mask=<LOG_DIR>/default/center_mask/version_0/diagnostics/regression/test_predictions.csv \
  --run black_to_gray=<LOG_DIR>/default/rgb_ablation_black_to_gray/version_0/diagnostics/regression/test_predictions.csv \
  --run black_to_mean=<LOG_DIR>/default/rgb_ablation_black_to_mean/version_0/diagnostics/regression/test_predictions.csv \
  --run black_to_blur=<LOG_DIR>/default/rgb_ablation_black_to_blur/version_0/diagnostics/regression/test_predictions.csv \
  --run soft_center_mask=<LOG_DIR>/default/rgb_ablation_soft_center_mask/version_0/diagnostics/regression/test_predictions.csv \
  --run inner_crop_resize=<LOG_DIR>/default/rgb_ablation_inner_crop_resize/version_0/diagnostics/regression/test_predictions.csv \
  --run border_black_feather=<LOG_DIR>/default/rgb_ablation_border_black_feather/version_0/diagnostics/regression/test_predictions.csv \
  --run border_black_to_gray=<LOG_DIR>/default/rgb_ablation_border_black_to_gray/version_0/diagnostics/regression/test_predictions.csv \
  --run center_mask_black_to_gray=<LOG_DIR>/default/rgb_ablation_center_mask_black_to_gray/version_0/diagnostics/regression/test_predictions.csv
```

统一表格的当前排序进一步支持以下判断：

- 整体 MAE 排名前三为 `center_mask_black_to_gray`、`center_mask`、`border_black_feather`；
- `center_mask` 仍保持最高 CCC 和接近原始 `rgb` 的 task consistency；
- `gray_scale`、`blur`、`inner_crop_resize` 和 `black_to_mean` 未能改善整体结果，说明过拟合不能简单解释为颜色、纹理或外围区域单因素；
- 所有较好变体的 `pred_std` 仍低于 `true_std`，severe 低估仍需作为 calibration / severity imbalance 问题单独处理。

## 2026-06-25 高优先级过拟合验证审查

当前不建议继续把主要精力放在新增 RGB mask 变体上。黑边/黑填充已被证明是可见风险入口，但不是单一充分解释；继续增加局部 mask 容易变成经验试错，论文价值不如系统审计过拟合机制。

下一批更高优先级任务如下：

1. **Split / subject integrity audit**：确认 train/val/test subject-disjoint，同一 subject 的 Freeform/Northwind 不跨 split，不存在重复视频目录、重复标签或 video_id 规范化错配。该项优先级最高，因为 split 一旦有问题，所有泛化结论都会被污染。
2. **Temporal sampling audit / ablation**：已接入实现，待服务器运行真实审计和六组训练消融。当前 `frame_count` / `sampled_frame_count` 与预测的相关性提示时序长度、截断或采样覆盖可能是隐藏捷径。
3. **Training overfit curve summary**：跨 run 汇总 best epoch、train/val RMSE gap、train/val MAE gap、val 最优后是否继续过拟合。该项可解释 behavior baseline 的 train-test gap，也可比较各 RGB 输入变体是否只是改变预测偏置。
4. **OpenFace alignment geometry audit**：统计 landmark bbox area/width/height/aspect、face center offset、eye distance 和 face scale，与标签、预测、残差和绝对误差相关。该项应从 P1 提到 P0，因为 `center_mask` 有效而 `inner_crop_resize` 变差，说明几何和尺度因素可能比单纯外围裁剪更关键。
5. **Embedding identity retrieval audit**：优先做同 subject Freeform/Northwind embedding paired retrieval，而不是直接训练 subject classifier。若同 subject embedding 高度互为近邻，说明 RGB backbone 编码了强身份/静态外观信息。
6. **Severity calibration verification**：使用 val predictions 拟合 post-hoc linear calibration，再应用到 test，检查 severe 低估和 minimal 高估是否缓解。该项用于区分输入捷径与 loss/标签分布导致的 prediction compression。
7. **Task inconsistency mixed-factor audit**：将 Freeform/Northwind prediction diff 与 frame_count、black-border、confidence、pose/gaze、alignment geometry 关联，判断任务不一致是否由可观测混杂变量驱动。

推荐执行顺序：

```text
split integrity audit
-> temporal sampling audit / ablation
-> training overfit curve summary
-> alignment geometry audit
-> embedding identity retrieval
-> severity calibration verification
-> task inconsistency mixed-factor audit
```

当前实验主线应写作“RGB 过拟合的多因素机制审计”，而不是“黑边导致过拟合”。黑边方向已足以支撑一个子结论：边界硬伪迹会影响泛化，但输入伪迹处理无法单独解决 severe 低估和 prediction compression。

## 2026-06-16 P0-A Split Integrity Audit 实现

P0-A split / subject integrity audit 已作为离线诊断能力落地。该任务优先级最高，因为如果 train/val/test 存在 subject 泄漏、同一 subject 的 Freeform/Northwind 跨 split、重复 video_id、label 错配或 prediction 与 split 无法唯一对齐，后续关于 RGB 过拟合、黑边、时序采样和 identity shortcut 的解释都会被污染。

新增实现：

```text
src/diagnostics/split_integrity.py
scripts/audit_split_integrity.py
tests/test_split_integrity.py
```

输出：

```text
tables/split_video_manifest.csv
tables/split_subject_overlap.csv
tables/split_label_distribution.csv
tables/split_prediction_alignment.csv  # 仅在提供 --predictions 时生成
reports/split_integrity_report.md
```

服务器运行示例：

```bash
python scripts/audit_split_integrity.py \
  --split-file /path/to/dataset_split.json \
  --label-dir /path/to/labels \
  --image-root /path/to/aligned/frame/root \
  --predictions <LOG_DIR>/default/rgb/version_0/diagnostics/regression/test_predictions.csv \
  --output-dir <LOG_DIR>/default/rgb/version_0/diagnostics/split_integrity
```

判读约束：

- `split_integrity_report.md` 中 `status: PASS` 才能继续把后续测试结果当作 subject-disjoint 泛化结果解释；
- 若存在 `subject_split_overlap`，必须先修复 split 或单独报告污染风险；
- 若 prediction alignment 出现 `missing_in_split` 或 `ambiguous`，不能继续解释对应 prediction-level 诊断；
- 该脚本只读 split、label、image root 和 prediction CSV，不改变训练 forward、loss、metric 或 checkpoint。

## 2026-06-16 P0-B Training Overfit Summary 实现

P0-B training overfit curve summary 已作为跨 run 离线审计能力落地。该任务用于判断输入变体或 behavior baseline 的表观提升是否伴随更大的 train/val gap，避免把测试预测偏置变化误解释为真正泛化提升。

新增实现：

```text
src/diagnostics/training_overfit.py
scripts/summarize_training_overfit.py
tests/test_training_overfit.py
```

输出：

```text
tables/training_overfit_summary.csv
tables/training_curve_gap_by_run.csv
reports/training_overfit_report.md
```

服务器运行示例：

```bash
python scripts/summarize_training_overfit.py \
  --output-dir analysis_outputs/training_overfit_summary \
  --run rgb=/path/to/rgb/metrics.csv \
  --run center_mask=/path/to/center_mask/metrics.csv \
  --run center_mask_black_to_gray=/path/to/center_mask_black_to_gray/metrics.csv \
  --run border_black_feather=/path/to/border_black_feather/metrics.csv \
  --run behavior=/path/to/behavior_baseline/metrics.csv
```

判读约束：

- best epoch 优先按 `val_RMSE_epoch` 选择；若缺失则回退到 `val_MAE_epoch`，再回退到 `val_loss`；
- gap 定义为 `validation - train`，正值越大表示泛化缺口越大；
- `overfit_after_best_val=True` 表示 best val epoch 之后 validation 变差而 train 指标继续改善；
- 该脚本只读取 Lightning `metrics.csv`，不读取 test prediction，不改变训练或 checkpoint。

## 2026-06-16 P0-C Alignment Geometry Audit 实现

P0-C OpenFace alignment geometry audit 已作为离线诊断能力落地。该任务用于把黑边之外的 OpenFace 对齐几何显式量化，检查 face scale、landmark bbox、face center offset、eye distance 和 landmark jitter 是否与 BDI、预测残差或绝对误差相关。

新增实现：

```text
src/diagnostics/alignment_geometry.py
scripts/audit_alignment_geometry.py
tests/test_alignment_geometry.py
```

输出：

```text
tables/alignment_geometry_summary.csv
tables/alignment_geometry_merged.csv
tables/alignment_geometry_correlation.csv
tables/alignment_geometry_group_summary.csv
reports/alignment_geometry_audit_report.md
```

服务器运行示例：

```bash
python scripts/audit_alignment_geometry.py \
  --predictions <LOG_DIR>/default/rgb/version_0/diagnostics/regression/test_predictions.csv \
  --openface-root /path/to/openface_csv_root \
  --output-dir <LOG_DIR>/default/rgb/version_0/diagnostics/alignment_geometry \
  --frame-width 112 \
  --frame-height 112 \
  --sample-step 1
```

判读约束：

- `alignment_geometry_correlation.csv` 用于检查几何变量与 `true_bdi`、`pred_bdi`、`residual`、`abs_error` 的线性关系；
- `alignment_geometry_group_summary.csv` 用于检查 severe 低估是否集中在异常 face scale、center offset 或 jitter；
- 如果 OpenFace landmark 坐标不是 aligned 112x112 坐标，应改用对应坐标尺度运行 `--frame-width` 和 `--frame-height`；
- 该脚本只读 OpenFace CSV 和 prediction CSV，不改变训练数据、split、loss、metric 或 checkpoint。

## 2026-06-16 P0-C Alignment Geometry 结果与坐标尺度修正

P0-C 已在真实 OpenFace CSV root 和当前 RGB `test_predictions.csv` 上运行，匹配质量正常：

```text
OpenFace videos summarized: 300
Matched prediction rows: 100
Missing prediction videos: 0
Max absolute correlation: 0.3746
```

主要相关性：

```text
landmark_bbox_height_mean  vs true_bdi:  r = 0.3746
normalized_face_scale_mean vs true_bdi:  r = 0.3410
landmark_bbox_area_mean    vs true_bdi:  r = 0.3410
landmark_bbox_width_mean   vs true_bdi:  r = 0.3041
eye_distance_mean          vs true_bdi:  r = 0.2991
landmark_bbox_height_mean  vs residual:  r = -0.2605
```

这说明 OpenFace landmark 检测几何与 BDI/severity 存在中等相关，并且更大的 bbox/scale 与更负 residual 相关，和 severe 低估方向一致。

坐标尺度检查确认：当前 OpenFace CSV 的 landmark 坐标不是 112x112 aligned face 坐标。示例 CSV 中 `x` 范围可达约 `150-643`，`y` 范围可达约 `-11-582`；而实际模型输入 jpg 为 `112 x 112`。OpenFace 日志中的 camera parameters `500,500,320,240` 提示源坐标系约为 `640 x 480`。因此，C 任务应解释为 **pre-alignment detection geometry confound**，而不是模型直接看到的 112x112 landmark geometry。

在使用 `--frame-width 640 --frame-height 480` 重跑后，当前可解释指标：

```text
landmark_bbox_width_mean
landmark_bbox_height_mean
landmark_bbox_area_mean
landmark_bbox_aspect_mean
eye_distance_mean
landmark_jitter_mean
normalized_face_scale_mean
face_center_offset_x_mean
face_center_offset_y_mean
```

这些指标反映 OpenFace 原始检测坐标系下的脸部尺度、检测几何和 landmark 稳定性。它们可能通过后续裁剪、对齐、缩放、黑边填充和插值过程间接影响 112x112 RGB 输入。

640x480 重跑后的 severity 分组显示：

```text
minimal:  true=4.96,  pred=11.85, residual=+6.89, scale=0.319
mild:     true=16.20, pred=14.34, residual=-1.86, scale=0.337
moderate: true=25.00, pred=17.34, residual=-7.66, scale=0.399
severe:   true=34.14, pred=17.64, residual=-16.50, scale=0.364
```

moderate / severe 的检测尺度明显大于 minimal，而模型预测没有随真实 BDI 同步上升，支持 prediction compression 与检测几何混杂并存的解释。后续仍建议增强 geometry 审计，显式输出原始 `landmark_x_min/x_max/y_min/y_max` 和相对比例指标，如 `eye_distance_to_bbox_height_ratio`，减少对固定 frame size 的依赖。

## 2026-06-16 P0-B Temporal Sampling Audit 真实运行结果

P0-B temporal sampling audit 已在当前 RGB `test_predictions.csv` 上完成真实运行，匹配质量正常：

```text
Videos summarized: 100
Matched prediction rows: 100
Missing videos: 0
Max absolute correlation: 0.2197
```

主要相关性显示，视频长度、截断和模型可见帧数与预测值存在弱到中等关系：

```text
truncated_frame_count vs pred_bdi: r = -0.2197
truncated_ratio       vs pred_bdi: r = -0.2090
raw_to_selected_ratio vs pred_bdi: r =  0.2090
sampled_frame_count   vs pred_bdi: r = -0.2073
frame_count           vs pred_bdi: r = -0.2073
```

该结果说明 temporal sampling 是 RGB 过拟合和 prediction compression 的候选混杂因素，但不是单独充分解释。最长视频四分位没有 padding，但平均预测更低、误差更高；这更像是 head-only 截断、长视频中性片段稀释或关键片段覆盖不足，而不是简单 padding 问题。Freeform 视频整体更长、padding 更少，Northwind 更短、padding 更多，但两类任务的平均绝对误差接近，因此任务级差异不能仅由 padding 解释。

当时建议运行已接入的 temporal sampling 训练消融：`uniform_256`、`uniform_512`、`uniform_1024`、`first_crop`、`middle_crop` 和 `random_crop`。在当前粗粒度 task-nuisance 路线下，这些实验保留为 task/context 混杂审计和 group-wise evaluation 证据，不作为模型路线。
## 2026-06-17 局部遮挡与静态外观捷径审计方向

眼镜、麦克风、胡须等因素应被纳入后续 RGB 过拟合机制审计，但它们不应替代当前多因素主线。更准确的定位是：它们位于 **identity/static appearance shortcut** 与 **local occlusion artifact** 的交叉处。

理论依据包括 shortcut learning、面部遮挡识别和遮挡鲁棒表情识别相关研究。对本项目而言，这些因素有三类风险：

- 作为 subject identity 线索：眼镜、胡须、发际线和局部纹理可能帮助模型识别 subject，而不是学习抑郁相关行为；
- 作为局部遮挡：麦克风或口鼻附近遮挡会破坏嘴部、下半脸和 AU/landmark 观测；
- 作为 patch-level artifact：眼镜反光、麦克风黑块和胡须边界可能形成高对比局部 patch，被 DeiT/ViT backbone 放大。

后续建议先做 case-study 与区域遮挡验证，而不是立即全局删除这些区域。优先检查 severe 低估、minimal 高估、高 task diff、以及 `middle_crop` 明显改善/恶化的样本，判断模型关注或遮挡敏感性是否集中在眼镜、麦克风、胡须、下半脸或边界残留上。

## 2026-06-18 OpenFace 边界硬突变和平滑过渡消融计划

人脸裁剪边界的硬像素突变值得作为 input artifact 子机制继续验证。当前 `border_black_feather` 已明显优于原始 `rgb` 和普通 `black_to_gray`，说明问题可能不只是黑色填充面积，而是黑区与脸部区域之间的高对比过渡被 DeiT/ViT patch backbone 放大。

该方向的定位是：**OpenFace aligned face hard-transition artifact**。它属于输入 artifact 与 ViT patch shortcut 的交叉机制，不应替代 identity、geometry、temporal 和 calibration 等主线。

下一轮不建议继续做全图 blur 或粗暴全黑替换，而应做精确边界消融：

```text
edge_soften_only              # 只降低边界连通黑区与脸部交界处的高梯度，不改变大面积黑区
border_blur_fill              # 用邻近非黑区域的模糊颜色填充边界连通黑区
border_reflect_fill           # 用边界邻近像素反射/延展填充黑区
border_feather_blur_fill      # feather mask + blur/inpaint-like fill，验证软过渡是否优于固定灰色
center_mask_soft_boundary_v2  # 在 center_mask 上使用更宽、更自然的软边界
```

优先只做两组：`edge_soften_only` 和 `border_blur_fill`。它们最直接回答：模型到底依赖黑色面积，还是依赖黑色-肤色之间的硬突变边缘。

对照组应固定为：

```text
rgb
center_mask
black_to_gray
border_black_feather
center_mask_black_to_gray
```

判读时必须同时报告 overall metrics、prediction std、severity bias、task consistency 和 pairwise improvement。若平滑边界改善 MAE 但继续加重 severe 低估，则应解释为 artifact mitigation，而不是完整泛化解决方案。

## 2026-06-18 Temporal Sampling 消融与过拟合结果

六组 temporal sampling 训练消融已经完成，并已通过 `scripts/summarize_prediction_runs.py` 与 `scripts/summarize_training_overfit.py` 汇总。结论是：temporal location 会影响预测，但简单替换采样策略不能解决 RGB 过拟合。

整体 test 指标：

```text
middle_crop   MAE=8.8014, RMSE=10.7291, Pearson=0.4192, CCC=0.3806, pred_std=7.3842
uniform_512   MAE=8.8661, RMSE=10.9208, Pearson=0.3606, CCC=0.2966, pred_std=6.1048
uniform_1024  MAE=8.8684, RMSE=10.9257, Pearson=0.3621, CCC=0.2981, pred_std=6.1294
uniform_256   MAE=8.8686, RMSE=10.9214, Pearson=0.3615, CCC=0.2974, pred_std=6.1142
first_crop    MAE=8.8746, RMSE=10.9844, Pearson=0.3272, CCC=0.2522, pred_std=5.4438
rgb           MAE=8.9145, RMSE=10.9530, Pearson=0.3526, CCC=0.2925, pred_std=6.1587
random_crop   MAE=9.0369, RMSE=11.2329, Pearson=0.3847, CCC=0.3631, pred_std=8.1834
```

关键判读：

- `middle_crop` 是本轮最有价值的策略，整体 MAE/RMSE/Pearson/CCC 均优于 `rgb`，并将 severe 组 bias 从 `-16.50` 缓解到 `-14.55`；
- `middle_crop` 同时显著恶化 task consistency，Freeform/Northwind 平均预测差异从 `2.89` 增至 `4.63`，说明中段片段可能引入任务语境混杂；
- `uniform_256/512/1024` 几乎等价，说明瓶颈不是简单的均匀采样帧数不足；
- `first_crop` 加重 prediction compression，severe bias 变为 `-17.42`，不应作为替代策略；
- `random_crop` 增大 `pred_std` 并缓解部分 severe 低估，但整体 MAE/RMSE 最差且 task consistency 最差，不稳定。

训练曲线过拟合汇总显示，所有 temporal run 都是 `overfit_after_best_val=True`：

```text
middle_crop   best_val_rmse=10.5861, best_epoch=11, last_gap=6.5087
rgb           best_val_rmse=10.7153, best_epoch=6,  last_gap=6.4181
uniform_256   best_val_rmse=10.8494, best_epoch=6,  last_gap=6.4019
uniform_512   best_val_rmse=10.8587, best_epoch=6,  last_gap=6.4607
uniform_1024  best_val_rmse=10.8717, best_epoch=6,  last_gap=6.4424
first_crop    best_val_rmse=11.1440, best_epoch=6,  last_gap=6.5589
random_crop   best_val_rmse=11.2404, best_epoch=9,  last_gap=6.5972
```

因此，temporal sampling 的论文结论应保持克制：**temporal location matters, but simple sampling replacement does not solve RGB overfitting**。后续不建议继续扩展 `uniform_2048` 或更多普通 crop；更高价值的是 task inconsistency mixed-factor audit、severity calibration 和 identity/static appearance retrieval。

## 2026-06-18 Identity Retrieval Multi-run 结果与任务走向

已对 `rgb`、`center_mask`、`center_mask_black_to_gray`、`border_black_feather` 和 `middle_crop` 分别在 test / val 上运行 embedding identity retrieval。该结果是当前 RGB shortcut 机制中最直接的证据之一。

核心 test 结果：

```text
rgb                       same_top1=0.66, top5=0.85, severity_agree=0.492, paired_rank_mean=5.09
center_mask               same_top1=0.67, top5=0.86, severity_agree=0.498, paired_rank_mean=4.12
center_mask_black_to_gray same_top1=0.68, top5=0.84, severity_agree=0.522, paired_rank_mean=5.12
border_black_feather      same_top1=0.75, top5=0.90, severity_agree=0.550, paired_rank_mean=2.91
middle_crop               same_top1=0.49, top5=0.65, severity_agree=0.454, paired_rank_mean=10.62
```

核心 val 结果：

```text
rgb                       same_top1=0.52, top5=0.73, severity_agree=0.518, paired_rank_mean=6.24
center_mask               same_top1=0.45, top5=0.80, severity_agree=0.480, paired_rank_mean=5.79
center_mask_black_to_gray same_top1=0.50, top5=0.76, severity_agree=0.462, paired_rank_mean=7.34
border_black_feather      same_top1=0.67, top5=0.80, severity_agree=0.494, paired_rank_mean=4.52
middle_crop               same_top1=0.45, top5=0.71, severity_agree=0.530, paired_rank_mean=7.02
```

主要判读：

- RGB baseline 的 same-subject top-1 为 `0.66`、top-5 为 `0.85`，paired-task median rank 为 `1`，说明 embedding 明显保留 subject identity / static appearance；
- `border_black_feather` 在 test 上 identity retrieval 最强，same-subject top-1 达 `0.75`，说明边界软化虽然改善部分预测指标，但并没有去身份化，可能只是降低 artifact noise 同时保留甚至强化稳定外观；
- `center_mask` 和 `center_mask_black_to_gray` 在 test 上并未明显降低 identity retrieval，因此不能把它们解释为 de-identification 方法；
- `middle_crop` 明显降低 same-subject retrieval，但它同时恶化 task consistency，说明降低身份稳定性不等于形成稳定 severity representation；
- severity neighbor agreement 没有随着 identity retrieval 降低而稳定提升，当前还没有任何变体同时满足低 identity、高 severity agreement、低 task inconsistency 和良好 BDI 指标。

当前任务走向：

1. 构建 `scripts/summarize_identity_retrieval_runs.py`，将多个 identity retrieval 输出汇总为统一表格和报告；
2. 输出 `identity_retrieval_run_summary.csv`、`identity_retrieval_severity_summary.csv` 和 `identity_retrieval_runs_report.md`；
3. 后续把 identity retrieval summary 与 prediction summary 合并，形成论文核心表：`MAE/RMSE/CCC/pred_std/severe_bias/task_diff/same_subject_top1/same_subject_top5/severity_agree/paired_rank_mean`；
4. 生成 high-identity / high-error case study，重点检查 `border_black_feather` severe 高身份样本、`middle_crop` 身份下降但 task diff 上升样本、moderate identity retrieval 失败样本；
5. 暂不把 `center_mask` 称为去身份化方法，应更准确地称为 input artifact / peripheral-region mitigation。

## 2026-06-18 RGB Baseline Severity Calibration 结果与下一步研究方向

已对 RGB baseline 运行 `severity_calibration` 诊断。该结果把 RGB 过拟合问题进一步具体化为 **prediction range compression / severity miscalibration**：模型不是完全没有 BDI 排序信号，而是输出范围明显被压缩到中间区间，导致 minimal 系统性高估、severe 系统性低估。

Validation-only 线性校准拟合：

```text
pred_calibrated = 0.870402 * pred + 2.689282
val_count=100, val_MAE=8.4228, val_RMSE=10.7286, val_Pearson=0.4452, val_CCC=0.3599
val_true_std=11.92, val_pred_std=6.10
```

Test 原始预测与校准后结果：

```text
original:   MAE=8.9145, RMSE=10.9530, Pearson=0.3526, CCC=0.2925, true_std=11.48, pred_std=6.13
calibrated: MAE=8.8460, RMSE=10.8278, Pearson=0.3526, CCC=0.2692, true_std=11.48, pred_std=5.33
```

Severity group bias：

```text
original minimal:  residual=+6.8856,  MAE= 8.2522
original mild:     residual=-1.8586,  MAE= 6.1840
original moderate: residual=-7.6645,  MAE= 7.7576
original severe:   residual=-16.5032, MAE=16.5032

calibrated minimal:  residual=+8.0397,  MAE= 8.6831
calibrated mild:     residual=-1.0279,  MAE= 5.4265
calibrated moderate: residual=-7.2219,  MAE= 7.2827
calibrated severe:   residual=-16.0999, MAE=16.0999
```

关键判读：

- RGB baseline 的 `pred_std=6.13`，仅为 `true_std=11.48` 的约一半，说明模型输出被明显压缩；
- validation-fit 线性校准几乎不改变 Pearson，只轻微改善 MAE/RMSE，但使 CCC 从 `0.2925` 降至 `0.2692`，说明简单后处理不能解决表征层面的 severity 失准；
- severe 组 residual 从 `-16.50` 到 `-16.10`，几乎未被修复；minimal 组反而从 `+6.89` 加重到 `+8.04`；
- 该结果支持“弱排序信号 + 强均值收缩”的机制，而不是“整体平移一下预测即可解决”；
- 结合 identity retrieval，当前 RGB 表征同时存在身份/静态外观保留和 severity 动态范围不足的问题。

下一步研究方向需要更具体地分成四条线：

1. **Severity calibration multi-run summary**：把 `rgb`、`middle_crop`、`border_black_feather`、`center_mask`、`center_mask_black_to_gray` 和关键 temporal/input 变体都跑同一 calibration 诊断，统一比较 `pred_std/true_std`、minimal residual、severe residual、CCC 变化和校准前后 trade-off。
2. **Identity-residual 联合分析**：将 prediction summary、identity retrieval summary 和 severity calibration summary 合并，形成论文核心机制表，检查 `same_subject_top1/top5` 是否与 severe 低估、prediction compression、task inconsistency 同向变化。
3. **Severity-aware training ablation**：在完成诊断汇总后再进入训练改进，优先测试 severity-balanced sampler、severity-weighted regression loss、ordinal severity auxiliary head，以及 MAE/Huber/CCC 或混合损失；目标是提高 `pred_std`、降低 severe underestimation，同时不显著增加 identity retrieval。
4. **Case study and representation check**：对 severe 高误差、高身份检索样本生成图组和 embedding 邻居列表，重点检查眼镜、麦克风、胡须、局部遮挡、OpenFace 尺度和任务片段是否共同出现。

当前不建议把线性校准作为最终模型方案。它的价值是证明 RGB baseline 的主要失败之一是 severity 动态范围不足，并为后续 severity-aware 训练和身份捷径联合分析提供依据。

## 2026-06-18 RGB Input / Identity / Calibration 三表联合结果

已完成 `rgb_input_ablation_summary`、`identity_retrieval_summary` 和 `severity_calibration_summary` 的联合审阅，覆盖同一组关键 run：`rgb`、`center_mask`、`border_black_feather`、`middle_crop`、`center_mask_black_to_gray`。当前结论已经从单点输入消融推进到多机制交叉解释。

整体性能显示，`center_mask_black_to_gray` 的 MAE 最低，但 `center_mask` 的 CCC 最高、prediction std 最大且 task consistency 接近 baseline：

```text
center_mask_black_to_gray: MAE=7.726, CCC=0.453, pred_std=7.020, task_diff=3.735
center_mask:               MAE=7.942, CCC=0.477, pred_std=8.132, task_diff=2.955
border_black_feather:      MAE=8.041, CCC=0.409, pred_std=6.264, task_diff=3.300
middle_crop:               MAE=8.801, CCC=0.381, pred_std=7.384, task_diff=4.634
rgb:                       MAE=8.915, CCC=0.293, pred_std=6.159, task_diff=2.890
```

Severity bias 显示输入变体主要是在重新分配不同严重程度组的误差，而不是消除 severe underestimation：

```text
center_mask:               minimal_bias=+4.61, moderate_bias=-2.08, severe_bias=-15.76
center_mask_black_to_gray: minimal_bias=+3.64, mild_bias=-0.18, severe_bias=-17.46
border_black_feather:      minimal_bias=+8.40, moderate_bias=-4.79, severe_bias=-12.73
rgb:                       minimal_bias=+6.89, moderate_bias=-7.66, severe_bias=-16.50
middle_crop:               minimal_bias=+6.56, moderate_bias=-6.48, severe_bias=-14.55
```

Identity retrieval 显示没有任何输入变体同时实现低身份检索、高 severity agreement、低 task inconsistency 和良好 BDI 指标：

```text
rgb:                       same_top1=0.66, same_top5=0.85, severity_agree=0.492
center_mask:               same_top1=0.67, same_top5=0.86, severity_agree=0.498
border_black_feather:      same_top1=0.75, same_top5=0.90, severity_agree=0.550
middle_crop:               same_top1=0.49, same_top5=0.65, severity_agree=0.454
center_mask_black_to_gray: same_top1=0.68, same_top5=0.84, severity_agree=0.522
```

Severity calibration summary 进一步确认 post-hoc linear calibration 不是解决方案。所有 run 的 calibration 都降低 CCC，并进一步压缩 prediction std；它们最多微调 MAE/RMSE，不能恢复 high-severity 端动态范围。因此 calibrated predictions 不应作为主结果。

当前各 run 的机制定位：

- `center_mask`：当前最健康的 input artifact mitigation 证据，CCC 最高，moderate 组改善明显，task consistency 接近 baseline；但 identity retrieval 未下降，不能称为去身份化。
- `center_mask_black_to_gray`：MAE 最低，minimal/mild 最好，但 severe bias 最差；可作为“MAE 可能误导模型选择”的关键反例。
- `border_black_feather`：severe 低估最轻，但 minimal 高估最重且 identity retrieval 最强；说明边界软化可能缓解 artifact noise，同时保留甚至强化 subject/static appearance shortcut。
- `middle_crop`：identity retrieval 明显下降，但 severity agreement 和 task consistency 变差；说明降低身份信息不等于获得稳定 severity representation，temporal/task context 是独立混杂因素。
- `rgb`：原始 baseline，表现出强 prediction compression、severe underestimation 和身份/静态外观保留。

当前阶段性论文结论：RGB input ablations can improve overall error by redistributing severity-group bias, but they do not eliminate identity shortcut or severe underestimation. Center-focused inputs provide the most stable artifact-mitigation evidence, while black replacement, boundary smoothing, and temporal cropping expose trade-offs among severity bias, identity retrieval, and task consistency.

## 历史阶段：2026-06-25 Shortcut-Regularized MTL 收束路线（已被粗粒度主线继承）

本节保留为历史阶段记录。该阶段曾将研究路线从“继续扩展输入滤镜/边界变体”收束为 **Shortcut-Regularized MTL**；截至 2026-07-06，它已被粗粒度 task-nuisance 主线继承，作为 identity-adversarial baseline、severity-balanced baseline 和稳健性验证的一部分。历史阶段判断为：RGB 静态输入过拟合可被定义为 subject-level shortcut 与 severity-level shortcut 的共同作用，并可用上层多任务约束做可验证干预。

当前正式实验计划只包含两类干预：

1. **identity-adversarial MTL**：保留 backbone 对面部视觉信息的全面提取能力，但通过上层 MTL 中的 identity adversarial branch / GRL 降低共享表征对 subject identity 的可用性。
2. **severity-balanced regression**：通过 score-bin / severity-bin 权重缓解标签分布不均，避免多数分数段主导反向传播；目标不是人为提高预测方差，而是降低 minimal / severe 等分段 bias，并保持 overall MAE、RMSE、CCC 稳定。

动态面部变化特征暂时降级为 **后续待考虑项**，不进入当前 Stage B 正式实验计划。它仍是潜在 Stage C 方向，用于在身份抑制与 severity imbalance 处理后，进一步验证 facial behavior dynamics 是否能减少 static appearance shortcut。

当前阶段安排：

```text
Stage A 收口诊断
  A1 layer-wise identity probe
  A2 prediction error x identity similarity coupling

Stage B 正式干预实验
  E0 RGB MTL-Lite baseline
  E1 + identity-adversarial MTL branch
  E2 + severity-balanced regression
  E3 + identity-adversarial MTL branch + severity-balanced regression

Stage C 待考虑
  feature delta / AU delta / landmark-pose-gaze delta / static-dynamic fusion
```

Stage A 只做收口，不再继续扩展普通黑边、灰度、模糊、mask 或输入级去身份滤镜。Stage A 完成条件是能够回答：身份信息在哪些层出现、是否与预测错误/严重程度偏置耦合、是否足以支持上层 identity-adversarial MTL。

Stage B 的最低判读指标固定为：

```text
MAE / RMSE / Pearson / CCC
pred_mean / pred_std / true_std
minimal / mild / moderate / severe MAE and bias
same_subject_top1 / same_subject_top5 / paired_rank_mean
severity_agree
task_diff_mean
train-val generalization gap
```

成功标准不是单一 MAE 下降，而是：identity retrieval 下降或不升高、少数 severity 分段误差与 bias 改善、CCC 不明显下降、task consistency 不恶化、train-val gap 不扩大。

## 历史阶段记录说明

以下 2026-06-19 相关段落保留为历史证据，说明 input-level identity suppression / boundary smoothing 2x2 的实现与当时路线。它们不再覆盖当前权威路线。当前下一步以本文开头的 `当前权威快照` 和 `TODO.md` 的 `当前立即执行任务（权威入口）` 为准。

## 2026-06-19 Identity x Boundary 2x2 变体实现

已实现并本地测试通过四个 2x2 机制消融变体：

- `edge_soften_only`：仅对边界连通黑区与脸部像素的过渡带做轻微模糊，保持大面积黑区不变。
- `border_blur_fill`：用邻近非黑像素的局部均值填充边界连通黑区。
- `identity_texture_suppressed`：对脸部区域做高频纹理抑制，弱化胡须/发际线/局部反光等静态身份线索，同时恢复边界连通黑区，避免与边界 artifact 混淆。
- `identity_texture_suppressed_edge_soften`：在上述身份抑制基础上再加入边界过渡带软化。

四者已加入 `src/datasets/input_variants.py`、`tests/test_input_variants.py`，并新增 override 配置：

- `configs/input_ablation/edge_soften_only.yaml`
- `configs/input_ablation/border_blur_fill.yaml`
- `configs/input_ablation/identity_texture_suppressed.yaml`
- `configs/input_ablation/identity_texture_suppressed_edge_soften.yaml`

历史当时计划：在服务器使用与 `rgb` / `center_mask` / `border_black_feather` 完全相同的 split、seed、训练入口和 checkpoint 策略运行四组 input-level 2x2 实验，并接入 `scripts/summarize_prediction_runs.py` / `scripts/summarize_identity_retrieval_runs.py` / `scripts/summarize_severity_calibration_runs.py` 进行三表联合判读。该计划已被当前粗粒度 task-nuisance 主线取代，不再作为下一步主线。

## 2026-06-19 Identity Suppression 方法调研与路线确认

已将现有 identity suppression / disentanglement / adversarial learning / behavior representation 研究映射到当前项目。历史当时结论是：输入级身份纹理弱化与边界平滑的 2x2 机制消融更适合作为第一步；该结论随后被 2026-07-06 的 task-nuisance 路线更新，并最终由 2026-07-10 的可审计、可证伪粗粒度信息分流目标取代。

原因：AVEC2014 样本小，RGB severity signal 与 subject/static appearance 可能纠缠。过早使用 adversarial identity removal 可能同时抹除有效行为线索。更稳妥的方式是先用 `edge_soften_only`、`border_blur_fill`、`identity_texture_suppressed`、`identity_texture_suppressed_edge_soften` 检验 identity shortcut 与 boundary artifact 是独立还是耦合。

“回到正轨”的判据被明确为：same-subject retrieval 不升高或下降、severity agreement 不下降、CCC 不下降、pred_std 不继续压缩、severe bias 改善且 task consistency 不恶化。
