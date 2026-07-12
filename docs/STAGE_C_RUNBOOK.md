# Stage C Runbook

> 文档职责：冻结 Stage C 的实施接口、配置矩阵、运行顺序、审计协议和停止条件。当前状态与阶段结论见 `CURRENT_STATUS.md`，任务完成状态见 `TODO.md`，机制理由见 `RGB_OVERFITTING_AUDIT_PLAN.md`。

## 1. 目标与主张边界

Stage C 验证以下结构假设：

```text
H0 -> z_dep, z_nuisance
BDI_pred = Head(z_dep)
H0_recon = Recon([z_dep, z_nuisance])
```

正式目标是 **Auditable and Falsifiable Coarse-Grained Task-Nuisance Information Separation**，即“可审计、可证伪的粗粒度任务-干扰信息分流”。

- `z_dep` 和 `z_nuisance` 是候选信息出口，不预设其获得了真实语义。
- prediction、ordinal auxiliary task 和任何后续 identity adversary 只能读取 `z_dep`。
- reconstruction、decorrelation、训练内 adversary 变弱或单一 probe 下降都不能单独证明信息分流成立。
- artifact、pose、quality、AU 等变量第一版只用于 probe、case study 和 group-wise evaluation，不建立对应显式 latent。
- `z_id`、task anchor、pose/artifact 弱监督和新的 GRL 均不进入第一版实现。

## 2. 实施顺序

```text
P0 训练协议冻结
-> C0 Stage C 规格与配置冻结
-> C1 最小 TaskNuisanceBlock 和多表征导出
-> C2 单 seed validation-only 筛选
-> C3 三 seed 稳定性闸门
-> Stage D 五 seed test 反证验证
```

在 C1 代码完成前，不创建可误运行的 Stage C 正式训练矩阵。配置骨架可以先建立，但未实现字段必须由配置校验明确拒绝。

## 3. P0 训练协议

### 3.1 Seed

共享配置增加：

```yaml
SEED: 42
```

`seed_everything` 必须在完整配置加载后调用，使 seed override 能进入 `resolved_config.yaml`。所有模型比较使用 paired seed，不允许某个候选单独更换 seed。

### 3.2 EarlyStopping

共享配置默认关闭，Stage C common override 开启：

```yaml
EARLY_STOPPING:
  ENABLE: false
  MONITOR: "val_RMSE_epoch"
  MODE: "min"
  PATIENCE: 8
  MIN_DELTA: 0.0
  STRICT: true
  CHECK_FINITE: true
```

约束：

- `ModelCheckpoint` 和 `EarlyStopping` 必须读取同一份 `MONITOR/MODE`。
- `check_val_every_n_epoch=1`、`save_top_k=1`、`save_last=true` 保持不变。
- `PROCESS_TEMPORAL.MAX_EPOCHS=40` 是硬上限，不是预期训练长度。
- test 始终使用 `ckpt_path="best"`。
- Stage C 的 `C-REF` 必须用同一 EarlyStopping 协议重跑 Stage B E2 结构；历史 Stage B 数值只作背景，不直接作为公平对照。

Stage B 的 12 个 run 最佳 validation epoch 全部位于 6-8，且之后均未恢复；`PATIENCE=8` 预计在约 14-16 epoch 停止，保留回弹窗口同时避免继续跑满 40 epoch。

## 4. C1 模型接口

### 4.1 维度

当前 `PROCESS_TEMPORAL.HIDDEN_DIM=192`，因此第一版固定：

```text
H0_DIM = 192
DEP_DIM = 96
NUISANCE_DIM = 96
DEP_DIM + NUISANCE_DIM = H0_DIM
```

`z_dep` 是 96 维 prediction bottleneck，`[z_dep, z_nuisance]` 的总宽度与原始 `H0` 相同。

### 4.2 TaskNuisanceBlock

建议新增 `src/models/task_nuisance.py`，最小结构为：

```text
dep_encoder:       Linear(192, 96) -> LayerNorm -> GELU
nuisance_encoder:  Linear(192, 96) -> LayerNorm -> GELU
reconstructor:     Linear(192, 192)
```

不加入 residual、gate、VAE、multi-level routing 或深 decoder。reconstructor 的输出不加激活，以便重建有符号的 `H0`。

### 4.3 配置契约

第一版配置键冻结为：

```yaml
MODEL:
  TASK_NUISANCE:
    ENABLE: false
    VARIANT: "split"  # "bottleneck" or "split"
    DEP_DIM: 96
    NUISANCE_DIM: 96
    RECONSTRUCTION_ENABLE: false
    CROSS_CORRELATION_ENABLE: false
    AUXILIARY_CALIBRATION_ONLY: false

LOSSES:
  RECONSTRUCTION_WEIGHT: 0.0
  CROSS_CORRELATION_WEIGHT: 0.0
```

约束：

- 配置节缺失或 `ENABLE=false` 时，Stage B/旧 MTL-Lite 行为与 state dict 不变。
- `VARIANT=bottleneck` 只实例化与 `dep_encoder` 同构的 `Linear -> LayerNorm -> GELU`，用于 `C-BN`。
- `VARIANT=split` 同时实例化两个 encoder 和 reconstructor；`DEP_DIM + NUISANCE_DIM` 必须等于 `H0_DIM`。
- 正式训练中，`RECONSTRUCTION_ENABLE=false` 时 `RECONSTRUCTION_WEIGHT` 必须为 `0.0`；开启时权重必须为正数。
- 正式训练中，`CROSS_CORRELATION_ENABLE=false` 时 `CROSS_CORRELATION_WEIGHT` 必须为 `0.0`；开启时权重必须为正数。
- `AUXILIARY_CALIBRATION_ONLY=true` 只允许 train split：两个原始辅助损失均计算并记录，但权重固定为 `0.0`，不进入优化目标；validation/test 不得参与权重选择。
- 未知 `TASK_NUISANCE` 字段、非法 variant、维度不匹配和开关/权重冲突必须 fail-fast，不能静默回退为 E2。

四组映射固定为：

| Run | ENABLE | VARIANT | Reconstruction | Cross-correlation |
|---|---:|---|---:|---:|
| `C-REF` | false | not applicable | off | off |
| `C-BN` | true | `bottleneck` | off | off |
| `C-REC` | true | `split` | on | off |
| `C-FULL` | true | `split` | on | on |

`configs/stage_c/` 在 C1 完成前使用明确的 spec-only 哨兵阻止 `C-BN/C-REC/C-FULL` 和 calibration support run 被误运行；只有 `C-REF` 可在 P0 后按新训练协议运行。

`calibration_train_only.yaml` 是非对照 support run：C1 完成后移除其哨兵，记录 seed 42 前 100 个 train batch 的原始主损失、reconstruction 和 cross-correlation；冻结权重后才把正数写入 `C-REC/C-FULL`。

### 4.4 输出契约

`MTLLiteOutput` 追加可选字段：

```text
h0_features
z_dep_features
z_nuisance_features
reconstructed_h0
```

兼容规则：

- Stage C 关闭时，新字段为 `None`，现有输出、loss 和 state dict 必须不变。
- `shared_features` 继续表示 prediction 实际读取的表征；Stage C 关闭时为 `H0`，开启时为 `z_dep`。
- `C-BN` 的 `z_nuisance_features` 和 `reconstructed_h0` 必须为 `None`/not applicable，不能为满足导出接口而实例化额外 nuisance 分支。
- layer-wise audit 中 `layer_shared` 继续表示 prediction 表征；另行导出 `H0/z_dep/z_nuisance`，避免旧 identity 审计静默改为攻击错误出口。
- 若未来同时启用 identity adversary，其输入必须是 `z_dep`，不能是 `H0`。

## 5. 辅助损失

```text
L_total = L_E2
        + lambda_rec * L_reconstruction
        + lambda_xcorr * L_cross_correlation
```

其中：

```text
L_reconstruction = MSE(H0_recon.float(), H0.detach().float())
```

重建 target 必须 stop-gradient，避免 target 与 decoder 同时移动形成伪收敛。

`L_cross_correlation` 使用 batch-centered、per-dimension standardized 的 `z_dep/z_nuisance`，计算交叉相关矩阵平方均值。该项使用 float32；batch size 小于 2 时返回可反传零值。

第一轮不做二维笛卡尔 sweep。使用 `calibration_train_only.yaml` 在 seed 42 的 train-only calibration 中记录前 100 个 batch 的原始 loss 量级，从以下候选中一次性冻结权重：

```text
lambda_rec:   0.001 or 0.01
lambda_xcorr: 0.001 or 0.01
```

选择规则：每个加权辅助项的中位贡献不超过主 BDI loss 的 10%，二者合计不超过 15%。选择过程不能读取 validation/test utility。

## 6. 公平对照矩阵

所有 Stage C 候选继承 E2 的 severity-balanced regression，第一轮固定 `POWER=0.5`。

| Run | 结构 | 作用 |
|---|---|---|
| `C-REF` | `H0[192] -> BDI` | 同协议重跑 E2，Stage C 强 reference |
| `C-BN` | `H0 -> z_task[96] -> BDI` | 等 prediction bottleneck，排除单纯降维收益 |
| `C-REC` | `H0 -> z_dep[96],z_nuisance[96]` + reconstruction | 与 full 等参数/等 bottleneck，decorrelation 关闭 |
| `C-FULL` | 与 `C-REC` 完全相同 + decorrelation | 第一版完整信息分流候选 |

解释规则：

- `C-FULL` 不优于 `C-BN`：收益主要可由 bottleneck 解释，信息分流假设未获支持。
- `C-FULL` 不优于 `C-REC`：decorrelation 没有新增证据，优先保留更简单的 `C-REC`。
- `C-REC/C-FULL` 均不优于 `C-REF`：停止增加结构复杂度。

## 7. 多表征导出

诊断脚本必须在同一次 checkpoint forward 中导出：

```text
features_h0
features_z_dep
features_z_nuisance
video_ids
subject_ids
task_names
true_bdi
pred_bdi
```

Stage C 关闭的 reference run 中，`features_z_dep` 取 prediction 表征，即与 `features_h0` 相同；`features_z_nuisance` 缺失并明确标记为 not applicable。

现有 `features` key 继续指向 prediction 表征，保证旧 embedding/identity 工具不被破坏。新增 representation bundle 供 leakage matrix 使用。

## 8. Leakage Matrix

固定矩阵：

```text
H0 / z_dep / z_nuisance
  x identity / task / pose / artifact / BDI
```

### 8.1 Probe split

- task、pose、artifact、BDI probe：probe 在 train representation 上训练，在 val 调参，最终只在 test 报告一次。
- identity class 在 train/val/test 不重叠，因此不能把 train subject classifier 直接用于 test。
- identity 主协议使用跨 subject 泛化的 pair verifier：train subject pairs 训练、val pairs 调参、test unseen-subject pairs 评估。
- 现有 paired-task retrieval 和 LOVO Ridge 继续作为补充视角。

### 8.2 Fresh multi-attacker

每个冻结 checkpoint 独立训练：

```text
non-parametric retrieval / kNN
linear or Ridge attacker
RBF-SVM attacker
2-layer MLP attacker
```

结论使用预注册 attacker 集合中的最强风险，不允许只选择表现最弱的 attacker。每类 probe 同时运行 shuffled-label control。

### 8.3 Primary artifact axes

第一版主审计轴固定为：

```text
openface_quality:pose_rz_mean
black_artifacts:black_border_ratio_mean
alignment_geometry:face_center_offset_y_mean
```

AU 变量单独作为 behavior-sensitive audit panel 报告，不进入强抑制成功条件。

## 9. Group-wise Evaluation

分组阈值只用 train split 估计并冻结，再应用到 val/test。至少报告：

```text
severity bins
Freeform / Northwind
pose_rz high / low
black-border high / low
face-offset-y high / low
```

MAE、bias、RMSE、可计算时的 CCC、worst-group gap 均按 subject bootstrap 计算区间，同一 subject 的两段任务视频必须一起重采样。

## 10. Seed 与数据使用规则

```text
C1 smoke: seed 42
C2 screening: seed 42, validation only
C3 gate: seeds 42, 43, 44, validation only
Stage D: seeds 42, 43, 44, 45, 46, frozen protocol, final test
```

同一阶段所有模型必须使用相同 split、seed、input variant、backbone、optimizer、precision、severity weighting、EarlyStopping 和 checkpoint policy。

在 Stage D 打开 test 之前，模型结构、loss 权重、attacker 集合、group threshold 和停止条件必须全部冻结。test 结果不得触发新的同族超参数 sweep。

## 11. 进入与停止条件

### 11.1 Utility non-inferiority

相对 paired-seed `C-REF`：

```text
delta_CCC >= -0.02
delta_MAE <= +0.25 BDI
delta_task_diff_mean <= +0.50 BDI
```

任一 seed 出现 `delta_CCC < -0.05` 或 `delta_MAE > +0.50`，视为该 seed 发生 utility failure。

### 11.2 Risk reduction

- identity primary metric 固定为预注册 fresh pair-verifier 中最强攻击器的 AUROC，记为 `R_id=max(AUROC)`；paired-task retrieval top1 为第二 primary metric。
- 相对 paired-seed `C-REF`，`R_id` 或 paired-task retrieval top1 至少一个绝对下降 `0.05`，另一个不得绝对上升超过 `0.02`。
- 其余预注册 attacker 的主指标不得绝对上升超过 `0.02`。
- primary pose/artifact high-risk group MAE 不得高于 `C-REF + 0.25 BDI`，worst-group MAE gap 不得高于 `C-REF + 0.50 BDI`。若只改善平均 MAE 而高风险组越过该界，不视为成功。

### 11.3 BDI leakage

若同一 probe family、同一 split 下的 fresh BDI probe 满足任一条件，则判定 `z_nuisance` 携带接近主要任务出口的 BDI 信息：

```text
CCC(z_dep) - CCC(z_nuisance) <= 0.05
MAE(z_nuisance) <= MAE(z_dep) + 0.50
```

触发后不得宣称任务信息已成功路由到 `z_dep`。

### 11.4 Stability

- C3 至少 2/3 seed 的风险方向一致，且全部 seed 满足 utility non-inferiority，才进入 Stage D。
- Stage D 报告 paired seed delta、均值/标准差和 subject-bootstrap CI。
- 若不优于 paired-seed `C-REF`、multi-seed 不稳定或风险下降以 utility/group robustness 恶化为代价，停止增加 `z_id`、门控、更多 latent 或训练监督。

## 12. 第一批实现文件

P0：

```text
scripts/train_mtl_lite.py
src/trainers/mtl_lite_runner.py
configs/avec2014_base.yaml
configs/stage_c/
tests/test_mtl_lite_training_policy.py
tests/test_stage_c_config_contract.py
```

C1：

```text
src/models/task_nuisance.py
src/models/mtl_lite.py
src/models/outputs.py
scripts/diagnose_mtl_lite.py
src/diagnostics/io.py
configs/stage_c/
tests/test_task_nuisance.py
tests/test_mtl_lite_forward.py
tests/test_mtl_lite_loss_backward.py
```

C2 前置审计：

```text
src/diagnostics/representation_leakage.py
src/diagnostics/group_robustness.py
scripts/audit_representation_leakage.py
```

Stage D 汇总：

```text
scripts/summarize_stage_c_runs.py
```

## 13. 验证命令

P0/C1 修改后优先运行：

```bash
python -m compileall src scripts tests
python -m pytest tests/test_mtl_lite_training_policy.py tests/test_stage_c_config_contract.py
python -m pytest tests/test_task_nuisance.py tests/test_mtl_lite_forward.py tests/test_mtl_lite_loss_backward.py
python scripts/train_mtl_lite.py --override configs/mtl_lite_debug_smoke.yaml
```

文档和配置还需运行：

```bash
git diff --check -- README.md configs docs
```
