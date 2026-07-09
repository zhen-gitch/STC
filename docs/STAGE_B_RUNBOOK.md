# Stage B 执行清单（B3 fixed-experiments → B4 report → B5 gate）

本文档是 Stage B 在服务器上执行的 runbook：从代码已落地的 E0–E3 配置出发，按统一协议跑完矩阵与 sweep，产出横向对照表，再对照 B5 门槛判定是否进入 Stage C。

前置条件：
- `configs/local_paths.yaml` 已配置真实 AVEC2014 数据路径、`LOG_DIR`，以及 backbone 权重路径 `EXTRACT_FEATURE.MODEL_WEIGHT_PATH`（从 `configs/local_paths.example.yaml` 复制后填入）。Stage B 的 `base_regression_only.yaml` 冻结 backbone，**必须**配权重路径，否则冻结的是随机初始化权重。
- GPU 可用；`configs/avec2014_base.yaml` 的 `DEVICES` / `PRECISION` 按机器调整。
- B1+B2 代码已提交（commit `9ea0a4c`）：`src/models/gradient_reversal.py`、`src/models/mtl_lite.py`、`src/trainers/mtl_lite_runner.py`、`src/models/outputs.py`、`configs/stage_b/*`。
- 本地 27 项测试全过：`python -m pytest tests/test_mtl_lite_*.py tests/test_gradient_reversal.py`。

---

## 1. 实验矩阵

四组配置的 override 栈是 `configs/avec2014_base.yaml` → `configs/local_paths.yaml` → `configs/stage_b/base_regression_only.yaml` → `eX_*.yaml`。`base_regression_only.yaml` 复现 Stage A 的 regression_only 基准（冻结 deit_tiny + 微调最后 2 层、ordinal 关闭、纯回归、40 epoch），E0–E3 在此之上只切换 `identity_adversarial` / `severity_balanced_regression` 两个开关，共用同一 split / seed / backbone / optimizer / precision / checkpoint 策略（`monitor=val_RMSE_epoch, mode=min`）。`EXPERIMENT_GROUP=stage_b` 把所有 run 归到同一目录便于横向对照。

| 实验 | 配置 | identity_adversarial | severity_balanced_regression |
|------|------|----------------------|------------------------------|
| E0 | `configs/stage_b/e0_rgb_mtl_lite.yaml` | False | False |
| E1 | `configs/stage_b/e1_identity_adversarial.yaml` | True | False |
| E2 | `configs/stage_b/e2_severity_balanced.yaml` | False | True |
| E3 | `configs/stage_b/e3_identity_adversarial_severity_balanced.yaml` | True | True |

## 2. Sweep 范围

- **lambda_id**（E1/E3）：`0.02 / 0.05 / 0.10 / 0.20`。`0.05` 是 base 配置值，sweep 脚本会跳过重复 run，只额外跑 `0.02 / 0.10 / 0.20`。优先选“不损伤 BDI utility 的最低有效强度”。
- **POWER**（E2/E3）：`0.5 / 1.0`。`0.5` 是 base 配置值，额外跑 `1.0` 对照。
- sweep 通过 2 行 override 文件叠加（自动生成在 `configs/stage_b/sweeps/`），不复制近似配置。

完整 sweep 总 run 数：E0×1 + E1×4 + E2×2 + E3×(4 lambda × 2 power 中只 sweep 默认组合之外的) —— 实际由 `run_stage_b_matrix.sh` 控制，默认 E3 的 lambda 与 power 各自独立 sweep（不笛卡尔积），共 E0=1, E1=4, E2=2, E3=4+1=5，合计 12 run。如需 E3 的 lambda×power 笛卡尔积，手动扩展脚本。

## 3. 执行命令

### 3.1 一键跑完整矩阵 + sweep + 诊断

```bash
# 在服务器项目根目录
bash scripts/stage_b/run_stage_b_matrix.sh
```

环境开关：
- `SKIP_SWEEPS=1`：只跑四组 base 配置，不跑 lambda/power sweep。
- `SKIP_TRAIN=1`：跳过训练，只对已有 run 跑诊断（用于补诊断）。
- `SKIP_DIAG=1`：只训练，不跑诊断。
- `DIAG_SPLITS="val test"`：每个 run 诊断哪些 split（默认 val+test）。
- 选择性跑：`bash scripts/stage_b/run_stage_b_matrix.sh e0 e1`。

### 3.2 单独跑一个实验

每个实验需要两层 override：`base_regression_only.yaml`（共享基准）+ 对应的 `eX_*.yaml`（开关）。

```bash
# E0 baseline
python scripts/train_mtl_lite.py \
  --override configs/stage_b/base_regression_only.yaml \
  --override configs/stage_b/e0_rgb_mtl_lite.yaml

# E1 + lambda_id=0.10 sweep
python scripts/train_mtl_lite.py \
  --override configs/stage_b/base_regression_only.yaml \
  --override configs/stage_b/e1_identity_adversarial.yaml \
  --override configs/stage_b/sweeps/e1_lambda_0.10.yaml
```

sweep override 文件由 `run_stage_b_matrix.sh` 自动生成；手动跑时可先用脚本生成一次：
```bash
bash scripts/stage_b/run_stage_b_matrix.sh e1   # 会生成 sweeps/e1_lambda_*.yaml
```

### 3.3 单 run 诊断（B4，per-run）

```bash
python scripts/diagnose_mtl_lite.py \
  --run-dir <LOG_DIR>/stage_b/e1_identity_adversarial/version_0 \
  --ckpt best --split val test \
  --enable-training-curves --enable-regression \
  --enable-embeddings --enable-correlation
```

`run_stage_b_matrix.sh` 会在每个 run 训练后自动跑这一步。`--enable-occlusion / --enable-keyframes / --enable-model-attention` 是昂贵诊断，默认不在矩阵里开；留给 case-study 单独跑。

## 4. 横向对照表（B4 聚合）

### 4.1 自动聚合

```bash
bash scripts/stage_b/aggregate_stage_b.sh
```

产出（`logs/stage_b/aggregate/` 下）：
- `prediction/prediction_run_summary.csv`：MAE/RMSE/Pearson/CCC、pred mean/std、train-val gap，以 E0 为 baseline 的 pairwise 改进。
- `severity_imbalance/severity_imbalance_summary.csv`：minimal/mild/moderate/severe 的 count/MAE/bias、compression、imbalance_ratio。
- `identity_retrieval/identity_retrieval_run_summary.csv`：A1 layer/shared retrieval top1/top5、paired rank。
- `severity_calibration/severity_calibration_run_summary.csv`：val 拟合 a/b、test original vs calibrated。
- `training_overfit/training_overfit_summary.csv`：best-val vs last epoch、train/val gap、`overfit_after_best_val` 标志。

**version-aware 聚合**：`aggregate_stage_b.sh` 通过 `scripts/stage_b/resolve_run_specs.py` 枚举每个实验的所有 `version_N`，按各自 `resolved_config.yaml` 的 `LAMBDA_ID`/`POWER` 打标签——base 配置 → `eN`，lambda sweep → `eN_lambda0.02`/`eN_lambda0.1`/`eN_lambda0.2`，power sweep → `eN_power1.0`，同一 sweep 签名的多个 version 只取最新。一张表覆盖 base + 全部 sweep。用 `RUN_E0=.../version_K` 可固定某个 version（跳过该实验的自动发现）。

### 4.1.1 全新重跑（清空旧 run）

当旧 run 会污染 version-aware 聚合时（例如修复前的 E1/E3 lambda sweep 实际都跑成 lambda=0.05，旧 version 的 `resolved_config.yaml` 仍读 0.05，会被标成 base 与真 base 去重碰撞），用 `clean_restart.sh` 清空后从 version_0 重跑：

```bash
bash scripts/stage_b/clean_restart.sh --dry-run   # 预览会清什么
bash scripts/stage_b/clean_restart.sh             # 默认备份旧 run 到 stage_b_archive_<时间戳>
bash scripts/stage_b/clean_restart.sh --delete    # 硬删除（省磁盘；checkpoint 较大）
bash scripts/stage_b/run_stage_b_matrix.sh        # 全矩阵重跑（version_0 起）
bash scripts/stage_b/aggregate_stage_b.sh
```

清空范围：Stage B group 目录（全部 4 个实验的所有 version，含 checkpoint/metrics/diagnostics/tables）、生成的 sweep override（`configs/stage_b/sweeps/`）、聚合输出（`logs/stage_b/aggregate/`）。默认移到带时间戳的备份目录（可逆），archive 目录不被聚合扫描。

### 4.2 per-run 审计的自动产出

`run_stage_b_matrix.sh` 的 `diagnose_run` 现在在每个 run 训练后自动跑完整 B4 审计链，无需手动补：

| 审计 | 脚本 | 产物 | 聚合消费方 |
|------|------|------|------------|
| 核心诊断 | `diagnose_mtl_lite.py` | `diagnostics/<split>/regression/<split>_predictions.csv` + 单层 features + plots | prediction / calibration 聚合 |
| A1 层级 identity | `audit_layerwise_identity_probe.py` | `diagnostics/test/a1_layerwise/test_layerwise_features.npz` + `tables/embedding_identity_summary.csv` | identity 聚合（+ A2 输入） |
| A2 error-identity coupling | `audit_error_identity_coupling.py` | `diagnostics/test/a2_coupling/` | B5 identity 判定 |
| severity calibration | `audit_severity_calibration.py` | `tables/severity_calibration_fit.csv` | calibration 聚合 |

关键点：A1/A2 需要**层级** NPZ，`diagnose_mtl_lite.py` 的单层 NPZ 不够，所以 `audit_layerwise_identity_probe.py` 会重新跑一遍模型（带 layer hooks）产出层级 NPZ —— 这与 Stage A 的 A1 同口径。这是诊断阶段最重的步骤（每个 run 多过一遍 test 数据）。

A1 的 identity summary 会被复制到 `<run_dir>/tables/embedding_identity_summary.csv`，`aggregate_stage_b.sh` 直接从这里读。

### 4.3 需手动跑的审计（matched-only A3）

A3 artifact weaklabel 依赖多个外部 summary（black_artifacts / alignment_geometry / openface_quality / temporal_sampling），依赖链较重，不在自动流程里。A3 在 B5 里是辅助判定（artifact-risk group），不是主门槛。按需对每个 run 跑：

```bash
# 先准备依赖 summary（若尚未生成）
python scripts/audit_black_artifacts.py --run-dir <RUN_DIR> --split test \
  --output-dir <RUN_DIR>/diagnostics/test/black_artifacts
python scripts/audit_alignment_geometry.py --run-dir <RUN_DIR> --split test \
  --output-dir <RUN_DIR>/diagnostics/test/alignment_geometry

# A3 weaklabel audit（接口是 --predictions，不是 --run-dir）
python scripts/audit_artifact_weaklabels.py \
  --predictions <RUN_DIR>/diagnostics/test/regression/test_predictions.csv \
  --black-artifacts <RUN_DIR>/diagnostics/test/black_artifacts/tables/black_artifact_summary.csv \
  --alignment-geometry <RUN_DIR>/diagnostics/test/alignment_geometry/tables/alignment_geometry_summary.csv \
  --output-dir <RUN_DIR>/diagnostics/test/a3_weaklabels \
  --coupling-threshold 0.2

# matched-only 汇总（跨 E0-E3）
python scripts/summarize_artifact_weaklabels_matched.py \
  --summary e0=<RUN_DIR_E0>/diagnostics/test/a3_weaklabels/artifact_weaklabel_summary.csv \
  --summary e1=<RUN_DIR_E1>/diagnostics/test/a3_weaklabels/artifact_weaklabel_summary.csv \
  --summary e2=<RUN_DIR_E2>/diagnostics/test/a3_weaklabels/artifact_weaklabel_summary.csv \
  --summary e3=<RUN_DIR_E3>/diagnostics/test/a3_weaklabels/artifact_weaklabel_summary.csv \
  --output-dir logs/stage_b/aggregate/a3_matched \
  --coupling-threshold 0.2
```

A3 只做 probe / case / group-wise 评估，**不把 A3 变量变成训练监督**。

## 5. 报告项（B4 五类，以 E0 为基准横向对照）

每组 run 完成后至少核对（前四类由 `aggregate_stage_b.sh` 自动产出横向对照表）：

1. **prediction**（自动）：MAE/RMSE/Pearson/CCC、pred mean/std、train-val gap；
2. **severity**（自动）：minimal/mild/moderate/severe 的 count、MAE、bias、compression；
3. **identity**（自动）：A1 layer/shared retrieval top1/top5、A2 residual-identity coupling；
4. **task consistency**（自动，随 prediction summary 产出 `task_consistency_summary.csv`）：同 subject Freeform/Northwind prediction diff 与 residual diff；
5. **artifact-risk group**（手动，见 4.3）：基于 matched-only A3 变量做分组评估。

subject attacker accuracy（identity 第三项）若需要，对 E1/E3 的 subject_id_head 单独评估 —— 训练日志的 `train_identity_loss` 已部分反映，离线 accuracy 可用 `audit_layerwise_identity_probe` 的 per-query 结果近似。

## 6. B5 阶段判定门槛

对照聚合表逐项判定（任一“有效”需同时满足主指标改善 + 代价不恶化）：

- **E1 有效**：identity risk（A1 top1/top5、A2 coupling、subject attacker accuracy）下降，且 MAE/RMSE/CCC、severity bias、task consistency、train-val gap 不明显恶化。sweep 行（`e1_lambda0.02/0.1/0.2`）需横向比较，选“不损伤 BDI utility 的最低有效 lambda”。
- **E2 有效**：minimal 高估 / severe 低估 / compression 改善，且 `pred_std/true_std` 更接近真实分布，CCC 与 identity risk 未明显恶化。sweep 行 `e2_power1.0` 与 base `e2`(POWER 0.5) 对照。
- **E3 有效**：同时优于或互补于 E1/E2，作为 Stage C 强 baseline。
- **train-val gap / 过拟合**：`training_overfit/training_overfit_summary.csv` 的 `overfit_after_best_val` 标志与 best-val→last 的 val 恶化幅度纳入代价评估；若所有 run 都在 best val 后持续恶化，支持后续加 EarlyStopping（不改变 B5 判定，但进 Stage C 前记录）。
- **均不能在可接受代价内改善风险** → 进入 Stage C 的粗粒度 `z_dep / z_nuisance` 解耦（`docs/TODO.md` C0-spec-before-code）。

判定结果写入 `docs/CURRENT_STATUS.md` 与 `docs/RGB_OVERFITTING_AUDIT_PLAN.md` 的 Stage B 段落，再决定 Stage C 是否开工。

## 7. 常见排雷

- **backbone 权重未加载**：`base_regression_only.yaml` 设 `FREEZE_BACKBONE=True`，必须配权重。启动日志应出现 `[BACKBONE] Loaded external weights for deit_tiny_patch16_224: ...`；若只看到 `no external weight path` 或 `weight file not found`，说明 `configs/local_paths.yaml` 的 `EXTRACT_FEATURE.MODEL_WEIGHT_PATH` 未设或路径错误 —— 此时冻结的是随机权重，结果无意义。
- **bf16 下 GRL 反向不稳定**：若 E1 出现 NaN，先用 `PRECISION: "32-true"` 跑一个 E1 smoke 确认是否精度问题；若是，记录并在 sweep 里跳过高 lambda。
- **subject 注入失败**：runner 启动时会打印 `[RUNNER] identity-adversarial ON: N train subjects`；若 N=0，检查 `DATASET_SPLIT_FILE` 与 `IMAGE_DIR` 是否指向真实 train split。
- **severity bin 计数为 0**：会打印 `[SEVERITY-BALANCE] Incomplete train bin counts ...` 并回退到 plain MSE；检查 `LABEL_DIR` 下 `<subject>_Depression.csv` 是否齐全。
- **sweep run 覆盖**：base 与 sweep 共享 `EXPERIMENT_NAME`，version 自增不覆盖；聚合按 sweep 签名去重取最新 version（见 4.1 version-aware）。
- **lambda sweep 之前不生效（已修复）**：早期 `run_stage_b_matrix.sh` 把 lambda override 写成字面带点 key `IDENTITY_ADVERSARIAL.LAMBDA_ID:`，OmegaConf 不展开，模型仍读 base 的 0.05 —— E1/E3 的所有 lambda sweep version 实际等价，best val RMSE / prediction 指标完全相同即为该症状。已改为嵌套 YAML。**修复前跑出的 E1/E3 sweep version 无效，需重跑**（见 4.4 重跑指引）。POWER sweep 一直是嵌套写法，不受影响。
- **忘了叠加 base_regression_only**：单独跑 `eX_*.yaml` 而不叠加 `base_regression_only.yaml` 会回落到 `avec2014_base.yaml` 默认（backbone 全解冻、无权重、ordinal 开、CLASS_STEP=2、80 epoch），与 Stage A 基准不一致，E0–E3 对照失效。`run_stage_b_matrix.sh` 已自动叠加；手动跑务必带 `--override configs/stage_b/base_regression_only.yaml`。
- **A1 layerwise probe 失败 / 层级为空**：`audit_layerwise_identity_probe.py` 依赖 backbone 暴露 `patch_embed`/`blocks`。deit_tiny 满足；若换 backbone 看到 `[LAYER-PROBE] Skipped unresolved layers`，identity 聚合会缺对应层。probe 失败时 identity 聚合会报 `embedding_identity_summary.csv not found`。
- **A1 产物文件名**：probe 产出的单 run schema 文件是 `tables/embedding_identity_summary.csv`（取 `layer_shared` 行）；多层文件是 `layerwise_identity_summary.csv`。`backfill_a1_identity.sh` 和 `diagnose_run` 都会把单 run 文件镜像到 `<run_dir>/tables/`，聚合从那里读。
- **聚合时 prediction CSV 找不到**：`aggregate_stage_b.sh` 现在指到 `<run_dir>/diagnostics/test/regression/test_predictions.csv`。若该文件缺失，说明 `diagnose_mtl_lite.py` 没跑或 test split 未诊断 —— 用 `SKIP_TRAIN=1 bash scripts/stage_b/run_stage_b_matrix.sh <exp>` 补诊断。
- **sweep version 的 A1 缺失**：聚合是 version-aware 的，会读每个 sweep version 的 `<run_dir>/tables/embedding_identity_summary.csv`。若某 sweep version 的 A1 没跑过（早于诊断修复），用 `ALL_VERSIONS=1 bash scripts/stage_b/backfill_a1_identity.sh` 一次性补全所有 version（不只是最新）。

### 4.4 重跑指引（lambda sweep 修复后）

修复 lambda sweep 后，E1/E3 的旧 sweep version 无效，需重跑。最小重跑集（E2/E0 不受影响，无需重跑）：

```bash
# 1. 拉取修复（lambda 嵌套 + version-aware 聚合 + overfit 汇总）
git pull origin dev

# 2. 重跑 E1 / E3 的完整矩阵（base + 3 个 lambda sweep 各一次）
#    （E3 还含 power sweep，但 power 一直有效；为干净起见整组重跑）
bash scripts/stage_b/run_stage_b_matrix.sh e1 e3

# 3. 重聚合（现在 version-aware，一张表含 base + 全部 sweep + overfit）
bash scripts/stage_b/aggregate_stage_b.sh
```

若不想重跑、只想给**已存在**的 version 补 A1 诊断（例如修复前跑的 version 想保留作对照），用 `ALL_VERSIONS=1 bash scripts/stage_b/backfill_a1_identity.sh`。但注意：旧 E1/E3 sweep version 的 `resolved_config.yaml` 里 `LAMBDA_ID` 仍是 0.05（因为 sweep 没生效），version-aware 聚合会把它们标成 base `e1`/`e3` 并与真 base 去重——不会产生假的 sweep 行。
