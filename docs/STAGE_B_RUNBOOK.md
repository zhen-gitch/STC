# Stage B 执行清单（B3 fixed-experiments → B4 report → B5 gate）

本文档是 Stage B 在服务器上执行的 runbook：从代码已落地的 E0–E3 配置出发，按统一协议跑完矩阵与 sweep，产出横向对照表，再对照 B5 门槛判定是否进入 Stage C。

前置条件：
- `configs/local_paths.yaml` 已配置真实 AVEC2014 数据路径与 `LOG_DIR`（从 `configs/local_paths.example.yaml` 复制）。
- GPU 可用；`configs/avec2014_base.yaml` 的 `DEVICES` / `PRECISION` 按机器调整。
- B1+B2 代码已提交（commit `9ea0a4c`）：`src/models/gradient_reversal.py`、`src/models/mtl_lite.py`、`src/trainers/mtl_lite_runner.py`、`src/models/outputs.py`、`configs/stage_b/*`。
- 本地 27 项测试全过：`python -m pytest tests/test_mtl_lite_*.py tests/test_gradient_reversal.py`。

---

## 1. 实验矩阵

四组配置都是 `configs/avec2014_base.yaml` 之上的 override 层，共用同一 split / seed / backbone / optimizer / precision / checkpoint 策略（`monitor=val_RMSE_epoch, mode=min`）。`EXPERIMENT_GROUP=stage_b` 把所有 run 归到同一目录便于横向对照。

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

```bash
# E0 baseline
python scripts/train_mtl_lite.py --override configs/stage_b/e0_rgb_mtl_lite.yaml

# E1 + lambda_id=0.10 sweep
python scripts/train_mtl_lite.py \
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

### 4.2 需手动跑的 per-run 审计（A2 / matched-only A3）

这两个审计输入较重（依赖 features_npz / weaklabel summary），不在自动聚合里，按需对每个 run 跑：

**A2 error-identity coupling**（每个 run）：
```bash
python scripts/audit_error_identity_coupling.py \
  --features-npz <RUN_DIR>/diagnostics/test/layer_features.npz \
  --predictions  <RUN_DIR>/diagnostics/test/test_predictions.csv \
  --output-dir   <RUN_DIR>/diagnostics/test/a2_coupling \
  --layers layer_shared,layer_block_6,layer_block_3
```

**A3 matched-only artifact weaklabels**（每个 run，需先有 `audit_artifact_weaklabels.py` 的 summary）：
```bash
python scripts/audit_artifact_weaklabels.py --run-dir <RUN_DIR> --split test \
  --output-dir <RUN_DIR>/diagnostics/test/a3_weaklabels
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

每组 run 完成后至少核对：

1. **prediction**：MAE/RMSE/Pearson/CCC、pred mean/std、train-val gap；
2. **severity**：minimal/mild/moderate/severe 的 count、MAE、bias、abs error、compression；
3. **identity**：A1 layer/shared retrieval top1/top5、A2 residual-identity coupling、subject attacker accuracy；
4. **task consistency**：同 subject Freeform/Northwind prediction diff 与 residual diff；
5. **artifact-risk group**：基于 matched-only A3 变量做分组评估。

task consistency 与 artifact-risk 若 `aggregate_stage_b.sh` 未覆盖，用 `compare_behavior_predictions.py` 与 A3 matched summary 手动补。

## 6. B5 阶段判定门槛

对照聚合表逐项判定（任一“有效”需同时满足主指标改善 + 代价不恶化）：

- **E1 有效**：identity risk（A1 top1/top5、A2 coupling、subject attacker accuracy）下降，且 MAE/RMSE/CCC、severity bias、task consistency、train-val gap 不明显恶化。
- **E2 有效**：minimal 高估 / severe 低估 / compression 改善，且 `pred_std/true_std` 更接近真实分布，CCC 与 identity risk 未明显恶化。
- **E3 有效**：同时优于或互补于 E1/E2，作为 Stage C 强 baseline。
- **均不能在可接受代价内改善风险** → 进入 Stage C 的粗粒度 `z_dep / z_nuisance` 解耦（`docs/TODO.md` C0-spec-before-code）。

判定结果写入 `docs/CURRENT_STATUS.md` 与 `docs/RGB_OVERFITTING_AUDIT_PLAN.md` 的 Stage B 段落，再决定 Stage C 是否开工。

## 7. 常见排雷

- **bf16 下 GRL 反向不稳定**：若 E1 出现 NaN，先用 `PRECISION: "32-true"` 跑一个 E1 smoke 确认是否精度问题；若是，记录并在 sweep 里跳过高 lambda。
- **subject 注入失败**：runner 启动时会打印 `[RUNNER] identity-adversarial ON: N train subjects`；若 N=0，检查 `DATASET_SPLIT_FILE` 与 `IMAGE_DIR` 是否指向真实 train split。
- **severity bin 计数为 0**：会打印 `[SEVERITY-BALANCE] Incomplete train bin counts ...` 并回退到 plain MSE；检查 `LABEL_DIR` 下 `<subject>_Depression.csv` 是否齐全。
- **sweep run 覆盖**：base 与 sweep 共享 `EXPERIMENT_NAME`，version 自增不覆盖；若手动跑同一配置多次，`resolve_run_dir` 总取最新 version。
