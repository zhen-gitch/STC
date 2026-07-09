# Stage B Experiment Configs

Stage B 最小干预实验矩阵的配置（`docs/MTL_LITE_DESIGN.md` 第 13 节 / `docs/TODO.md` B0-spec-first）。四组实验都锚定到 **regression_only 基准**（Stage A 收集 A1/A2/A4 证据时用的同一基座），只切换 `identity_adversarial` 与 `severity_balanced_regression` 两个开关。

## 配置结构

```
configs/stage_b/
  base_regression_only.yaml                              # 共享 base：复现 regression_only 基准
  e0_rgb_mtl_lite.yaml                                   # E0：两开关 OFF（baseline）
  e1_identity_adversarial.yaml                           # E1：identity_adversarial ON
  e2_severity_balanced.yaml                              # E2：severity_balanced ON
  e3_identity_adversarial_severity_balanced.yaml         # E3：两开关 ON
  sweeps/                                                # sweep override（B3 运行时自动生成）
```

`base_regression_only.yaml` 复现服务器 `configs/experiments/regression_only.yaml` 的有效字段：冻结 deit_tiny + 微调最后 2 层、`MAX_SEQ_LEN=2000`、`CLASS_STEP=10`、`MAX_EPOCHS=40`、ordinal 关闭（纯回归）、`ORDINAL_WEIGHT=0.1`、`CCC_WEIGHT=0.0`。legacy full-model 字段（`ENABLE_CGC`/`ENABLE_PCGRAD` 等）对 MTL-Lite 是 no-op，已省略。E0–E3 在此之上只写 `EXPERIMENT_NAME` + 各自开关。

**权重路径**：`base_regression_only.yaml` 不设 `MODEL_WEIGHT_PATH`（避免覆盖 local_paths）。在 `configs/local_paths.yaml`（gitignored）里设 `EXTRACT_FEATURE.MODEL_WEIGHT_PATH`，例如服务器上 `/usr/local/conda/zhen/stc/weights/deit_tiny_patch16_224/model.pth`。`FREEZE_BACKBONE=True` 必须配权重，否则冻结的是随机初始化的 backbone。

## 实验矩阵

| 实验 | 配置 | identity_adversarial | severity_balanced_regression |
|------|------|----------------------|------------------------------|
| E0 | `e0_rgb_mtl_lite.yaml` | False | False |
| E1 | `e1_identity_adversarial.yaml` | True | False |
| E2 | `e2_severity_balanced.yaml` | False | True |
| E3 | `e3_identity_adversarial_severity_balanced.yaml` | True | True |

四组共享同一 base，差异仅在开关 —— 满足 B0 规格“同一 split/seed/backbone/optimizer/precision/checkpoint”。

## 运行方式

每个实验需要两层 override：`base_regression_only.yaml` + 对应的 `eX_*.yaml`：

```bash
# E0 baseline
python scripts/train_mtl_lite.py \
  --override configs/stage_b/base_regression_only.yaml \
  --override configs/stage_b/e0_rgb_mtl_lite.yaml

# E1 identity-adversarial（默认 lambda_id=0.05）
python scripts/train_mtl_lite.py \
  --override configs/stage_b/base_regression_only.yaml \
  --override configs/stage_b/e1_identity_adversarial.yaml

# E2 severity-balanced（默认 power=0.5）
python scripts/train_mtl_lite.py \
  --override configs/stage_b/base_regression_only.yaml \
  --override configs/stage_b/e2_severity_balanced.yaml

# E3 identity-adversarial + severity-balanced
python scripts/train_mtl_lite.py \
  --override configs/stage_b/base_regression_only.yaml \
  --override configs/stage_b/e3_identity_adversarial_severity_balanced.yaml
```

一键跑完整矩阵 + sweep + 诊断：

```bash
bash scripts/stage_b/run_stage_b_matrix.sh
```

`run_stage_b_matrix.sh` 会自动为每个 run 叠加 `base_regression_only.yaml`。

## Sweep

- **lambda_id**（E1/E3）：`0.02 / 0.05 / 0.10 / 0.20`。`0.05` 是 base 值，脚本额外跑 `0.02 / 0.10 / 0.20`。优先选“不损伤 BDI utility 的最低有效强度”。
- **POWER**（E2/E3）：`0.5 / 1.0`。`0.5` 是 base 值，额外跑 `1.0`。

sweep 通过 `configs/stage_b/sweeps/` 下的 2 行 override 文件叠加，不复制近似配置。手动示例：

```bash
python scripts/train_mtl_lite.py \
  --override configs/stage_b/base_regression_only.yaml \
  --override configs/stage_b/e1_identity_adversarial.yaml \
  --override configs/stage_b/sweeps/e1_lambda_0.10.yaml
```

## 设计约束

- **开关默认关闭**：`IDENTITY_ADVERSARIAL.ENABLE` 与 `SEVERITY_BALANCED_REGRESSION.ENABLE` 默认 False。E0 显式写 False 以便 resolved config 自文档化。
- **subject 类别与 bin 权重仅来自 train split**：runner 的 `build_train_subject_index` / `build_train_severity_bin_counts` 只读 train split。
- **bin 边界同口径**：`SEVERITY_BALANCED_REGRESSION.EDGES` 默认 `[13, 19, 28]`，与 `src/diagnostics/io.py:severity_group` 一致。
- **只加权 regression MSE**：B2 第一版不改 CCC / ordinal auxiliary loss。
- **monitor 不变**：`ModelCheckpoint(monitor=val_RMSE_epoch, mode=min)` 四组一致。
- **权重路径私有**：不硬编码进 repo，通过 `configs/local_paths.yaml` 注入。

完整执行清单见 `docs/STAGE_B_RUNBOOK.md`。
