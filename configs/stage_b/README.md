# Stage B Experiment Configs

Stage B 最小干预实验矩阵的配置草案（`docs/MTL_LITE_DESIGN.md` 第 13 节 / `docs/TODO.md` B0-spec-first）。

## 实验矩阵

| 实验 | 配置 | identity_adversarial | severity_balanced_regression | 状态 |
|------|------|----------------------|------------------------------|------|
| E0 | `e0_rgb_mtl_lite.yaml` | False | False | ✓ |
| E1 | `e1_identity_adversarial.yaml` | True | False | ✓ |
| E2 | `e2_severity_balanced.yaml` | False | True | ✓ |
| E3 | `e3_identity_adversarial_severity_balanced.yaml` | True | True | ✓ |

四组配置与对应代码均已落地（B1 identity-adversarial + B2 severity-balanced）。

## 运行方式

所有 Stage B 配置都是 `configs/avec2014_base.yaml` 之上的 override 层：

```bash
# E0 baseline
python scripts/train_mtl_lite.py --override configs/stage_b/e0_rgb_mtl_lite.yaml

# E1 identity-adversarial (默认 lambda_id=0.05)
python scripts/train_mtl_lite.py --override configs/stage_b/e1_identity_adversarial.yaml

# E2 severity-balanced regression (默认 power=0.5)
python scripts/train_mtl_lite.py --override configs/stage_b/e2_severity_balanced.yaml

# E3 identity-adversarial + severity-balanced
python scripts/train_mtl_lite.py --override configs/stage_b/e3_identity_adversarial_severity_balanced.yaml
```

四组实验必须共用同一 split、seed（`train_mtl_lite.py` 内 `pl.seed_everything(42)`）、backbone、optimizer、precision、checkpoint 策略（`monitor=val_RMSE_epoch, mode=min`）与输出目录结构。`EXPERIMENT_GROUP=stage_b` 把所有 Stage B run 归到同一目录下便于横向对照。

## lambda_id sweep

E1 的 `LAMBDA_ID` 固定四个取值：`0.02 / 0.05 / 0.10 / 0.20`，优先选“不损伤 BDI utility 的最低有效强度”。sweep 不复制多份近似配置：编辑 `e1_identity_adversarial.yaml` 的 `LAMBDA_ID`，或额外叠加一个 2 行 override 文件：

```bash
python scripts/train_mtl_lite.py \
  --override configs/stage_b/e1_identity_adversarial.yaml \
  --override configs/stage_b/sweeps/e1_lambda_0.02.yaml
```

`sweeps/` 目录在 B3 实际执行 sweep 时按需创建。

## POWER sweep (E2/E3)

E2/E3 的 `POWER` 先跑 `0.5`，再比较 `1.0`。与 `lambda_id` sweep 对称：编辑配置文件，或叠加 2 行 override：

```bash
python scripts/train_mtl_lite.py \
  --override configs/stage_b/e2_severity_balanced.yaml \
  --override configs/stage_b/sweeps/e2_power_1.0.yaml
```

## 设计约束

- **开关默认关闭**：`IDENTITY_ADVERSARIAL.ENABLE` 与 `SEVERITY_BALANCED_REGRESSION.ENABLE` 均默认 False。E0 显式写 False 以便 resolved config 自文档化，数值上与既有 MTL-Lite baseline 完全一致。
- **subject 类别与 bin 权重仅来自 train split**：runner 的 `build_train_subject_index` / `build_train_severity_bin_counts` 只读 train split，val/test 不参与类别表或权重计算。
- **bin 边界同口径**：`SEVERITY_BALANCED_REGRESSION.EDGES` 默认 `[13, 19, 28]`，与 `src/diagnostics/io.py:severity_group` 一致，保证训练侧重加权与 A4/severity 诊断可比。
- **只加权 regression MSE**：B2 第一版不改 CCC / ordinal auxiliary loss。
- **monitor 不变**：`ModelCheckpoint(monitor=val_RMSE_epoch, mode=min)` 在四组间一致。
