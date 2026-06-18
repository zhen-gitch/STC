# 实验脚本使用手册

本手册汇总当前项目所有可直接运行的训练、诊断、审计和汇总脚本。所有路径默认采用新的实验输出布局：

```text
experiment/<EXPERIMENT_GROUP>/<EXPERIMENT_NAME>/version_N/
```

其中 `version_N` 由 `src/config.py` 自动按已有目录递增（`version_0`、`version_1`、...）。

---

## 1. 配置栈

实验配置按以下顺序合并：

1. `configs/avec2014_base.yaml`（MTL-Lite 主线最小基线）
2. `configs/local_paths.yaml`（机器相关路径，**不提交到 git**，从 `configs/local_paths.example.yaml` 复制）
3. 一个或多个 override YAML（如 `configs/mtl_lite_baseline.yaml`）

创建本地路径配置：

```bash
cp configs/local_paths.example.yaml configs/local_paths.yaml
# 编辑 configs/local_paths.yaml 填入真实路径
```

`LOG_DIR` 仅作为旧端到端 runner 的 fallback；新实验使用 `EXPERIMENT_ROOT / EXPERIMENT_GROUP / EXPERIMENT_NAME`。

---

## 2. 训练入口

### 2.1 准备本地 backbone 权重（推荐）

```bash
python scripts/prepare_backbone_weights.py \
  --model-name deit_tiny_patch16_224 \
  --timm-model-name deit_tiny_patch16_224.fb_in1k \
  --output weights/deit_tiny_patch16_224/model.pth \
  --verify
```

然后在 override 中指定：

```yaml
EXTRACT_FEATURE:
  MODEL_NAME: "deit_tiny_patch16_224"
  TIMM_PRETRAINED: False
  MODEL_WEIGHT_PATH: "weights/deit_tiny_patch16_224/model.pth"
```

### 2.2 MTL-Lite baseline

```bash
python scripts/train_mtl_lite.py --override configs/mtl_lite_baseline.yaml
```

输出：

```text
experiment/default/mtl_lite/version_0/
  metrics.csv
  hparams.yaml
  resolved_config.yaml
  checkpoints/
```

### 2.3 Regression-only baseline

```bash
python scripts/train_mtl_lite.py --override configs/regression_only_baseline.yaml
```

输出：

```text
experiment/default/regression_only/version_0/
```

### 2.4 MTL-Lite debug smoke

```bash
python scripts/train_mtl_lite.py --override configs/mtl_lite_debug_smoke.yaml
```

输出：

```text
experiment/default/mtl_lite_debug_smoke/version_0/
```

### 2.5 Behavior-only baseline

需要额外提供 `DATASET.OPENFACE_ROOT`：

```bash
python scripts/train_behavior_baseline.py \
  --override configs/behavior_baseline.yaml \
  --override configs/local_openface_paths.yaml
```

示例 `configs/local_openface_paths.yaml`：

```yaml
DATASET:
  OPENFACE_ROOT: "/path/to/openface_csv_root"
```

输出：

```text
experiment/default/behavior_baseline/version_0/
  metrics.csv
  checkpoints/
  diagnostics/behavior/val_predictions.csv
  diagnostics/behavior/test_predictions.csv
```

---

## 3. 离线诊断入口（`diagnose_mtl_lite.py`）

### 3.1 默认只诊断 test split

```bash
python scripts/diagnose_mtl_lite.py \
  --run-dir experiment/default/mtl_lite/version_0 \
  --ckpt best
```

输出：

```text
experiment/default/mtl_lite/version_0/diagnostics/
  regression/test_predictions.csv
  embeddings/test_features.npz
  training/training_curves.png
  correlation/...
  reports/diagnostic_report.md
```

### 3.2 同时诊断 val 和 test

```bash
python scripts/diagnose_mtl_lite.py \
  --run-dir experiment/default/mtl_lite/version_0 \
  --ckpt best \
  --split val test
```

输出：

```text
experiment/default/mtl_lite/version_0/diagnostics/
  val/regression/val_predictions.csv
  val/embeddings/val_features.npz
  val/reports/diagnostic_report.md
  test/regression/test_predictions.csv
  test/embeddings/test_features.npz
  test/reports/diagnostic_report.md
  reports/diagnostic_summary.md
```

### 3.3 只跑部分诊断（按需开启）

```bash
python scripts/diagnose_mtl_lite.py \
  --run-dir experiment/default/mtl_lite/version_0 \
  --ckpt best \
  --split test \
  --enable-regression \
  --enable-embeddings \
  --enable-training-curves
```

---

## 4. 审计入口

所有审计脚本都是**只读**的，不改变训练/验证/测试结果。输入为 prediction CSV 或 OpenFace/aligned frame 元数据。

### 4.1 Split / subject integrity audit

```bash
python scripts/audit_split_integrity.py \
  --split-file /path/to/dataset_split.json \
  --label-dir /path/to/labels \
  --image-root /path/to/aligned/frame/root \
  --predictions experiment/default/mtl_lite/version_0/diagnostics/regression/test_predictions.csv \
  --output-dir experiment/default/mtl_lite/version_0/diagnostics/split_integrity
```

### 4.2 Temporal sampling audit

```bash
python scripts/audit_temporal_sampling.py \
  --predictions experiment/default/mtl_lite/version_0/diagnostics/regression/test_predictions.csv \
  --image-root /path/to/AVEC2014/face_images \
  --output-dir experiment/default/mtl_lite/version_0/diagnostics/temporal_sampling \
  --sample-step 10 \
  --max-seq-len 2000 \
  --sampling-strategy stride_head
```

`--sample-step` / `--max-seq-len` / `--sampling-strategy` 必须与对应实验的 `resolved_config.yaml` 一致。

### 4.3 OpenFace alignment geometry audit

```bash
python scripts/audit_alignment_geometry.py \
  --predictions experiment/default/mtl_lite/version_0/diagnostics/regression/test_predictions.csv \
  --openface-root /path/to/openface_csv_root \
  --output-dir experiment/default/mtl_lite/version_0/diagnostics/alignment_geometry \
  --frame-width 640 \
  --frame-height 480 \
  --sample-step 1
```

> 注意：当前 OpenFace CSV 坐标系约为 640x480，不是 aligned 112x112。

### 4.4 Black artifact audit

```bash
python scripts/audit_black_artifacts.py \
  --predictions experiment/default/mtl_lite/version_0/diagnostics/regression/test_predictions.csv \
  --image-root /path/to/AVEC2014/face_images \
  --output-dir experiment/default/mtl_lite/version_0/diagnostics/black_artifacts \
  --sample-step 10
```

### 4.5 Shortcut audit（OpenFace quality + shortcut predictor）

```bash
python scripts/audit_shortcuts.py \
  --predictions experiment/default/mtl_lite/version_0/diagnostics/regression/test_predictions.csv \
  --openface-root /path/to/openface_csv_root \
  --output-dir experiment/default/mtl_lite/version_0/diagnostics/shortcut_audit
```

### 4.6 Embedding identity retrieval audit

**单 split 默认布局：**

```bash
python scripts/audit_identity_retrieval.py \
  --features-npz experiment/default/mtl_lite/version_0/diagnostics/embeddings/test_features.npz \
  --predictions experiment/default/mtl_lite/version_0/diagnostics/regression/test_predictions.csv \
  --output-dir experiment/default/mtl_lite/version_0/diagnostics/identity_retrieval \
  --top-k 5
```

**多 split 布局：**

```bash
python scripts/audit_identity_retrieval.py \
  --features-npz experiment/default/mtl_lite/version_0/diagnostics/test/embeddings/test_features.npz \
  --predictions experiment/default/mtl_lite/version_0/diagnostics/test/regression/test_predictions.csv \
  --output-dir experiment/default/mtl_lite/version_0/diagnostics/test/identity_retrieval \
  --top-k 5
```

### 4.7 Severity calibration verification

需要先在 val split 上导出预测：

```bash
python scripts/diagnose_mtl_lite.py \
  --run-dir experiment/default/mtl_lite/version_0 \
  --ckpt best \
  --split val test
```

然后运行校准审计：

```bash
python scripts/audit_severity_calibration.py \
  --val-predictions experiment/default/mtl_lite/version_0/diagnostics/val/regression/val_predictions.csv \
  --test-predictions experiment/default/mtl_lite/version_0/diagnostics/test/regression/test_predictions.csv \
  --output-dir experiment/default/mtl_lite/version_0/diagnostics/severity_calibration
```

### 4.8 Task inconsistency mixed-factor audit

```bash
python scripts/audit_task_inconsistency.py \
  --predictions experiment/default/mtl_lite/version_0/diagnostics/test/regression/test_predictions.csv \
  --black-artifacts-summary experiment/default/mtl_lite/version_0/diagnostics/black_artifacts/tables/black_artifact_summary.csv \
  --openface-quality-summary experiment/default/mtl_lite/version_0/diagnostics/shortcut_audit/tables/openface_quality_summary.csv \
  --alignment-geometry-summary experiment/default/mtl_lite/version_0/diagnostics/alignment_geometry/tables/alignment_geometry_summary.csv \
  --temporal-sampling-summary experiment/default/mtl_lite/version_0/diagnostics/temporal_sampling/tables/temporal_sampling_summary.csv \
  --output-dir experiment/default/mtl_lite/version_0/diagnostics/task_inconsistency \
  --top-n 20
```

所有 `--*-summary` 参数都是可选的；缺少的变量将不会进入相关分析。

---

## 5. 跨 run 汇总入口

### 5.1 Prediction runs summary

```bash
python scripts/summarize_prediction_runs.py \
  --output-dir analysis_outputs/rgb_input_ablation_summary \
  --baseline rgb \
  --run rgb=experiment/default/rgb/version_0/diagnostics/regression/test_predictions.csv \
  --run center_mask=experiment/default/center_mask/version_0/diagnostics/regression/test_predictions.csv \
  --run border_black_feather=experiment/default/border_black_feather/version_0/diagnostics/regression/test_predictions.csv
```

`PATH` 可以是 CSV 文件，也可以是包含 `test_predictions.csv` 的目录。使用 `--split val` 可汇总验证集预测。

输出：

```text
analysis_outputs/rgb_input_ablation_summary/
  tables/prediction_run_summary.csv
  tables/severity_bias_summary.csv
  tables/task_consistency_summary.csv
  tables/pairwise_baseline_improvement.csv
  reports/prediction_runs_report.md
```

### 5.2 Training overfit summary

```bash
python scripts/summarize_training_overfit.py \
  --output-dir analysis_outputs/training_overfit_summary \
  --run rgb=experiment/default/rgb/version_0/metrics.csv \
  --run center_mask=experiment/default/center_mask/version_0/metrics.csv \
  --run behavior=experiment/default/behavior_baseline/version_0/metrics.csv
```

### 5.3 Identity retrieval multi-run summary

```bash
python scripts/summarize_identity_retrieval_runs.py \
  --output-dir analysis_outputs/identity_retrieval_summary \
  --run rgb=experiment/default/rgb/version_0/diagnostics/identity_retrieval \
  --run center_mask=experiment/default/center_mask/version_0/diagnostics/identity_retrieval \
  --run border_black_feather=experiment/default/border_black_feather/version_0/diagnostics/identity_retrieval
```

### 5.4 Severity calibration multi-run summary

```bash
python scripts/summarize_severity_calibration_runs.py \
  --output-dir analysis_outputs/severity_calibration_summary \
  --run rgb=experiment/default/rgb/version_0/diagnostics/severity_calibration \
  --run center_mask=experiment/default/center_mask/version_0/diagnostics/severity_calibration \
  --run border_black_feather=experiment/default/border_black_feather/version_0/diagnostics/severity_calibration
```

### 5.5 Mechanism summary（合并预测 / identity / calibration）

```bash
python scripts/summarize_mechanism.py \
  --prediction-summary analysis_outputs/rgb_input_ablation_summary/tables/prediction_run_summary.csv \
  --severity-bias-summary analysis_outputs/rgb_input_ablation_summary/tables/severity_bias_summary.csv \
  --task-consistency-summary analysis_outputs/rgb_input_ablation_summary/tables/task_consistency_summary.csv \
  --identity-summary analysis_outputs/identity_retrieval_summary/tables/identity_retrieval_run_summary.csv \
  --calibration-summary analysis_outputs/severity_calibration_summary/tables/severity_calibration_run_summary.csv \
  --output-dir analysis_outputs/mechanism_summary
```

---

## 6. RGB vs behavior 预测比较

```bash
python scripts/compare_behavior_predictions.py \
  --rgb-predictions experiment/default/mtl_lite/version_0/diagnostics/regression/test_predictions.csv \
  --behavior-predictions experiment/default/behavior_baseline/version_0/diagnostics/behavior/test_predictions.csv \
  --output-dir analysis_outputs/rgb_behavior_comparison
```

---

## 7. 完整实验工作流示例

### 7.1 跑一组 RGB 输入消融并汇总

```bash
# 1. 训练 rgb baseline
python scripts/train_mtl_lite.py \
  --override configs/regression_only_baseline.yaml \
  --override configs/input_ablation/rgb.yaml

# 2. 训练 center_mask
python scripts/train_mtl_lite.py \
  --override configs/regression_only_baseline.yaml \
  --override configs/input_ablation/center_mask.yaml

# 3. 对两个 run 生成 test 诊断（单 split 默认即可）
python scripts/diagnose_mtl_lite.py \
  --run-dir experiment/default/rgb/version_0 --ckpt best

python scripts/diagnose_mtl_lite.py \
  --run-dir experiment/default/center_mask/version_0 --ckpt best

# 4. 汇总比较
python scripts/summarize_prediction_runs.py \
  --output-dir analysis_outputs/rgb_vs_center_mask \
  --baseline rgb \
  --run rgb=experiment/default/rgb/version_0/diagnostics/regression/test_predictions.csv \
  --run center_mask=experiment/default/center_mask/version_0/diagnostics/regression/test_predictions.csv
```

### 7.2 完整多因素审计工作流

```bash
RUN_DIR=experiment/default/mtl_lite/version_0

# 1. 导出 val + test 预测
python scripts/diagnose_mtl_lite.py --run-dir $RUN_DIR --ckpt best --split val test

# 2. 各项审计
python scripts/audit_split_integrity.py \
  --split-file /path/to/dataset_split.json \
  --label-dir /path/to/labels \
  --image-root /path/to/aligned/frame/root \
  --predictions $RUN_DIR/diagnostics/test/regression/test_predictions.csv \
  --output-dir $RUN_DIR/diagnostics/test/split_integrity

python scripts/audit_temporal_sampling.py \
  --predictions $RUN_DIR/diagnostics/test/regression/test_predictions.csv \
  --image-root /path/to/AVEC2014/face_images \
  --output-dir $RUN_DIR/diagnostics/test/temporal_sampling \
  --sample-step 10 --max-seq-len 2000 --sampling-strategy stride_head

python scripts/audit_alignment_geometry.py \
  --predictions $RUN_DIR/diagnostics/test/regression/test_predictions.csv \
  --openface-root /path/to/openface_csv_root \
  --output-dir $RUN_DIR/diagnostics/test/alignment_geometry \
  --frame-width 640 --frame-height 480 --sample-step 1

python scripts/audit_black_artifacts.py \
  --predictions $RUN_DIR/diagnostics/test/regression/test_predictions.csv \
  --image-root /path/to/AVEC2014/face_images \
  --output-dir $RUN_DIR/diagnostics/test/black_artifacts \
  --sample-step 10

python scripts/audit_shortcuts.py \
  --predictions $RUN_DIR/diagnostics/test/regression/test_predictions.csv \
  --openface-root /path/to/openface_csv_root \
  --output-dir $RUN_DIR/diagnostics/test/shortcut_audit

python scripts/audit_identity_retrieval.py \
  --features-npz $RUN_DIR/diagnostics/test/embeddings/test_features.npz \
  --predictions $RUN_DIR/diagnostics/test/regression/test_predictions.csv \
  --output-dir $RUN_DIR/diagnostics/test/identity_retrieval \
  --top-k 5

python scripts/audit_severity_calibration.py \
  --val-predictions $RUN_DIR/diagnostics/val/regression/val_predictions.csv \
  --test-predictions $RUN_DIR/diagnostics/test/regression/test_predictions.csv \
  --output-dir $RUN_DIR/diagnostics/test/severity_calibration

python scripts/audit_task_inconsistency.py \
  --predictions $RUN_DIR/diagnostics/test/regression/test_predictions.csv \
  --black-artifacts-summary $RUN_DIR/diagnostics/test/black_artifacts/tables/black_artifact_summary.csv \
  --openface-quality-summary $RUN_DIR/diagnostics/test/shortcut_audit/tables/openface_quality_summary.csv \
  --alignment-geometry-summary $RUN_DIR/diagnostics/test/alignment_geometry/tables/alignment_geometry_summary.csv \
  --temporal-sampling-summary $RUN_DIR/diagnostics/test/temporal_sampling/tables/temporal_sampling_summary.csv \
  --output-dir $RUN_DIR/diagnostics/test/task_inconsistency \
  --top-n 20
```

---

## 8. 输出目录速查表

| 脚本 | 主要输出目录 | 关键产物 |
|---|---|---|
| `train_mtl_lite.py` | `experiment/<group>/<name>/version_N/` | `metrics.csv`、`checkpoints/`、`resolved_config.yaml` |
| `train_behavior_baseline.py` | `experiment/<group>/<name>/version_N/diagnostics/behavior/` | `val_predictions.csv`、`test_predictions.csv` |
| `diagnose_mtl_lite.py`（单 split） | `<run_dir>/diagnostics/` | `regression/test_predictions.csv`、`embeddings/test_features.npz` |
| `diagnose_mtl_lite.py`（多 split） | `<run_dir>/diagnostics/<split>/` | `<split>/regression/<split>_predictions.csv`、`<split>/embeddings/<split>_features.npz` |
| `audit_split_integrity.py` | `<output_dir>/` | `tables/split_integrity_report.md` 等 |
| `audit_temporal_sampling.py` | `<output_dir>/` | `tables/temporal_sampling_summary.csv`、报告 |
| `audit_alignment_geometry.py` | `<output_dir>/` | `tables/alignment_geometry_summary.csv`、报告 |
| `audit_black_artifacts.py` | `<output_dir>/` | `tables/black_artifact_summary.csv`、报告 |
| `audit_shortcuts.py` | `<output_dir>/` | `tables/openface_quality_summary.csv`、`shortcut_audit_report.md` |
| `audit_identity_retrieval.py` | `<output_dir>/` | `tables/embedding_identity_summary.csv`、相似度图、报告 |
| `audit_severity_calibration.py` | `<output_dir>/` | `tables/severity_calibration_fit.csv`、测试摘要、报告 |
| `audit_task_inconsistency.py` | `<output_dir>/` | `tables/task_inconsistency_manifest.csv`、`task_artifact_correlation.csv`、报告 |
| `summarize_prediction_runs.py` | `<output_dir>/` | `prediction_run_summary.csv`、`severity_bias_summary.csv`、报告 |
| `summarize_training_overfit.py` | `<output_dir>/` | `training_overfit_summary.csv`、报告 |
| `summarize_identity_retrieval_runs.py` | `<output_dir>/` | `identity_retrieval_run_summary.csv`、报告 |
| `summarize_severity_calibration_runs.py` | `<output_dir>/` | `severity_calibration_run_summary.csv`、报告 |
| `summarize_mechanism.py` | `<output_dir>/` | `mechanism_summary.csv`、`mechanism_report.md` |

---

## 9. 常见问题

**Q: 训练输出没有生成 `version_0` 而是覆盖到了旧目录？**

A: 检查 `configs/local_paths.yaml` 是否仍设置了 `LOG_DIR` 并启用了旧 runner。新 runner 使用 `EXPERIMENT_ROOT / EXPERIMENT_GROUP / EXPERIMENT_NAME`；确认 override 中 `EXPERIMENT_NAME` 与预期一致。

**Q: `diagnose_mtl_lite.py` 报错找不到 `test_predictions.csv`？**

A: 如果使用 `--split val test`，预测文件在 `<run_dir>/diagnostics/test/regression/test_predictions.csv`；如果单 split 默认，文件在 `<run_dir>/diagnostics/regression/test_predictions.csv`。

**Q: behavior baseline 报错 `DATASET.OPENFACE_ROOT is required`？**

A: 必须额外提供一个 override YAML 或在 `configs/local_paths.yaml` 中设置 `DATASET.OPENFACE_ROOT`。

**Q: 旧端到端训练（`scripts/train.py`）还能用吗？**

A: 当前旧端到端模型 `src/models/end_to_end.py` 缺失，legacy 训练入口不可用。如需复现旧结果，需先恢复 legacy 模块或显式要求修复。

**Q: 配置加载失败？**

A: 确认 `configs/local_paths.yaml` 存在，或运行训练脚本时加上 `--allow-missing-local-paths`。但真实数据训练必须提供 `IMAGE_DIR`、`LABEL_DIR`、`DATASET_SPLIT_FILE`。
