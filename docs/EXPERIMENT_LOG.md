# EXPERIMENT_LOG.md

This log records completed project maintenance, smoke validation, and experiment
workflow milestones. Keep entries concise and reproducible.

## 2026-06-12

### Debug smoke and config-path maintenance

- Completed config loader validation in the real project environment.
- Created/prepared `configs/local_paths.yaml` on the target machine from the
  ignored local paths workflow.
- Restored/fixed `src.datasets.dataset` imports required before smoke training.
- Audited import dependencies after restoring `src.datasets.dataset`.
- Verified that debug smoke training can run end to end in the server
  environment.
- Verified `scripts.diagnose` import checks:
  - `import scripts.diagnose`
  - `from scripts.diagnose import build_parser, load_config_from_args, run_diagnostic`
  - `python scripts/diagnose.py --help`

## 2026-06-13

### OpenFace behavior-representation research direction

- Reviewed recent experiment behavior from regression-only and backbone
  freeze/high-layer finetuning runs.
- Interpreted the generalization issue as likely shortcut learning inside
  OpenFace aligned face frames rather than only raw background overfitting.
- Decided to prioritize OpenFace quality diagnostics, input ablations,
  landmark/AU/pose/gaze baselines, and RGB + behavior late fusion before
  further backbone-layer search.
- Added `docs/RESEARCH_NOTES.md` to archive related papers and the next
  experiment roadmap.

### Shortcut Audit design

- Defined the Shortcut Audit Framework for validating non-depression shortcuts
  in OpenFace aligned face inputs.
- Scoped the first implementation to offline diagnostics: OpenFace quality
  summary, prediction-residual correlation, heatmaps, and markdown report.
- Added `docs/SHORTCUT_AUDIT_DESIGN.md` as the implementation blueprint.

### Shortcut Audit video-level alignment

- Found that short `subject_id` values such as `203_1` are ambiguous because
  OpenFace exports may contain both Freeform and Northwind files for the same
  subject/session.
- Updated the planned diagnostic alignment to prefer full `video_id` and only
  fall back to `subject_id` when the match is unique.

### Latest Shortcut Audit file review

- Reviewed the latest exported diagnostic files:
  - `test_predictions.csv`
  - `openface_quality_summary.csv`
  - `shortcut_audit_report.md`
  - `shortcut_merged.csv`
  - `shortcut_correlation.csv`
  - `shortcut_predictor_results.csv`
- Found that `test_predictions.csv` used IDs such as
  `203_2_Freeform_video_aligned`, while OpenFace summaries used IDs such as
  `203_2_Freeform_video`.
- The current Shortcut Audit matched 0 samples, so the generated shortcut risk
  level is invalid and must not be interpreted as evidence of low shortcut
  risk.
- A manual normalization check that removes the `_aligned` suffix matched
  100/100 prediction samples, confirming that the problem is an ID
  normalization gap rather than missing OpenFace files.
- Current prediction diagnostics still show strong range compression:
  MAE about 8.91, RMSE about 10.95, Pearson about 0.35, CCC about 0.29.
- Group-wise bias remains important:
  minimal samples are overpredicted on average, while severe samples are
  strongly underpredicted on average.
- OpenFace pose/gaze/AU/quality features show non-trivial diagnostic signal
  after manual matching; they should remain offline audit variables until a
  behavior-only baseline and leakage-safe grouped validation are established.

### Shortcut Audit `_aligned` video-id normalization fix

- Updated Shortcut Audit matching so prediction IDs such as
  `203_2_Freeform_video_aligned` normalize to `203_2_Freeform_video`.
- Added a focused test to ensure aligned prediction IDs match the correct
  Freeform/Northwind OpenFace summary row without falling back to ambiguous
  short `subject_id` matching.
- Verified with the latest local logs that 100/100 prediction rows match the
  OpenFace quality summary after normalization.

### Valid Shortcut Audit and representation analysis

- Re-ran Shortcut Audit after the `_aligned` video-id normalization fix.
- Verified that `shortcut_audit_report.md` now reports `Matched samples: 100`
  and `shortcut_merged.csv` contains 100 rows matched by full `video_id`.
- Current Shortcut Audit risk level is medium, with maximum absolute
  correlation about 0.418 between `AU07_c_mean` and `true_bdi`.
- Current RGB/MTL-Lite prediction remains range-compressed:
  MAE about 8.91, RMSE about 10.95, Pearson about 0.35, CCC about 0.29,
  prediction std about 6.13 versus true BDI std about 11.48.
- Group-wise bias:
  minimal samples are overpredicted on average by about +6.89, while severe
  samples are underpredicted on average by about -16.50.
- Most severe high-error cases include `246_1`, `359_1`, `237_1`, and
  `315_2`; these should be prioritized for case-study visualization.
- Freeform and Northwind aggregate metrics are similar, but same-subject task
  predictions can differ substantially. The largest observed task-pair
  difference is about 13.61, and high-difference cases include `237_1`,
  `247_1`, `247_3`, `224_1`, and `212_1`.
- In-sample shortcut-only linear/ridge predictors are very strong, but this is
  not a valid generalization estimate because the diagnostic currently has 100
  samples and 94 numeric OpenFace features.
- A manual grouped-CV check suggests shortcut-only ridge remains close to, but
  slightly weaker than, the current RGB model. This supports treating OpenFace
  behavior/quality features as a medium shortcut risk and motivates a formal
  grouped-CV diagnostic plus behavior-only baseline.

### Grouped-CV shortcut-only predictor implementation

- Added subject-level grouped CV design to `docs/SHORTCUT_AUDIT_DESIGN.md`.
- Implemented grouped-CV shortcut-only predictor diagnostics in Shortcut Audit.
- The grouped CV logic keeps all videos from the same `subject_id` in the same
  fold to reduce Freeform/Northwind leakage.
- Shortcut Audit now writes both the combined `shortcut_predictor_results.csv`
  and a focused `shortcut_predictor_grouped_cv.csv`.
- The markdown report now separates in-sample predictor diagnostics from
  grouped-CV predictor diagnostics.
- Local validation:
  - `python -m compileall src scripts tests` passed.
  - Direct grouped-CV smoke passed.
  - Temporary end-to-end Shortcut Audit smoke generated the grouped-CV CSV and
    report section successfully.
- Local bundled Python still lacks `pytest`, so full pytest validation should
  be run in the server environment.

### P0 剩余任务设计归档

- 根据最新 grouped-CV shortcut-only predictor 结果，当前 shortcut 风险维持为
  medium：OpenFace 统计特征与标签、预测和误差存在中等相关，但不能单独接近
  RGB/MTL-Lite 测试表现。
- 后续 P0 不再只围绕 in-sample shortcut predictor 或 backbone 解冻层数展开，
  而是优先定位预测范围压缩、severe 系统性低估、minimal 系统性高估和
  Freeform/Northwind 同一 subject 预测不一致。
- 已将 P0-2 case study manifest、P0-3 input ablation protocol、
  P0-4 behavior-only baseline interface 的目标、输出、字段和判读原则写入
  `docs/TODO.md`、`docs/SHORTCUT_AUDIT_DESIGN.md`、`docs/CURRENT_STATUS.md`
  和 `docs/CODEX_CONTEXT.md`。
- 本次仅进行文档设计，不修改训练代码、不修改训练超参数、不修改
  `configs/local_paths.yaml`，也不删除或覆盖任何实验结果。

### P0-2 case study manifest implementation

- Added offline case-study manifest generation for high-error and
  task-inconsistency analysis.
- New module: `src/diagnostics/case_studies.py`.
- Regression diagnostics now emit `case_study_manifest.csv` and
  `case_study_manifest.md` next to the regression plots.
- Shortcut Audit now emits `tables/case_study_manifest.csv` and
  `reports/case_study_manifest.md`.
- Manifest case types:
  - `severe_underestimate`
  - `minimal_overestimate`
  - `task_inconsistency`
  - `low_error_reference`
- Added tests for manifest selection and output wiring.
- Validation:
  - `python -m compileall src scripts tests` passed with the bundled Codex
    Python runtime.
  - Direct manifest smoke passed.
  - Direct Shortcut Audit smoke generated `case_study_manifest.csv` and
    `case_study_manifest.md`.
  - Import check for `case_studies`, `regression`, and `shortcut_audit` passed.
  - Local bundled Python still lacks `pytest`; run the focused pytest command
    on the server environment.

### P0-3 input ablation variant implementation

- Added optional RGB input ablation support through `DATASET.INPUT_VARIANT`.
- New module: `src/datasets/input_variants.py`.
- `AVECDataset` now applies the selected variant before resize/normalize and
  before training augmentations.
- Default behavior remains `rgb`, so existing configs keep the same data path.
- Supported variants:
  - `rgb`
  - `grayscale`
  - `blur`
  - `center_mask`
  - `boundary_erased`
- `landmark_heatmap` is intentionally reserved for the OpenFace
  landmark/behavior baseline route; configuring it in the RGB dataset raises a
  clear error instead of silently generating a fake landmark input.
- Added `DATASET.INPUT_VARIANT: "rgb"` to `configs/avec2014_base.yaml`.
- Added `tests/test_input_variants.py`.
- Validation:
  - `python -m compileall src scripts tests` passed with the bundled Codex
    Python runtime.
  - Local bundled Python lacks `torch`, `pytorch_lightning`, and `pytest`; run
    input-variant pytest and dataset import checks on the server environment.

### P0-4 behavior-only baseline interface implementation

- Added independent OpenFace behavior-only baseline route.
- New dataset module: `src/datasets/openface_features.py`.
  - Matches OpenFace CSV files to split video IDs.
  - Builds AU/pose/gaze/landmark/quality temporal features.
  - Appends temporal delta by default and optional acceleration.
  - Computes normalization statistics from the training split only.
- New model module: `src/models/behavior_baseline.py`.
  - Feature projection + GRU temporal encoder + mask-aware pooling.
  - BDI regression head and optional ordinal auxiliary head.
- New runner and entry:
  - `src/trainers/behavior_baseline_runner.py`
  - `scripts/train_behavior_baseline.py`
- New config:
  - `configs/behavior_baseline.yaml`
- New tests:
  - `tests/test_openface_features.py`
  - `tests/test_behavior_baseline.py`
- Validation:
  - `python -m compileall src scripts tests` passed with the bundled Codex
    Python runtime.
  - Local bundled Python lacks `torch`, `pytorch_lightning`, and `pytest`; run
    focused behavior baseline tests and import checks on the server environment.
