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

## 2026-06-14

### Behavior-only baseline result review

- Reviewed the latest `behavior_metrics.csv` exported from the OpenFace
  behavior-only baseline run.
- Test metrics:
  - MAE about 9.93.
  - RMSE about 12.86.
  - CCC about 0.151.
- Best validation RMSE occurred around epoch 65:
  - val MAE about 9.94.
  - val RMSE about 12.38.
  - val CCC about 0.324.
  - corresponding train MAE about 2.17, train RMSE about 2.74, train CCC about
    0.975.
- Interpretation:
  - The behavior-only baseline currently overfits strongly.
  - Complete OpenFace feature sets should not be treated as clean behavioral
    representations.
  - Raw landmark coordinates and static facial geometry may carry identity or
    subject-specific shortcuts.
  - Late fusion should wait until feature-group ablation identifies a stable
    behavior subset.
- Updated documentation:
  - `docs/CURRENT_STATUS.md`
  - `docs/TODO.md`
  - `docs/CODEX_CONTEXT.md`
  - `docs/RESEARCH_NOTES.md`
  - `docs/SHORTCUT_AUDIT_DESIGN.md`
  - `docs/BUG_LOG.md`

### P0 behavior prediction export implementation

- Added behavior baseline val/test prediction export after best-checkpoint test
  evaluation.
- Prediction CSV files are written under:
  `behavior_baseline_csv/version_*/diagnostics/behavior/`.
- Exported files:
  - `val_predictions.csv`
  - `test_predictions.csv`
- Prediction rows now include `video_id`, `subject_id`, `task_name`,
  `true_bdi`, `pred_bdi`, `residual`, `abs_error`, and `severity_group`.
- Extended the shared prediction-table writer with an optional `task_name`
  field while preserving existing callers that do not provide task names.
- This change does not alter model forward, losses, metrics, training
  hyperparameters, checkpoint selection, or `configs/local_paths.yaml`.

### P0 behavior feature-set ablation interface

- Added `BEHAVIOR_FEATURES.FEATURE_SET` with default value `custom`.
- Supported named feature sets:
  - `quality_only`
  - `au_only`
  - `pose_gaze_only`
  - `raw_landmark_only`
  - `landmark_delta_only`
  - `au_landmark_delta`
  - `all_without_raw_landmarks`
- The named feature sets make future ablations runnable with a minimal override
  instead of maintaining many near-duplicate YAML files.
- `landmark_delta_only` excludes raw landmark coordinates, and
  `all_without_raw_landmarks` keeps non-landmark raw features while retaining
  temporal deltas for all selected features.
- Default `custom` behavior preserves the existing boolean feature flags and
  therefore does not change previous behavior baseline runs.

### P0 RGB-vs-behavior prediction comparison interface

- Added offline comparison module:
  - `src/diagnostics/behavior_comparison.py`
  - `scripts/compare_behavior_predictions.py`
- The comparison aligns RGB/MTL-Lite and behavior-only predictions by
  normalized `video_id`, including compatibility with `_aligned` processing
  suffixes.
- Outputs:
  - `rgb_behavior_prediction_comparison.csv`
  - `rgb_behavior_prediction_summary.csv`
- The summary reports RGB and behavior MAE/RMSE/Pearson/CCC, severity-group
  metrics, and counts where RGB, behavior, or neither is better.

## 2026-06-15

### RGB input ablation result review

- Reviewed the first RGB input ablation batch:
  - `rgb`
  - `grayscale`
  - `blur`
  - `center_mask`
  - `boundary_erased`
- All reviewed runs used comparable core settings: same split, seed, MTL-Lite
  regression-only route, DeiT-tiny backbone weights, frozen backbone with the
  last 2 transformer blocks trainable, max sequence length 2000, and the same
  checkpoint-based evaluation style.
- Key test results:
  - `center_mask`: MAE about `7.94`, RMSE about `10.16`, Pearson about `0.51`,
    CCC about `0.48`.
  - `rgb`: MAE about `8.91`, RMSE about `10.95`, Pearson about `0.35`, CCC
    about `0.29`.
  - `boundary_erased`: close to or slightly better than `rgb`, but weaker than
    `center_mask`.
  - `blur` and `grayscale`: worse than `rgb`.
- Interpretation:
  - The improvement from `center_mask` suggests that peripheral/crop/alignment
    artifacts are hurting generalization.
  - The degradation from `grayscale` and `blur` suggests that simple color
    shortcut or fine texture shortcut is not the only explanation.
  - Severe underestimation remains unresolved, so input artifact mitigation is
    only one part of the failure analysis.

### OpenFace black-padding artifact hypothesis

- Reviewed an OpenFace aligned face sample showing pure black fill around the
  face contour and black microphone occlusion inside the crop.
- Updated the research hypothesis from generic "background shortcut" to a more
  specific OpenFace artifact mechanism:
  - black padding around aligned face contours;
  - black occluder regions such as microphones;
  - hard pixel discontinuities at crop and mask boundaries;
  - possible ViT/DeiT sensitivity to these structured high-contrast edges.
- Current priority is to explain RGB model overfitting before adding RGB +
  behavior late fusion or additional auxiliary tasks.

### Black artifact ablation implementation

- Extended `src/datasets/input_variants.py` with black artifact variants:
  - `black_to_gray`
  - `black_to_mean`
  - `black_to_blur`
  - `soft_center_mask`
  - `inner_crop_resize`
- Added aliases:
  - `black_fill_gray`
  - `black_fill_mean`
  - `black_fill_blur`
  - `soft_mask`
  - `inner_crop`
- Added input ablation configs:
  - `configs/input_ablation/black_to_gray.yaml`
  - `configs/input_ablation/black_to_mean.yaml`
  - `configs/input_ablation/black_to_blur.yaml`
  - `configs/input_ablation/soft_center_mask.yaml`
  - `configs/input_ablation/inner_crop_resize.yaml`
- Added tests for black replacement, soft masks, inner crop behavior, and
  aliases in `tests/test_input_variants.py`.

### Black artifact diagnostic implementation

- Added offline audit module:
  - `src/diagnostics/black_artifacts.py`
  - `scripts/audit_black_artifacts.py`
- The audit samples aligned frames and writes:
  - `tables/black_artifact_summary.csv`
  - `tables/black_artifact_merged.csv`
  - `tables/black_artifact_correlation.csv`
  - `reports/black_artifact_audit_report.md`
- Per-video features include:
  - mean/std black pixel ratio;
  - border black pixel ratio;
  - center black pixel ratio;
  - black-boundary edge ratio;
  - frame-to-frame black ratio delta.
- The merged audit correlates these artifact statistics with `true_bdi`,
  `pred_bdi`, `residual`, and `abs_error`.

### Validation

- Local compile validation passed for:
  - `src/datasets/input_variants.py`
  - `src/diagnostics/black_artifacts.py`
  - `scripts/audit_black_artifacts.py`
  - `tests/test_input_variants.py`
- Local bundled Python still lacks `torch` and `pytest`; focused pytest should
  be run on the server:

```bash
python -m pytest tests/test_input_variants.py
```

### Black artifact ablation result review

- Reviewed second-round RGB artifact ablations:
  - `black_to_gray`
  - `black_to_mean`
  - `black_to_blur`
  - `soft_center_mask`
  - `inner_crop_resize`
- Compared them against previous `rgb`, `center_mask`, and
  `boundary_erased` runs under the same core training settings.
- Main results:
  - `center_mask` remains best overall: MAE about `7.94`, RMSE about `10.16`,
    Pearson about `0.51`, CCC about `0.48`.
  - `black_to_gray` is the best new black-artifact variant: MAE about `8.34`,
    RMSE about `10.62`, CCC about `0.39`.
  - `soft_center_mask` improves severe bias more than most variants but
    overpredicts minimal samples, leading to worse overall MAE than
    `center_mask`.
  - `black_to_mean` and `inner_crop_resize` degrade test performance and should
    not be prioritized as the next main direction.
- Interpretation:
  - Replacing black pixels helps compared with raw `rgb`, but does not explain
    the full `center_mask` gain.
  - The artifact risk is likely a mixture of boundary fill, crop shape,
    peripheral non-behavior regions, face contour, pose/scale remnants, and
    subject-specific appearance.

### Black artifact audit result review

- Reviewed `black_artifact_audit_report.md`,
  `black_artifact_summary.csv`, `black_artifact_merged.csv`, and
  `black_artifact_correlation.csv`.
- The audit matched all expected test videos:
  - Videos summarized: `100`.
  - Missing videos: `0`.
  - Matched prediction rows: `100`.
- Black regions are common in aligned frames:
  - `black_ratio_mean` average about `0.24`.
  - `black_border_ratio_mean` average about `0.44`.
  - `black_center_ratio_mean` average about `0.02`.
- Maximum absolute correlation is weak, about `0.207`, so black artifacts are
  not a strong single-variable explanation for RGB overfitting.
- However, high-border-black quartile samples show larger average error than
  low-border-black quartile samples, about `12.29` versus `7.45`.
- Important correction:
  - `black_border_ratio_mean` is the cleaner OpenFace artifact indicator.
  - `black_center_ratio_mean` is semantically mixed because center black pixels
    may represent nostrils, mouth shadows, beard, natural facial shadows,
    microphones, or true occlusions.
- Next experiment plan:
  - Implement `border_black_to_gray`.
  - Implement `border_black_feather`.
  - Implement `center_mask_black_to_gray`.
  - Keep center black pixels unchanged unless a case study confirms they are
    preprocessing artifacts rather than facial structure or real occlusion.

### Border-connected black artifact ablation implementation

- Added three border-connected black artifact input variants:
  - `border_black_to_gray`
  - `border_black_feather`
  - `center_mask_black_to_gray`
- Implementation location:
  - `src/datasets/input_variants.py`
- New configs:
  - `configs/input_ablation/border_black_to_gray.yaml`
  - `configs/input_ablation/border_black_feather.yaml`
  - `configs/input_ablation/center_mask_black_to_gray.yaml`
- Added focused tests in `tests/test_input_variants.py`:
  - border-connected black pixels are replaced;
  - center black pixels remain unchanged;
  - feathering softens boundary-adjacent pixels;
  - `center_mask_black_to_gray` uses a gray outside region rather than a hard
    black outside region.
- Local validation:
  - `python -m compileall src/datasets/input_variants.py tests/test_input_variants.py` passed with the bundled Codex Python runtime.
  - Local Python environment still lacks `pytest` and `torch`; run the focused
    pytest command on the server environment.
