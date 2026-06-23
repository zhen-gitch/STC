# EXPERIMENT_LOG.md

> 文档职责：按时间追加实验和实现记录，用于追溯证据。当前权威结论见 `CURRENT_STATUS.md` / `RGB_OVERFITTING_AUDIT_PLAN.md`；文档导航见 `DOCS_GUIDE.md`。

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
  `<LOG_DIR>/default/behavior_baseline/version_0/diagnostics/behavior/`.
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

### RGB overfitting factor map documentation

- Consolidated the broader RGB overfitting hypothesis beyond black artifacts.
- Added a multi-factor research map to `docs/RESEARCH_NOTES.md`, covering:
  - identity and static appearance shortcuts;
  - OpenFace alignment geometry and crop artifacts;
  - pose, gaze, confidence, success, and tracking quality;
  - video length, temporal sampling, and padding;
  - Freeform/Northwind task-context differences;
  - prediction compression and severity calibration;
  - ViT/DeiT patch-level shortcut sensitivity;
  - targeted robustness augmentation.
- Expanded `docs/SHORTCUT_AUDIT_DESIGN.md` with audit modules for:
  - identity/static appearance;
  - alignment geometry;
  - pose/gaze/tracking quality;
  - temporal sampling;
  - task consistency;
  - severity calibration.
- Added follow-up task queues to `docs/TODO.md`.
- Updated `docs/CURRENT_STATUS.md` and `docs/CODEX_CONTEXT.md` so future work
  treats black borders as one risk factor rather than a complete explanation.

### P0 temporal sampling audit implementation

- Implemented the first P0 multi-factor audit for video length, temporal
  sampling, truncation, and padding.
- New module:
  - `src/diagnostics/temporal_sampling.py`
- New CLI:
  - `scripts/audit_temporal_sampling.py`
- New tests:
  - `tests/test_temporal_sampling_audit.py`
- The audit follows the actual `AVECDataset` selection rule:
  - `sampled_frame_count = len(frames[::SAMPLE_STEP])`
  - `model_max_len = MAX_SEQ_LEN // SAMPLE_STEP`
  - `selected_frame_count = min(sampled_frame_count, model_max_len)`
  - padding and truncation are computed from the model-visible temporal length.
- Outputs:
  - `tables/temporal_sampling_summary.csv`
  - `tables/temporal_sampling_merged.csv`
  - `tables/temporal_sampling_correlation.csv`
  - `tables/temporal_sampling_group_summary.csv`
  - `reports/temporal_sampling_audit_report.md`
- Local validation:
  - `python -m compileall src/diagnostics/temporal_sampling.py scripts/audit_temporal_sampling.py tests/test_temporal_sampling_audit.py` passed with the bundled Codex Python runtime.
  - Direct smoke test generated all expected outputs.
  - Local Python still lacks `pytest`; run focused pytest on the server.

### P0 temporal sampling ablation implementation

- Added configurable temporal sampling strategies while preserving the
  historical default behavior.
- New module:
  - `src/datasets/temporal_sampling.py`
- Updated dataset:
  - `src/datasets/dataset.py`
- New config key:
  - `PROCESS_TEMPORAL.SAMPLING_STRATEGY`
- Supported strategies:
  - `stride_head`: historical `frames[::SAMPLE_STEP][:MAX_SEQ_LEN // SAMPLE_STEP]`;
  - `uniform`: uniformly sample model-visible frames across the whole video;
  - `first`: contiguous first crop;
  - `middle`: contiguous middle crop;
  - `random`: deterministic random contiguous crop keyed by video path.
- New training overrides:
  - `configs/temporal_sampling/uniform_256.yaml`
  - `configs/temporal_sampling/uniform_512.yaml`
  - `configs/temporal_sampling/uniform_1024.yaml`
  - `configs/temporal_sampling/first_crop.yaml`
  - `configs/temporal_sampling/middle_crop.yaml`
  - `configs/temporal_sampling/random_crop.yaml`
- New tests:
  - `tests/test_temporal_sampling.py`
  - extended `tests/test_temporal_sampling_audit.py`
- Local validation:
  - Compile validation passed.
  - Direct strategy smoke passed.
  - Direct uniform temporal audit smoke passed.
  - Server still needs focused pytest and actual training ablations.

### Border-connected black ablation result review

- Reviewed:
  - `rgb_ablation_border_black_to_gray`
  - `rgb_ablation_border_black_feather`
  - `rgb_ablation_center_mask_black_to_gray`
- All three runs are comparable to previous RGB ablations in core settings.
- Main results:
  - `center_mask_black_to_gray`: MAE about `7.73`, RMSE about `10.02`,
    Pearson about `0.52`, CCC about `0.45`.
  - `center_mask`: MAE about `7.94`, RMSE about `10.16`, Pearson about
    `0.51`, CCC about `0.48`.
  - `border_black_feather`: MAE about `8.04`, RMSE about `10.13`, Pearson
    about `0.50`, CCC about `0.41`.
  - `border_black_to_gray`: weaker than feathering, MAE about `8.69`.
- Interpretation:
  - `center_mask_black_to_gray` is currently best by MAE/RMSE/Pearson, but it
    mainly improves minimal/mild samples and worsens severe underestimation.
  - `center_mask` remains more balanced and retains the best CCC.
  - `border_black_feather` supports the boundary-softening hypothesis better
    than hard gray replacement.
  - Input artifact mitigation and severity calibration must be treated as
    separate problems.

### P0 prediction run summary implementation

- Added reusable multi-run prediction summary diagnostics:
  - `src/diagnostics/prediction_runs.py`
  - `scripts/summarize_prediction_runs.py`
  - `tests/test_prediction_runs.py`
- Outputs:
  - `tables/prediction_run_summary.csv`
  - `tables/severity_bias_summary.csv`
  - `tables/task_consistency_summary.csv`
  - `tables/pairwise_baseline_improvement.csv`
  - `reports/prediction_runs_report.md`
- The tool records true/pred mean/std, overall MAE/RMSE/Pearson/CCC,
  severity bias, Freeform/Northwind task consistency, and pairwise improvement
  against an optional baseline run.

### P0 RGB input ablation unified summary

- Ran `scripts/summarize_prediction_runs.py` on all currently available RGB
  input ablation prediction CSV files.
- Included runs:
  - `rgb`
  - `gray_scale`
  - `blur`
  - `boundary_erased`
  - `center_mask`
  - `black_to_gray`
  - `black_to_mean`
  - `black_to_blur`
  - `soft_center_mask`
  - `inner_crop_resize`
  - `border_black_feather`
  - `border_black_to_gray`
  - `center_mask_black_to_gray`
- Generated unified tables:
  - `prediction_run_summary.csv`
  - `severity_bias_summary.csv`
  - `task_consistency_summary.csv`
  - `pairwise_baseline_improvement.csv`
  - `prediction_runs_report.md`
- Current ranking by test MAE:
  `center_mask_black_to_gray` < `center_mask` < `border_black_feather` <
  `black_to_gray` < `soft_center_mask` < `border_black_to_gray` < `rgb`.
- Interpretation remains conservative:
  - `center_mask_black_to_gray` is best overall by MAE/RMSE/Pearson, but
    worsens severe underestimation.
  - `center_mask` remains the most balanced candidate by CCC and task
    consistency.
  - `gray_scale`, `blur`, `inner_crop_resize`, and `black_to_mean` do not
    support a single-factor explanation based only on color, texture, or
    peripheral cropping.
  - Severity calibration remains a separate P0/P2 issue from input artifact
    mitigation.

### High-priority overfitting audit review

- Reviewed the remaining overfitting-related ablation and validation queue.
- Decision: do not keep prioritizing additional RGB mask variants. The black
  artifact line has enough evidence to support a bounded sub-conclusion, but
  it cannot explain prediction compression or severe underestimation alone.
- Promoted the following items to the next high-priority queue:
  - split / subject integrity audit;
  - temporal sampling audit and fixed-frame / temporal-crop ablations;
  - training overfit curve summary across runs;
  - OpenFace alignment geometry audit;
  - embedding identity paired-task retrieval;
  - severity calibration verification;
  - task inconsistency mixed-factor audit.
- Updated `docs/TODO.md`, `docs/CURRENT_STATUS.md`,
  `docs/RESEARCH_NOTES.md`, `docs/SHORTCUT_AUDIT_DESIGN.md`, and
  `docs/CODEX_CONTEXT.md` with purpose, outputs, interpretation rules, and
  recommended execution order.
- Key interpretation:
  RGB overfitting should now be treated as a multi-factor mechanism involving
  OpenFace alignment geometry, identity/static appearance, temporal sampling,
  task context, and label-distribution compression. Border artifacts remain an
  important visible entry point, not a sufficient explanation.

### RGB overfitting audit plan consolidation

- Added canonical plan document:
  - `docs/RGB_OVERFITTING_AUDIT_PLAN.md`
- Reorganized related documentation roles:
  - `RGB_OVERFITTING_AUDIT_PLAN.md`: authoritative research plan and
    experiment ordering;
  - `CURRENT_STATUS.md`: current status summary;
  - `TODO.md`: executable task checklist;
  - `SHORTCUT_AUDIT_DESIGN.md`: audit input/output/field specification;
  - `RESEARCH_NOTES.md`: paper background and research rationale;
  - `CODEX_CONTEXT.md`: handoff context for future Codex sessions.
- Clarified that black border / black padding is one possible overfitting
  factor, not the main thesis.
- Updated the current execution order to:
  split integrity -> temporal sampling -> training overfit curves ->
  alignment geometry -> embedding identity retrieval -> severity calibration ->
  task inconsistency mixed-factor audit.

### P0-A split integrity audit implementation

- Implemented offline split / subject integrity audit:
  - `src/diagnostics/split_integrity.py`
  - `scripts/audit_split_integrity.py`
  - `tests/test_split_integrity.py`
- The audit checks subject overlap across train/val/test, duplicate video ids,
  missing or invalid BDI label files, optional aligned image directory
  existence, and optional prediction-to-split alignment.
- Expected outputs:
  - `tables/split_video_manifest.csv`
  - `tables/split_subject_overlap.csv`
  - `tables/split_label_distribution.csv`
  - `tables/split_prediction_alignment.csv`
  - `reports/split_integrity_report.md`
- Interpretation rule:
  `split_integrity_report.md` should report `status: PASS` before later RGB
  overfitting diagnostics are interpreted as valid subject-disjoint
  generalization evidence.
- Local validation:
  compileall passed for `src/diagnostics/split_integrity.py`,
  `scripts/audit_split_integrity.py`, and `tests/test_split_integrity.py`;
  direct smoke passed on synthetic split/label/image/prediction data. Local
  bundled Python does not include `pytest`, so formal pytest remains
  server-side.
- Next server-side commands:

```bash
python -m pytest tests/test_split_integrity.py

python scripts/audit_split_integrity.py \
  --split-file /path/to/dataset_split.json \
  --label-dir /path/to/labels \
  --image-root /path/to/aligned/frame/root \
  --predictions <LOG_DIR>/default/rgb/version_0/diagnostics/regression/test_predictions.csv \
  --output-dir <LOG_DIR>/default/rgb/version_0/diagnostics/split_integrity
```

### P0-B training overfit summary implementation

- Implemented offline training overfit summary:
  - `src/diagnostics/training_overfit.py`
  - `scripts/summarize_training_overfit.py`
  - `tests/test_training_overfit.py`
- The audit reads Lightning-style `metrics.csv`, aggregates sparse metric rows
  by epoch, selects the best validation epoch by `val_RMSE_epoch` when
  available, then falls back to `val_MAE_epoch` and `val_loss`.
- Expected outputs:
  - `tables/training_overfit_summary.csv`
  - `tables/training_curve_gap_by_run.csv`
  - `reports/training_overfit_report.md`
- Interpretation rule:
  if a run improves test MAE but has a larger train/val gap or
  `overfit_after_best_val=True`, treat the result as possible prediction bias
  or checkpoint behavior rather than direct evidence of better generalization.
- Local validation:
  compileall passed for `src/diagnostics/training_overfit.py`,
  `scripts/summarize_training_overfit.py`, and `tests/test_training_overfit.py`;
  direct smoke passed on synthetic Lightning-style metrics rows. Local bundled
  Python does not include `pytest`, so formal pytest remains server-side.
- Next server-side commands:

```bash
python -m pytest tests/test_training_overfit.py

python scripts/summarize_training_overfit.py \
  --output-dir analysis_outputs/training_overfit_summary \
  --run rgb=/path/to/rgb/metrics.csv \
  --run center_mask=/path/to/center_mask/metrics.csv \
  --run center_mask_black_to_gray=/path/to/center_mask_black_to_gray/metrics.csv \
  --run border_black_feather=/path/to/border_black_feather/metrics.csv \
  --run behavior=/path/to/behavior_baseline/metrics.csv
```

### P0-C alignment geometry audit implementation

- Implemented offline OpenFace alignment geometry audit:
  - `src/diagnostics/alignment_geometry.py`
  - `scripts/audit_alignment_geometry.py`
  - `tests/test_alignment_geometry.py`
- The audit reads OpenFace landmark columns `x_*` and `y_*`, computes landmark
  bbox width/height/area/aspect, face center offset, eye distance,
  normalized face scale, confidence/success summary, and landmark jitter.
- Expected outputs:
  - `tables/alignment_geometry_summary.csv`
  - `tables/alignment_geometry_merged.csv`
  - `tables/alignment_geometry_correlation.csv`
  - `tables/alignment_geometry_group_summary.csv`
  - `reports/alignment_geometry_audit_report.md`
- Interpretation rule:
  if geometry variables correlate with `residual` or `abs_error`, treat
  OpenFace alignment geometry as a shortcut or mixed-factor risk rather than
  attributing RGB overfitting only to black padding artifacts.
- Local validation:
  compileall passed for `src/diagnostics/alignment_geometry.py`,
  `scripts/audit_alignment_geometry.py`, and `tests/test_alignment_geometry.py`;
  direct smoke passed on synthetic OpenFace CSV and prediction rows. Local
  bundled Python does not include `pytest`, so formal pytest remains
  server-side.
- Next server-side commands:

```bash
python -m pytest tests/test_alignment_geometry.py

python scripts/audit_alignment_geometry.py \
  --predictions <LOG_DIR>/default/rgb/version_0/diagnostics/regression/test_predictions.csv \
  --openface-root /path/to/openface_csv_root \
  --output-dir <LOG_DIR>/default/rgb/version_0/diagnostics/alignment_geometry \
  --frame-width 112 \
  --frame-height 112 \
  --sample-step 1
```

### P0-A/B/C server results and geometry scale correction

Split integrity audit:

- `status: PASS`
- `300` videos, `150` subjects, `3` splits.
- `overlapping_subjects = 0`, `duplicate_video_rows = 0`, `missing_labels = 0`,
  `invalid_labels = 0`, `missing_image_dirs = 0`.
- Prediction alignment matched `100/100` test prediction rows.
- Interpretation: current RGB overfitting analysis is not explained by split
  leakage, label missingness, duplicate video ids, or prediction alignment
  mismatch.

Training overfit summary:

- RGB-family runs all reported `overfit_after_best_val=True`.
- `rgb` best validation RMSE occurred early at epoch `6`; after that, train
  RMSE kept improving while validation RMSE degraded.
- `center_mask_black_to_gray` had the best validation RMSE among the compared
  runs but still overfit after best validation.
- Behavior baseline showed a very large train/val gap, supporting the concern
  that full OpenFace behavior features contain strong subject/static geometry
  memorization signals.

Alignment geometry audit:

- `300` OpenFace videos summarized.
- `100/100` test prediction rows matched.
- Max absolute correlation: about `0.3746`.
- Top findings:
  - `landmark_bbox_height_mean` vs `true_bdi`: `r = 0.3746`
  - `landmark_bbox_area_mean` vs `true_bdi`: `r = 0.3410`
  - `landmark_bbox_width_mean` vs `true_bdi`: `r = 0.3041`
  - `eye_distance_mean` vs `true_bdi`: `r = 0.2991`
  - `landmark_bbox_height_mean` vs `residual`: `r = -0.2605`

Coordinate scale correction:

- Current OpenFace CSV landmark coordinates are not `112 x 112` aligned-frame
  coordinates.
- Observed examples: `x` range approximately `150-643`, `y` range
  approximately `-11-582`, while aligned jpg inputs are `112 x 112`.
- OpenFace run metadata reported camera parameters `500,500,320,240`, which
  indicates a likely source coordinate frame of approximately `640 x 480`.
- The alignment geometry audit was rerun with `--frame-width 640` and
  `--frame-height 480`.
- Therefore, the current geometry result should be interpreted as
  `pre-alignment detection geometry confound`, not as direct 112x112 input
  landmark geometry.
- Directly interpretable metrics: bbox width/height/area/aspect, eye distance,
  and landmark jitter in the OpenFace detection coordinate system.
- After the `640 x 480` rerun, `normalized_face_scale_mean` and
  `face_center_offset_*` are interpretable as relative pre-alignment detection
  geometry metrics. They still must not be described as 112x112 input-space
  landmark coordinates.
- Severity group means after the `640 x 480` rerun:
  - minimal: `scale=0.319`, `residual=+6.89`
  - mild: `scale=0.337`, `residual=-1.86`
  - moderate: `scale=0.399`, `residual=-7.66`
  - severe: `scale=0.364`, `residual=-16.50`
- Recommended follow-up implementation: add raw coordinate ranges and relative
  geometry ratios such as `eye_distance_to_bbox_height_ratio`.

### P0 temporal sampling audit server result

- Reviewed the RGB temporal sampling audit output generated on the current
  test prediction CSV.
- Matching quality was valid:
  - Videos summarized: `100`.
  - Matched prediction rows: `100`.
  - Missing videos: `0`.
  - Maximum absolute correlation: about `0.2197`.
- Main correlations:
  - `truncated_frame_count` vs `pred_bdi`: about `r = -0.2197`.
  - `truncated_ratio` vs `pred_bdi`: about `r = -0.2090`.
  - `raw_to_selected_ratio` vs `pred_bdi`: about `r = 0.2090`.
  - `sampled_frame_count` / `frame_count` vs `pred_bdi`: about
    `r = -0.2073`.
- Interpretation:
  - Temporal length and truncation are weak-to-moderate confounds.
  - Longest videos tend to receive lower predictions and higher errors, even
    though they have no padding; this points toward truncation, head-biased
    sampling, or diluted key behavior rather than padding alone.
  - Freeform videos are longer and less padded, while Northwind videos are
    shorter and more padded, but task-level absolute errors are similar.
  - Temporal sampling should remain a P0 ablation target, but it should not be
    treated as the sole cause of RGB overfitting or severe underestimation.
- Next:
  - Run `uniform_256`, `uniform_512`, `uniform_1024`, `first_crop`,
    `middle_crop`, and `random_crop` training ablations.
  - Summarize each run with `scripts/summarize_prediction_runs.py`.
### Local occlusion and static-appearance shortcut direction

- Added a focused research direction for eyeglasses, microphones, beard, local reflections, and mouth-region occluders.
- Interpretation:
  - These factors should not replace the broader multi-factor overfitting explanation.
  - They sit between identity/static appearance shortcut and local occlusion artifact.
  - They may act as subject identifiers, corrupt visible facial behavior regions, or create high-contrast ViT patches.
- Updated documentation:
  - `docs/RGB_OVERFITTING_AUDIT_PLAN.md`
  - `docs/SHORTCUT_AUDIT_DESIGN.md`
  - `docs/RESEARCH_NOTES.md`
  - `docs/TODO.md`
  - `docs/CURRENT_STATUS.md`
  - `docs/CODEX_CONTEXT.md`
- Recommended next step is case-study and spatial occlusion verification before adding global preprocessing rules.

### Systematic overfitting mechanism roadmap

- Added `docs/OVERFITTING_MECHANISM_ROADMAP.md` as a higher-level route for RGB overfitting mechanism research.
- The roadmap organizes current evidence into layers: data validity, prediction compression, input artifact/local occlusion, identity/static appearance, OpenFace geometry/quality, temporal/task context, and model optimization.
- It also adds a decision tree and stop rules to avoid scattered experiment accumulation.
- Updated references from `RGB_OVERFITTING_AUDIT_PLAN.md`, `TODO.md`, `CURRENT_STATUS.md`, and `CODEX_CONTEXT.md`.

### Temporal sampling ablation and overfit result review

- Reviewed temporal sampling prediction summary and training overfit summary.
- Main prediction result: `middle_crop` achieved the best overall temporal result with MAE about `8.80`, RMSE about `10.73`, Pearson about `0.42`, and CCC about `0.38`.
- `middle_crop` reduced severe underestimation but increased Freeform/Northwind task inconsistency.
- `uniform_256`, `uniform_512`, and `uniform_1024` were nearly identical, so insufficient uniform frame count is unlikely to be the main bottleneck.
- `first_crop` worsened prediction compression; `random_crop` increased prediction variance but was unstable.
- All temporal runs were still `overfit_after_best_val=True`; sampling replacement did not fix late training memorization.

### OpenFace boundary hard-transition smoothing plan

- Added a focused plan for boundary hard-transition smoothing.
- Priority variants: `edge_soften_only` and `border_blur_fill`.
- Goal: distinguish whether the RGB model is sensitive to black padding area itself or to the high-contrast transition between OpenFace black fill and face pixels.
- This remains an input artifact submechanism, not a replacement for identity, geometry, task context, or calibration analyses.

### Identity retrieval multi-run result review

- Reviewed identity retrieval outputs for `rgb`, `center_mask`, `center_mask_black_to_gray`, `border_black_feather`, and `middle_crop` on test and val splits.
- `rgb_test` showed strong identity retrieval: same-subject top-1 about `0.66`, top-5 about `0.85`, paired-task median rank `1`.
- `border_black_feather_test` had even stronger identity retrieval, with same-subject top-1 about `0.75` and top-5 about `0.90`, so artifact smoothing should not be interpreted as de-identification.
- `middle_crop_test` reduced same-subject top-1 to about `0.49`, but this coincided with worse task consistency; it likely reflects temporal/task-context confounding rather than a clean severity representation.
- No current variant simultaneously lowers identity retrieval, increases severity-neighbor agreement, improves BDI metrics, and preserves task consistency.
- Added follow-up task to build `scripts/summarize_identity_retrieval_runs.py` and a diagnostics module for multi-run identity retrieval summaries.

### RGB baseline severity calibration result review

- Reviewed `logs/severity_calibration` for the RGB baseline.
- Validation-only calibration fitted `pred_calibrated = 0.870402 * pred + 2.689282` on 100 validation records.
- Test original metrics: MAE `8.9145`, RMSE `10.9530`, Pearson `0.3526`, CCC `0.2925`, true std `11.48`, pred std `6.13`.
- Test calibrated metrics: MAE `8.8460`, RMSE `10.8278`, Pearson `0.3526`, CCC `0.2692`, pred std `5.33`.
- Severe underestimation remained almost unchanged: residual `-16.50 -> -16.10`; minimal overestimation worsened: `+6.89 -> +8.04`.
- Interpretation: RGB baseline has weak ranking signal but strong prediction range compression. Simple post-hoc linear calibration is not a solution; next work should run multi-run severity calibration summary and then test severity-aware training methods with identity/task-consistency checks.
### RGB input / identity / calibration joint summary review

- Reviewed `rgb_input_ablation_summary`, `identity_retrieval_summary`, and `severity_calibration_summary` together for `rgb`, `center_mask`, `border_black_feather`, `middle_crop`, and `center_mask_black_to_gray`.
- `center_mask_black_to_gray` had the best MAE (`7.726`) but the worst severe bias (`-17.46`), showing that overall MAE can be dominated by minimal/mild improvements.
- `center_mask` had the best CCC (`0.477`), largest prediction std (`8.132`), strong moderate-bias improvement, and near-baseline task consistency, making it the most stable input artifact mitigation evidence.
- `border_black_feather` reduced severe bias most (`-12.73`) but had the strongest identity retrieval (`same_top1=0.75`, `same_top5=0.90`), so boundary smoothing is not de-identification.
- `middle_crop` reduced identity retrieval (`same_top1=0.49`) but worsened task consistency and severity agreement, supporting temporal/task-context confounding.
- All linear calibration variants reduced CCC and compressed prediction variance, so post-hoc calibration remains diagnostic only.
### Revised feasible roadmap: identity suppression x boundary smoothing

- Reorganized the next research stage around a 2x2 mechanism ablation rather than more isolated RGB variants.
- The proposed factors are identity/static appearance suppression and OpenFace boundary hard-transition smoothing.
- Priority variants: `edge_soften_only`, `border_blur_fill`, `identity_texture_suppressed`, and `identity_texture_suppressed_edge_soften`.
- The goal is to determine whether identity shortcut and boundary artifact are independent mechanisms or coupled effects.
- Severity-aware training is deferred until this mechanism split is evaluated with prediction, identity retrieval, and severity calibration summaries.
### Identity suppression literature mapping

- Reviewed identity suppression, domain-adversarial learning, disentanglement, shortcut learning, and facial behavior representation methods.
- Mapped them to the current OpenFace aligned face setting.
- Current recommendation: prioritize input-level `identity_texture_suppressed` and boundary smoothing 2x2 ablation before subject-adversarial GRL or full disentanglement.
- Added explicit criteria for whether a method moves the model back toward depression-relevant behavior: identity retrieval should not rise, severity agreement and CCC should not drop, prediction std should not compress further, severe bias should improve, and task consistency should not degrade.

### Shortcut-Regularized MTL planning update

- Reframed the next project stage from additional input-filter expansion to a two-mechanism Shortcut-Regularized MTL route.
- Current formal mechanisms:
  - subject-level shortcut: handled by an identity-adversarial MTL branch with Gradient Reversal Layer;
  - severity-level label imbalance: handled by severity-balanced regression loss.
- Clarified that severity-balanced loss is intended to mitigate score-bin imbalance, not to directly maximize prediction variance.
- Moved dynamic facial-change features to Stage C as a later candidate direction, not part of the current experiment plan.
- Updated the planned experimental sequence:
  - Stage A closure: layer-wise identity probe and prediction error x identity similarity coupling;
  - Stage B intervention: E0 baseline, E1 severity-balanced regression, E2 identity-adversarial MTL, E3 combined;
  - Stage C deferred: feature delta / AU delta / landmark-pose-gaze delta / static-dynamic fusion.
- Updated `CURRENT_STATUS.md`, `RGB_OVERFITTING_AUDIT_PLAN.md`, `OVERFITTING_MECHANISM_ROADMAP.md`, `TODO.md`, and `CODEX_CONTEXT.md` accordingly.
