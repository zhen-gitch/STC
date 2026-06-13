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
