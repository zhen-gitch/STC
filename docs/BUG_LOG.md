# BUG_LOG.md

This log tracks known problems that should be fixed before relying on training
or experiment results. Keep entries concise, reproducible, and tied to files.

## Open Issues

### RISK-001: Behavior-only baseline strongly overfits full OpenFace feature set

- Status: open
- Severity: high
- Files:
  - `src/datasets/openface_features.py`
  - `src/models/behavior_baseline.py`
  - `configs/behavior_baseline.yaml`
- Evidence:
  - Latest behavior-only run reports test MAE about `9.93`, RMSE about `12.86`,
    and CCC about `0.151`.
  - Best validation RMSE is about `12.38`, while the corresponding train RMSE
    is about `2.74`.
  - Train CCC reaches about `0.975`, but test CCC remains about `0.151`.
- Impact:
  - The full OpenFace feature set is not yet a reliable behavior representation
    for downstream fusion.
  - Raw landmark coordinates, static facial geometry, quality variables, or
    subject-specific acquisition conditions may be acting as shortcuts.
  - Direct RGB + behavior late fusion may amplify overfitting if performed
    before feature-group ablation.
- Recommended fix:
  - Export behavior val/test predictions with the same schema as RGB/MTL-Lite
    predictions.
  - Run feature-group ablations: quality-only, AU-only, pose+gaze-only,
    raw-landmark-only, landmark-delta-only, AU+landmark-delta, and
    all-without-raw-landmarks.
  - Compare RGB and behavior predictions at case level before attempting late
    fusion or behavior auxiliary MTL.

### BUG-004: Legacy default config contains absolute local paths

- Status: open
- Severity: low
- Files:
  - `configs/pre/default_config.yaml`
- Evidence:
  - The historical complete config under `configs/pre/default_config.yaml`
    contains machine-specific absolute paths.
  - The canonical workflow now uses `configs/avec2014_base.yaml`,
    ignored `configs/local_paths.yaml`, and optional overrides.
- Impact:
  - Low risk for normal runs if users follow the canonical config stack.
  - Legacy `--config configs/pre/default_config.yaml` runs may still be
    machine-specific.
- Recommended fix:
  - Keep `configs/pre/default_config.yaml` documented as legacy-only, or
    migrate it into the base/local/override config organization.
  - Do not silently change dataset paths without an explicit migration step.

### BUG-006: Debug smoke device selection is machine-specific

- Status: open
- Severity: low
- Files:
  - `configs/debug_smoke.yaml`
- Evidence:
  - `configs/debug_smoke.yaml` sets `DEVICES: [4]`.
- Impact:
  - Debug smoke works on the current server setup but may fail on machines
    without GPU index 4.
- Recommended fix:
  - Keep server-specific device overrides local, or add separate
    machine-neutral smoke configs such as `debug_smoke_gpu0.yaml` or
    `debug_smoke_cpu.yaml`.

## Resolved Issues

### BUG-007: Shortcut Audit does not match `_aligned` prediction video IDs

- Status: resolved
- Severity: high
- Resolved on: 2026-06-13
- Files:
  - `src/diagnostics/shortcut_audit.py`
  - `tests/test_shortcut_audit.py`
  - `docs/SHORTCUT_AUDIT_DESIGN.md`
- Evidence:
  - Latest `test_predictions.csv` uses IDs such as
    `203_2_Freeform_video_aligned`.
  - Latest `openface_quality_summary.csv` uses IDs such as
    `203_2_Freeform_video`.
  - Before the fix, `shortcut_audit_report.md` reported `Matched samples: 0`.
  - `_normalize_video_id()` now strips the `_aligned` processing suffix before
    matching.
  - A real-log merge check matched 100/100 prediction samples against the
    OpenFace quality summary after the fix.
  - After rerunning Shortcut Audit, `shortcut_audit_report.md` reports
    `Matched samples: 100`, so the generated correlation and predictor tables
    are now valid for offline risk interpretation.

### BUG-001: Training entry imports missing `src.paths`

- Status: resolved
- Severity: blocking
- Resolved on: 2026-06-12
- Files:
  - `scripts/train.py`
- Evidence:
  - `scripts/train.py` now imports from `src.config` and no longer depends on
    `src.paths`.
  - Config loading supports base, local paths, and override YAML files.

### BUG-002: Dataset module is missing from the checked-in tree

- Status: resolved
- Severity: blocking
- Resolved on: 2026-06-12
- Files:
  - `src/datasets/dataset.py`
  - `scripts/diagnose.py`
  - `src/trainers/end_to_end_runner.py`
  - `src/models/end_to_end.py`
- Evidence:
  - `src/datasets/dataset.py` has been restored.
  - Debug smoke training can run end to end in the server environment.

### BUG-003: Smoke-test config exists but is not wired into an entry point

- Status: resolved
- Severity: high
- Resolved on: 2026-06-12
- Files:
  - `configs/debug_smoke.yaml`
  - `scripts/train.py`
  - `src/config.py`
- Evidence:
  - Training can be launched with `scripts/train.py --override
    configs/debug_smoke.yaml`.
  - The server environment has completed debug smoke training.

### BUG-005: TODO list is stale for `debug_smoke.yaml`

- Status: resolved
- Severity: low
- Resolved on: 2026-06-12
- Files:
  - `docs/TODO.md`
  - `configs/debug_smoke.yaml`
- Evidence:
  - Completed smoke-config tasks have been removed from `docs/TODO.md` and
    recorded in `docs/EXPERIMENT_LOG.md`.

## Verification Needed After Fixes

Use this order after future code or config refactors:

1. `python -m compileall scripts src`
2. Import checks for `scripts.train`, `scripts.diagnose`,
   `src.trainers.end_to_end_runner`, and `src.models.end_to_end`.
3. Config loading check for base, local paths, and debug override configs.
4. One-batch smoke training using the debug config on the target machine.
5. Model forward and backward tests, including non-zero regression-head
   gradients.
