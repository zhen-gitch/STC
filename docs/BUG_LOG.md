# BUG_LOG.md

This log tracks known problems that should be fixed before relying on training
or experiment results. Keep entries concise, reproducible, and tied to files.

## Open Issues

### BUG-001: Training entry imports missing `src.paths`

- Status: open
- Severity: blocking
- Files:
  - `scripts/train.py`
- Evidence:
  - `scripts/train.py` imports `from src.paths import *`.
  - The current `src/` tree does not contain `paths.py`.
- Impact:
  - `scripts/train.py` cannot be assumed importable or runnable.
  - `CONFIG_DIR` is undefined unless provided by an untracked/local file.
- Recommended fix:
  - Restore or implement `src/paths.py` with minimal project-root/config-root
    helpers.
  - Preserve backward compatibility for `CONFIG_DIR`.
  - Avoid hard-coding machine-specific dataset paths in Python.

### BUG-002: Dataset module is missing from the checked-in tree

- Status: open
- Severity: blocking
- Files:
  - `scripts/diagnose.py`
  - `src/trainers/end_to_end_runner.py`
  - `src/models/end_to_end.py`
- Evidence:
  - These files import `src.datasets.dataset`.
  - The current `src/` tree does not contain a `datasets/` package.
- Impact:
  - Training, diagnostics, and parts of the model are likely not importable.
  - Smoke tests cannot run until `AVECDataModule` and
    `generate_soft_spatial_mask` are available.
- Recommended fix:
  - Restore the dataset package or implement the minimal missing interfaces.
  - Do not change split logic, label normalization, label formats, or dataset
    path semantics unless explicitly requested.
  - Add an import test before attempting training.

### BUG-003: Smoke-test config exists but is not wired into an entry point

- Status: open
- Severity: high
- Files:
  - `configs/debug_smoke.yaml`
  - `scripts/train.py`
  - `configs/avec2014_base.yaml`
  - `configs/default_config.yaml`
- Evidence:
  - `configs/debug_smoke.yaml` exists.
  - `scripts/train.py` directly loads `configs/default_config.yaml`.
  - No checked-in script currently merges base, local paths, and debug override
    configs.
- Impact:
  - The project cannot yet provide a reproducible one-batch smoke command.
  - Future changes are harder to validate safely.
- Recommended fix:
  - Add a config loading path that supports explicit config selection or ordered
    OmegaConf merges.
  - Keep `default_config.yaml` behavior backward compatible.
  - Record the resolved config in each run directory.

### BUG-004: Committed default config contains absolute local paths

- Status: open
- Severity: medium
- Files:
  - `configs/default_config.yaml`
- Evidence:
  - `IMAGE_DIR`, `LABEL_DIR`, `DATASET_SPLIT_FILE`, and `LOG_DIR` are absolute
    local Linux-style paths.
- Impact:
  - Experiments may fail across machines.
  - Machine-specific paths are harder to keep reproducible and private.
- Recommended fix:
  - Prefer ignored `configs/local_paths.yaml` or environment-variable expansion
    for local paths.
  - Do not change dataset paths silently; migrate with a compatibility path.

### BUG-005: TODO list is stale for `debug_smoke.yaml`

- Status: open
- Severity: low
- Files:
  - `docs/TODO.md`
  - `configs/debug_smoke.yaml`
- Evidence:
  - `docs/TODO.md` still lists `Add configs/debug_smoke.yaml`.
  - `configs/debug_smoke.yaml` already exists.
- Impact:
  - Task tracking may mislead future work.
- Recommended fix:
  - Update `docs/TODO.md` after the user allows edits outside the currently
    requested documentation files.

## Verification Needed After Fixes

After BUG-001 and BUG-002 are fixed, run checks in this order:

1. `python -m compileall scripts src`
2. Import check for `scripts.train`, `src.trainers.end_to_end_runner`, and
   `src.models.end_to_end`.
3. Config loading check for the default and debug configurations.
4. One-batch smoke training using the debug config.
5. Model forward and backward tests, including non-zero regression-head
   gradients.
