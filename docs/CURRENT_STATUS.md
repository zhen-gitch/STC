# CURRENT_STATUS.md

## Status Date

2026-06-12

## Active Goal

Build a reproducible Codex-assisted workflow for paper experiments and code
maintenance, with a near-term focus on making training importable, smoke-testable,
and safe to use for paper experiments.

## Active Checkout

- Project path: `C:\CodeXWorkSpace\PaperWork\STC`
- Current branch: `dev`
- Upstream: `origin/dev`
- Current short commit: `97efceb`
- Working tree at last check: clean

## Project Summary

This is a PyTorch / PyTorch Lightning research codebase for AVEC2014-style
depression assessment. The current intended mode is end-to-end visual training
with BDI score regression, auxiliary task heads, temporal/multi-task blocks,
diagnostic visualization, CSV logging, and TensorBoard logging.

## Current Structure Snapshot

- `scripts/train.py`: main training entry, currently loads
  `configs/default_config.yaml`.
- `scripts/diagnose.py`: attribution/diagnostic entry.
- `src/trainers/end_to_end_runner.py`: Lightning train/test runner.
- `src/models/end_to_end.py`: main model.
- `src/models/backbone_factory.py`: backbone construction.
- `src/models/mtl_blocks.py`: temporal and multi-task blocks.
- `src/models/task_heads.py`: task heads.
- `src/losses/losses.py`: losses.
- `src/metrics/metrics.py`: metrics.
- `src/utils/`: PCGrad, visualization, adaptive mask, label distribution, and
  decomposition utilities.
- `configs/default_config.yaml`: current direct training config.
- `configs/avec2014_base.yaml`: cleaner base-style config.
- `configs/debug_smoke.yaml`: smoke-test override config.

## Current Priority

1. Restore importability of the training pipeline.
2. Confirm or rebuild dataset/path modules without changing split semantics.
3. Add a minimal import and config-loading check.
4. Wire a one-batch smoke test using `configs/debug_smoke.yaml`.
5. Add focused forward/backward tests for the model and regression head.
6. Standardize experiment logging, including merged config snapshots.
7. Export metrics for paper tables.

## Last Read-Only Audit

Read-only checks performed:

- Read `AGENTS.md`, `README.md`, `docs/CODEX_CONTEXT.md`,
  `docs/CURRENT_STATUS.md`, and `docs/BUG_LOG.md`.
- Checked git status and branch.
- Listed tracked project files.
- Searched for missing-module imports and config-loading paths.

Observed state:

- Branch is `dev`.
- `dev` tracks `origin/dev`.
- Working tree was clean before these documentation edits.
- `src/paths.py` is not present.
- `src/datasets/dataset.py` is not present.
- `scripts/train.py` directly loads `configs/default_config.yaml`.
- No checked-in entry point currently merges `avec2014_base.yaml`,
  `local_paths.yaml`, and `debug_smoke.yaml`.

## Last Verified Runtime Command

No training, import, test, or smoke command has been verified yet in the current
checkout.

## Immediate Risks

- Main training entry is likely blocked by missing `src.paths`.
- Training, diagnostics, and model import are likely blocked by missing
  `src.datasets.dataset`.
- `configs/default_config.yaml` contains machine-specific absolute paths.
- `configs/debug_smoke.yaml` exists but is not currently wired into a reproducible
  smoke-test command.
- `docs/TODO.md` still lists `Add configs/debug_smoke.yaml` even though the file
  now exists.
- Datasets, checkpoints, logs, and local credentials must remain uncommitted.

## Next Recommended Command Sequence

After the missing modules are restored or implemented, validate in this order:

1. `python -m compileall scripts src`
2. Import/config loading check for the selected config path.
3. One-batch smoke training with a merged debug config.
4. Focused model forward test.
5. Backward test confirming non-zero regression-head gradients.

Do not run full training until importability and smoke checks pass.
