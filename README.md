# STC Research

This repository supports reproducible depression assessment experiments on
AVEC2014-style aligned facial video and multimodal data.

## Current Route

The current research route is documented in:

1. `docs/DOCS_GUIDE.md` - document navigation and source-of-truth rules.
2. `docs/TODO.md` - current executable task queue.
3. `docs/CURRENT_STATUS.md` - current status snapshot.
4. `docs/RGB_OVERFITTING_AUDIT_PLAN.md` - main shortcut/overfitting research plan.
5. `docs/OVERFITTING_MECHANISM_ROADMAP.md` - mechanism map and decision logic.
6. `docs/STAGE_C_RUNBOOK.md` - frozen Stage C implementation and evaluation protocol.

As of 2026-07-10, the project objective is **Auditable and Falsifiable
Coarse-Grained Task-Nuisance Information Separation**. The `z_dep` / `z_nuisance`
split is treated as a testable structural hypothesis, not as proof of semantic
disentanglement. Historical RPDF-Net notes are retained as background and
should not override the current route.

Stage A/Stage B, C0 specification, and the P0 seed/EarlyStopping policy are
complete. The C1 minimal task-nuisance block, auxiliary losses, and multi-
representation export are implemented locally. Server debug smoke and the
train-only weight calibration remain before C1 can close and C2 can start.

## Main Entry Points

- MTL-Lite training: `scripts/train_mtl_lite.py`
- MTL-Lite diagnostics: `scripts/diagnose_mtl_lite.py`
- Behavior baseline: `scripts/train_behavior_baseline.py`
- Common configs: `configs/avec2014_base.yaml` plus local paths and overrides

Machine-specific paths belong in `configs/local_paths.yaml`, which is ignored
by git. Do not commit datasets, checkpoints, logs, weights, or private paths.
