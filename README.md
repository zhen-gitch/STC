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

As of 2026-07-06, the model direction is **Shortcut-aware
Task-Nuisance Disentangled Representation Learning**. Historical RPDF-Net
notes are retained as background and should not override the current route.

## Main Entry Points

- MTL-Lite training: `scripts/train_mtl_lite.py`
- MTL-Lite diagnostics: `scripts/diagnose_mtl_lite.py`
- Behavior baseline: `scripts/train_behavior_baseline.py`
- Common configs: `configs/avec2014_base.yaml` plus local paths and overrides

Machine-specific paths belong in `configs/local_paths.yaml`, which is ignored
by git. Do not commit datasets, checkpoints, logs, weights, or private paths.
