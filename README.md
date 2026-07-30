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
6. `docs/GLOBAL_LOCAL_AU_EXPERIMENT_PLAN.md` - current global/local + AU architecture and hybrid ablation authority.
7. `docs/PRIVILEGED_BEHAVIOR_ALIGNMENT_PLAN.md` - PB source and P0 data-contract authority.
8. `docs/STAGE_C_RUNBOOK.md` - historical frozen Stage C implementation and evaluation protocol.

As of 2026-07-30, the project objective is **Auditable and Falsifiable
Coarse-Grained Task-Nuisance Information Separation**. The `z_dep` / `z_nuisance`
split is treated as a testable structural hypothesis, not as proof of semantic
disentanglement. Historical RPDF-Net notes are retained as background and
should not override the current route.

Stage C and its follow-up capacity/regularization analyses are closed after
negative utility and robustness results. The active intervention is a shared
backbone over one global face plus eye-brow, nose-cheek, and mouth-lower-face
views, with eligible AU/head signals used only as train-time auxiliary targets.
The experiment order is maximum eligible `GLA-FULL` first, dependency-aware
subtractive ablation, bounded additive confirmation, and paired multi-seed
evaluation. PB-P0A3 full-rich source coverage is complete; P0B-P0E, model code,
and training remain unauthorized.

Before any code, config, test, data-generation, run, commit, or push package,
follow the mandatory disclosure and explicit-authorization gate in `AGENTS.md`.

## Main Entry Points

- MTL-Lite training: `scripts/train_mtl_lite.py`
- MTL-Lite diagnostics: `scripts/diagnose_mtl_lite.py`
- Behavior baseline: `scripts/train_behavior_baseline.py`
- Common configs: `configs/avec2014_base.yaml` plus local paths and overrides

Machine-specific paths belong in `configs/local_paths.yaml`, which is ignored
by git. Do not commit datasets, checkpoints, logs, weights, or private paths.
