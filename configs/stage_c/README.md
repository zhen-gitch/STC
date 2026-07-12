# Stage C Experiment Configs

Stage C uses the authoritative protocol in `docs/STAGE_C_RUNBOOK.md`. The
override order is fixed:

```text
common.yaml -> experiment yaml -> seed yaml -> optional debug_smoke.yaml
```

`c_ref_e2.yaml` is the only runnable Stage C experiment before C1 is
implemented. It reproduces the Stage B E2 structure under the new configured
seed and EarlyStopping policy.

`c_bn_bottleneck.yaml`, `c_rec_split_recon.yaml`, and
`c_full_split_full.yaml` intentionally set `MODE: stage_c_spec_only`. The
current model does not implement `MODEL.TASK_NUISANCE`; the sentinel makes the
training entry fail before an unimplemented candidate can be silently run as
E2. Remove the sentinel only after C1 config validation, forward, loss, and
backward tests pass.

`calibration_train_only.yaml` is also spec-only until C1. Once enabled, it
computes and logs both raw auxiliary losses while keeping their contribution to
the optimization objective at zero. The first 100 train batches determine the
single frozen weight pair; validation and test remain closed.

The reconstruction and cross-correlation weights in the split candidates are
`null` until the seed-42 train-only calibration freezes one value from
`0.001 / 0.01`. Validation and test metrics must not be used for this choice.

Reference run:

```bash
python scripts/train_mtl_lite.py \
  --override configs/stage_c/common.yaml \
  --override configs/stage_c/c_ref_e2.yaml \
  --override configs/stage_c/seeds/seed_42.yaml
```

Reference smoke:

```bash
python scripts/train_mtl_lite.py \
  --override configs/stage_c/common.yaml \
  --override configs/stage_c/c_ref_e2.yaml \
  --override configs/stage_c/seeds/seed_42.yaml \
  --override configs/stage_c/debug_smoke.yaml
```
