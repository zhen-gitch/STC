# Stage C Experiment Configs

Stage C uses the authoritative protocol in `docs/STAGE_C_RUNBOOK.md`. The
override order is fixed:

```text
common.yaml -> experiment yaml -> seed yaml -> optional debug_smoke.yaml
```

`c_ref_e2.yaml` reproduces the Stage B E2 structure under the configured seed
and EarlyStopping policy. `c_bn_bottleneck.yaml` is the runnable prediction-
bottleneck control.

`c_rec_split_recon.yaml` and `c_full_split_full.yaml` intentionally retain
`MODE: stage_c_spec_only` while their calibrated weights are `null`. The
sentinel prevents either candidate from being run before the train-only
calibration freezes its weight values.

`calibration_train_only.yaml` computes and logs both raw auxiliary losses while
keeping their contribution to the optimization objective at zero. The runner
stops at `CALIBRATION_STEPS=100`, disables sanity/validation, and skips test.
Those train batches determine the single frozen weight pair.

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

Prediction-bottleneck smoke:

```bash
python scripts/train_mtl_lite.py \
  --override configs/stage_c/common.yaml \
  --override configs/stage_c/c_bn_bottleneck.yaml \
  --override configs/stage_c/seeds/seed_42.yaml \
  --override configs/stage_c/debug_smoke.yaml
```

Train-only calibration:

```bash
python scripts/train_mtl_lite.py \
  --override configs/stage_c/common.yaml \
  --override configs/stage_c/calibration_train_only.yaml \
  --override configs/stage_c/seeds/seed_42.yaml
```
