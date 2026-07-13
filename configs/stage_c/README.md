# Stage C Experiment Configs

Stage C uses the authoritative protocol in `docs/STAGE_C_RUNBOOK.md`. The
override order is fixed:

```text
common.yaml -> experiment yaml -> seed yaml -> optional debug_smoke.yaml
```

`c_ref_e2.yaml` reproduces the Stage B E2 structure under the configured seed
and EarlyStopping policy. `c_bn_bottleneck.yaml` is the runnable prediction-
bottleneck control.

`c_rec_split_recon.yaml` uses the frozen reconstruction weight `0.001`.
`c_full_split_full.yaml` uses reconstruction `0.001` and cross-correlation
`0.01`. Both are runnable, but the seed-42 utility screen failed relative to
`C-REF`; do not start C3 seeds without a newly preregistered route.

`calibration_train_only.yaml` computes and logs both raw auxiliary losses while
keeping their contribution to the optimization objective at zero. The runner
stops at `CALIBRATION_STEPS=100`, disables sanity/validation, and skips test.
Those train batches determine the single frozen weight pair.

The weights were selected only from the seed-42 train-only calibration loss
scale. Validation/test utility was not used to choose them.

`common.yaml` sets `RUN_TEST_AFTER_FIT: False`. C2/C3 therefore finish after
validation and keep test closed. Only a frozen final-test protocol may
explicitly override this field to `True`.

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

## Postmortem Capacity Audit

The bounded C-BN capacity audit is separate from C2/C3. It runs the full 40
epochs with EarlyStopping disabled, still selects/logs validation checkpoints,
and keeps test closed. Fixed dimensions are `96/128/160/192`; do not add
intermediate points after viewing results.

```bash
python scripts/train_mtl_lite.py \
  --override configs/stage_c/common.yaml \
  --override configs/stage_c/c_bn_bottleneck.yaml \
  --override configs/stage_c/capacity_audit/common_full40.yaml \
  --override configs/stage_c/capacity_audit/dep_128.yaml \
  --override configs/stage_c/seeds/seed_42.yaml
```
