# Validation/Test Role-Swap Audit

This is an exploratory split-sensitivity experiment on the current P0/E2
protocol. It compares the normal logical roles with a condition that exchanges
the physical validation and test sources while keeping the training split,
seed, backbone, optimizer, loss, precision, and early-stopping policy fixed.

| Logical role | Reference | Swapped condition |
|---|---|---|
| `train` | original `train` | original `train` |
| `val` (checkpoint selection) | original `val` | original `test` |
| `test` (post-fit evaluation) | original `test` | original `val` |

The raw split JSON is not modified. `DATASET.SWAP_VAL_TEST` is a strict
boolean consumed by the MTL-Lite `AVECDataModule`; the runner still selects
the best checkpoint from logical `val` and evaluates logical `test` only after
fit. This switch is intended for RGB/MTL-Lite experiments. The OpenFace
behavior baseline has a separate split reader and is not covered by this
configuration.

Run the paired conditions in this order:

```bash
git branch --show-current
git rev-parse HEAD

python scripts/train_mtl_lite.py \
  --override configs/photometric_normalization/common.yaml \
  --override configs/photometric_normalization/p0_rgb.yaml \
  --override configs/split_sensitivity/reference.yaml

python scripts/train_mtl_lite.py \
  --override configs/photometric_normalization/common.yaml \
  --override configs/photometric_normalization/p0_rgb.yaml \
  --override configs/split_sensitivity/val_test_swapped.yaml
```

For a short smoke, append `configs/split_sensitivity/debug_smoke.yaml` to
either command. The resolved config and console output record the role mapping;
prediction files named `val`/`test` always refer to logical roles, not the
original physical split names. Outputs are written below
`<LOG_DIR>/split_sensitivity/e2_p0_original_roles/` and
`<LOG_DIR>/split_sensitivity/e2_p0_val_test_swapped/`.

Report both directions with MAE, RMSE, Pearson, CCC, prediction standard
deviation, best epoch, best-to-last validation degradation, severity-group
bias, task consistency, and identity probes. Keep the raw split-integrity
report labeled by physical split.

This is not a new unbiased test: the original test is used for checkpoint
selection in the swapped condition, and the original validation set has
already influenced earlier project decisions. Name the result
`split-sensitivity` or `role-swap robustness audit`, not a final test-set
generalization claim. Confirm the direction with at least three seeds before
using it in a paper.
