# TODO.md

## Near-term validation

- [ ] Add `tests/test_model_forward.py`
- [ ] Add `tests/test_loss_backward.py` to confirm `reg_task_head` gradients are non-zero
- [ ] Add a regression-head initialization test to ensure no train-label statistics are used
- [ ] Audit `loss_dist` batch mean/std regularizer against MAE/RMSE/CCC reporting
- [ ] Verify `validation_step` forward can run under `no_grad` without changing metrics
- [ ] Confirm LDS label weighting is training-only and never touches val/test labels
- [ ] Verify `MODEL_WEIGHT_PATH` loading for raw `state_dict` and `{"state_dict": ...}` checkpoints

## Config and backbone workflow

- [ ] Decide whether configs/pre/default_config.yaml remains legacy-only or should be migrated into base/local/override configs
- [ ] Verify timm backbone factory with deit_tiny_patch16_224
- [ ] Add BACKBONE_OUT_DIMS entries before testing new timm backbones
- [ ] Add a backbone factory import/create smoke test
- [ ] Document supported backbone naming and dimension requirements
- [ ] Add documented China mirror workflow for downloading timm/HuggingFace backbone weights
- [ ] Standardize local backbone weight format under weights/backbones/<model_name>.pth

## Distributed and precision checks

- [ ] Verify DDP metric logging and best-weight saving on multi-GPU
- [ ] Add test for finite predictions/loss under bf16-mixed precision

## Later structure cleanup

- [ ] Decide whether src/utils/decomposition.py and src/utils/adaptive_mask.py should move to model/temporal modules
- [ ] Consider splitting src/utils/visualize.py into visualization and diagnostics modules
- [ ] Review redundant module names src/losses/losses.py and src/metrics/metrics.py
- [ ] Check stale path comment in src/models/mtl_blocks.py

## Paper utilities

- [ ] Standardize experiment log format
- [ ] Add export script for metrics.csv to LaTeX table

## Codex task queue

### Task 1

Add a model forward test.

### Task 2

Add a backward test to ensure regression head gradients are non-zero.

### Task 3

Add regression-head initialization and LDS isolation tests.

### Task 4

Add script to export experiment metrics to paper table.
