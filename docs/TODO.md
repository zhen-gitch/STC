# TODO.md

## High priority

- [ ] Verify new config loader with real project environment
- [ ] Create configs/local_paths.yaml from configs/local_paths.example.yaml on the target machine
- [ ] Run import checks for scripts.train and scripts.diagnose
- [ ] Decide whether configs/pre/default_config.yaml remains legacy-only or should be migrated into base/local/override configs
- [ ] Fix missing src.datasets.dataset imports before smoke training
- [ ] Audit remaining import dependencies after restoring src.datasets.dataset
- [ ] Decide whether src/utils/decomposition.py and src/utils/adaptive_mask.py should move to model/temporal modules
- [ ] Consider splitting src/utils/visualize.py into visualization and diagnostics modules
- [ ] Review redundant module names src/losses/losses.py and src/metrics/metrics.py
- [ ] Check stale path comment in src/models/mtl_blocks.py
- [ ] Verify timm backbone factory with deit_tiny_patch16_224
- [ ] Add BACKBONE_OUT_DIMS entries before testing new timm backbones
- [ ] Add a backbone factory import/create smoke test
- [ ] Document supported backbone naming and dimension requirements
- [ ] Add documented China mirror workflow for downloading timm/HuggingFace backbone weights
- [ ] Standardize local backbone weight format under weights/backbones/<model_name>.pth
- [ ] Verify MODEL_WEIGHT_PATH loading for raw state_dict and {"state_dict": ...} checkpoints
- [ ] Add test_model_forward.py
- [ ] Add test_loss_backward.py
- [ ] Standardize experiment log format
- [ ] Add export script for metrics.csv to LaTeX table

## Codex task queue

### Task 1

Add a one-batch smoke test config.

### Task 2

Add a model forward test.

### Task 3

Add a backward test to ensure regression head gradients are non-zero.

### Task 4

Add script to export experiment metrics to paper table.
