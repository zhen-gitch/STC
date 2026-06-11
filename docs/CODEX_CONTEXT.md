# CODEX_CONTEXT.md

## Project

This project supports a research paper on depression assessment using AVEC2014-style data.

## Main pipeline

- Input: face images or multimodal features
- Output: BDI/depression score regression
- Training framework: PyTorch / PyTorch Lightning
- Configuration: YAML files under configs/
- Metrics: MAE, RMSE, PCC or task-specific metrics

## Important constraints

- Do not change dataset split logic unless explicitly requested.
- Do not change label normalization unless explicitly requested.
- Do not modify absolute local dataset paths directly in code.
- Prefer environment variables or configs/local_paths.yaml for machine-specific paths.
- Every experiment should be reproducible from config + commit + seed.

## Known recurring issues

- bdi_pred may become constant.
- reg_head gradients may become zero.
- softmax without dim may appear.
- DDP may report unused parameters.
- mixed precision may cause instability.