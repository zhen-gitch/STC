# AGENTS.md

## Project context

This repository is for a depression assessment research project using AVEC2014-style visual or multimodal data.

The main goals are:

- maintain reproducible experiments
- support research paper writing
- keep training, evaluation, and ablation code stable
- avoid accidental data leakage
- avoid uncontrolled refactoring

## Coding rules

- Prefer minimal, localized changes.
- Do not rewrite unrelated modules.
- Do not rename public files, classes, functions, or config keys unless explicitly requested.
- Do not change dataset paths, split files, or label formats unless explicitly requested.
- Preserve backward compatibility with existing configs.
- Avoid introducing new dependencies unless necessary.
- Do not commit datasets, checkpoints, logs, or private credentials.

## Deep learning rules

When reviewing or modifying training code, pay special attention to:

- tensor shapes
- train/val/test split leakage
- loss and metric mismatch
- frozen gradients
- accidental detach
- constant predictions
- wrong sigmoid or softmax dimension
- mixed precision instability
- DDP unused parameters
- logging errors
- validation/test contamination

## Experiment rules

Every experiment must be reproducible from:

- git commit hash
- branch name
- config file
- random seed
- dataset split file
- command line
- GPU/device
- precision setting
- metrics output path

## Validation rules

Before claiming completion, provide:

- root cause or design explanation
- changed files
- exact diff summary
- validation commands
- remaining risks

For code changes, prefer running or proposing:

- import check
- config loading check
- one-batch smoke test
- relevant pytest test
- short training command using configs/debug_smoke.yaml