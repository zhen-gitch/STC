# CODEX_CONTEXT.md

This file is the durable working context for Codex-assisted maintenance of this
repository. Read it before making changes, then update `CURRENT_STATUS.md` and
`BUG_LOG.md` when the project state changes.

## Project Purpose

This repository supports a research paper on depression assessment using
AVEC2014-style visual data.

The intended experiment target is BDI/depression score prediction, primarily as
a regression task, with auxiliary classification, contrastive, and visualization
components in the current model code.

The main engineering goal is reproducibility: future experiments should be
traceable from a git commit, branch, config, seed, split file, command line,
device, precision setting, log directory, and metric output.

## Repository Location

The active project checkout is:

`C:\CodeXWorkSpace\PaperWork\STC`

The Codex default shell may open in another directory, so always confirm the
working directory before running commands.

## Current Project Shape

- `AGENTS.md`: project rules for Codex and future agents.
- `README.md`: currently minimal; not a reliable source of setup instructions.
- `configs/`: YAML experiment configs.
- `scripts/`: executable entry scripts.
- `src/models/`: backbone factory, end-to-end model, temporal/multi-task blocks,
  task heads, and iResNet implementation.
- `src/trainers/`: PyTorch Lightning runner for end-to-end training.
- `src/losses/`: loss functions.
- `src/metrics/`: metrics.
- `src/utils/`: visualization, PCGrad, label distribution, decomposition, and
  adaptive mask helpers.
- `docs/`: project status, Codex context, bug log, experiment log, and TODO list.

## Main Pipeline

- Input: face image sequences.
- Output: BDI/depression score prediction.
- Framework: PyTorch and PyTorch Lightning.
- Configuration: OmegaConf YAML files under `configs/`.
- Expected metrics: MAE, RMSE, CCC, and task-specific diagnostics.
- Logging target: CSVLogger and TensorBoardLogger under `LOG_DIR`.Another file is console output log.

## Data Flow
face image sequences -> backbone -> project -> time series net -> cgc ->depression regression tower/ depression classification tower/ contrastive learning tower

## Main Entry Points

### Training

`scripts/train.py`

Current behavior:

- Imports `CONFIG_DIR` from `src.paths`.
- Loads `configs/default_config.yaml`.
- Seeds PyTorch Lightning with `42`.
- Requires `MODE: "full"`.
- Calls `src.trainers.end_to_end_runner.run_end2end(cfgs)`.

Important: the currently checked-in tree does not contain `src/paths.py`, so this
entry point cannot be assumed runnable until that module is restored or replaced.

### End-To-End Runner

`src/trainers/end_to_end_runner.py`

Current behavior:

- Imports `AVECDataModule` from `src.datasets.dataset`.
- Builds `EndToEndDepressionModel`.
- Builds a Lightning `Trainer` with accelerator, devices, precision, max epochs,
  CSVLogger, TensorBoardLogger, RichProgressBar, and LearningRateMonitor.
- Mirrors stdout/stderr into the CSVLogger version directory.
- Runs `trainer.fit(...)`.
- Loads segmented best weights from the run's `weights` directory when possible.
- Runs `trainer.test(...)`.
- Plots training curves from CSV logs.

Important: the currently checked-in tree does not contain `src/datasets/`, so the
runner cannot be assumed importable until the dataset module is restored.

### Diagnostics / Attribution

`scripts/diagnose.py`

Current behavior:

- Accepts GPU, project path, logger version, sample count, prediction CSV, and
  sample selection strategy.
- Loads `configs/default_config.yaml` from the provided project path.
- Forces batch size to 1 and `MODE: "full"`.
- Loads validation data and a trained model checkpoint.
- Generates GradCAM or ViT attention heatmaps plus occlusion sensitivity maps.

Important: this script also depends on `src.datasets.dataset`, so it shares the
same missing-module risk as training.

## Configuration Organization

### Existing Files

- `configs/default_config.yaml`
  - The only config currently loaded by `scripts/train.py`.
  - Contains absolute dataset and log paths.
  - Contains hardware settings, precision, model choice, feature extraction,
    temporal/multi-task settings, and visualization intervals.

- `configs/avec2014_base.yaml`
  - A cleaner base-style experiment config.
  - Contains hardware, model, training, loss, mask, and visualization settings.
  - Does not currently include dataset or log paths.

- `configs/debug_smoke.yaml`
  - A small override-style config for smoke testing.
  - Reduces batch size, chunk size, sequence length, epochs, and visualization.
  - It exists, but no checked-in entry point currently merges it with a base
    config.

### Desired Direction

Keep machine-specific values out of committed experiment logic. Prefer:

1. A shared base config, such as `configs/avec2014_base.yaml`.
2. An ignored local path config, such as `configs/local_paths.yaml`.
3. Optional overrides, such as `configs/debug_smoke.yaml`.
4. A clearly logged merged config for every experiment run.

Until this is implemented, avoid changing dataset paths, split files, or label
formats unless the user explicitly asks.

## Reproducibility Requirements

Every completed experiment should record:

- git branch and commit hash
- exact config file or merged config snapshot
- random seed
- dataset split file
- command line
- GPU/device selection
- precision setting
- log directory
- metrics output path
- checkpoint or weight path used for evaluation

## Codex Working Rules

- Make minimal, localized changes.
- Do not rename public files, classes, functions, or config keys unless asked.
- Do not modify dataset split logic unless explicitly requested.
- Do not change label normalization unless explicitly requested.
- Do not hard-code new absolute dataset paths in Python code.
- Do not commit datasets, checkpoints, logs, private credentials, or machine-only
  configs.
- Before editing training code, inspect tensor shapes, loss/metric alignment,
  gradients, detach points, sigmoid/softmax dimensions, DDP behavior, precision,
  and validation/test separation.
- Before claiming a code task is complete, provide changed files, diff summary,
  validation commands, and remaining risks.

## Known Recurring Model Risks

- `bdi_pred` may collapse to a constant value.
- regression head gradients may become zero.
- softmax may be called without an explicit `dim`.
- DDP may report unused parameters.
- mixed precision may cause instability.
- validation/test contamination can invalidate reported results.
- loss terms and reported metrics may optimize different targets.
- visualization or diagnostic code may accidentally run on the wrong checkpoint.

## Current Blocking Risks

See `docs/BUG_LOG.md` for the actionable list. The most important current risks
are:

- `scripts/train.py` imports missing `src.paths`.
- training, diagnostics, and the model import missing `src.datasets.dataset`.
- `debug_smoke.yaml` exists but is not wired into an executable smoke-test path.
- committed default config contains absolute local dataset/log paths.

## Recommended Next Work Order

1. Restore or implement the missing path and dataset modules with minimal changes.
2. Add an import/config loading check.
3. Wire a one-batch smoke test using `configs/debug_smoke.yaml`.
4. Add model forward and backward tests, including a non-zero regression-head
   gradient check.
5. Standardize experiment logging and merged-config snapshots.
6. Add metrics export for paper tables.
