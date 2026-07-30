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

## Mandatory pre-implementation disclosure and authorization

Before creating or modifying code, configs, schemas, scripts, tests, build/dependency files, executable notebooks, generators, or executable pipelines, the agent must first send the user a standalone implementation disclosure with a package ID/version, end that turn, and wait for explicit authorization of that same package. The initial request to "implement", "start", or "execute the next step" does not waive the first disclosure. The same gate applies before data materialization/extraction, smoke or full training/evaluation, detached/background jobs, dependency installation, commit, or push.

The disclosure must state, concretely:

- the requested authorization boundary, included work, excluded work, and non-goals
- the current architecture and proposed architecture, including ownership boundaries and data flow
- model inputs/outputs, tensor shapes, masks, losses, gradient paths, and train/val/test behavior when relevant
- preprocessing, dataset, sampler, model, trainer, evaluator, and diagnostic responsibilities
- files/modules/config keys/schemas/public interfaces to add or change, with backward-compatibility behavior
- dataset root, split, labels, manifests, hashes, and leakage controls that remain unchanged
- exact commands to run, whether they write or launch background work, and their expected duration, device, precision, storage, and output paths
- validation plan, acceptance criteria, stop conditions, rollback path, and remaining risks or unknowns
- whether sub-agents will be used and the exact read/write/run boundary delegated to them

Authorization is package-specific and non-transitive. Design/docs, code changes, data generation, smoke/debug runs, full experiments, commit, and push are separate boundaries unless the user explicitly approves them together. A request such as "start" or "execute" authorizes work only when it clearly refers to a previously disclosed package. Sub-agents may not silently expand the parent agent's authorization or perform writes/runs outside that package.

If architecture, files, data usage, commands, compute cost, outputs, or risks materially change after authorization, stop before the new work, report the delta, and obtain renewed authorization.

Before authorization, agents and sub-agents are limited to read-only discovery: reading code/docs/logs/artifacts, inspecting git status/diff, and non-mutating static checks. They must not apply patches, run formatters with fixes, generate or update fixtures/snapshots, install dependencies, launch training/extraction, or create commits. A user-requested docs-only synchronization is allowed in the same turn only after it is explicitly identified as docs-only; it must not add executable configs or imply that code, data, or experiments were authorized.

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