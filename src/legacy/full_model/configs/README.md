# Configs

Recommended merge order for new experiments:

1. `avec2014_base.yaml`
2. `local_paths.yaml`
3. experiment or debug overrides, such as `debug_smoke.yaml`

`local_paths.yaml` stores machine-specific dataset and log paths. It is ignored
by git and should be created from `local_paths.example.yaml` on each machine.

`pre/default_config.yaml` is retained as a historical complete config for
compatibility. Prefer the base + local paths + override layout for new runs.
