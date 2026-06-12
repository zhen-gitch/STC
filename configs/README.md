# Configs

Recommended merge order for new experiments:

1. `avec2014_base.yaml`
2. `local_paths.yaml`
3. experiment or debug overrides, such as `debug_smoke.yaml`

`local_paths.yaml` stores machine-specific dataset and log paths. It is ignored
by git and should be created from `local_paths.example.yaml` on each machine.

`pre/default_config.yaml` is retained as a historical complete config for
compatibility. Prefer the base + local paths + override layout for new runs.

MTL-Lite mainline overrides:

- `regression_only_baseline.yaml`: BDI regression only.
- `mtl_lite_baseline.yaml`: BDI regression plus ordinal severity classification.
- `mtl_lite_debug_smoke.yaml`: short MTL-Lite smoke run.

Run the new mainline with:

```bash
python scripts/train_mtl_lite.py --override configs/mtl_lite_debug_smoke.yaml
```
