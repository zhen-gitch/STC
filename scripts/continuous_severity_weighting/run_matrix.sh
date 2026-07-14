#!/usr/bin/env bash
# Fixed continuous severity-density weighting audit.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

BASE="configs/stage_b/base_regression_only.yaml"
COMMON="configs/continuous_severity_weighting/common_full40.yaml"
RUN_NAMES=(continuous_alpha025 continuous_alpha05)
RUN_OVERRIDES=(
  "configs/continuous_severity_weighting/alpha025.yaml"
  "configs/continuous_severity_weighting/alpha05.yaml"
)

extra_overrides=()
if [[ "${DEBUG:-0}" == "1" ]]; then
  extra_overrides+=("configs/continuous_severity_weighting/debug_smoke.yaml")
fi

train_run() {
  local override="$1"
  local args=(--override "$BASE" --override "$COMMON" --override "$override")
  for extra in "${extra_overrides[@]}"; do
    args+=(--override "$extra")
  done
  python scripts/train_mtl_lite.py "${args[@]}"
}

resolve_run_dir() {
  local override="$1"
  python - "$override" "${extra_overrides[@]}" <<'PY'
import sys
from pathlib import Path

from src.config import load_experiment_config, resolve_experiment_save_dir

overrides = [
    "configs/stage_b/base_regression_only.yaml",
    "configs/continuous_severity_weighting/common_full40.yaml",
    sys.argv[1],
    *sys.argv[2:],
]
cfg = load_experiment_config(overrides=overrides)
root = resolve_experiment_save_dir(cfg)
versions = sorted(
    (path for path in root.glob("version_*") if path.is_dir()),
    key=lambda path: int(path.name.split("_")[-1]),
)
print(versions[-1] if versions else root)
PY
}

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  for override in "${RUN_OVERRIDES[@]}"; do
    train_run "$override"
  done
fi

declare -a run_dirs=()
for override in "${RUN_OVERRIDES[@]}"; do
  run_dir="$(resolve_run_dir "$override")"
  if [[ ! -f "$run_dir/metrics.csv" ]]; then
    echo "[ERROR] metrics.csv was not found: $run_dir" >&2
    exit 1
  fi
  run_dirs+=("$run_dir")
done

GROUP_DIR="$(dirname "$(dirname "${run_dirs[0]}")")"
ANALYSIS_DIR="$GROUP_DIR/analysis"
summary_args=()
for index in "${!RUN_NAMES[@]}"; do
  summary_args+=(--run "${RUN_NAMES[$index]}=${run_dirs[$index]}")
done
python scripts/summarize_training_overfit.py \
  "${summary_args[@]}" \
  --output-dir "$ANALYSIS_DIR/training_overfit"

if [[ "${SKIP_DIAG:-0}" == "1" ]]; then
  echo "[CONTINUOUS-SEVERITY] diagnostics skipped"
  exit 0
fi

for run_dir in "${run_dirs[@]}"; do
  python scripts/diagnose_mtl_lite.py \
    --run-dir "$run_dir" \
    --ckpt best \
    --split train val \
    --enable-training-curves \
    --enable-regression \
    --enable-embeddings
done

prediction_args=()
group_args=()
leakage_args=()
for index in "${!RUN_NAMES[@]}"; do
  run_name="${RUN_NAMES[$index]}"
  run_dir="${run_dirs[$index]}"
  prediction_args+=(--run "${run_name}=${run_dir}")
  group_args+=(--run "$run_name" \
    "$run_dir/diagnostics/train/regression/train_predictions.csv" \
    "$run_dir/diagnostics/val/regression/val_predictions.csv")
  leakage_args+=(--run "$run_name" \
    "$run_dir/diagnostics/train/embeddings/train_features.npz" \
    "$run_dir/diagnostics/val/embeddings/val_features.npz")
done

python scripts/summarize_prediction_runs.py \
  "${prediction_args[@]}" \
  --split val \
  --baseline continuous_alpha025 \
  --output-dir "$ANALYSIS_DIR/prediction"

python scripts/audit_group_robustness.py \
  "${group_args[@]}" \
  --reference-run continuous_alpha025 \
  --seed 42 \
  --output-dir "$ANALYSIS_DIR/group_robustness"

python scripts/audit_representation_leakage.py \
  "${leakage_args[@]}" \
  --output-dir "$ANALYSIS_DIR/representation_leakage"

echo "[CONTINUOUS-SEVERITY] complete"
for index in "${!RUN_NAMES[@]}"; do
  echo "  ${RUN_NAMES[$index]}: ${run_dirs[$index]}"
done
echo "  analysis: $ANALYSIS_DIR"
