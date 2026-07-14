#!/usr/bin/env bash
# Fixed four-run explicit L1/L2 regularization overfit audit.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

BASE="configs/stage_b/base_regression_only.yaml"
COMMON="configs/regularization_audit/common_full40.yaml"
RUN_NAMES=(reference l1 l2 elastic)
RUN_OVERRIDES=(
  "configs/regularization_audit/reference.yaml"
  "configs/regularization_audit/l1.yaml"
  "configs/regularization_audit/l2.yaml"
  "configs/regularization_audit/elastic.yaml"
)

extra_overrides=()
if [[ "${DEBUG:-0}" == "1" ]]; then
  extra_overrides+=("configs/regularization_audit/debug_smoke.yaml")
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
    "configs/regularization_audit/common_full40.yaml",
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
  echo "[REGULARIZATION-AUDIT] diagnostics skipped"
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

echo "[REGULARIZATION-AUDIT] complete"
for index in "${!RUN_NAMES[@]}"; do
  echo "  ${RUN_NAMES[$index]}: ${run_dirs[$index]}"
done
echo "  analysis: $ANALYSIS_DIR"
