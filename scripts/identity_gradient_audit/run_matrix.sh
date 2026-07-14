#!/usr/bin/env bash
# Paired regression-only versus BDI + identity-adversarial gradient audit.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

BASE="configs/stage_b/base_regression_only.yaml"
COMMON="configs/identity_gradient_audit/common_full40.yaml"
REF="configs/identity_gradient_audit/reference.yaml"
ADV="configs/identity_gradient_audit/bdi_identity_adversarial.yaml"

train_run() {
  local override="$1"
  python scripts/train_mtl_lite.py \
    --override "$BASE" \
    --override "$COMMON" \
    --override "$override"
}

resolve_run_dir() {
  local override="$1"
  python - "$override" <<'PY'
import sys
from pathlib import Path

from src.config import load_experiment_config, resolve_experiment_save_dir

override = sys.argv[1]
cfg = load_experiment_config(
    overrides=[
        "configs/stage_b/base_regression_only.yaml",
        "configs/identity_gradient_audit/common_full40.yaml",
        override,
    ]
)
root = resolve_experiment_save_dir(cfg)
versions = sorted(
    (path for path in root.glob("version_*") if path.is_dir()),
    key=lambda path: int(path.name.split("_")[-1]),
)
print(versions[-1] if versions else root)
PY
}

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  train_run "$REF"
  train_run "$ADV"
fi

REF_DIR="$(resolve_run_dir "$REF")"
ADV_DIR="$(resolve_run_dir "$ADV")"
if [[ ! -f "$REF_DIR/metrics.csv" || ! -f "$ADV_DIR/metrics.csv" ]]; then
  echo "[ERROR] paired metrics.csv files were not found"
  echo "  reference:   $REF_DIR"
  echo "  adversarial: $ADV_DIR"
  exit 1
fi

GROUP_DIR="$(dirname "$(dirname "$REF_DIR")")"
ANALYSIS_DIR="$GROUP_DIR/analysis"

python scripts/summarize_identity_gradient_audit.py \
  --run "adversarial=$ADV_DIR" \
  --output-dir "$ANALYSIS_DIR/gradient_conflict"

python scripts/summarize_training_overfit.py \
  --run "reference=$REF_DIR" \
  --run "adversarial=$ADV_DIR" \
  --output-dir "$ANALYSIS_DIR/training_overfit"

if [[ "${SKIP_DIAG:-0}" == "1" ]]; then
  echo "[IDENTITY-GRADIENT-AUDIT] diagnostics skipped"
  exit 0
fi

for run_dir in "$REF_DIR" "$ADV_DIR"; do
  python scripts/diagnose_mtl_lite.py \
    --run-dir "$run_dir" \
    --ckpt best \
    --split train val \
    --enable-training-curves \
    --enable-regression \
    --enable-embeddings
done

REF_TRAIN_NPZ="$REF_DIR/diagnostics/train/embeddings/train_features.npz"
REF_VAL_NPZ="$REF_DIR/diagnostics/val/embeddings/val_features.npz"
ADV_TRAIN_NPZ="$ADV_DIR/diagnostics/train/embeddings/train_features.npz"
ADV_VAL_NPZ="$ADV_DIR/diagnostics/val/embeddings/val_features.npz"
REF_TRAIN_PRED="$REF_DIR/diagnostics/train/regression/train_predictions.csv"
REF_VAL_PRED="$REF_DIR/diagnostics/val/regression/val_predictions.csv"
ADV_TRAIN_PRED="$ADV_DIR/diagnostics/train/regression/train_predictions.csv"
ADV_VAL_PRED="$ADV_DIR/diagnostics/val/regression/val_predictions.csv"

python scripts/audit_representation_leakage.py \
  --run reference "$REF_TRAIN_NPZ" "$REF_VAL_NPZ" \
  --run adversarial "$ADV_TRAIN_NPZ" "$ADV_VAL_NPZ" \
  --output-dir "$ANALYSIS_DIR/representation_leakage"

python scripts/audit_group_robustness.py \
  --run reference "$REF_TRAIN_PRED" "$REF_VAL_PRED" \
  --run adversarial "$ADV_TRAIN_PRED" "$ADV_VAL_PRED" \
  --reference-run reference \
  --seed 42 \
  --output-dir "$ANALYSIS_DIR/group_robustness"

echo "[IDENTITY-GRADIENT-AUDIT] complete"
echo "  reference:   $REF_DIR"
echo "  adversarial: $ADV_DIR"
echo "  analysis:    $ANALYSIS_DIR"
