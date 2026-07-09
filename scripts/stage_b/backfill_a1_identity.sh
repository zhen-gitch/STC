#!/usr/bin/env bash
# =============================================================================
# Backfill A1 layerwise identity probe for existing Stage B runs.
#
# Use this when training + diagnose_mtl_lite already ran (so prediction CSVs
# exist) but audit_layerwise_identity_probe was NOT run -- i.e. the identity
# aggregation fails with "Could not find embedding_identity_summary.csv".
#
# Loads each run's best checkpoint and re-runs the model with layer hooks to
# produce a layer-wise NPZ + embedding_identity_summary.csv (the A1 tool used
# in Stage A).  Does NOT retrain and does NOT re-run diagnose_mtl_lite /
# calibration.  After this, re-run aggregate_stage_b.sh and the identity
# summary will be found.
#
# Usage:
#   bash scripts/stage_b/backfill_a1_identity.sh           # all 4 base runs
#   bash scripts/stage_b/backfill_a1_identity.sh e1 e3      # only named runs
#   CKPT=last bash scripts/stage_b/backfill_a1_identity.sh  # use last instead of best
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

CKPT="${CKPT:-best}"

declare -A EXP_NAME=(
  [e0]="e0_rgb_mtl_lite"
  [e1]="e1_identity_adversarial"
  [e2]="e2_severity_balanced"
  [e3]="e3_identity_adversarial_severity_balanced"
)

if [[ $# -gt 0 ]]; then
  EXPERIMENTS=("$@")
else
  EXPERIMENTS=(e0 e1 e2 e3)
fi

# Resolve the latest run_dir for one experiment (mirrors aggregate_stage_b.sh).
resolve_run_dir() {
  python - "$1" <<'PY'
import sys, glob
from pathlib import Path
from omegaconf import OmegaConf
from src.config import resolve_config_path, load_yaml_config, DEFAULT_BASE_CONFIG
name = sys.argv[1]
cfg = load_yaml_config(DEFAULT_BASE_CONFIG)
lp = resolve_config_path("configs/local_paths.yaml")
if lp.exists():
    cfg = OmegaConf.merge(cfg, OmegaConf.load(lp))
cfg = OmegaConf.merge(cfg, OmegaConf.load("configs/stage_b/base_regression_only.yaml"))
ov = glob.glob(f"configs/stage_b/{name}_*.yaml")
if ov:
    cfg = OmegaConf.merge(cfg, OmegaConf.load(ov[0]))
log_dir = getattr(cfg, "LOG_DIR", None)
if not log_dir:
    sys.exit("LOG_DIR not set")
root = Path(log_dir)
if not root.is_absolute():
    root = Path.cwd() / root
base = root / str(getattr(cfg, "EXPERIMENT_GROUP", "default")) / str(getattr(cfg, "EXPERIMENT_NAME", name))
if not base.exists():
    sys.exit(f"run dir not found: {base}")
versions = sorted([p for p in base.glob("version_*") if p.is_dir()],
                  key=lambda p: int(p.name.split("_")[1]))
print(versions[-1])
PY
}

for exp in "${EXPERIMENTS[@]}"; do
  ename="${EXP_NAME[$exp]:-}"
  if [[ -z "$ename" ]]; then
    echo "[WARN] unknown experiment: $exp (skip)"; continue
  fi
  run_dir="$(resolve_run_dir "$exp")"
  echo "================================================================"
  echo "[A1-BACKFILL] $exp -> $run_dir"
  echo "================================================================"
  if [[ ! -d "$run_dir" ]]; then
    echo "[ERROR] run dir not found, skipping: $run_dir"
    continue
  fi

  a1_dir="$run_dir/diagnostics/test/a1_layerwise"
  test_pred="$run_dir/diagnostics/test/regression/test_predictions.csv"

  # Run the layerwise identity probe (loads best checkpoint, re-runs test with
  # hooks).  --predictions enriches the per-query rows with video_id/task.
  python scripts/audit_layerwise_identity_probe.py \
    --run-dir "$run_dir" --ckpt "$CKPT" \
    --split test \
    --output-dir "$a1_dir" \
    ${test_pred:+--predictions "$test_pred"} \
    || { echo "[ERROR] A1 probe failed for $exp"; continue; }

  # Mirror the identity summary into <run_dir>/tables/ so aggregate_stage_b.sh
  # (which looks under <run_dir>/tables/) finds it.
  if [[ -f "$a1_dir/tables/embedding_identity_summary.csv" ]]; then
    mkdir -p "$run_dir/tables"
    cp "$a1_dir/tables/embedding_identity_summary.csv" "$run_dir/tables/embedding_identity_summary.csv"
    # Also mirror the per-query CSV so summarize_identity_retrieval_runs can
    # pick up rank/agreement signals if it looks for them.
    if [[ -f "$a1_dir/tables/embedding_identity_retrieval.csv" ]]; then
      cp "$a1_dir/tables/embedding_identity_retrieval.csv" "$run_dir/tables/embedding_identity_retrieval.csv"
    fi
    echo "[A1-BACKFILL] $exp done: $run_dir/tables/embedding_identity_summary.csv"
  else
    echo "[ERROR] $exp: probe ran but no summary produced under $a1_dir/tables/"
  fi
done

echo ""
echo "[A1-BACKFILL] complete.  Re-run aggregation:"
echo "  bash scripts/stage_b/aggregate_stage_b.sh"
