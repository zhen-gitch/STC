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
#   bash scripts/stage_b/backfill_a1_identity.sh           # latest version of each base run
#   bash scripts/stage_b/backfill_a1_identity.sh e1 e3      # only named experiments
#   ALL_VERSIONS=1 bash scripts/stage_b/backfill_a1_identity.sh   # every version (base + sweeps)
#   CKPT=last bash scripts/stage_b/backfill_a1_identity.sh  # use last instead of best
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

CKPT="${CKPT:-best}"
RESOLVER="$PROJECT_ROOT/scripts/stage_b/resolve_run_specs.py"

if [[ $# -gt 0 ]]; then
  EXPERIMENTS=("$@")
else
  EXPERIMENTS=(e0 e1 e2 e3)
fi

# Resolve target runs:
#   ALL_VERSIONS=1 -> every version per experiment, labeled by sweep
#                    (eN, eN_lambda0.02, eN_power1.0, ...).  Needed so
#                    aggregation can read identity summaries for sweep versions
#                    that predate the per-version diagnose fix.
#   default        -> latest version per experiment only (the training matrix
#                    already diagnoses each version right after it trains).
RESOLVER_FLAGS=()
if [[ -n "${ALL_VERSIONS:-}" ]]; then
  RESOLVER_FLAGS=()           # version-aware (default behavior of the resolver)
else
  RESOLVER_FLAGS=("--latest-only")
fi
mapfile -t TARGETS < <(python "$RESOLVER" "${RESOLVER_FLAGS[@]}" "${EXPERIMENTS[@]}" 2>/dev/null || true)

for spec in "${TARGETS[@]}"; do
  [[ -z "$spec" ]] && continue
  label="${spec%%=*}"
  run_dir="${spec#*=}"
  echo "================================================================"
  echo "[A1-BACKFILL] $label -> $run_dir"
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
    || { echo "[ERROR] A1 probe failed for $label"; continue; }

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
    echo "[A1-BACKFILL] $label done: $run_dir/tables/embedding_identity_summary.csv"
  else
    echo "[ERROR] $label: probe ran but no summary produced under $a1_dir/tables/"
  fi
done

echo ""
echo "[A1-BACKFILL] complete.  Re-run aggregation:"
echo "  bash scripts/stage_b/aggregate_stage_b.sh"
