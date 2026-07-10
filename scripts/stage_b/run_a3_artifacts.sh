#!/usr/bin/env bash
# =============================================================================
# A3 artifact weak-label chain across Stage B runs (manual / standalone).
#
# Runs the full A3 chain (black_artifacts + alignment_geometry + openface_quality
# + temporal_sampling + weaklabel join) on each Stage B run, then builds the
# cross-run matched-only summary.  This is the same per-run logic diagnose_run
# runs automatically as Stage 3d; this script is for backfilling A3 on existing
# runs without re-running training/diagnostics, or for re-running A3 alone after
# changing OPENFACE_ROOT.
#
# Version-aware: enumerates every version per experiment (base + sweeps),
# labeled by sweep config, via scripts/stage_b/resolve_run_specs.py.
#
# Usage:
#   bash scripts/stage_b/run_a3_artifacts.sh                 # all versions, all exps
#   bash scripts/stage_b/run_a3_artifacts.sh e1 e3           # only named experiments
#   ALL_VERSIONS=1 bash scripts/stage_b/run_a3_artifacts.sh  # (default; every version)
#   bash scripts/stage_b/run_a3_artifacts.sh --latest-only e1
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

RESOLVER="$PROJECT_ROOT/scripts/stage_b/resolve_run_specs.py"
OUT_ROOT="${OUT_ROOT:-logs/stage_b/aggregate/a3_matched}"
mkdir -p "$OUT_ROOT"

# Parse --latest-only (pass-through to resolver); remainder are experiment names.
LATEST_FLAG=()
EXPERIMENTS=()
for arg in "$@"; do
  case "$arg" in
    --latest-only) LATEST_FLAG=("--latest-only") ;;
    *) EXPERIMENTS+=("$arg") ;;
  esac
done
[[ ${#EXPERIMENTS[@]} -eq 0 ]] && EXPERIMENTS=(e0 e1 e2 e3)

mapfile -t TARGETS < <(python "$RESOLVER" "${LATEST_FLAG[@]}" "${EXPERIMENTS[@]}" 2>/dev/null || true)

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  echo "[A3] no runs resolved; nothing to do."
  exit 0
fi

# Per-run chain (reuses the diagnose_run Stage 3d helper).
for spec in "${TARGETS[@]}"; do
  [[ -z "$spec" ]] && continue
  label="${spec%%=*}"
  run_dir="${spec#*=}"
  test_pred="$run_dir/diagnostics/test/regression/test_predictions.csv"
  echo "================================================================"
  echo "[A3] $label -> $run_dir"
  echo "================================================================"
  if [[ ! -f "$test_pred" ]]; then
    echo "[A3] test predictions missing ($test_pred) -- run diagnose first; skipping"
    continue
  fi
  bash "$PROJECT_ROOT/scripts/stage_b/_run_a3_for_run.sh" "$run_dir" "$test_pred" \
    || echo "[A3] chain failed for $label"
done

# Cross-run matched-only summary.
echo ""
echo "[A3] building matched-only cross-run summary ..."
SUMMARY_ARGS=()
for spec in "${TARGETS[@]}"; do
  [[ -z "$spec" ]] && continue
  label="${spec%%=*}"
  run_dir="${spec#*=}"
  wl="$run_dir/diagnostics/test/a3_weaklabels/tables/artifact_weaklabel_summary.csv"
  [[ -f "$wl" ]] && SUMMARY_ARGS+=(--summary "${label}=${wl}")
done

if [[ ${#SUMMARY_ARGS[@]} -ge 2 ]]; then
  python scripts/summarize_artifact_weaklabels_matched.py \
    "${SUMMARY_ARGS[@]}" \
    --output-dir "$OUT_ROOT" \
    --coupling-threshold 0.2 || echo "[A3] matched summary failed"
  echo "[A3] matched summary under $OUT_ROOT/"
else
  echo "[A3] fewer than 2 weaklabel summaries found; skipping matched summary"
fi
