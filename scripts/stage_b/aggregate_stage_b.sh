#!/usr/bin/env bash
# =============================================================================
# Stage B aggregation (B4) — build the E0-anchored comparison table.
#
# Consumes per-run diagnostics produced by run_stage_b_matrix.sh and emits
# cross-run summary tables comparing E0/E1/E2/E3 (+ sweeps) on:
#   - prediction (MAE/RMSE/Pearson/CCC, pred mean/std, train-val gap)
#   - severity imbalance (minimal/mild/moderate/severe count/MAE/bias)
#   - identity retrieval (A1 layer/shared top1/top5, paired rank)
#   - severity calibration (val fits a,b; test evaluates)
#   - training overfit (best-val vs last epoch, train/val gap)
#
# Version-aware: every version_N per experiment is included, labeled by sweep
# config (eN for base, eN_lambda0.02 / eN_power1.0 for sweeps) via
# scripts/stage_b/resolve_run_specs.py.  Pin a specific version with RUN_E0=...
# to skip discovery for that experiment.
#
# A2 (error-identity coupling) and matched-only A3 (artifact weaklabels) are
# per-run audits with heavier inputs (features_npz, weaklabel summaries); see
# docs/STAGE_B_RUNBOOK.md for their command templates — they are not run here.
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Output root for the cross-run tables.
OUT_ROOT="${OUT_ROOT:-logs/stage_b/aggregate}"
mkdir -p "$OUT_ROOT"

# Resolve run dirs version-aware: enumerate every version_N per experiment and
# label by sweep config (eN for the base config, eN_lambda0.02 / eN_power1.0
# for sweeps), de-duplicated to the latest version per sweep signature.  This
# lets one aggregation table cover the base configs AND the lambda/power sweeps
# (previously only the latest version per experiment was read, which silently
# dropped E2's base POWER=0.5 when a POWER=1.0 sweep was newer).
#
# Pin a specific version with RUN_E0=.../version_K (labeled e0); that
# experiment then skips auto-discovery.
RESOLVER="scripts/stage_b/resolve_run_specs.py"

ALL_RUNS=()
# Explicit RUN_E* pins first (one exact dir, labeled eN).
for name in e0 e1 e2 e3; do
  dirvar="RUN_${name^^}"
  dir="${!dirvar:-}"
  if [[ -n "$dir" && -d "$dir" ]]; then
    ALL_RUNS+=("${name}=${dir}")
  fi
done
# Discover versions for experiments not pinned above.
DISCOVER=()
for name in e0 e1 e2 e3; do
  dirvar="RUN_${name^^}"
  [[ -z "${!dirvar:-}" ]] && DISCOVER+=("$name")
done
if [[ ${#DISCOVER[@]} -gt 0 ]]; then
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    ALL_RUNS+=("$line")
  done < <(python "$RESOLVER" "${DISCOVER[@]}" 2>/dev/null || true)
fi

echo "[STAGE-B-AGG] runs (version-aware):"
for spec in "${ALL_RUNS[@]}"; do echo "  $spec"; done

# RUNS      : --run NAME=<run_dir>  for identity / calibration / overfit
#             (they read summaries under <run_dir>/tables/ or <run_dir>/metrics.csv).
# PRED_RUNS : --run NAME=<pred_csv> for prediction
#             (diagnostics/test/regression/test_predictions.csv).
# LABELS    : bare run names for severity_imbalance (selects by name).
RUNS=(); PRED_RUNS=(); LABELS=()
for spec in "${ALL_RUNS[@]}"; do
  label="${spec%%=*}"
  dir="${spec#*=}"
  RUNS+=("--run" "${label}=${dir}")
  LABELS+=("$label")
  pred_csv="$dir/diagnostics/test/regression/test_predictions.csv"
  if [[ -f "$pred_csv" ]]; then
    PRED_RUNS+=("--run" "${label}=${pred_csv}")
  else
    echo "[WARN] ${label}: test prediction CSV missing ($pred_csv); excluded from prediction summary"
  fi
done

if [[ ${#RUNS[@]} -lt 2 ]]; then
  echo "[ERROR] need at least 2 runs to aggregate; found: ${ALL_RUNS[*]:-none}"
  exit 1
fi

# ----- prediction (MAE/RMSE/Pearson/CCC, pred mean/std, train-val gap) -----
# Also writes severity_bias_summary.csv + task_consistency_summary.csv, which
# the severity_imbalance step below consumes.
echo "[STAGE-B-AGG] prediction summary ..."
if [[ ${#PRED_RUNS[@]} -ge 2 ]]; then
  python scripts/summarize_prediction_runs.py \
    "${PRED_RUNS[@]}" --baseline e0 \
    --split test \
    --output-dir "$OUT_ROOT/prediction" || echo "[WARN] prediction aggregation failed"
else
  echo "[WARN] fewer than 2 prediction CSVs found; skipping prediction summary"
fi

# ----- identity retrieval (A1: layer/shared top1/top5, paired rank) --------
# Reads <run_dir>/tables/embedding_identity_summary.csv from audit_identity_retrieval.
echo "[STAGE-B-AGG] identity retrieval summary ..."
python scripts/summarize_identity_retrieval_runs.py \
  "${RUNS[@]}" \
  --output-dir "$OUT_ROOT/identity_retrieval" || echo "[WARN] identity aggregation failed"

# ----- severity calibration (val fits a,b; test evaluates) -----------------
# Reads <run_dir>/tables/severity_calibration_fit.csv from audit_severity_calibration.
echo "[STAGE-B-AGG] severity calibration summary ..."
python scripts/summarize_severity_calibration_runs.py \
  "${RUNS[@]}" \
  --output-dir "$OUT_ROOT/severity_calibration" || echo "[WARN] calibration aggregation failed"

# ----- severity imbalance (A4: minimal/mild/moderate/severe count/MAE/bias) -
# Runs LAST: it consumes prediction_run_summary.csv + severity_bias_summary.csv
# (from the prediction step above) and severity_calibration_run_summary.csv
# (from the calibration step above).
echo "[STAGE-B-AGG] severity imbalance summary ..."
PRED_SUMMARY="$OUT_ROOT/prediction/tables/prediction_run_summary.csv"
SEVERITY_BIAS="$OUT_ROOT/prediction/tables/severity_bias_summary.csv"
CALIB_SUMMARY="$OUT_ROOT/severity_calibration/tables/severity_calibration_run_summary.csv"
# severity_imbalance selects rows by run name, so pass every discovered label
# (base + sweeps) -- not just e0-e3 -- otherwise sweep rows are silently dropped.
SEV_RUNS=()
for label in "${LABELS[@]}"; do SEV_RUNS+=("--run" "$label"); done
python scripts/summarize_severity_imbalance.py \
  ${PRED_SUMMARY:+--prediction-summary "$PRED_SUMMARY"} \
  ${SEVERITY_BIAS:+--severity-bias "$SEVERITY_BIAS"} \
  ${CALIB_SUMMARY:+--calibration-summary "$CALIB_SUMMARY"} \
  --output-dir "$OUT_ROOT/severity_imbalance" \
  "${SEV_RUNS[@]}" || echo "[WARN] severity imbalance aggregation failed"

# ----- training overfit (best-val vs last epoch, train/val gap) -------------
# Reads <run_dir>/metrics.csv (auto-resolved by summarize_training_overfit).
# Surfaces whether each run keeps degrading after best val (supports the B5
# train-val-gap criterion and a future EarlyStopping decision).
echo "[STAGE-B-AGG] training overfit summary ..."
python scripts/summarize_training_overfit.py \
  "${RUNS[@]}" \
  --output-dir "$OUT_ROOT/training_overfit" || echo "[WARN] overfit aggregation failed"

echo ""
echo "[STAGE-B-AGG] done.  Tables under $OUT_ROOT/:"
echo "  prediction/prediction_run_summary.csv   (MAE/RMSE/Pearson/CCC, E0-anchored)"
echo "  severity_imbalance/severity_imbalance_summary.csv"
echo "  identity_retrieval/identity_retrieval_run_summary.csv"
echo "  severity_calibration/severity_calibration_run_summary.csv"
echo "  training_overfit/training_overfit_summary.csv  (best-val vs last, train/val gap)"
echo ""
echo "Next: review the E0-anchored deltas against the B5 gate criteria in"
echo "      docs/STAGE_B_RUNBOOK.md section 'B5 gate'."
