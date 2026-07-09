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

# Each run: NAME=run_dir.  Resolve run dirs lazily via the same python helper.
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

# Collect run dirs for the base experiments.  Override by setting RUN_E0=...
# (useful when sweeps created additional version_N under the same name).
E0_DIR="${RUN_E0:-$(resolve_run_dir e0 2>/dev/null || true)}"
E1_DIR="${RUN_E1:-$(resolve_run_dir e1 2>/dev/null || true)}"
E2_DIR="${RUN_E2:-$(resolve_run_dir e2 2>/dev/null || true)}"
E3_DIR="${RUN_E3:-$(resolve_run_dir e3 2>/dev/null || true)}"

echo "[STAGE-B-AGG] runs:"
echo "  E0: ${E0_DIR:-(missing)}"
echo "  E1: ${E1_DIR:-(missing)}"
echo "  E2: ${E2_DIR:-(missing)}"
echo "  E3: ${E3_DIR:-(missing)}"

# Build the --run NAME=PATH list, skipping any experiment whose dir is empty.
RUNS=()
for name in e0 e1 e2 e3; do
  dirvar="RUN_${name^^}"
  dir="${!dirvar:-$(resolve_run_dir "$name" 2>/dev/null || true)}"
  if [[ -n "$dir" && -d "$dir" ]]; then
    RUNS+=("--run" "${name}=${dir}")
  fi
done

if [[ ${#RUNS[@]} -lt 2 ]]; then
  echo "[ERROR] need at least 2 runs to aggregate; found: ${RUNS[*]:-none}"
  exit 1
fi

# ----- prediction (MAE/RMSE/Pearson/CCC, pred mean/std, train-val gap) -----
echo "[STAGE-B-AGG] prediction summary ..."
python scripts/summarize_prediction_runs.py \
  "${RUNS[@]}" --baseline e0 \
  --split test \
  --output-dir "$OUT_ROOT/prediction" || echo "[WARN] prediction aggregation failed"

# ----- severity imbalance (A4: minimal/mild/moderate/severe count/MAE/bias) -
echo "[STAGE-B-AGG] severity imbalance summary ..."
# severity_imbalance consumes prediction_run_summary.csv + severity_bias_summary.csv
# + severity_calibration_run_summary.csv produced by the per-run audits.  We
# point it at the prediction summary dir from the step above plus any per-run
# severity_bias / calibration summaries if present.
PRED_SUMMARY="$OUT_ROOT/prediction/prediction_run_summary.csv"
SEVERITY_BIAS=""
CALIB_SUMMARY=""
# Prefer aggregated calibration summary if it exists (built below), else null.
python scripts/summarize_severity_imbalance.py \
  ${PRED_SUMMARY:+--prediction-summary "$PRED_SUMMARY"} \
  ${CALIB_SUMMARY:+--calibration-summary "$CALIB_SUMMARY"} \
  ${SEVERITY_BIAS:+--severity-bias "$SEVERITY_BIAS"} \
  --output-dir "$OUT_ROOT/severity_imbalance" \
  --run e0 --run e1 --run e2 --run e3 || echo "[WARN] severity imbalance aggregation failed"

# ----- identity retrieval (A1: layer/shared top1/top5, paired rank) --------
echo "[STAGE-B-AGG] identity retrieval summary ..."
python scripts/summarize_identity_retrieval_runs.py \
  "${RUNS[@]}" \
  --output-dir "$OUT_ROOT/identity_retrieval" || echo "[WARN] identity aggregation failed"

# ----- severity calibration (val fits a,b; test evaluates) -----------------
echo "[STAGE-B-AGG] severity calibration summary ..."
python scripts/summarize_severity_calibration_runs.py \
  "${RUNS[@]}" \
  --output-dir "$OUT_ROOT/severity_calibration" || echo "[WARN] calibration aggregation failed"

echo ""
echo "[STAGE-B-AGG] done.  Tables under $OUT_ROOT/:"
echo "  prediction/prediction_run_summary.csv   (MAE/RMSE/Pearson/CCC, E0-anchored)"
echo "  severity_imbalance/severity_imbalance_summary.csv"
echo "  identity_retrieval/identity_retrieval_run_summary.csv"
echo "  severity_calibration/severity_calibration_run_summary.csv"
echo ""
echo "Next: review the E0-anchored deltas against the B5 gate criteria in"
echo "      docs/STAGE_B_RUNBOOK.md section 'B5 gate'."
