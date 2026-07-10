#!/usr/bin/env bash
# =============================================================================
# Stage B fixed experiment matrix (B3) — E0/E1/E2/E3 + sweeps.
#
# Runs the four Stage B experiments under one unified protocol (same split,
# seed, backbone, optimizer, precision, checkpoint policy) and then runs the
# B4 offline diagnostics on each.  Intended to run on the server with real
# AVEC2014 data + GPU; configs/local_paths.yaml must exist.
#
# Usage:
#   bash scripts/stage_b/run_stage_b_matrix.sh            # all 4 base runs + sweeps
#   bash scripts/stage_b/run_stage_b_matrix.sh e0 e1       # only named experiments
#   SKIP_SWEEPS=1 bash scripts/stage_b/run_stage_b_matrix.sh  # base configs only
#
# Environment knobs:
#   LOG_ROOT        override LOG_DIR (default: from local_paths.yaml)
#   SKIP_SWEEPS=1   skip lambda_id / POWER sweeps, run base configs only
#   SKIP_TRAIN=1    skip training (assume runs already exist), only diagnose
#   SKIP_DIAG=1     skip diagnostics, only train
#   DIAG_SPLITS     "val test" (default) — splits to diagnose per run
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Shared Stage B base (reproduces the regression_only baseline).  Every
# experiment layers its own thin override on top of this.
STAGE_BASE="configs/stage_b/base_regression_only.yaml"

# Experiments: name -> thin override config (EXPERIMENT_NAME + switches).
# Sweeps are layered on top as 2-line override files (configs/stage_b/sweeps/).
declare -A EXP_CONFIG=(
  [e0]="configs/stage_b/e0_rgb_mtl_lite.yaml"
  [e1]="configs/stage_b/e1_identity_adversarial.yaml"
  [e2]="configs/stage_b/e2_severity_balanced.yaml"
  [e3]="configs/stage_b/e3_identity_adversarial_severity_balanced.yaml"
)

# lambda_id sweep targets for E1/E3.
LAMBDA_SWEEP=(0.02 0.05 0.10 0.20)
# POWER sweep targets for E2/E3.
POWER_SWEEP=(0.5 1.0)

DIAG_SPLITS="${DIAG_SPLITS:-val test}"

# Select which experiments to run (default: all).
if [[ $# -gt 0 ]]; then
  EXPERIMENTS=("$@")
else
  EXPERIMENTS=(e0 e1 e2 e3)
fi

# -----------------------------------------------------------------------------
# Resolve the run directory for one experiment (latest version_N).
# Mirrors src/config.resolve_experiment_save_dir: <LOG_DIR>/<group>/<name>/.
# -----------------------------------------------------------------------------
resolve_run_dir() {
  local name="$1"
  # EXPERIMENT_GROUP is "stage_b" for all stage_b configs; name is the exp id.
  python - "$name" <<'PY'
import sys, os
from pathlib import Path
from omegaconf import OmegaConf
from src.config import resolve_config_path, load_yaml_config, DEFAULT_BASE_CONFIG
name = sys.argv[1]
cfg = OmegaConf.merge(
    load_yaml_config(DEFAULT_BASE_CONFIG),
    load_yaml_config(resolve_config_path("configs/local_paths.yaml")),
    OmegaConf.load("configs/stage_b/base_regression_only.yaml"),
)
# Load the actual stage_b override to get EXPERIMENT_NAME.
import glob
ov = glob.glob(f"configs/stage_b/{name}_*.yaml")
if ov:
    cfg = OmegaConf.merge(cfg, OmegaConf.load(ov[0]))
log_dir = getattr(cfg, "LOG_DIR", None)
if not log_dir:
    sys.exit("LOG_DIR not set in local_paths.yaml")
root = Path(log_dir)
if not root.is_absolute():
    root = Path.cwd() / root
group = getattr(cfg, "EXPERIMENT_GROUP", "default")
ename = getattr(cfg, "EXPERIMENT_NAME", name)
base = root / str(group) / str(ename)
if not base.exists():
    print(base); sys.exit(0)
versions = sorted([p for p in base.glob("version_*") if p.is_dir()],
                  key=lambda p: int(p.name.split("_")[1]))
print(versions[-1] if versions else base)
PY
}

# -----------------------------------------------------------------------------
# Train one run.  $1=experiment id, $2=optional extra override (sweep file).
# -----------------------------------------------------------------------------
train_run() {
  local exp="$1"; local extra="${2:-}"
  local overrides=("--override" "$STAGE_BASE" "--override" "${EXP_CONFIG[$exp]}")
  if [[ -n "$extra" ]]; then
    overrides+=("--override" "$extra")
  fi
  echo "================================================================"
  echo "[STAGE-B] TRAIN $exp  extra=${extra:-none}"
  echo "================================================================"
  python scripts/train_mtl_lite.py "${overrides[@]}"
}

# -----------------------------------------------------------------------------
# Diagnose one run (B4).  $1=experiment id.
#
# Three stages:
#  1. diagnose_mtl_lite.py -- per-split prediction CSV + single-layer features
#     + plots.  Writes <run_dir>/diagnostics/<split>/regression/<split>_predictions.csv
#     (consumed by prediction + calibration aggregation).
#  2. audit_layerwise_identity_probe.py -- re-runs the model with layer hooks
#     to produce a LAYER-WISE NPZ + tables/embedding_identity_summary.csv.
#     This is the A1 tool (same as Stage A), needed because diagnose_mtl_lite's
#     NPZ is single-layer and cannot feed the per-layer A1/A2 audits.  Output
#     goes to <run_dir>/diagnostics/test/a1_layerwise/ ; the identity summary
#     is also copied/located so aggregate_stage_b.sh finds it under <run_dir>.
#  3. A2 error-identity coupling + severity calibration, both reading step 1/2
#     outputs.
#
# Calibration needs BOTH val and test prediction CSVs, so DIAG_SPLITS must
# include both (default "val test").  The layerwise probe is test-only (A1
# identity risk is evaluated on test, matching Stage A).
# -----------------------------------------------------------------------------
diagnose_run() {
  local exp="$1"
  local run_dir
  run_dir="$(resolve_run_dir "$exp")"
  echo "----------------------------------------------------------------"
  echo "[STAGE-B] DIAGNOSE $exp  run_dir=$run_dir"
  echo "----------------------------------------------------------------"
  if [[ ! -d "$run_dir" ]]; then
    echo "[WARN] run dir not found for $exp, skipping diagnostics: $run_dir"
    return
  fi

  # Stage 1: core diagnostics (prediction CSV + single-layer features + plots).
  python scripts/diagnose_mtl_lite.py \
    --run-dir "$run_dir" --ckpt best \
    --split $DIAG_SPLITS \
    --enable-training-curves --enable-regression \
    --enable-embeddings --enable-correlation

  local test_pred="$run_dir/diagnostics/test/regression/test_predictions.csv"
  local val_pred="$run_dir/diagnostics/val/regression/val_predictions.csv"
  local a1_dir="$run_dir/diagnostics/test/a1_layerwise"
  local layerwise_npz="$a1_dir/test_layerwise_features.npz"

  # Stage 2: A1 layerwise identity probe (produces layer-wise NPZ + identity
  # summary).  This is heavier (re-runs the model with hooks) but is the only
  # way to get per-layer identity retrieval comparable to Stage A.
  python scripts/audit_layerwise_identity_probe.py \
    --run-dir "$run_dir" --ckpt best \
    --split test \
    --output-dir "$a1_dir" \
    ${test_pred:+--predictions "$test_pred"} \
    || echo "[WARN] A1 layerwise identity probe failed for $exp"

  # Mirror the identity summary (+ per-query retrieval) into <run_dir>/tables/
  # so aggregate_stage_b.sh (which looks under <run_dir>/tables/) finds them
  # without path special-casing.  The per-query CSV feeds the per-severity
  # identity summary; without it that sub-table is silently skipped.
  if [[ -f "$a1_dir/tables/embedding_identity_summary.csv" ]]; then
    mkdir -p "$run_dir/tables"
    cp "$a1_dir/tables/embedding_identity_summary.csv" "$run_dir/tables/embedding_identity_summary.csv"
    if [[ -f "$a1_dir/tables/embedding_identity_retrieval.csv" ]]; then
      cp "$a1_dir/tables/embedding_identity_retrieval.csv" "$run_dir/tables/embedding_identity_retrieval.csv"
    fi
  fi

  # Stage 3a: A2 error-identity coupling (uses the LAYER-WISE NPZ from step 2,
  # not diagnose_mtl_lite's single-layer NPZ).
  if [[ -f "$layerwise_npz" && -f "$test_pred" ]]; then
    python scripts/audit_error_identity_coupling.py \
      --features-npz "$layerwise_npz" \
      --predictions "$test_pred" \
      --output-dir "$run_dir/diagnostics/test/a2_coupling" \
      || echo "[WARN] A2 error-identity coupling audit failed for $exp"
  else
    echo "[WARN] layerwise NPZ or test predictions missing for $exp, skipping A2 audit"
  fi

  # Stage 3b: severity calibration (val fits a,b; test evals).  Uses the
  # prediction CSVs from step 1.  Writes tables/severity_calibration_fit.csv
  # under <run_dir>, which aggregate_stage_b.sh consumes.
  if [[ -f "$val_pred" && -f "$test_pred" ]]; then
    python scripts/audit_severity_calibration.py \
      --val-predictions "$val_pred" \
      --test-predictions "$test_pred" \
      --output-dir "$run_dir" || echo "[WARN] severity calibration audit failed for $exp"
  else
    echo "[WARN] val/test prediction CSV missing for $exp, skipping calibration audit"
  fi

  # Stage 3c: subject attacker accuracy (fresh linear-probe on z_dep).  Reads
  # the A1 layerwise NPZ (Stage 2 output) -- no checkpoint/GPU needed.  AVEC2014
  # is subject-disjoint, so the jointly-trained identity head is unusable on
  # test (coverage=0); the fresh LOVO Ridge attacker trains on test subjects
  # themselves, giving coverage=100% and a real top1/top3 for every run
  # (including E0/E2 baseline).  Requires Stage 2 (A1 probe) to have run.
  local attacker_dir="$run_dir/diagnostics/test/subject_attacker"
  python scripts/audit_subject_attacker.py \
    --run-dir "$run_dir" --split test \
    --output-dir "$attacker_dir" \
    || echo "[WARN] subject attacker audit failed/skipped for $exp"
  # Mirror the summary into <run_dir>/tables/ for aggregate_stage_b.sh.
  if [[ -f "$attacker_dir/tables/subject_attacker_summary.csv" ]]; then
    mkdir -p "$run_dir/tables"
    cp "$attacker_dir/tables/subject_attacker_summary.csv" "$run_dir/tables/subject_attacker_summary.csv"
  fi

  # Stage 3d: A3 artifact weak-label chain (graceful degradation).  Reads
  # IMAGE_DIR + DATASET.OPENFACE_ROOT from the run's resolved_config.yaml.
  # black_artifacts + temporal_sampling need only IMAGE_DIR; alignment_geometry
  # + openface_quality need OPENFACE_ROOT.  When OPENFACE_ROOT is unset, those
  # steps (and the weaklabel join) are skipped with a warning rather than
  # failing the whole diagnose pass.  A3 is a B5 auxiliary signal, not a gate.
  if [[ -n "${SKIP_A3:-}" ]]; then
    echo "[STAGE-B] SKIP_A3 set, skipping A3 artifact chain for $exp"
  else
    bash "${PROJECT_ROOT}/scripts/stage_b/_run_a3_for_run.sh" "$run_dir" "$test_pred" \
      || echo "[WARN] A3 artifact chain failed for $exp"
  fi
}

# =============================================================================
# Main
# =============================================================================
mkdir -p configs/stage_b/sweeps

for exp in "${EXPERIMENTS[@]}"; do
  cfg="${EXP_CONFIG[$exp]:-}"
  if [[ -z "$cfg" ]]; then
    echo "[WARN] unknown experiment: $exp (skip)"; continue
  fi

  # Base config run.
  if [[ -z "${SKIP_TRAIN:-}" ]]; then
    train_run "$exp"
  fi
  if [[ -z "${SKIP_DIAG:-}" ]]; then
    diagnose_run "$exp"
  fi

  # Sweeps (E1/E3: lambda_id; E2/E3: POWER).  Skipped when SKIP_SWEEPS=1.
  if [[ -z "${SKIP_SWEEPS:-}" ]]; then
    case "$exp" in
      e1|e3)
        for lam in "${LAMBDA_SWEEP[@]}"; do
          # 0.05 is the base config value; skip the duplicate run.
          [[ "$lam" == "0.05" ]] && continue
          ov="configs/stage_b/sweeps/${exp}_lambda_${lam}.yaml"
          # LAMBDA_ID lives under IDENTITY_ADVERSARIAL; must be NESTED YAML.
          # A flat literal dotted key (IDENTITY_ADVERSARIAL.LAMBDA_ID) is NOT
          # expanded by OmegaConf, so the model would keep reading the base
          # 0.05 and every lambda sweep would silently run at the same value.
          mkdir -p "$(dirname "$ov")"
          cat > "$ov" <<EOF
MODEL:
  IDENTITY_ADVERSARIAL:
    LAMBDA_ID: ${lam}
EOF
          if [[ -z "${SKIP_TRAIN:-}" ]]; then train_run "$exp" "$ov"; fi
          if [[ -z "${SKIP_DIAG:-}" ]]; then diagnose_run "$exp"; fi
        done
        ;;
    esac
    case "$exp" in
      e2|e3)
        for pw in "${POWER_SWEEP[@]}"; do
          [[ "$pw" == "0.5" ]] && continue  # 0.5 is the base config value
          ov="configs/stage_b/sweeps/${exp}_power_${pw}.yaml"
          # POWER lives under SEVERITY_BALANCED_REGRESSION; nested key.
          mkdir -p "$(dirname "$ov")"
          cat > "$ov" <<EOF
MODEL:
  SEVERITY_BALANCED_REGRESSION:
    POWER: ${pw}
EOF
          if [[ -z "${SKIP_TRAIN:-}" ]]; then train_run "$exp" "$ov"; fi
          if [[ -z "${SKIP_DIAG:-}" ]]; then diagnose_run "$exp"; fi
        done
        ;;
    esac
  fi
done

echo ""
echo "[STAGE-B] matrix complete.  Next: run scripts/stage_b/aggregate_stage_b.sh"
echo "          to build the E0-anchored comparison table for B5 gating."
