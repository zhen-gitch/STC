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
  # Core B4 diagnostics: prediction/severity/identity/correlation + training
  # curves.  Occlusion/keyframes/model-attention are expensive and left for a
  # focused pass on selected cases, not the full matrix.
  python scripts/diagnose_mtl_lite.py \
    --run-dir "$run_dir" --ckpt best \
    --split $DIAG_SPLITS \
    --enable-training-curves --enable-regression \
    --enable-embeddings --enable-correlation
}

# -----------------------------------------------------------------------------
# Write a 2-line sweep override file on the fly (avoids pre-creating many
# near-duplicate configs).  $1=path, $2=key, $3=value.
# -----------------------------------------------------------------------------
write_sweep_override() {
  local path="$1" key="$2" value="$3"
  mkdir -p "$(dirname "$path")"
  cat > "$path" <<EOF
MODEL:
  ${key}: ${value}
EOF
  echo "$path"
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
          write_sweep_override "$ov" "IDENTITY_ADVERSARIAL.LAMBDA_ID" "$lam"
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
