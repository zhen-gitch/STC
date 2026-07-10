#!/usr/bin/env bash
# =============================================================================
# A3 artifact weak-label chain for ONE run (internal helper).
#
# Shared by run_stage_b_matrix.sh's diagnose_run (Stage 3d) and the standalone
# run_a3_artifacts.sh.  Reads IMAGE_DIR + DATASET.OPENFACE_ROOT from the run's
# resolved_config.yaml, then runs the 4 upstream artifact/quality summaries +
# the weaklabel join.
#
# Graceful degradation:
#   - IMAGE_DIR missing -> black_artifacts + temporal_sampling skipped (they
#     need aligned frames); the whole chain degrades to a no-op with a warning.
#   - OPENFACE_ROOT missing -> alignment_geometry + openface_quality skipped
#     (audit_openface_quality writes an empty summary; alignment_geometry is
#     skipped outright); the weaklabel join still runs on whatever summaries
#     exist (black + temporal), so a partial A3 is produced.
#
# Args: $1=run_dir  $2=test_predictions.csv
# =============================================================================
set -uo pipefail

run_dir="$1"
test_pred="$2"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Resolve IMAGE_DIR + OPENFACE_ROOT from resolved_config.yaml.
read -r image_dir openface_root < <(python - "$run_dir" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf
cfg = OmegaConf.load(Path(sys.argv[1]) / "resolved_config.yaml")
img = str(getattr(cfg, "IMAGE_DIR", "") or "")
ofr = ""
try:
    ofr = str(getattr(cfg.DATASET, "OPENFACE_ROOT", "") or "")
except (AttributeError, KeyError):
    pass
print(img, ofr)
PY
)

base="$run_dir/diagnostics/test"
black_dir="$base/black_artifacts"
align_dir="$base/alignment_geometry"
ofq_dir="$base/openface_quality"
temp_dir="$base/temporal_sampling"
a3_dir="$base/a3_weaklabels"

if [[ -z "$image_dir" || ! -d "$image_dir" ]]; then
  echo "[A3] IMAGE_DIR missing/invalid ($image_dir) for $run_dir -- skipping A3 chain"
  exit 0
fi

# black_artifacts (needs IMAGE_DIR)
python "$project_root/scripts/audit_black_artifacts.py" \
  --predictions "$test_pred" --image-root "$image_dir" --output-dir "$black_dir" \
  2>&1 | tail -2 || echo "[A3] black_artifacts failed"

# temporal_sampling (needs IMAGE_DIR)
python "$project_root/scripts/audit_temporal_sampling.py" \
  --predictions "$test_pred" --image-root "$image_dir" --output-dir "$temp_dir" \
  2>&1 | tail -2 || echo "[A3] temporal_sampling failed"

# alignment_geometry (needs OPENFACE_ROOT -- reads OpenFace landmark CSVs)
if [[ -n "$openface_root" && -d "$openface_root" ]]; then
  python "$project_root/scripts/audit_alignment_geometry.py" \
    --predictions "$test_pred" --openface-root "$openface_root" --output-dir "$align_dir" \
    2>&1 | tail -2 || echo "[A3] alignment_geometry failed"
else
  echo "[A3] OPENFACE_ROOT missing -- skipping alignment_geometry"
fi

# openface_quality (writes empty summary if root missing, so the weaklabel join
# finds the file and skips openface fields cleanly)
python "$project_root/scripts/audit_openface_quality.py" \
  --run-dir "$run_dir" --output-dir "$ofq_dir" \
  2>&1 | tail -2 || echo "[A3] openface_quality failed"

# weaklabel join: pass whatever summaries exist.
black_csv="$black_dir/tables/black_artifact_summary.csv"
align_csv="$align_dir/tables/alignment_geometry_summary.csv"
ofq_csv="$ofq_dir/tables/openface_quality_summary.csv"
temp_csv="$temp_dir/tables/temporal_sampling_summary.csv"

join_args=(--predictions "$test_pred" --output-dir "$a3_dir")
[[ -f "$black_csv" ]] && join_args+=(--black-artifacts "$black_csv")
[[ -f "$align_csv" ]] && join_args+=(--alignment-geometry "$align_csv")
[[ -f "$ofq_csv" ]] && join_args+=(--openface-quality "$ofq_csv")
[[ -f "$temp_csv" ]] && join_args+=(--temporal-sampling "$temp_csv")

python "$project_root/scripts/audit_artifact_weaklabels.py" "${join_args[@]}" \
  2>&1 | tail -2 || echo "[A3] weaklabel join failed"

echo "[A3] done for $run_dir (summaries under $base/{black_artifacts,alignment_geometry,openface_quality,temporal_sampling,a3_weaklabels})"
