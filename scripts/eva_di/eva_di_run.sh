#!/usr/bin/env bash
# EVA-DI server runner (RIB-STD-v1; runbook: docs/EVA_DI_SERVER_RUNBOOK.md).
# ONE-CLICK steps, NOT executed locally; formal implementation runs on the
# server.  Resumable + provenance-pinned (git commit + config sha + weight sha
# inside artifacts).  Rib alignment: machine paths ONLY via EVA_DI_* env
# (paths.server.yaml private override), offline weight pinning, bounded
# writable run root outside the source tree, fail-closed preflight.
#
# Required environment (see configs/eva_di/paths.server.example.yaml):
#   EVA_DI_OF3_ROOT       OF3 formal masked-v2 feature root (read-only)
#   EVA_DI_SPLIT_FILE     dataset_split.json (read-only, locked)
#   EVA_DI_LABEL_DIR      avec2014 depression_labels (read-only)
#   EVA_DI_NORM_ROOT      behavior_normalization_train_v1 (read-only)
#   EVA_DI_WEIGHT         pinned EVA02 weight file, sha d3d632...0c2f
#   EVA_DI_CACHE_ROOT     extraction cache            (writable run root)
#   EVA_DI_LOG_ROOT       run outputs <log_root>/<run_id>/ (writable)
#   EVA_DI_OUTPUT_ROOT    exports/matrix tables        (writable)
#   EVA_DI_TRANS_ROOT     trans checkout (oracle reference; default /home/zhen/code/trans)
#   PY                    python (default conda env "light")
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-/home/zhen/miniconda3/envs/light/bin/python}"
WEIGHT_SHA=d3d632352efbd0a0a8269dce114ab8a214833f85ce6f738860221f11ab0f0c2f
STEP="${1:-}"

require() {  # name must be set + exist
  [ -n "${!1:-}" ] || { echo "MISSING ENV: $1"; exit 2; }
  [ -e "${!1}" ] || { echo "MISSING PATH: $1=${!1}"; exit 2; }
}

case "$STEP" in
  env-check)   # rib-style fail-closed preflight
    for v in EVA_DI_OF3_ROOT EVA_DI_SPLIT_FILE EVA_DI_LABEL_DIR EVA_DI_NORM_ROOT \
             EVA_DI_WEIGHT EVA_DI_CACHE_ROOT EVA_DI_LOG_ROOT EVA_DI_OUTPUT_ROOT; do
      require "$v"
    done
    echo "sha256sum: $EVA_DI_WEIGHT"
    sha256sum "$EVA_DI_WEIGHT" | grep -q "^$WEIGHT_SHA " || { echo "WEIGHT SHA MISMATCH"; exit 1; }
    df -h "$EVA_DI_CACHE_ROOT" "$EVA_DI_LOG_ROOT" "$EVA_DI_OUTPUT_ROOT"
    avail_kb=$(df --output=avail -k "$EVA_DI_LOG_ROOT" | tail -1 | tr -d ' ')
    [ "$avail_kb" -ge $((20 * 1024 * 1024)) ] || { echo "DISK < 20G on log root"; exit 1; }
    PYTHONPATH=.:src "$PY" - <<'PYX'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
      "| visible devices:", torch.cuda.device_count())
assert torch.cuda.is_available(), "no visible CUDA device (CUDA_VISIBLE_DEVICES?)"
PYX
    git rev-parse HEAD | sed 's/^/git commit: /'; git status --porcelain | head -3
    echo "ENV_CHECK_OK"
    ;;
  extract)   # full cache extraction (EVA-DI-EXTRACT package, server-authorized)
    "$PY" -m src.eva_di.extract.runner --config configs/eva_di/extract_uniform512.yaml --device cuda
    ;;
  oracle)    # equivalence gate over the just-extracted cache; rc!=0 => FAILED
    "$PY" -m src.eva_di.extract.oracle_equivalence --config configs/eva_di/extract_uniform512.yaml --device cuda
    ;;
  plan)      # matrix dry-run (prints arms/status/commands; launches nothing)
    "$PY" -m src.eva_di.matrix --spec configs/eva_di/matrix_v1.yaml --runs-root "${EVA_DI_LOG_ROOT:?}"
    ;;
  launch)    # matrix sequential training, stop-on-first-failure
    "$PY" -m src.eva_di.matrix --spec configs/eva_di/matrix_v1.yaml --runs-root "${EVA_DI_LOG_ROOT:?}" --launch --device cuda
    ;;
  identity)  # external identity metrics for one finished run
    RUN_ID="${2:?usage: $0 identity <run_id>}"
    E="${EVA_DI_LOG_ROOT:?}/${RUN_ID}/embeddings"
    "$PY" -m src.eva_di.identity_metrics --npz "$E/p_mean_val.npz" "$E/v_h0_val.npz" \
        --output-dir "${EVA_DI_LOG_ROOT}/${RUN_ID}/identity"
    ;;
  viz)       # per-run training curves + cross-arm comparison (read-only inputs)
    "$PY" -m src.eva_di.viz_training --runs-root "${EVA_DI_LOG_ROOT:?}" \
        --output-dir "${EVA_DI_OUTPUT_ROOT:?}/viz"
    ;;
  report)    # cross-run matrix table (utility + external identity metrics)
    "$PY" -m src.eva_di.matrix_report --runs-root "${EVA_DI_LOG_ROOT:?}" --output-dir "${EVA_DI_LOG_ROOT}/_matrix"
    ;;
  archive)   # rib_transfer-style bundle: tar.zst + sha256 sidecar of the run root
    OUT="${2:-${EVA_DI_OUTPUT_ROOT:?}/eva_di_run_$(date +%Y%m%d).tar.zst}"
    tar -C "$(dirname "$EVA_DI_LOG_ROOT")" --zstd -cf "$OUT" "$(basename "$EVA_DI_LOG_ROOT")"
    sha256sum "$OUT" > "$OUT.sha256"
    echo "archived: $OUT ($(du -h "$OUT" | cut -f1)) + $(basename "$OUT").sha256"
    ;;
  *) echo "usage: $0 {env-check|extract|oracle|plan|launch|identity <run_id>|viz|report|archive}"; exit 2;;
esac
