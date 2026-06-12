import argparse
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytorch_lightning as pl
import torch

from src.config import (
    DEFAULT_BASE_CONFIG,
    DEFAULT_LOCAL_PATHS_CONFIG,
    LEGACY_DEFAULT_CONFIG,
    load_experiment_config,
    load_yaml_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run end-to-end BDI training.")
    parser.add_argument(
        "--base-config",
        default=str(DEFAULT_BASE_CONFIG),
        help="Shared base YAML config.",
    )
    parser.add_argument(
        "--local-paths",
        default=str(DEFAULT_LOCAL_PATHS_CONFIG),
        help="Machine-local YAML config with dataset and log paths.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Optional override YAML config. Can be provided multiple times.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Load one complete config directly. This bypasses base/local/override "
            "merging and is intended for legacy configs."
        ),
    )
    parser.add_argument(
        "--allow-missing-local-paths",
        action="store_true",
        help="Allow config loading without configs/local_paths.yaml.",
    )
    return parser.parse_args()


def load_config_from_args(args):
    if args.config:
        return load_yaml_config(args.config)

    return load_experiment_config(
        base_config=args.base_config,
        local_paths_config=args.local_paths,
        overrides=args.override,
        require_local_paths=not args.allow_missing_local_paths,
    )


def run_from_config(cfgs):
    mode = str(cfgs.MODE)
    if mode != "full":
        raise ValueError(f"MODE must be 'full' for end-to-end training, got: {mode}")

    from src.trainers.end_to_end_runner import run_end2end

    print("[INFO] START RUNNING END-TO-END STREAM LINE...")
    run_end2end(cfgs)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(42, workers=True)
    args = parse_args()
    try:
        if args.config:
            print(f"[INFO] LOADING COMPLETE CONFIG FILE: {args.config}")
        else:
            print(f"[INFO] LOADING BASE CONFIG FILE: {args.base_config}")
            print(f"[INFO] LOADING LOCAL PATHS FILE: {args.local_paths}")
            if args.override:
                print(f"[INFO] LOADING OVERRIDE CONFIG FILES: {args.override}")
            if args.allow_missing_local_paths:
                print("[WARNING] configs/local_paths.yaml is optional for this run.")

        cfgs = load_config_from_args(args)
        run_from_config(cfgs)

    except FileNotFoundError as e:
        print(f"\n[CONFIG ERROR] {e}")
        print(f"[HINT] For legacy behavior, try: --config {LEGACY_DEFAULT_CONFIG}")
        sys.exit(1)

    except torch.cuda.OutOfMemoryError:
        print("\n[FATAL ERROR] CUDA out of memory.")
        print("[HINT] Reduce EXTRACT_FEATURE.BATCH_SIZE or PROCESS_TEMPORAL.MAX_SEQ_LEN.")
        sys.exit(1)

    except ValueError as e:
        print(f"\n[VALUE ERROR] {e}")
        sys.exit(1)

    except Exception as e:
        print("\n[UNKNOWN ERROR] Training was interrupted by an unexpected error.")
        print(f"[DETAIL] {e}")
        traceback.print_exc()
        sys.exit(1)
