#!/usr/bin/env python
"""RPDF Stage A1: layer-wise identity probe.

Loads a trained MTL-Lite checkpoint, registers forward hooks on backbone
sub-modules, runs one split through ``forward(return_layer_features=True)``,
saves a multi-key per-layer embedding NPZ, and runs the paired-task identity
retrieval audit independently on each layer.

The retrieval logic is reused from ``src.diagnostics.identity_retrieval``; this
script only adds the layer loop and aggregation via
``src.diagnostics.layerwise_identity``.

Example:

    python scripts/audit_layerwise_identity_probe.py \\
        --run-dir <LOG_DIR>/default/rgb/version_0 \\
        --ckpt best \\
        --split test \\
        --output-dir <LOG_DIR>/default/rgb/version_0/diagnostics/layerwise_identity
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.io import ensure_dir


VALID_SPLITS = {"train", "val", "test"}


def build_parser():
    parser = argparse.ArgumentParser(
        description="RPDF Stage A1: layer-wise identity probe."
    )
    parser.add_argument("--run-dir", required=True, help="MTL-Lite CSVLogger version directory.")
    parser.add_argument("--ckpt", default="best", help="'best', 'last', or an explicit checkpoint path.")
    parser.add_argument(
        "--split",
        default="test",
        choices=sorted(VALID_SPLITS),
        help="Dataset split to probe (default: test).",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for tables, NPZ, and reports.")
    parser.add_argument("--config", default=None, help="Resolved config YAML. Defaults to <run-dir>/resolved_config.yaml.")
    parser.add_argument("--base-config", default="configs/avec2014_base.yaml", help="Shared base YAML config fallback.")
    parser.add_argument("--local-paths", default="configs/local_paths.yaml", help="Machine-local YAML config fallback.")
    parser.add_argument("--override", action="append", default=[], help="Optional fallback override YAML.")
    parser.add_argument("--allow-missing-local-paths", action="store_true")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or a torch device string.")
    parser.add_argument("--batch-size", type=int, default=1, help="Diagnostic dataloader batch size.")
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated layer names. Defaults to LAYERWISE_PROBE_LAYERS.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Neighborhood size for agreement metrics.")
    parser.add_argument("--predictions", default=None, help="Optional prediction CSV for metadata enrichment.")
    return parser


def load_config(args):
    from omegaconf import OmegaConf

    from src.config import load_experiment_config

    run_dir = Path(args.run_dir)
    config_path = Path(args.config) if args.config else run_dir / "resolved_config.yaml"
    if config_path.exists():
        return OmegaConf.load(config_path)
    return load_experiment_config(
        base_config=args.base_config,
        local_paths_config=args.local_paths,
        overrides=args.override,
        require_local_paths=not args.allow_missing_local_paths,
    )


def resolve_device(device_arg):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_model(cfgs, checkpoint_path, device):
    import torch

    from src.models.mtl_lite import MTLLiteDepressionModel

    model = MTLLiteDepressionModel(cfgs)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def build_data_module(cfgs, batch_size):
    from src.datasets.dataset import AVECDataModule

    cfgs.EXTRACT_FEATURE.BATCH_SIZE = int(batch_size)
    cfgs.DATASET.RETURN_MULTI_VIEW_TRAIN = False
    data_module = AVECDataModule(cfgs)
    data_module.setup()
    return data_module


def get_split_loader(data_module, split):
    if split == "train":
        return data_module.train_dataloader()
    if split == "val":
        return data_module.val_dataloader()
    return data_module.test_dataloader()


def _labels_to_list(values):
    if isinstance(values, str):
        return [values]
    return [str(item) for item in values]


def _video_ids_from_labels(labels):
    video_ids = labels.get("video_id", labels["subject_id"])
    return _labels_to_list(video_ids)


def collect_layerwise_features(model, data_loader, device, layer_names):
    """Run the split through the model and collect per-layer video embeddings.

    Returns a dict mapping ``layer_name -> (N, D)`` numpy array, plus aligned
    metadata lists in dataloader order.
    """
    import numpy as np
    import torch

    all_video_ids = []
    all_subject_ids = []
    all_task_names = []
    all_targets = []
    all_preds = []
    layer_stacks = {name: [] for name in layer_names}

    model.register_layer_hooks(layer_names)
    try:
        with torch.no_grad():
            for video_tensor, mask, labels in data_loader:
                video_tensor = video_tensor.to(device)
                mask = mask.to(device)
                outputs = model(video_tensor, mask, return_layer_features=True)
                preds = model.prediction_for_metrics(outputs.bdi_pred).detach().cpu().numpy()
                targets = labels["bdi_score"].detach().cpu().numpy()

                all_video_ids.extend(_video_ids_from_labels(labels))
                all_subject_ids.extend(_labels_to_list(labels["subject_id"]))
                all_task_names.extend(_labels_to_list(labels.get("task_name", "")))
                all_targets.extend(targets.tolist())
                all_preds.extend(preds.tolist())

                if outputs.layer_features is None:
                    raise RuntimeError(
                        "Model returned no layer_features. Ensure register_layer_hooks "
                        "was called and the backbone exposes hookable sub-modules."
                    )
                for name in layer_names:
                    if name not in outputs.layer_features:
                        # Layer could not be resolved/captured on this backbone; skip.
                        continue
                    vec = outputs.layer_features[name].detach().cpu().numpy()
                    layer_stacks[name].append(vec)
    finally:
        model.clear_layer_hooks()

    layer_features = {}
    for name in layer_names:
        if layer_stacks[name]:
            layer_features[name] = np.concatenate(layer_stacks[name], axis=0)

    return (
        layer_features,
        all_video_ids,
        all_subject_ids,
        all_task_names,
        np.asarray(all_targets),
        np.asarray(all_preds),
    )


def main():
    args = build_parser().parse_args()
    from src.diagnostics.io import find_checkpoint
    from src.diagnostics.layerwise_identity import (
        run_layerwise_identity_audit,
        save_layerwise_features_npz,
    )
    from src.models.mtl_lite import LAYERWISE_PROBE_LAYERS

    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = ensure_dir(Path(args.output_dir))
    layer_names = (
        [name.strip() for name in args.layers.split(",") if name.strip()]
        if args.layers
        else list(LAYERWISE_PROBE_LAYERS)
    )

    cfgs = load_config(args)
    checkpoint_path = find_checkpoint(run_dir, args.ckpt)
    device = resolve_device(args.device)
    model = load_model(cfgs, checkpoint_path, device)
    data_module = build_data_module(cfgs, args.batch_size)
    data_loader = get_split_loader(data_module, args.split)

    print(f"[LAYER-PROBE] Split: {args.split} | layers: {layer_names}")
    (
        layer_features,
        video_ids,
        subject_ids,
        task_names,
        targets,
        preds,
    ) = collect_layerwise_features(model, data_loader, device, layer_names)

    if not layer_features:
        raise RuntimeError(
            "No layer features were captured. The backbone may not expose "
            "patch_embed/blocks; check the [LAYER-PROBE] skipped-layers warning."
        )

    n = len(subject_ids)
    print(f"[LAYER-PROBE] Collected {n} samples across {len(layer_features)} layers.")

    features_npz = output_dir / f"{args.split}_layerwise_features.npz"
    save_layerwise_features_npz(
        features_npz,
        layer_features=layer_features,
        subject_ids=subject_ids,
        targets=targets,
        preds=preds,
        video_ids=video_ids,
    )
    print(f"[LAYER-PROBE] Saved layer-wise NPZ: {features_npz}")

    generated = run_layerwise_identity_audit(
        features_npz=features_npz,
        output_dir=output_dir,
        predictions_csv=args.predictions,
        top_k=args.top_k,
        layer_order=layer_names,
    )
    print("[LAYER-PROBE] Generated files:")
    for path in generated:
        print(f"  - {path}")
    print(f"[LAYER-PROBE] Output directory: {output_dir}")


if __name__ == "__main__":
    main()
