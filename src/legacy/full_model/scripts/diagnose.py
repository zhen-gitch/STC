import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEFAULT_BASE_CONFIG,
    DEFAULT_LOCAL_PATHS_CONFIG,
    load_experiment_config,
    load_yaml_config,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Generate visual attribution diagnostics.")
    parser.add_argument(
        "--gpu",
        type=str,
        default="3",
        help="Physical GPU id used for diagnostics, for example '0', '1', or '2'.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help=(
            "Legacy project/run root. If provided, logs are read from "
            "<path>/logs/csv_log/version_<version>."
        ),
    )
    parser.add_argument("--version", type=int, default=15, help="CSVLogger version id.")
    parser.add_argument("--num-samples", type=int, default=10, help="Maximum samples to diagnose.")
    parser.add_argument(
        "--prediction-csv",
        type=str,
        default=None,
        help="Optional predictions.csv used to select attribution samples.",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="high_error",
        choices=["high_error", "low_error", "severity_balanced"],
        help="Attribution sample selection strategy.",
    )
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
        help="Load one complete config directly instead of merging base/local/override configs.",
    )
    parser.add_argument(
        "--allow-missing-local-paths",
        action="store_true",
        help="Allow config loading without configs/local_paths.yaml.",
    )
    return parser


def load_config_from_args(args):
    if args.config:
        return load_yaml_config(args.config)

    return load_experiment_config(
        base_config=args.base_config,
        local_paths_config=args.local_paths,
        overrides=args.override,
        require_local_paths=not args.allow_missing_local_paths,
    )


def tensor_frame_to_bgr(frame_tensor):
    """Convert a normalized RGB frame tensor to an OpenCV BGR uint8 image."""
    import cv2
    import numpy as np

    raw_img = (frame_tensor.detach().cpu().float().permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255.0
    return cv2.cvtColor(np.clip(raw_img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def resolve_log_root(cfgs, legacy_path=None):
    if legacy_path:
        return Path(legacy_path) / "logs"
    return Path(str(cfgs.LOG_DIR))


def run_diagnostic(cfgs, log_root, version_id, num_samples, prediction_csv=None, selection="high_error"):
    """Generate visual attribution maps for selected validation samples."""
    import cv2
    import torch

    from src.datasets.dataset import AVECDataModule
    from src.models.end_to_end import EndToEndDepressionModel
    from src.utils.visualize import (
        BackboneGradCAM,
        ViTAttentionCAM,
        map_occlusion_sensitivity,
        select_attribution_subjects,
    )

    cfgs.EXTRACT_FEATURE.BATCH_SIZE = 1
    cfgs.MODE = "full"
    cfgs.EXTRACT_FEATURE.CHUNK_SIZE = 1000

    data_module = AVECDataModule(cfgs)
    data_module.setup()
    val_loader = data_module.val_dataloader()

    model = EndToEndDepressionModel(cfgs)
    version_dir = Path(log_root) / "csv_log" / f"version_{version_id}"
    model.load_segmented_weights(str(version_dir / "weights"), tag="best")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"[DEVICE STATUS] diagnostic model device: {model.device} | mode: {cfgs.MODE}")

    if hasattr(model.backbone, "blocks"):
        cam_engine = ViTAttentionCAM(model)
        is_vit_mode = True
    else:
        target_layer = model.backbone.layer4[-1]
        cam_engine = BackboneGradCAM(model, target_layer)
        is_vit_mode = False

    selected_subjects = set(select_attribution_subjects(prediction_csv, selection, num_samples))
    if selected_subjects:
        print(f"[DIAGNOSIS] selected subjects with {selection}: {sorted(selected_subjects)}")

    generated = 0
    for batch_idx, batch in enumerate(val_loader):
        if generated >= num_samples:
            break

        video_tensor, masks, labels = batch
        subject_id = labels["subject_id"][0]
        if selected_subjects and subject_id not in selected_subjects:
            continue

        video_tensor = video_tensor.to(model.device)
        masks = masks.to(model.device)
        true_bdi = labels["bdi_score"][0].item()

        target_frame_idx = min(10, video_tensor.size(1) - 1)

        if is_vit_mode:
            with torch.no_grad():
                model(video_tensor, masks)
            raw_img_bgr = tensor_frame_to_bgr(video_tensor[0, target_frame_idx])
            fused_cam = cam_engine.generate_heatmap(raw_img_bgr, frame_idx=target_frame_idx)
        else:
            with torch.enable_grad():
                bdi_preds, _, _ = model(video_tensor, masks)
                raw_img_bgr = tensor_frame_to_bgr(video_tensor[0, target_frame_idx])
                fused_cam = cam_engine.generate_heatmap(bdi_preds.mean(), raw_img_bgr)

        fused_occlusion = map_occlusion_sensitivity(model, video_tensor.cpu(), masks.cpu())

        save_path = version_dir / "diagnose_plots" / str(subject_id)
        save_path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path / "gradcam_attribution.png"), fused_cam)
        cv2.imwrite(str(save_path / "occlusion_sensitivity.png"), fused_occlusion)
        print(f"   [ID: {subject_id}] true BDI: {true_bdi:.1f} | saved to {save_path}")
        generated += 1

    cam_engine.remove()


def main():
    args = build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cfgs = load_config_from_args(args)
    log_root = resolve_log_root(cfgs, legacy_path=args.path)
    run_diagnostic(
        cfgs=cfgs,
        log_root=log_root,
        version_id=args.version,
        num_samples=args.num_samples,
        prediction_csv=args.prediction_csv,
        selection=args.selection,
    )


if __name__ == "__main__":
    main()
