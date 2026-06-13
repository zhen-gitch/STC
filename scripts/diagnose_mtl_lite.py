import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.correlation import (
    plot_metrics_correlation_heatmap,
    plot_predictions_correlation_heatmap,
)
from src.diagnostics.embeddings import plot_embedding_diagnostics
from src.diagnostics.io import (
    ensure_dir,
    find_checkpoint,
    read_prediction_table,
    save_features_npz,
    select_records,
    write_prediction_table,
)
from src.diagnostics.regression import plot_regression_diagnostics
from src.diagnostics.reports import write_diagnostic_report
from src.diagnostics.training_curves import plot_training_curves


def build_parser():
    parser = argparse.ArgumentParser(description="Generate offline diagnostics for MTL-Lite runs.")
    parser.add_argument("--run-dir", required=True, help="MTL-Lite CSVLogger version directory.")
    parser.add_argument("--ckpt", default="best", help="'best', 'last', or an explicit checkpoint path.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to diagnose.")
    parser.add_argument("--config", default=None, help="Resolved config YAML. Defaults to <run-dir>/resolved_config.yaml.")
    parser.add_argument("--base-config", default="configs/avec2014_base.yaml", help="Shared base YAML config fallback.")
    parser.add_argument("--local-paths", default="configs/local_paths.yaml", help="Machine-local YAML config fallback.")
    parser.add_argument("--override", action="append", default=[], help="Optional fallback override YAML.")
    parser.add_argument("--allow-missing-local-paths", action="store_true")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or a torch device string.")
    parser.add_argument("--batch-size", type=int, default=1, help="Diagnostic dataloader batch size.")
    parser.add_argument("--num-samples", type=int, default=5, help="Samples for expensive visual diagnostics.")
    parser.add_argument(
        "--selection",
        default="high_error",
        choices=["high_error", "low_error", "severity_balanced"],
        help="Sample selection strategy for expensive diagnostics.",
    )
    parser.add_argument("--occlusion-patch-size", type=int, default=24)
    parser.add_argument("--occlusion-stride", type=int, default=12)
    parser.add_argument("--temporal-window", type=int, default=5)
    parser.add_argument("--attention-method", default="auto", choices=["auto", "gradcam", "input_gradient"])
    parser.add_argument("--target-frame", default="key", help="'key', 'middle', or an integer frame index.")

    parser.add_argument("--enable-training-curves", action="store_true")
    parser.add_argument("--enable-regression", action="store_true")
    parser.add_argument("--enable-embeddings", action="store_true")
    parser.add_argument("--enable-correlation", action="store_true")
    parser.add_argument("--enable-occlusion", action="store_true")
    parser.add_argument("--enable-keyframes", action="store_true")
    parser.add_argument("--enable-model-attention", action="store_true")
    return parser


def resolve_enabled(args):
    flags = {
        "training_curves": args.enable_training_curves,
        "regression": args.enable_regression,
        "embeddings": args.enable_embeddings,
        "correlation": args.enable_correlation,
        "occlusion": args.enable_occlusion,
        "keyframes": args.enable_keyframes,
        "model_attention": args.enable_model_attention,
    }
    if not any(flags.values()):
        return {key: True for key in flags}
    return flags


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


def _subject_ids_from_labels(labels):
    subject_ids = labels["subject_id"]
    if isinstance(subject_ids, str):
        return [subject_ids]
    return [str(item) for item in subject_ids]


def collect_predictions_and_features(model, data_loader, device):
    import torch

    all_subject_ids = []
    all_targets = []
    all_preds = []
    all_features = []

    with torch.no_grad():
        for video_tensor, mask, labels in data_loader:
            video_tensor = video_tensor.to(device)
            mask = mask.to(device)
            outputs = model(video_tensor, mask, return_features=True)
            preds = model.prediction_for_metrics(outputs.bdi_pred).detach().cpu().numpy()
            targets = labels["bdi_score"].detach().cpu().numpy()
            features = outputs.shared_features.detach().cpu().numpy()

            all_subject_ids.extend(_subject_ids_from_labels(labels))
            all_targets.extend(targets.tolist())
            all_preds.extend(preds.tolist())
            all_features.append(features)

    import numpy as np

    return all_subject_ids, np.asarray(all_targets), np.asarray(all_preds), np.concatenate(all_features, axis=0)


def _find_batch_for_subject(data_loader, subject_id):
    for video_tensor, mask, labels in data_loader:
        subject_ids = _subject_ids_from_labels(labels)
        if subject_id in subject_ids:
            idx = subject_ids.index(subject_id)
            return video_tensor[idx:idx + 1], mask[idx:idx + 1], labels
    return None, None, None


def _target_frame_from_arg(target_frame, key_frame, mask):
    valid_count = int(mask[0].sum().item())
    if target_frame == "key":
        return key_frame
    if target_frame == "middle":
        return max(0, valid_count // 2)
    return max(0, min(int(target_frame), max(0, valid_count - 1)))


def run_expensive_diagnostics(args, enabled, model, data_loader, records, output_root, device):
    generated_files = []
    selected = select_records(records, args.selection, args.num_samples)
    if not selected:
        return generated_files

    from src.diagnostics.keyframes import save_keyframe_diagnostics
    from src.diagnostics.model_attention import save_model_attention_diagnostic
    from src.diagnostics.occlusion import save_spatial_occlusion_diagnostic

    for record in selected:
        subject_id = record["subject_id"]
        video_tensor, mask, _labels = _find_batch_for_subject(data_loader, subject_id)
        if video_tensor is None:
            continue
        video_tensor = video_tensor.to(device)
        mask = mask.to(device)

        key_frame = None
        if enabled["keyframes"] or args.target_frame == "key":
            key_frame, paths = save_keyframe_diagnostics(
                model=model,
                video_tensor=video_tensor,
                mask=mask,
                save_dir=output_root / "keyframes",
                subject_id=subject_id,
                temporal_window=args.temporal_window,
            )
            if enabled["keyframes"]:
                generated_files.extend(paths)

        if key_frame is None:
            valid_count = int(mask[0].sum().item())
            key_frame = max(0, valid_count // 2)
        target_frame = _target_frame_from_arg(args.target_frame, key_frame, mask)

        if enabled["occlusion"]:
            generated_files.append(
                save_spatial_occlusion_diagnostic(
                    model=model,
                    video_tensor=video_tensor,
                    mask=mask,
                    frame_idx=target_frame,
                    save_path=output_root / "occlusion" / f"occlusion_impact_subject_{subject_id}.png",
                    patch_size=args.occlusion_patch_size,
                    stride=args.occlusion_stride,
                )
            )

        if enabled["model_attention"]:
            generated_files.append(
                save_model_attention_diagnostic(
                    model=model,
                    video_tensor=video_tensor,
                    mask=mask,
                    frame_idx=target_frame,
                    save_path=output_root / "model_attention" / f"model_attention_subject_{subject_id}.png",
                    method=args.attention_method,
                )
            )

    return generated_files


def main():
    args = build_parser().parse_args()
    enabled = resolve_enabled(args)
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_root = ensure_dir(run_dir / "diagnostics")
    generated_files = []

    metrics_csv = run_dir / "metrics.csv"
    if enabled["training_curves"] and metrics_csv.exists():
        path = plot_training_curves(metrics_csv, output_root / "training")
        if path:
            generated_files.append(path)

    cfgs = load_config(args)
    checkpoint_path = find_checkpoint(run_dir, args.ckpt)
    device = resolve_device(args.device)
    model = load_model(cfgs, checkpoint_path, device)
    data_module = build_data_module(cfgs, args.batch_size)
    data_loader = get_split_loader(data_module, args.split)

    subject_ids, targets, preds, features = collect_predictions_and_features(model, data_loader, device)
    prediction_csv = output_root / "regression" / f"{args.split}_predictions.csv"
    records = write_prediction_table(prediction_csv, subject_ids, targets, preds)
    generated_files.append(prediction_csv)

    features_npz = output_root / "embeddings" / f"{args.split}_features.npz"
    save_features_npz(features_npz, features, subject_ids, targets, preds)
    generated_files.append(features_npz)

    if enabled["regression"]:
        generated_files.extend(plot_regression_diagnostics(prediction_csv, output_root / "regression"))

    if enabled["embeddings"]:
        generated_files.extend(plot_embedding_diagnostics(features_npz, output_root / "embeddings"))

    if enabled["correlation"]:
        if metrics_csv.exists():
            path = plot_metrics_correlation_heatmap(metrics_csv, output_root / "correlation")
            if path:
                generated_files.append(path)
        path = plot_predictions_correlation_heatmap(prediction_csv, output_root / "correlation")
        if path:
            generated_files.append(path)

    expensive_files = run_expensive_diagnostics(
        args=args,
        enabled=enabled,
        model=model,
        data_loader=get_split_loader(data_module, args.split),
        records=read_prediction_table(prediction_csv),
        output_root=output_root,
        device=device,
    )
    generated_files.extend(expensive_files)

    report_path = write_diagnostic_report(
        output_root / "reports" / "diagnostic_report.md",
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        records=read_prediction_table(prediction_csv),
        generated_files=generated_files,
    )
    print(f"[DIAGNOSTICS] Report saved to: {report_path}")
    print(f"[DIAGNOSTICS] Output directory: {output_root}")


if __name__ == "__main__":
    main()
