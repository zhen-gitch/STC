import argparse
import os


parser = argparse.ArgumentParser(description="端到端模型视觉归因诊断脚本")
parser.add_argument(
    "--gpu",
    type=str,
    default="3",
    help="指定用于特征诊断的物理 GPU ID，例如 '0', '1', '2'"
)
parser.add_argument("--path", type=str, default="/usr/local/conda/zhen/test05", help="项目运行根目录")
parser.add_argument("--version", type=int, default=15, help="CSVLogger version 编号")
parser.add_argument("--num-samples", type=int, default=10, help="最多生成多少个样本的归因图")
parser.add_argument(
    "--prediction-csv",
    type=str,
    default=None,
    help="由评估流程生成的 predictions.csv；提供后按 selection 策略选择样本"
)
parser.add_argument(
    "--selection",
    type=str,
    default="high_error",
    choices=["high_error", "low_error", "severity_balanced"],
    help="归因样本选择策略"
)
args, unknown = parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import cv2
import numpy as np
from omegaconf import OmegaConf
from src.datasets.dataset import AVECDataModule
from src.models.end_to_end import EndToEndDepressionModel
from src.utils.visualize import BackboneGradCAM, map_occlusion_sensitivity, select_attribution_subjects


def tensor_frame_to_bgr(frame_tensor):
    """Convert a normalized RGB frame tensor to an OpenCV BGR uint8 image."""
    raw_img = (frame_tensor.detach().cpu().float().permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255.0
    return cv2.cvtColor(np.clip(raw_img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def run_diagnostic(path, version_id, num_samples, prediction_csv=None, selection="high_error"):
    """Generate visual attribution maps for selected validation samples."""
    cfgs = OmegaConf.load(f"{path}/configs/default_config.yaml")
    cfgs.EXTRACT_FEATURE.BATCH_SIZE = 1  # 诊断必须单样本串行进行
    cfgs.MODE = "full"
    cfgs.EXTRACT_FEATURE.CHUNK_SIZE = 1000

    data_module = AVECDataModule(cfgs)
    data_module.setup()
    val_loader = data_module.val_dataloader()

    model = EndToEndDepressionModel(cfgs)
    version_id = version_id
    model.load_segmented_weights(f"{path}/logs/csv_log/version_{version_id}/weights", tag="best")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"[DEVICE STATUS] 诊断模型当前运行在安全逻辑设备: {model.device} | 运行模式: {cfgs.MODE}")

    if hasattr(model.backbone, 'blocks'):
        from src.utils.visualize import ViTAttentionCAM
        cam_engine = ViTAttentionCAM(model)
        is_vit_mode = True
    else:
        from src.utils.visualize import BackboneGradCAM
        target_layer = model.backbone.layer4[-1]
        cam_engine = BackboneGradCAM(model, target_layer)
        is_vit_mode = False

    selected_subjects = set(select_attribution_subjects(prediction_csv, selection, num_samples))
    if selected_subjects:
        print(f"[DIAGNOSIS] 使用 {selection} 策略选择归因样本: {sorted(selected_subjects)}")

    generated = 0
    for batch_idx, batch in enumerate(val_loader):
        if generated >= num_samples:
            break

        video_tensor, masks, labels = batch
        subject_id = labels['subject_id'][0]
        if selected_subjects and subject_id not in selected_subjects:
            continue

        video_tensor = video_tensor.to(model.device)
        masks = masks.to(model.device)
        true_bdi = labels['bdi_score'][0].item()

        target_frame_idx = min(10, video_tensor.size(1) - 1)

        if is_vit_mode:
            with torch.no_grad():
                bdi_preds, _, _ = model(video_tensor, masks)

            raw_img_bgr = tensor_frame_to_bgr(video_tensor[0, target_frame_idx])

            fused_cam = cam_engine.generate_heatmap(raw_img_bgr, frame_idx=target_frame_idx)
        else:
            with torch.enable_grad():
                bdi_preds, _, _ = model(video_tensor, masks)
                raw_img_bgr = tensor_frame_to_bgr(video_tensor[0, target_frame_idx])
                fused_cam = cam_engine.generate_heatmap(bdi_preds.mean(), raw_img_bgr)

        fused_occlusion = map_occlusion_sensitivity(model, video_tensor.cpu(), masks.cpu())

        save_path = f"{path}/logs/csv_log/version_{version_id}/diagnose_plots/{subject_id}/"
        os.makedirs(save_path, exist_ok=True)
        cv2.imwrite(f"{save_path}/gradcam_attribution.png", fused_cam)
        cv2.imwrite(f"{save_path}/occlusion_sensitivity.png", fused_occlusion)
        print(f"   [ID: {subject_id}] 真实分数: {true_bdi:.1f} | 视觉特征图已成功映射投射至 {save_path}")
        generated += 1

    cam_engine.remove()


if __name__ == "__main__":
    run_diagnostic(
        path=args.path,
        version_id=args.version,
        num_samples=args.num_samples,
        prediction_csv=args.prediction_csv,
        selection=args.selection
    )
