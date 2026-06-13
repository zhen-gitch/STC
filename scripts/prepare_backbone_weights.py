import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Prepare local PyTorch backbone weights for MODEL_WEIGHT_PATH."
    )
    parser.add_argument(
        "--model-name",
        default="deit_tiny_patch16_224",
        help="timm model name used by EXTRACT_FEATURE.MODEL_NAME.",
    )
    parser.add_argument(
        "--timm-model-name",
        default=None,
        help=(
            "Optional exact timm/HuggingFace model id to download. "
            "Example: deit_tiny_patch16_224.fb_in1k."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .pth path. Defaults to weights/<model-name>/model.pth.",
    )
    parser.add_argument(
        "--hf-endpoint",
        default="https://hf-mirror.com",
        help="HuggingFace endpoint mirror. Use an empty string to leave HF_ENDPOINT unchanged.",
    )
    parser.add_argument("--img-size", type=int, default=112, help="Image size used when creating transformer backbones.")
    parser.add_argument(
        "--convert-safetensors",
        default=None,
        help="Convert an existing .safetensors file to .pth instead of downloading through timm.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the saved .pth can be loaded by src.models.backbone_factory.build_feature_backbone.",
    )
    return parser


def resolve_output_path(model_name, output):
    if output:
        return Path(output).expanduser().resolve()
    return (PROJECT_ROOT / "weights" / model_name / "model.pth").resolve()


def set_hf_endpoint(endpoint):
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
        print(f"[WEIGHTS] HF_ENDPOINT={endpoint}")


def convert_safetensors_to_pth(safetensors_path, output_path):
    import torch
    from safetensors.torch import load_file

    safetensors_path = Path(safetensors_path).expanduser().resolve()
    if not safetensors_path.exists():
        raise FileNotFoundError(f"safetensors file not found: {safetensors_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = load_file(str(safetensors_path))
    torch.save(state_dict, output_path)
    print(f"[WEIGHTS] Converted safetensors to PyTorch checkpoint: {output_path}")
    return output_path


def download_timm_state_dict(timm_model_name, output_path, img_size):
    import timm
    import torch

    print(f"[WEIGHTS] Creating timm model with pretrained weights: {timm_model_name}")
    try:
        model = timm.create_model(
            timm_model_name,
            pretrained=True,
            num_classes=0,
            img_size=img_size,
        )
    except TypeError as exc:
        if "img_size" not in str(exc):
            raise
        model = timm.create_model(
            timm_model_name,
            pretrained=True,
            num_classes=0,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"[WEIGHTS] Saved local PyTorch state_dict: {output_path}")
    return output_path


def verify_weight_loading(model_name, weight_path, img_size):
    from src.models.backbone_factory import build_feature_backbone

    print("[WEIGHTS] Verifying project loader...")
    build_feature_backbone(
        model_name=model_name,
        weight_path=str(weight_path),
        timm_pretrained=False,
        img_size=img_size,
    )
    print("[WEIGHTS] Verification passed.")


def main():
    args = build_parser().parse_args()
    output_path = resolve_output_path(args.model_name, args.output)

    if args.convert_safetensors:
        saved_path = convert_safetensors_to_pth(args.convert_safetensors, output_path)
    else:
        set_hf_endpoint(args.hf_endpoint)
        timm_model_name = args.timm_model_name or args.model_name
        saved_path = download_timm_state_dict(timm_model_name, output_path, args.img_size)

    if args.verify:
        verify_weight_loading(args.model_name, saved_path, args.img_size)

    print("\n[WEIGHTS] Use this in your override config:")
    print("EXTRACT_FEATURE:")
    print(f'  MODEL_NAME: "{args.model_name}"')
    print("  TIMM_PRETRAINED: False")
    print(f'  MODEL_WEIGHT_PATH: "{saved_path}"')


if __name__ == "__main__":
    main()
