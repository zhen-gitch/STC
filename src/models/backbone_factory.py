"""
Backbone construction utilities shared by end-to-end training and feature extraction.
"""

import os

import timm
import torch

from src.models.iresnet import iresnet50


def _load_optional_weights(model, model_name, weight_path):
    if not weight_path:
        print(f"[BACKBONE] {model_name} 未指定外部权重路径，使用模型默认初始化/预训练权重。")
        return model

    if not os.path.exists(weight_path):
        print(f"❌ [WARNING] 找不到模型权重文件: {weight_path}")
        return model

    state_dict = torch.load(weight_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    cleaned_state_dict = {
        key.replace('module.', ''): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(cleaned_state_dict, strict=False)
    print(f"成功加载 {model_name} 预训练权重！")
    return model


def build_feature_backbone(model_name: str, weight_path=None):
    """
    Create a visual backbone that returns feature embeddings instead of logits.

    Supported names:
    - iresnet50
    - resnet50
    - timm ViT models whose names start with ``vit_``
    """
    try:
        if model_name == 'iresnet50':
            model = iresnet50()
        elif model_name == 'resnet50':
            model = timm.create_model(
                model_name=model_name,
                pretrained=True,
                num_classes=0,
            )
        elif model_name.startswith('vit_'):
            model = timm.create_model(
                model_name=model_name,
                pretrained=True,
                img_size=112,
                num_classes=0,
            )
            print(f"成功创建 {model_name} 视觉 Transformer 骨干网络！")
        else:
            raise ValueError(f"model name is illegal: {model_name}")

        return _load_optional_weights(model, model_name, weight_path)

    except RuntimeError:
        raise
    except Exception as e:
        print(f"❌ [Error] timm 创建模型失败，对于 timm 非法的模型名称 {model_name}！")
        raise e
