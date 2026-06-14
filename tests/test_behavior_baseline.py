import torch
from omegaconf import OmegaConf

from src.models.behavior_baseline import BehaviorBaselineModel


def _config():
    return OmegaConf.create(
        {
            "EXTRACT_FEATURE": {
                "MAX_SCORE": 63,
            },
            "PROCESS_TEMPORAL": {
                "CLASS_STEP": 10,
                "LEARNING_RATE": 1e-4,
                "WEIGHT_DECAY": 5e-4,
            },
            "BEHAVIOR_MODEL": {
                "HIDDEN_DIM": 8,
                "NUM_LAYERS": 1,
                "BIDIRECTIONAL": True,
                "DROPOUT": 0.0,
            },
            "MODEL": {
                "AUXILIARY_TASKS": {
                    "ORDINAL_CLASSIFICATION": True,
                }
            },
            "LOSSES": {
                "ORDINAL_WEIGHT": 0.5,
                "CCC_WEIGHT": 0.0,
            },
        }
    )


def test_behavior_baseline_forward_and_loss_are_finite():
    model = BehaviorBaselineModel(_config(), input_dim=6)
    features = torch.randn(2, 5, 6)
    mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
        ],
        dtype=torch.bool,
    )
    labels = {
        "bdi_score": torch.tensor([10.0, 20.0]),
        "class_label": torch.tensor([1, 2]),
    }

    outputs = model(features, mask, return_features=True)
    losses = model.compute_losses(outputs, labels)

    assert outputs.bdi_pred.shape == (2,)
    assert outputs.ordinal_logits.shape == (2, model.num_classes - 1)
    assert outputs.shared_features.shape == (2, 16)
    assert torch.isfinite(losses.total)


def test_behavior_baseline_regression_head_gets_gradients():
    model = BehaviorBaselineModel(_config(), input_dim=6)
    features = torch.randn(2, 5, 6)
    mask = torch.ones(2, 5, dtype=torch.bool)
    labels = {
        "bdi_score": torch.tensor([10.0, 20.0]),
        "class_label": torch.tensor([1, 2]),
    }

    outputs = model(features, mask)
    losses = model.compute_losses(outputs, labels)
    losses.total.backward()

    grad_norm = 0.0
    for param in model.reg_task_head.parameters():
        if param.grad is not None:
            grad_norm += float(param.grad.abs().sum().item())

    assert grad_norm > 0.0
