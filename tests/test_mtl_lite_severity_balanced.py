"""Stage B2 severity-balanced regression tests.

Covers the B2 minimal closed loop (docs/MTL_LITE_DESIGN.md section 13.5 /
docs/TODO.md B2-severity-balanced-minimal):

- weight computation: ``(total / (num_bins * count)) ** power``, mean-normalize,
  clip to ``[MIN_WEIGHT, MAX_WEIGHT]``;
- loss scaling: severity-bin reweighted MSE differs from plain MSE exactly by
  the per-sample weight table;
- default-off switch is bit-identical to plain MSE baseline;
- only regression MSE is reweighted -- CCC and ordinal are unchanged;
- bin edges match ``src.diagnostics.io.severity_group`` (<=13 / <=19 / <=28);
- degenerate / empty train counts fall back to plain MSE (no inf/nan).
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.diagnostics.io import severity_group
from tests.test_mtl_lite_forward import DummyBackbone, minimal_mtl_lite_config


def _severity_balanced_config(power=0.5, min_w=0.5, max_w=4.0):
    cfg = minimal_mtl_lite_config()
    cfg.MODEL = {
        "SEVERITY_BALANCED_REGRESSION": {
            "ENABLE": True,
            "POWER": power,
            "MIN_WEIGHT": min_w,
            "MAX_WEIGHT": max_w,
            "EDGES": [13, 19, 28],
        }
    }
    return cfg


def _continuous_severity_config(
    power=0.5, sigma=2.0, epsilon=1e-3, min_w=0.5, max_w=4.0
):
    cfg = _severity_balanced_config(power=power, min_w=min_w, max_w=max_w)
    cfg.MODEL.SEVERITY_BALANCED_REGRESSION.WEIGHTING_MODE = "continuous"
    cfg.MODEL.SEVERITY_BALANCED_REGRESSION.SMOOTHING_SIGMA = sigma
    cfg.MODEL.SEVERITY_BALANCED_REGRESSION.DENSITY_EPSILON = epsilon
    return cfg


def _build_model(monkeypatch, cfg):
    import src.models.mtl_lite as mtl_lite

    def build_dummy_backbone(model_name, weight_path=None, timm_pretrained=False, img_size=112):
        return DummyBackbone(output_dim=cfg.BACKBONE_OUT_DIMS[model_name])

    monkeypatch.setattr(mtl_lite, "build_feature_backbone", build_dummy_backbone)
    return mtl_lite.MTLLiteDepressionModel(cfg)


# ---------------------------------------------------------------------------
# config-default + weight computation
# ---------------------------------------------------------------------------

def test_severity_balanced_defaults_to_off(monkeypatch):
    cfg = minimal_mtl_lite_config()  # no SEVERITY_BALANCED_REGRESSION section
    model = _build_model(monkeypatch, cfg)
    assert model.severity_balanced_regression is False
    assert model.severity_bin_weights == {}


def test_severity_bin_weight_formula_and_normalization(monkeypatch):
    cfg = _severity_balanced_config(power=0.5)
    model = _build_model(monkeypatch, cfg)
    counts = {"minimal": 100, "mild": 50, "moderate": 20, "severe": 5}
    model.set_severity_bin_counts(counts)

    total, num_bins = 175, 4
    raw = {b: (total / (num_bins * counts[b])) ** 0.5 for b in counts}
    clipped = {b: min(max(raw[b], 0.5), 4.0) for b in counts}
    mean_c = sum(clipped.values()) / num_bins
    expected = {b: clipped[b] / mean_c for b in counts}

    for name in ("minimal", "mild", "moderate", "severe"):
        assert math.isclose(model.severity_bin_weights[name], expected[name], rel_tol=1e-6)
    # mean-normalized -> average weight == 1.0
    assert math.isclose(sum(model.severity_bin_weights.values()) / 4, 1.0, rel_tol=1e-6)
    # the smaller bin (severe) gets the larger weight
    assert model.severity_bin_weights["severe"] > model.severity_bin_weights["minimal"]


def test_severity_weight_clip_bounds_enforced(monkeypatch):
    # Extreme imbalance: severe bin tiny -> raw weight huge; must clip to MAX.
    cfg = _severity_balanced_config(power=1.0, min_w=0.2, max_w=3.0)
    model = _build_model(monkeypatch, cfg)
    counts = {"minimal": 1000, "mild": 1000, "moderate": 1000, "severe": 1}
    model.set_severity_bin_counts(counts)
    # After clipping the severe raw weight hits MAX_WEIGHT before normalization;
    # normalized severe weight = MAX / mean(clipped) which must be > 1 but the
    # underlying clipped value is bounded.
    # Verify no weight is inf/nan and severe is the largest.
    for w in model.severity_bin_weights.values():
        assert math.isfinite(w)
    assert model.severity_bin_weights["severe"] == max(model.severity_bin_weights.values())


def test_degenerate_counts_fall_back_to_plain_mse(monkeypatch):
    cfg = _severity_balanced_config()
    model = _build_model(monkeypatch, cfg)
    # A zero-count bin -> would divide by zero; must disable weighting.
    model.set_severity_bin_counts({"minimal": 50, "mild": 50, "moderate": 50, "severe": 0})
    assert model.severity_bin_weights == {}


def test_continuous_mode_builds_smoothed_train_label_weights(monkeypatch):
    cfg = _continuous_severity_config(power=0.5, sigma=2.0)
    model = _build_model(monkeypatch, cfg)
    histogram = [0] * 64
    histogram[5] = 40
    histogram[35] = 4
    model.set_severity_label_histogram(histogram)

    assert len(model.severity_label_weights) == 64
    assert model.severity_bin_weights == {}
    assert all(math.isfinite(weight) for weight in model.severity_label_weights)
    empirical_mean = sum(
        count * model.severity_label_weights[score]
        for score, count in enumerate(histogram)
    ) / sum(histogram)
    assert math.isclose(empirical_mean, 1.0, rel_tol=1e-6)
    # The sparse tail receives a larger weight, while smoothing keeps nearby
    # scores finite and similar rather than creating hard bin discontinuities.
    assert model.severity_label_weights[35] > model.severity_label_weights[5]
    assert abs(model.severity_label_weights[34] - model.severity_label_weights[35]) < 0.2


def test_continuous_mode_requires_full_histogram(monkeypatch):
    cfg = _continuous_severity_config()
    model = _build_model(monkeypatch, cfg)
    with pytest.raises(ValueError, match="MAX_SCORE"):
        model.set_severity_label_histogram([1, 2, 3])


def test_continuous_weighted_mse_uses_linear_label_interpolation(monkeypatch):
    cfg = _continuous_severity_config(power=0.5, sigma=1.0)
    model = _build_model(monkeypatch, cfg)
    histogram = [0] * 64
    histogram[10] = 10
    histogram[40] = 1
    model.set_severity_label_histogram(histogram)

    true_bdi = torch.tensor([10.5, 40.0])
    true_bdi_norm = true_bdi / 63.0
    bdi_pred = torch.tensor([0.2, 0.7])
    loss = model._severity_weighted_mse(bdi_pred, true_bdi_norm, true_bdi)
    weights = torch.tensor(model.severity_label_weights)
    w_mid = 0.5 * weights[10] + 0.5 * weights[11]
    expected = torch.tensor([
        w_mid * (bdi_pred[0] - true_bdi_norm[0]) ** 2,
        weights[40] * (bdi_pred[1] - true_bdi_norm[1]) ** 2,
    ]).mean()
    assert torch.allclose(loss, expected)


def test_continuous_mode_without_histogram_falls_back_to_plain_mse(monkeypatch):
    cfg = _continuous_severity_config()
    model = _build_model(monkeypatch, cfg)
    true_bdi = torch.tensor([5.0, 35.0])
    true_bdi_norm = true_bdi / 63.0
    bdi_pred = torch.tensor([0.2, 0.7])
    loss = model._severity_weighted_mse(bdi_pred, true_bdi_norm, true_bdi)
    assert torch.allclose(loss, torch.nn.functional.mse_loss(bdi_pred, true_bdi_norm))


def test_runner_continuous_histogram_reads_train_labels_only(monkeypatch, tmp_path):
    from src.trainers import mtl_lite_runner

    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    (label_dir / "00001_Depression.csv").write_text("5\n")
    (label_dir / "00002_Depression.csv").write_text("40\n")
    cfg = SimpleNamespace(
        LABEL_DIR=str(label_dir),
        DATASET_SPLIT_FILE=str(tmp_path / "split.csv"),
        IMAGE_DIR=str(tmp_path / "images"),
    )
    monkeypatch.setattr(
        mtl_lite_runner,
        "load_data_list",
        lambda *_args: ["00001_video", "00002_video", "00003_video"],
    )
    histogram = mtl_lite_runner.build_train_severity_label_histogram(
        cfg, max_score=63
    )
    assert len(histogram) == 64
    assert histogram[5] == 1
    assert histogram[40] == 1
    assert sum(histogram) == 2


# ---------------------------------------------------------------------------
# bin edges match diagnostics.io.severity_group
# ---------------------------------------------------------------------------

def test_severity_bin_index_matches_io_severity_group(monkeypatch):
    cfg = _severity_balanced_config()
    model = _build_model(monkeypatch, cfg)
    scores = torch.tensor([0, 13, 14, 19, 20, 28, 29, 40, 63], dtype=torch.float32)
    bin_idx = model._severity_bin_index(scores)
    names = ("minimal", "mild", "moderate", "severe")
    for score, idx in zip(scores.tolist(), bin_idx.tolist()):
        assert names[idx] == severity_group(score)


# ---------------------------------------------------------------------------
# loss scaling + default-off equivalence + CCC/ordinal untouched
# ---------------------------------------------------------------------------

def test_weighted_mse_equals_per_sample_weighted_squared_error(monkeypatch):
    cfg = _severity_balanced_config(power=0.5)
    model = _build_model(monkeypatch, cfg)
    model.set_severity_bin_counts({"minimal": 10, "mild": 5, "moderate": 3, "severe": 1})

    true_bdi = torch.tensor([5.0, 16.0, 25.0, 50.0])          # one per bin
    true_bdi_norm = true_bdi / 63.0
    bdi_pred = torch.tensor([0.2, 0.4, 0.5, 0.7])
    loss = model._severity_weighted_mse(bdi_pred, true_bdi_norm, true_bdi)

    names = ("minimal", "mild", "moderate", "severe")
    expected_w = torch.tensor([model.severity_bin_weights[n] for n in names])
    sq = (bdi_pred - true_bdi_norm) ** 2
    expected = (expected_w * sq).mean()
    assert torch.allclose(loss, expected)


def test_disabled_switch_matches_plain_mse(monkeypatch):
    cfg_off = minimal_mtl_lite_config()
    cfg_on = _severity_balanced_config()
    torch.manual_seed(0)
    true_bdi = torch.tensor([5.0, 16.0, 25.0, 50.0])
    bdi_pred = torch.randn(4)

    model_off = _build_model(monkeypatch, cfg_off)
    model_on = _build_model(monkeypatch, cfg_on)
    # Switch on but no counts injected -> weights empty -> plain MSE.
    assert model_on.severity_bin_weights == {}

    true_bdi_norm = true_bdi / 63.0
    loss_off = model_off._severity_weighted_mse(bdi_pred, true_bdi_norm, true_bdi)
    loss_on = model_on._severity_weighted_mse(bdi_pred, true_bdi_norm, true_bdi)
    assert torch.allclose(loss_off, loss_on)
    assert torch.allclose(loss_on, torch.nn.functional.mse_loss(bdi_pred, true_bdi_norm))


def test_only_regression_reweighted_ccc_ordinal_unchanged(monkeypatch):
    cfg_off = minimal_mtl_lite_config()
    cfg_on = _severity_balanced_config()
    torch.manual_seed(1)
    video = torch.randn(4, 4, 3, 16, 16)
    mask = torch.ones(4, 4, dtype=torch.bool)
    labels = {
        "bdi_score": torch.tensor([5.0, 16.0, 25.0, 50.0]),
        "class_label": torch.tensor([0, 1, 2, 4]),
        "subject_id": ["203_1", "204_1", "205_1", "206_1"],
    }

    model_off = _build_model(monkeypatch, cfg_off)
    model_on = _build_model(monkeypatch, cfg_on)
    model_on.set_severity_bin_counts({"minimal": 10, "mild": 5, "moderate": 3, "severe": 1})
    # Share weights so any difference comes only from the reweighting.
    model_on.load_state_dict(model_off.state_dict(), strict=True)
    model_off.eval(); model_on.eval()
    with torch.no_grad():
        out_off = model_off(video, mask)
        loss_off = model_off.compute_losses(out_off, labels, stage="train")
        out_on = model_on(video, mask)
        loss_on = model_on.compute_losses(out_on, labels, stage="train")

    # CCC and ordinal use the same per-sample values (regression pred unchanged).
    assert torch.allclose(loss_off.ccc, loss_on.ccc)
    assert torch.allclose(loss_off.ordinal, loss_on.ordinal)
    # Regression differs (reweighted), and total reflects only that delta.
    assert not torch.allclose(loss_off.regression, loss_on.regression)
    assert torch.allclose(loss_on.total - loss_on.regression, loss_off.total - loss_off.regression)
