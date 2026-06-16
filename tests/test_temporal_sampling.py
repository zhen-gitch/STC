import pytest

from src.datasets.temporal_sampling import (
    model_max_len,
    normalize_temporal_sampling_strategy,
    select_temporal_indices,
)


def test_model_max_len_uses_dataset_convention():
    assert model_max_len(max_seq_len=2000, sample_step=10) == 200
    assert model_max_len(max_seq_len=256, sample_step=1) == 256
    assert model_max_len(max_seq_len=3, sample_step=10) == 1


def test_stride_head_matches_historical_selection():
    indices = select_temporal_indices(30, sample_step=3, max_seq_len=12, strategy="stride_head")

    assert indices == [0, 3, 6, 9]


def test_uniform_spreads_indices_across_full_video():
    indices = select_temporal_indices(10, sample_step=1, max_seq_len=4, strategy="uniform")

    assert indices == [0, 3, 6, 9]


def test_first_middle_and_random_crops_return_contiguous_windows():
    first = select_temporal_indices(10, sample_step=1, max_seq_len=4, strategy="first")
    middle = select_temporal_indices(10, sample_step=1, max_seq_len=4, strategy="middle")
    random_a = select_temporal_indices(10, sample_step=1, max_seq_len=4, strategy="random", seed="video")
    random_b = select_temporal_indices(10, sample_step=1, max_seq_len=4, strategy="random", seed="video")

    assert first == [0, 1, 2, 3]
    assert middle == [3, 4, 5, 6]
    assert random_a == random_b
    assert len(random_a) == 4
    assert random_a == list(range(random_a[0], random_a[0] + 4))


def test_temporal_sampling_aliases_and_errors():
    assert normalize_temporal_sampling_strategy("default") == "stride_head"
    assert normalize_temporal_sampling_strategy("uniform_fixed") == "uniform"
    assert normalize_temporal_sampling_strategy("middle_crop") == "middle"

    with pytest.raises(ValueError, match="Unsupported"):
        normalize_temporal_sampling_strategy("unknown")
