import pytest
from omegaconf import OmegaConf

from src.datasets.dataset import AVECDataModule, resolve_dataset_split_roles


_UNSET = object()


def _config(swap=_UNSET):
    values = {
        "EXTRACT_FEATURE": {"NUM_WORKERS": 0, "BATCH_SIZE": 1},
    }
    if swap is not _UNSET:
        values["DATASET"] = {"SWAP_VAL_TEST": swap}
    return OmegaConf.create(values)


def test_split_roles_default_preserves_original_mapping():
    expected = {
        "train": "train",
        "val": "val",
        "test": "test",
    }
    assert resolve_dataset_split_roles(_config()) == expected
    assert resolve_dataset_split_roles(_config(False)) == expected


def test_split_roles_swap_only_exchanges_validation_and_test():
    assert resolve_dataset_split_roles(_config(True)) == {
        "train": "train",
        "val": "test",
        "test": "val",
    }


@pytest.mark.parametrize("value", [1, "true", None])
def test_split_roles_rejects_non_boolean_switch(value):
    with pytest.raises(ValueError, match="SWAP_VAL_TEST must be a boolean"):
        resolve_dataset_split_roles(_config(value))


@pytest.mark.parametrize(
    ("swap", "expected"),
    [
        (False, ["train", "val", "test"]),
        (True, ["train", "test", "val"]),
    ],
)
def test_data_module_setup_uses_logical_role_mapping(monkeypatch, swap, expected):
    created = []

    class FakeDataset:
        def __init__(self, configs, dataset):
            created.append(dataset)

    monkeypatch.setattr("src.datasets.dataset.AVECDataset", FakeDataset)

    data_module = AVECDataModule(_config(swap))
    data_module.setup()

    assert created == expected
