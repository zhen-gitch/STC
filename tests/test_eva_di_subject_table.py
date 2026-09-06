"""subject_table: train-only table, -1 mapping, derange negative-control utility."""

import numpy as np
import pytest

from src.eva_di.subject_table import (build_subject_table, derange_subject_labels,
                                      identity_of, subject_of)


def test_label_key_is_session_and_identity_is_person():
    # two granularities, deliberately separate (decision 2026-09-05): labels
    # are session files (203_1_Depression.csv), identity classes are persons.
    assert subject_of("203_1_Freeform_video") == "203_1"
    assert identity_of("203_1_Freeform_video") == "203"


def test_table_is_train_only_and_maps_unseen_to_minus_one():
    table = build_subject_table(["203_1_Freeform_video", "205_2_Northwind_video"])
    assert table.classes == ("203", "205")
    # person granularity: ANY session of a train person maps (the attacker
    # question is whether held-out sessions of the same person stay
    # recognisable); unseen persons map to -1.
    assert table("203_1_Northwind_video") == 0
    assert table("203_9_Northwind_video") == 0
    assert table("999_1_Freeform_video") == -1
    assert len(table) == 2


def test_empty_train_rejected():
    with pytest.raises(ValueError):
        build_subject_table([])


def _freq(values):
    from collections import Counter
    return sorted(Counter(values).values())


def test_derange_is_consistent_fixed_point_free_bijection():
    # R2-P1-2 semantics: a class-space bijection -- every recording of one
    # subject gets the SAME new label, no class keeps its label, and the
    # per-class frequency multiset is preserved exactly (plan L77).
    labels = [0, 0, 1, 2, 2, 2, 3]
    a = derange_subject_labels(labels, np.random.default_rng(7))
    b = derange_subject_labels(labels, np.random.default_rng(7))
    assert a == b  # deterministic under the same seed
    assert _freq(a) == _freq(labels)
    for original, mapped in zip(labels, a):
        assert mapped != original          # fixed-point free
    consistency = {}
    for original, mapped in zip(labels, a):
        consistency.setdefault(original, set()).add(mapped)
    assert all(len(v) == 1 for v in consistency.values())  # one donor per class
    assert sorted(set(a)) == sorted(set(labels))           # onto the same class set


def test_derange_permutes_full_class_space_under_truncation():
    from collections import Counter as _Counter  # noqa: F401
    # a max_recordings-truncated train may show 1-2 classes; the bijection
    # still runs over ALL classes (n_classes=50 here)
    out = derange_subject_labels([3, 3, 3], np.random.default_rng(1), n_classes=50)
    assert out[0] == out[1] == out[2] != 3
    assert 0 <= out[0] < 50


def test_derange_single_class_and_empty_rejected():
    from src.eva_di.contracts import EvaDiSchemaError
    with pytest.raises(EvaDiSchemaError):
        derange_subject_labels([], np.random.default_rng(7))
    with pytest.raises(EvaDiSchemaError, match="at least|>= 2|needs"):
        derange_subject_labels([2, 2, 2], np.random.default_rng(7))
    with pytest.raises(EvaDiSchemaError, match="outside class space"):
        derange_subject_labels([0, 4], np.random.default_rng(7), n_classes=4)

