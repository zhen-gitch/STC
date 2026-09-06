"""Train-only subject category table.

Classes are derived from ``video_id[:SUBJECT_PREFIX_LEN]`` (mirroring the
existing AVEC contract, ``src/datasets/dataset.py:277``) over the **physical
train split only** (docs/DUAL_LEVEL_IDENTITY_ADVERSARIAL_PLAN.md section 0.2).
Videos outside the train mapping return ``-1`` so consumers can exclude them at
*sample* level (never batch-wide).

Two granularities are deliberately separate (user decision 2026-09-05):
``subject_of`` uses ``SUBJECT_PREFIX_LEN = 5`` because the BDI *label files*
are session-level (``depression_labels/203_1_Depression.csv``), matching the
legacy contract ``src/datasets/dataset.py:276-280``; ``identity_of`` uses
``IDENTITY_PREFIX_LEN = 3`` so identity classes are PERSONS (``203``).
Persons repeat across splits by design (train∩val = 26): the identity head
is trained on train persons and the *external* attacker measures whether the
features still fingerprint a person in held-out sessions -- cross-split
person recurrence is the phenomenon under study, not a label leak (labels
and target stats remain train-only at session granularity).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.eva_di.contracts import (IDENTITY_PREFIX_LEN, SUBJECT_PREFIX_LEN,
                                  EvaDiSchemaError)


def subject_of(video_id: str) -> str:
    """Session-level label key (``203_1``); label files and BDI lookup only."""
    return str(video_id)[:SUBJECT_PREFIX_LEN]


def identity_of(video_id: str) -> str:
    """Person-level identity class key (``203``), user decision 2026-09-05."""
    return str(video_id)[:IDENTITY_PREFIX_LEN]


@dataclass(frozen=True)
class SubjectTable:
    classes: tuple[str, ...]  # sorted, train-only
    index: dict[str, int]

    def __len__(self) -> int:
        return len(self.classes)

    def __call__(self, video_id: str) -> int:
        return self.index.get(identity_of(video_id), -1)

    def classes_repr(self) -> list[str]:
        return list(self.classes)


def build_subject_table(train_videos: list[str]) -> SubjectTable:
    if not train_videos:
        raise ValueError("subject table requires a non-empty train split")
    classes = tuple(sorted({identity_of(video) for video in train_videos}))
    index = {name: i for i, name in enumerate(classes)}
    return SubjectTable(classes=classes, index=index)


def derange_subject_labels(subjects: list[int], rng, *,
                           n_classes: int | None = None) -> list[int]:
    """DI-DERM negative control: a fixed-point-free BIJECTION of the subject
    label space, applied consistently (every recording of one subject gets the
    same new label; audit R2-P1-2).

    The plan (DUAL_LEVEL_IDENTITY_ADVERSARIAL_PLAN.md L77) pins "labels
    deranged within train, preserving class count/frequency"; a class-space
    bijection preserves both exactly, while a positional per-recording shuffle
    would split each subject across several labels and merge unrelated ones --
    a different control.  This mirrors the repo's established subject-deranged
    convention (GLOBAL_LOCAL plan: "对 subject 做固定无自映射置换").

    ``n_classes`` overrides the class space (pass ``len(subject_table)`` so a
    ``data.max_recordings``-truncated run still permutes ALL classes).
    ``rng`` must be seeded.
    """
    array = np.asarray(subjects, dtype=np.int64)
    if array.size == 0:
        raise EvaDiSchemaError("cannot derange an empty subject list")
    classes = np.arange(int(n_classes)) if n_classes else np.unique(array)
    if classes.size < 2:
        raise EvaDiSchemaError(
            "DI-DERM needs >= 2 train subject classes; a single class admits "
            "no fixed-point-free permutation")
    if int(array.max()) >= int(classes.size):
        raise EvaDiSchemaError(
            f"subject label {int(array.max())} outside class space of size {classes.size}")
    identity = np.arange(classes.size)
    perm = None
    for _ in range(1000):
        candidate = rng.permutation(classes.size)
        if not np.any(candidate == identity):
            perm = candidate
            break
    if perm is None:  # pragma: no cover - probability ~1e-434 at 50 classes
        raise EvaDiSchemaError("could not draw a fixed-point-free permutation")
    remap = {int(classes[i]): int(classes[perm[i]]) for i in range(classes.size)}
    out = [remap[int(s)] for s in array.tolist()]
    in_counts = sorted(int(c) for c in np.bincount(array, minlength=classes.size))
    out_counts = sorted(int(c) for c in np.bincount(np.asarray(out, dtype=np.int64),
                                                    minlength=classes.size))
    if in_counts != out_counts:
        raise AssertionError("derangement changed the per-class frequency multiset")
    return out
