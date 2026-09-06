"""Path registry: frozen read-only roots + explicit writable roots + sha helpers.

Resolution order for every root: explicit override argument > EVA_DI_* env var
> contract default.  ``~`` and ``${VAR}`` are expanded (the design doc promised
``${EVA_DI_*}``; the implementation used to expand only ``~``, audit R2-P2-5).
After resolution :meth:`PathSet.resolve_read_only` asserts existence of every
root the MODE consumes (weights are only needed for extraction, labels/norm
stats only for training, audit R2-P2-6); writable roots are created lazily by
their writers (never here).  Runtime code must not write anywhere else (design
doc section 1 rule 2 and section 4).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, fields
from pathlib import Path

from src.eva_di.contracts import (
    AVEC_ROOT_DEFAULT,
    BEHAVIOR_NORM_SUBDIR,
    CACHE_ROOT_SUBDIR,
    EVA_WEIGHT_REVISION,
    OF3_ROOT_DEFAULT,
)

ENV_PREFIX = "EVA_DI_"


def _env_or(name: str, fallback: str | None) -> str | None:
    value = os.environ.get(ENV_PREFIX + name)
    return value if value not in (None, "") else fallback


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive int, got {chunk_size!r}")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PathSet:
    """Resolved eva_di path registry.  Writable: cache_root/log_root/output_root."""

    of3_root: Path
    avec_root: Path
    split_file: Path
    label_dir: Path
    cache_root: Path
    weight_path: Path
    behavior_norm_root: Path
    log_root: Path
    output_root: Path

    READ_ONLY_ATTRS = (
        "of3_root", "avec_root", "split_file", "label_dir",
        "behavior_norm_root", "weight_path",
    )
    # mode -> roots that must exist for that entry point (audit R2-P2-6):
    # train/export never open the weight file; extraction never reads labels
    # or behavior-norm stats.  avec_root is only the DEFAULT PARENT of
    # split_file/label_dir/cache_root and is not itself existence-checked.
    REQUIRED_READ_ONLY = {
        "train": ("of3_root", "split_file", "label_dir", "behavior_norm_root"),
        "extract": ("of3_root", "split_file", "weight_path"),
    }
    _ROOT_IS_FILE = ("split_file", "weight_path")

    @classmethod
    def build(
        cls,
        *,
        overrides: dict[str, str | Path | None] | None = None,
        repo_root: Path | None = None,
    ) -> "PathSet":
        overrides = overrides or {}
        repo_root = Path(repo_root) if repo_root is not None else Path.cwd()

        def expand(value: str | Path) -> Path:
            return Path(os.path.expandvars(str(value))).expanduser()

        def pick(name: str, default: str | Path | None) -> Path:
            explicit = overrides.get(name)
            if explicit not in (None, ""):
                return expand(explicit)
            from_env = _env_or(name.upper(), None)
            if from_env is not None:
                return expand(from_env)
            if default is None:
                raise ValueError(f"no default and no override for path {name!r}")
            return expand(default)

        avec_root = pick("avec_root", AVEC_ROOT_DEFAULT)
        of3_root = pick("of3_root", OF3_ROOT_DEFAULT)
        return cls(
            of3_root=of3_root,
            avec_root=avec_root,
            split_file=pick("split_file", avec_root / "dataset_split.json"),
            label_dir=pick("label_dir", avec_root / "depression_labels"),
            cache_root=pick("cache_root", avec_root / CACHE_ROOT_SUBDIR),
            weight_path=pick(
                "weight_path",
                Path.home() / ".cache" / "huggingface" / "hub"
                / "models--timm--eva02_small_patch14_224.mim_in22k"
                / "snapshots" / EVA_WEIGHT_REVISION / "model.safetensors",
            ),
            behavior_norm_root=pick(
                "behavior_norm_root", of3_root / BEHAVIOR_NORM_SUBDIR
            ),
            log_root=pick("log_root", repo_root / "logs" / "eva_di"),
            output_root=pick("output_root", repo_root / "outputs" / "eva_di"),
        )

    def resolve_read_only(self, *, mode: str = "train") -> "PathSet":
        from src.eva_di.contracts import EvaDiPathError

        if mode not in self.REQUIRED_READ_ONLY:
            raise ValueError(f"unknown mode {mode!r}; expected one of "
                             f"{sorted(self.REQUIRED_READ_ONLY)}")
        for attr in self.REQUIRED_READ_ONLY[mode]:
            path = getattr(self, attr)
            if attr in self._ROOT_IS_FILE:
                if not path.is_file():
                    raise EvaDiPathError(f"read-only root {attr} missing (file): {path}")
            elif not path.is_dir():
                raise EvaDiPathError(f"read-only root {attr} missing (directory): {path}")
        return self

    def as_dict(self) -> dict[str, str]:
        return {f.name: str(getattr(self, f.name)) for f in fields(self)}
