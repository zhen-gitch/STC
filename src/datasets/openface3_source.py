"""Read-only OpenFace 3.0 aligned RGB source manifests."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class OpenFace3Frame:
    """One valid, temporally indexed aligned RGB frame."""

    path: Path
    frame_index_one: int
    sha256: str


class OpenFace3Source:
    """Resolve OpenFace3 manifests without silently repairing them."""

    def __init__(
        self,
        root: str | Path,
        *,
        verify_hashes: bool = True,
        min_contiguous_frames: int = 2,
        min_retained_valid_ratio: float = 0.95,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"OpenFace3 root does not exist: {self.root}")
        if min_contiguous_frames < 1:
            raise ValueError("min_contiguous_frames must be positive")
        if not 0.0 < float(min_retained_valid_ratio) <= 1.0:
            raise ValueError("min_retained_valid_ratio must be in (0, 1]")
        self.verify_hashes = bool(verify_hashes)
        self.min_contiguous_frames = int(min_contiguous_frames)
        self.min_retained_valid_ratio = float(min_retained_valid_ratio)

    def _sample_dir(self, video_id: str) -> Path:
        candidates = (self.root / "samples" / video_id, self.root / video_id)
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError(
            f"OpenFace3 sample directory not found for {video_id!r} under {self.root}"
        )

    def _features_path(self, video_id: str) -> Path:
        path = self._sample_dir(video_id) / "features.csv"
        if not path.is_file():
            raise FileNotFoundError(f"OpenFace3 features.csv not found: {path}")
        return path

    def _resolve_image_path(self, sample_dir: Path, value: str) -> Path:
        relative = Path(value)
        candidates = [
            relative if relative.is_absolute() else self.root / relative,
            sample_dir / relative,
            sample_dir / "aligned" / relative.name,
        ]
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"OpenFace3 aligned image does not exist: {value!r} for {sample_dir.name}"
        )

    def read_rows(self, video_id: str) -> list[dict[str, str]]:
        sample_dir = self._sample_dir(video_id)
        rows: list[dict[str, str]] = []
        seen: set[int] = set()
        with self._features_path(video_id).open(
            "r", newline="", encoding="utf-8-sig"
        ) as handle:
            reader = csv.DictReader(handle)
            required = {
                "sample_id",
                "frame_index_one",
                "aligned_image_path",
                "aligned_image_sha256",
                "image_valid",
            }
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"OpenFace3 features.csv missing columns for {video_id}: "
                    f"{sorted(missing)}"
                )
            for row in reader:
                if row.get("sample_id") != video_id:
                    raise ValueError(
                        f"OpenFace3 sample_id mismatch: expected {video_id}, "
                        f"got {row.get('sample_id')!r}"
                    )
                try:
                    frame_index = int(row["frame_index_one"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid frame_index_one for {video_id}: "
                        f"{row.get('frame_index_one')!r}"
                    ) from exc
                if frame_index < 1 or frame_index in seen:
                    raise ValueError(
                        f"Duplicate or invalid frame_index_one for {video_id}: {frame_index}"
                    )
                seen.add(frame_index)
                if not _as_bool(row.get("image_valid")):
                    continue
                if not row.get("aligned_image_path") or not row.get(
                    "aligned_image_sha256"
                ):
                    raise ValueError(
                        f"Valid OpenFace3 frame has no image/hash for "
                        f"{video_id}:{frame_index}"
                    )
                image_path = self._resolve_image_path(
                    sample_dir, row["aligned_image_path"]
                )
                row["_resolved_image_path"] = str(image_path)
                row["_frame_index_one_int"] = str(frame_index)
                rows.append(row)
        rows.sort(key=lambda item: int(item["_frame_index_one_int"]))
        if not rows:
            raise ValueError(f"No valid OpenFace3 aligned frames for {video_id}")
        return rows

    @staticmethod
    def _runs(rows: Iterable[dict[str, str]]) -> list[list[dict[str, str]]]:
        runs: list[list[dict[str, str]]] = []
        for row in rows:
            frame_index = int(row["_frame_index_one_int"])
            if not runs or frame_index != int(runs[-1][-1]["_frame_index_one_int"]) + 1:
                runs.append([row])
            else:
                runs[-1].append(row)
        return runs

    def frame_entries(self, video_id: str) -> list[OpenFace3Frame]:
        rows = self.read_rows(video_id)
        eligible = [
            run
            for run in self._runs(rows)
            if len(run) >= self.min_contiguous_frames
        ]
        if not eligible:
            raise ValueError(
                f"No OpenFace3 contiguous run reaches "
                f"{self.min_contiguous_frames} frames for {video_id}"
            )
        run = max(
            eligible,
            key=lambda candidate: (
                len(candidate),
                -int(candidate[0]["_frame_index_one_int"]),
            ),
        )
        retained_ratio = len(run) / len(rows)
        if retained_ratio < self.min_retained_valid_ratio:
            raise ValueError(
                f"OpenFace3 contiguous coverage below threshold for {video_id}: "
                f"{retained_ratio:.6f} < {self.min_retained_valid_ratio:.6f}"
            )
        return [
            OpenFace3Frame(
                path=Path(row["_resolved_image_path"]),
                frame_index_one=int(row["_frame_index_one_int"]),
                sha256=row["aligned_image_sha256"].strip().lower(),
            )
            for row in run
        ]

    def verify_frame(self, frame: OpenFace3Frame) -> None:
        if not self.verify_hashes:
            return
        actual = _sha256_file(frame.path)
        if actual != frame.sha256:
            raise ValueError(
                f"OpenFace3 aligned image hash mismatch at frame "
                f"{frame.frame_index_one}: expected {frame.sha256}, got {actual}"
            )


__all__ = ["OpenFace3Frame", "OpenFace3Source"]
