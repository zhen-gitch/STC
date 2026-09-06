"""Oracle equivalence harness (EVA-DI-EXTRACT-v1 runtime).

Purpose (validation plan section 3 row 6 / design section 10): prove that
``Eva02FrozenEncoder``'s re-implemented transform + forward is numerically
equivalent to the read-only RIB-Former reference encoder, per frame,
``cos(cls) >= 0.999``, before any downstream result is trusted.  Evidence lands
under ``cache_root/_equivalence`` and is never part of a metric set.

Faithful reference chain (cited, NOT imported into this process):
``trans/src/ribformer/data/clips.py:1485-1516`` (cv2 decode -> float32/255) and
``trans/src/ribformer/models/image.py:300-333`` (bicubic ``_transform`` +
pretrained mean/std + ``forward_features``).  It runs in a SUBPROCESS with
``cwd/PYTHONPATH`` pointed at the trans checkout so the two ``src`` trees never
collide; the trans package is never imported inside training/inference.

Our side reuses the runner's exact PIL loader and the encoder with
``autocast="none"`` (fp32) so the only differences measured are decode library
(PIL vs cv2) and re-implementation drift -- which is precisely what the gate
exists to bound.  Frame sources are the SAME cached, sha-verified jpgs, taken
from the OF3 registry for samples already covered by the cache manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.eva_di.contracts import EVA_SOURCE_SIZE, EvaDiError, EvaDiPathError

COSINE_ACCEPT = 0.999
FRAMES_PER_VIDEO = 8
VIDEOS = 3
TRANS_ROOT_DEFAULT = "/home/zhen/code/trans"

REFERENCE_CITATION = ("trans/src/ribformer/data/clips.py:1485-1516 "
                      "(cv2 decode, float32/255); "
                      "trans/src/ribformer/models/image.py:300-333 "
                      "(bicubic _transform + mean/std + forward_features)")

_REFERENCE_SCRIPT = r'''
import sys
from pathlib import Path
import numpy as np, torch, cv2
trans_root, weight, frames_file, out, device = sys.argv[1:6]
sys.path.insert(0, str(Path(trans_root) / "src"))
from ribformer.models.image import ViTImageRegionEncoder  # noqa: E402
enc = ViTImageRegionEncoder(
    backbone_name="eva02_small_patch14_224.mim_in22k", pretrained=True,
    pretrained_weight_path=Path(weight)).to(device).eval()
paths = [Path(l) for l in Path(frames_file).read_text().split("\n") if l]
tens = []
for p in paths:
    decoded = cv2.imdecode(np.frombuffer(p.read_bytes(), dtype=np.uint8),
                           cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape != (112, 112, 3) or decoded.dtype != np.uint8:
        raise SystemExit(f"reference decode failed: {p}")
    rgb = np.ascontiguousarray(decoded[:, :, ::-1])
    tens.append(torch.from_numpy(rgb).permute(2, 0, 1).to(torch.float32).div_(255.0))
batch = torch.stack(tens).to(device)
with torch.no_grad():
    feats = enc.backbone.forward_features(enc._transform(batch))
np.savez(out, cls=feats[:, 0].float().cpu().numpy(),
         gap=feats[:, 1:].mean(dim=1).float().cpu().numpy())
'''


def equivalence_report_skeleton(*, cache_root: Path) -> dict:
    return {
        "status": "NOT_RUN",
        "reason": "oracle equivalence has not been executed for this cache yet",
        "accept": {"cosine_min": COSINE_ACCEPT, "videos": VIDEOS,
                   "frames_per_video": FRAMES_PER_VIDEO, "input": EVA_SOURCE_SIZE},
        "evidence_dir": str(Path(cache_root) / "_equivalence"),
    }


def select_oracle_frames(of3_root: Path, cache_sample_ids, *,
                         videos: int, frames_per_video: int) -> list[dict]:
    """First ``videos`` cache-covered samples x first ``frames_per_video``
    image-valid registry frames; every frame carries its registry sha."""
    from src.eva_di.of3_registry import load_video_record

    picked: list[dict] = []
    for sid in sorted(cache_sample_ids):
        if len(picked) // max(frames_per_video, 1) >= videos and picked:
            break
        rec = load_video_record(of3_root, sid)
        chosen = 0
        for i in range(rec.n_rows):
            if not bool(rec.image_valid[i]):
                continue
            rel = rec.aligned_image_path[i]
            path = Path(rel) if Path(rel).is_absolute() else Path(of3_root) / rel
            picked.append({"sample_id": sid, "frame_index": int(rec.frame_index_zero[i]),
                           "path": str(path), "sha256": rec.aligned_image_sha256[i]})
            chosen += 1
            if chosen >= frames_per_video:
                break
        if len(picked) // frames_per_video >= videos:
            break
    if len(picked) < videos * frames_per_video:
        raise EvaDiError(
            f"only {len(picked)} oracle frames available from {len(cache_sample_ids)} "
            f"cached samples; need {videos * frames_per_video}")
    return picked[: videos * frames_per_video]


def verify_frame_shas(frames: list[dict]) -> None:
    for f in frames:
        digest = hashlib.sha256(Path(f["path"]).read_bytes()).hexdigest()
        if digest != f["sha256"]:
            raise EvaDiPathError(
                f"{f['sample_id']} frame {f['frame_index']}: aligned jpg sha256 "
                f"drifted ({digest} != registry {f['sha256']})")


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b / (na * nb))


def compare_and_record(frames: list[dict], our: tuple[np.ndarray, np.ndarray],
                       ref: tuple[np.ndarray, np.ndarray], *,
                       evidence_dir: Path, extra: dict | None = None) -> dict:
    """Per-frame cos(cls/gap), accept only if min cos(cls) >= COSINE_ACCEPT;
    evidence JSON is written on BOTH outcomes (a failure is also evidence)."""
    our_cls, our_gap = our
    ref_cls, ref_gap = ref
    if our_cls.shape != ref_cls.shape or our_cls.shape[0] != len(frames):
        raise EvaDiError(
            f"oracle shape mismatch: ours {our_cls.shape} ref {ref_cls.shape} "
            f"frames {len(frames)}")
    rows = [{"sample_id": f["sample_id"], "frame_index": f["frame_index"],
             "sha256": f["sha256"],
             "cos_cls": _cos(our_cls[i], ref_cls[i]),
             "cos_gap": _cos(our_gap[i], ref_gap[i])}
            for i, f in enumerate(frames)]
    min_cls = min(r["cos_cls"] for r in rows)
    report = {
        "status": "PASSED" if min_cls >= COSINE_ACCEPT else "FAILED",
        "accept": {"cosine_min": COSINE_ACCEPT, "input": EVA_SOURCE_SIZE},
        "reference": REFERENCE_CITATION,
        "n_frames": len(rows),
        "min_cos_cls": min_cls,
        "mean_cos_cls": math.fsum(r["cos_cls"] for r in rows) / len(rows),
        "min_cos_gap": min(r["cos_gap"] for r in rows),
        "frames": rows,
        "run_at": datetime.now(timezone.utc).isoformat(),
    }
    report.update(extra or {})
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (evidence_dir / f"oracle_equivalence_{stamp}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")
    return report


def _reference_features(frames: list[dict], *, trans_root: Path, weight_path: Path,
                        device: str) -> tuple[np.ndarray, np.ndarray]:
    with tempfile.TemporaryDirectory(prefix="eva_di_oracle_") as td:
        frames_file = Path(td) / "frames.txt"
        frames_file.write_text("\n".join(f["path"] for f in frames), encoding="utf-8")
        script = Path(td) / "reference_chain.py"
        script.write_text(_REFERENCE_SCRIPT, encoding="utf-8")
        out_npz = Path(td) / "ref.npz"
        env = dict(os.environ, PYTHONPATH=str(Path(trans_root) / "src"),
                   HF_HUB_OFFLINE="1")
        proc = subprocess.run(
            [sys.executable, str(script), str(trans_root), str(weight_path),
             str(frames_file), str(out_npz), device],
            cwd=str(trans_root), env=env, timeout=1800)
        if proc.returncode != 0:
            raise EvaDiError(
                f"reference subprocess failed (rc={proc.returncode}); the trans "
                "checkout or its env is not usable for the oracle gate")
        with np.load(out_npz) as data:
            return data["cls"].astype(np.float64), data["gap"].astype(np.float64)


def run_oracle_equivalence(*, of3_root: Path, cache_root: Path, weight_path: Path,
                           device: str = "cuda", trans_root: str = TRANS_ROOT_DEFAULT,
                           videos: int = VIDEOS, frames_per_video: int = FRAMES_PER_VIDEO,
                           our_encode=None, ref_encode=None) -> dict:
    """Full gate. ``our_encode``/``ref_encode`` are injectable callables
    (frames -> (cls, gap) float arrays) for hermetic testing; production uses
    the real encoder and the guarded trans subprocess."""
    from src.eva_di.cache_reader import CacheReader

    cache_ids = CacheReader(Path(cache_root)).sample_ids()
    frames = select_oracle_frames(Path(of3_root), cache_ids,
                                  videos=videos, frames_per_video=frames_per_video)
    verify_frame_shas(frames)
    evidence_dir = Path(cache_root) / "_equivalence"

    if our_encode is None:
        import torch
        from src.eva_di.extract.encoder import Eva02FrozenEncoder
        from src.eva_di.extract.runner import _load_image_tensor
        encoder = Eva02FrozenEncoder(Path(weight_path), device=device,
                                     autocast="none")

        def our_encode(fr: list[dict]) -> tuple[np.ndarray, np.ndarray]:
            array = _load_image_tensor([f["path"] for f in fr], Path(of3_root),
                                       "oracle")
            cls, gap = encoder.encode(torch.from_numpy(array))
            return cls.numpy().astype(np.float64), gap.numpy().astype(np.float64)

    if ref_encode is None:
        def ref_encode(fr: list[dict]) -> tuple[np.ndarray, np.ndarray]:
            return _reference_features(fr, trans_root=Path(trans_root),
                                       weight_path=Path(weight_path), device=device)

    our = our_encode(frames)
    ref = ref_encode(frames)
    return compare_and_record(frames, our, ref, evidence_dir=evidence_dir,
                              extra={"device": device,
                                     "trans_root": str(trans_root),
                                     "cache_root": str(cache_root)})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Oracle equivalence gate for the EVA-DI extraction cache")
    parser.add_argument("--config", required=True, help="extract-config yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trans-root", default=os.environ.get("EVA_DI_TRANS_ROOT",
                                                               TRANS_ROOT_DEFAULT))
    parser.add_argument("--videos", type=int, default=VIDEOS)
    parser.add_argument("--frames-per-video", type=int, default=FRAMES_PER_VIDEO)
    args = parser.parse_args(argv)

    from src.eva_di.config import load_config
    from src.eva_di.paths import resolve_read_only

    cfg = load_config(Path(args.config))
    paths = resolve_read_only(cfg, mode="extract")
    report = run_oracle_equivalence(
        of3_root=paths.of3_root, cache_root=paths.cache_root,
        weight_path=paths.weight_path, device=args.device,
        trans_root=args.trans_root, videos=args.videos,
        frames_per_video=args.frames_per_video)
    print(json.dumps({k: v for k, v in report.items() if k != "frames"},
                     ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
