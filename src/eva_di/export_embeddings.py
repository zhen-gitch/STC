"""Export exit-P / exit-V embeddings for the *external* identity attacker.

The model carries **no identity metric**: identity is judged only by
``src/diagnostics/subject_attacker.py`` (fresh LOVO Ridge), pair-AUROC and A1
retrieval on the exported npz (mechanism section 6, design section 10).  Test
export is refused; locked-benchmark export is a future separate package.

Audit R2 hardening (P1-6): the export entry point now enforces the same
guards training does -- ``run.mode == "train"``, full cache fingerprint +
of3_root pin, and a cross-check of the checkpoint's recorded provenance
(ablation mode, identity flags, cache manifest sha) against the config, so an
ablated forward can no longer masquerade as the full model's features.  Every
npz carries the run/checkpoint/config provenance and is written atomically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.eva_di.config import load_config
from src.eva_di.contracts import (
    DECISION_SPLITS,
    EVA_FEATURE_DIM,
    EVA_WEIGHT_REVISION,
    SCHEMA_FRAME_CACHE,
    EvaDiError,
)
from src.eva_di.paths import PathSet, sha256_file

EXPORT_SCHEMA = "eva_di_embeddings_v1"


def _expected_cache_fingerprint(cfg) -> dict:
    """Same pin set as train.py's entry (kept in sync deliberately)."""
    return {
        "schema_version": SCHEMA_FRAME_CACHE,
        "selection_policy": cfg.extraction.selection_policy,
        "frame_budget": cfg.extraction.frame_budget,
        "backbone": cfg.extraction.backbone,
        "weight_sha256": cfg.extraction.weight_sha256,
        "weight_revision": EVA_WEIGHT_REVISION,
        "input_size": cfg.extraction.input_size,
        "feature_dim": EVA_FEATURE_DIM,
        "feature_dtype": "float32",
    }


def _assert_checkpoint_matches(cfg, payload, cache) -> None:
    prov = payload.get("provenance") or {}
    if not isinstance(prov, dict) or not prov:
        return  # pre-provenance checkpoint: npz still records its own sha
    if prov.get("ablation") is not None and prov["ablation"] != cfg.ablation.input_mode:
        raise EvaDiError(
            f"checkpoint was trained with ablation {prov['ablation']!r} but "
            f"config declares {cfg.ablation.input_mode!r}; exported features "
            "would be the wrong forward (audit R2-P1-6)")
    ident = prov.get("identity") or {}
    if ident and (ident.get("t1") != cfg.identity.t1.enable
                  or ident.get("t3") != cfg.identity.t3.enable):
        raise EvaDiError(
            f"checkpoint identity flags {ident!r} disagree with config "
            f"t1={cfg.identity.t1.enable} t3={cfg.identity.t3.enable}")
    ckpt_cache = prov.get("cache_manifest_sha256")
    if ckpt_cache and ckpt_cache != cache.manifest_sha256:
        raise EvaDiError(
            f"checkpoint was trained on cache manifest {ckpt_cache} but the "
            f"resolved cache is {cache.manifest_sha256}; features would mix "
            "two extraction generations")


def export_embeddings(cfg, *, split: str, checkpoint: Path, device: str = "cpu") -> list[Path]:
    import torch

    from src.eva_di.batching import collate
    from src.eva_di.behavior import load_pinned_stats
    from src.eva_di.cache_reader import FrozenFrameCache
    from src.eva_di.dataset import build_recording_index
    from src.eva_di.model import DualStreamDI

    if getattr(cfg.run, "mode", None) != "train":
        raise EvaDiError(
            f"embedding export requires a train-mode config, got "
            f"{cfg.run.mode!r} (audit R2-P1-6: extract/foreign configs carry "
            "different identity semantics)")
    if split not in DECISION_SPLITS:
        raise EvaDiError(f"embedding export split must be one of {DECISION_SPLITS}, got {split!r}")
    paths = PathSet.build(overrides=cfg.paths.overrides).resolve_read_only()
    cache = FrozenFrameCache(paths.cache_root,
                             expected=_expected_cache_fingerprint(cfg),
                             expected_of3_root=paths.of3_root)
    stats = load_pinned_stats(paths.behavior_norm_root)
    # One index build only: subject tables/target stats are train-only, so the
    # index is always assembled over (train, split) and the export consumes
    # by_split(split).  (An earlier split-only build crashed the train-split
    # requirement inside build_recording_index for split='val'.)
    index = build_recording_index(
        paths, splits=("train", split), behavior_stats=stats, cache=cache,
        max_recordings=cfg.data.max_recordings,
    )
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise EvaDiError(
            f"device {device!r} requested but CUDA is unavailable; "
            "pass device='cpu' explicitly if that is intended")
    device_obj = torch.device(device)
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise EvaDiError(f"checkpoint missing: {checkpoint}")
    checkpoint_sha = sha256_file(checkpoint)
    # Mirror the training architecture exactly (identity flags + subject count)
    # so a strict load matches the checkpoint's state_dict.  The lazy identity
    # heads are materialized before loading even though the export only reads
    # exit-P / exit-V -- otherwise their saved weights are "unexpected keys".
    # Per-flag heads (model.py) make strict loading itself flag-sensitive now.
    model = DualStreamDI(
        cfg.model, input_mode=cfg.ablation.input_mode, n_subjects=len(index.subject_table),
        t1_enable=cfg.identity.t1.enable, t3_enable=cfg.identity.t3.enable,
    ).to(device_obj)
    model.materialize_identity_heads()
    try:
        payload = torch.load(checkpoint, map_location=device_obj, weights_only=False)
    except TypeError as exc:  # older torch without weights_only kwarg
        raise EvaDiError(f"cannot load checkpoint {checkpoint}: {exc}") from exc
    if not isinstance(payload, dict) or "model" not in payload:
        raise EvaDiError(
            f"checkpoint payload must be a dict with a 'model' state_dict; "
            f"got {type(payload).__name__}")
    _assert_checkpoint_matches(cfg, payload, cache)
    model.load_state_dict(payload["model"])
    model.eval()
    out = checkpoint.parent.parent / "embeddings"
    out.mkdir(parents=True, exist_ok=True)
    p_mean: list[np.ndarray] = []
    v_h0: list[np.ndarray] = []
    ids: list[str] = []
    prov = payload.get("provenance") or {}
    with torch.no_grad():
        for rec in index.by_split(split):
            batch = collate([rec], use_gap=cfg.model.use_gap, device=device_obj)
            out_fwd = model(batch)
            mask = batch.frame_mask.unsqueeze(-1).to(out_fwd.p.dtype)
            counts = mask.sum(1)
            if bool((counts <= 0).any()):
                raise EvaDiError(f"{rec.sample_id}: zero valid frames at export")
            # masked mean over time -> one [D] vector per recording
            p_mean.append(((out_fwd.p * mask).sum(1) / counts)[0].float().cpu().numpy())
            v_h0.append(out_fwd.h0[0].float().cpu().numpy())
            ids.append(rec.sample_id)
    provenance_keys = {
        "export_schema": EXPORT_SCHEMA,
        "run_id": cfg.run.run_id or "",
        "config_name": cfg.config_name,
        "config_source_sha256": cfg.source_sha256,
        "config_resolved_sha256": cfg.resolved_sha256,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "git_commit": str(prov.get("git_commit") or ""),
        "seed": int(cfg.run.seed),
        "input_mode": cfg.ablation.input_mode,
        "derange": bool(cfg.identity.derange),
        "precision": "fp32 (eval, autocast disabled at export)",
        "cache_manifest_sha256": cache.manifest_sha256,
        "max_recordings": -1 if cfg.data.max_recordings is None else int(cfg.data.max_recordings),
        "n_exported": len(ids),
    }
    empty = np.zeros((0, cfg.model.proj_dim), np.float32)
    dests = []
    for name, vectors in ((f"p_mean_{split}.npz", p_mean), (f"v_h0_{split}.npz", v_h0)):
        dest = out / name
        # np.savez appends ".npz" unless the name already ends with it -- the
        # tmp name MUST keep the suffix for atomic replace to find the file.
        tmp = out / f".{name}.tmp.npz"
        np.savez(tmp, schema=EXPORT_SCHEMA, split=split,
                 sample_ids=np.asarray(ids),
                 embeddings=np.stack(vectors) if vectors else empty,
                 **provenance_keys)
        tmp.replace(dest)
        dests.append(dest)
    return dests


def main(argv=None) -> int:  # pragma: no cover - runtime entry
    parser = argparse.ArgumentParser(description="Export DI exit embeddings for the external attacker")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=list(DECISION_SPLITS))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if "envs/light/" not in __import__("sys").executable.replace("\\", "/"):
        raise EvaDiError(f"export must run in conda env 'light'; got {__import__('sys').executable}")
    cfg = load_config(args.config)
    dests = export_embeddings(cfg, split=args.split, checkpoint=Path(args.checkpoint),
                              device=args.device)
    for dest in dests:
        print(str(dest))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
