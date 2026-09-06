"""Training entry point (EVA-DI-CODE-v1: code only; smoke/full runs are separate boundaries).

Enforces the mechanism invariants end to end:

* splits come from ``dataset_split.json`` and are ``{train, val}`` only; ``test``
  is refused at config parse and again here;
* severity weights, target/behavior normalization, subject table = train-only;
* SINGLE warmup on the CE-weight side (plan L61; user decision 2026-09-05);
  GRL lambda stays constant; config still pins t1==t3 warmup keys (R2-P1-1);
* identity heads are materialized BEFORE the optimizer parameter snapshot --
  otherwise the lazy heads would never be trained (audit R2-P0);
* identity heads are training-only (model gates on ``self.training``);
* every run writes ``run_provenance.json`` (SCHEMA_PROVENANCE) before training;
  a reused ``run_id`` directory is RESET (metrics.jsonl/checkpoints) instead of
  silently appending across runs (audit R2-P1-5);
* every JSON artifact is strict-JSON: non-finite floats become ``null`` and
  ``best_defined`` records whether any epoch was selectable.

This module is import-safe (no torch at import time beyond what the model needs)
so the CPU test suite can exercise the batch/loss/model graph; ``main`` is the
GPU/light-env entry and is excluded from coverage here.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np

from src.eva_di.config import EvaDiConfig, load_config
from src.eva_di.contracts import (
    EVA_FEATURE_DIM,
    EVA_WEIGHT_REVISION,
    SCHEMA_FRAME_CACHE,
    SCHEMA_PROVENANCE,
    EvaDiError,
)
from src.eva_di.paths import PathSet


def _json_safe(obj):
    """Recursively map non-finite floats to None so run artifacts are valid
    strict JSON (json.dumps' -Infinity/NaN literals broke json.load; R2-P2)."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.floating):
        obj = float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def build_provenance(cfg: EvaDiConfig, paths: PathSet, *, cache_manifest_sha: str,
                     split_file_sha: str, behavior_norm_sha256: str = "",
                     device: str = "cpu", run_stats: dict | None = None,
                     rerun_overwrite: bool = False) -> dict:
    import os
    import subprocess
    import sys

    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(["git", *args], cwd=str(Path.cwd()),
                                    capture_output=True, text=True, timeout=10)
        except Exception:
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    toplevel = git("rev-parse", "--show-toplevel")
    checkout_root = str(Path(__file__).resolve().parents[2])
    if toplevel is not None and toplevel != checkout_root:
        raise EvaDiError(
            f"training must run inside the code checkout: git toplevel "
            f"{toplevel!r} != {checkout_root!r} (provenance would record the "
            "wrong commit/dirty state)")
    git_commit = git("rev-parse", "HEAD")
    torch_version, cudnn_deterministic, device_name = "unknown", None, "n/a"
    try:
        import torch

        torch_version = torch.__version__
        cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
        if str(device).startswith("cuda") and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(torch.device(device))
    except Exception:
        pass
    return {
        "schema": SCHEMA_PROVENANCE,
        "plan_id": _plan_id(),
        "git_commit": git_commit,
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": None if git_commit is None else bool(git("status", "--porcelain")),
        "git_toplevel": toplevel,
        "argv": list(os.sys.argv),
        "python": sys.version,
        "torch_version": torch_version,
        "seed": cfg.run.seed,
        "device": device,
        "device_name": device_name,
        "precision": cfg.training.precision,
        "autocast_enabled": str(device).startswith("cuda"),
        "cudnn_deterministic": cudnn_deterministic,
        "config_name": cfg.config_name,
        "config_source_sha256": cfg.source_sha256,
        "config_resolved_sha256": cfg.resolved_sha256,
        "cache_manifest_sha256": cache_manifest_sha,
        "split_file_sha256": split_file_sha,
        "behavior_norm_sha256": behavior_norm_sha256,
        "splits": list(cfg.data.splits),
        "ablation": cfg.ablation.input_mode,
        "identity": {"t1": cfg.identity.t1.enable, "t3": cfg.identity.t3.enable,
                     "derange": cfg.identity.derange,
                     "warmup_epochs": cfg.identity.t1.warmup_epochs},
        "rerun_overwrite": bool(rerun_overwrite),
        "run_stats": dict(run_stats or {}),
        "paths": paths.as_dict(),
    }


def _plan_id() -> str:
    from src.eva_di.contracts import PLAN_ID

    return PLAN_ID


def run_training(cfg: EvaDiConfig, *, device: str = "cuda") -> dict:  # pragma: no cover - runtime
    import torch

    from src.eva_di.batching import collate
    from src.eva_di.behavior import assert_stats_consistency, load_pinned_stats
    from src.eva_di.cache_reader import FrozenFrameCache
    from src.eva_di.dataset import build_recording_index, load_split_ids
    from src.eva_di.losses import (
        build_severity_weight_table,
        ce_masked,
        mse_norm,
        severity_weighted_l2,
        total_loss_terms,
    )
    from src.eva_di.model import DualStreamDI
    from src.eva_di.subject_table import derange_subject_labels

    if "test" in cfg.data.splits:
        raise EvaDiError("test split is locked; training may not touch it")
    if "val" not in cfg.data.splits:
        raise EvaDiError("data.splits must include 'val': early stopping and "
                         "model selection are val-only by design")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        # fail fast BEFORE any csv/cache IO (audit R2-P3: the check used to sit
        # after the full train-split recompute).
        raise EvaDiError(
            f"device {device!r} requested but CUDA is unavailable; "
            "pass device='cpu' explicitly if that is intended")
    torch.manual_seed(cfg.run.seed)
    np.random.seed(cfg.run.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    paths = PathSet.build(overrides=cfg.paths.overrides).resolve_read_only()
    paths.output_root.mkdir(parents=True, exist_ok=True)
    run_id = cfg.run.run_id or (
        f"{time.strftime('%Y%m%d', time.gmtime())}-{cfg.config_name}-seed{cfg.run.seed}"
    )
    run_dir = paths.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    # A pinned run_id re-run must not inherit the previous run's artifacts:
    # metrics.jsonl used to APPEND and a stale best.pt could pair one run's
    # weights with another run's summary (audit R2-P1-5/P2-6).
    rerun_overwrite = (run_dir / "run_provenance.json").exists()
    for stale in ("metrics.jsonl", "summary.json", "val_metrics.json",
                  "checkpoints/best.pt"):
        (run_dir / stale).unlink(missing_ok=True)
    import yaml

    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(cfg.resolved_payload, sort_keys=True), encoding="utf-8")

    # The cache fingerprint must match the config's extraction pins (design §6:
    # "train entry asserts against the manifest, then ignores the section").
    cache = FrozenFrameCache(
        paths.cache_root,
        expected={
            "schema_version": SCHEMA_FRAME_CACHE,
            "selection_policy": cfg.extraction.selection_policy,
            "frame_budget": cfg.extraction.frame_budget,
            "backbone": cfg.extraction.backbone,
            "weight_sha256": cfg.extraction.weight_sha256,
            "weight_revision": EVA_WEIGHT_REVISION,
            "input_size": cfg.extraction.input_size,
            "feature_dim": EVA_FEATURE_DIM,
            "feature_dtype": "float32",
        },
        expected_of3_root=paths.of3_root,
    )
    split_ids = load_split_ids(paths.split_file)
    stats = load_pinned_stats(paths.behavior_norm_root)  # source_sha256 = its own file
    index = build_recording_index(
        paths, splits=cfg.data.splits, behavior_stats=stats,
        max_recordings=cfg.data.max_recordings, cache=cache,
    )
    from src.eva_di.paths import sha256_file

    train_records = index.by_split("train")
    val_records = index.by_split("val")

    # Behavior-stat integrity (validation plan §1.2 / design §6).  User decision
    # 2026-09-05: ``reuse`` (production default) takes the OF3-pinned, frozen,
    # train-only stats AS-IS and MUST NOT pay a full train-split recompute on the
    # run path (that re-reads every train features.csv each run for no semantic
    # change -- the pinned stats are hash-pinned into provenance below).  The
    # independent consistency recompute is therefore an AUDIT lever, run only
    # when the config explicitly selects ``behavior_norm: recompute``.
    if cfg.data.behavior_norm == "recompute":
        from src.eva_di.of3_registry import load_video_record

        assert_stats_consistency(
            stats, [load_video_record(paths.of3_root, sid)
                    for sid in split_ids["train"]],
        )
        consistency_note = ("recompute audit PASSED (full train-split recompute "
                            "within tolerance; features fed from pinned stats)")
    else:
        consistency_note = ("reuse: OF3-pinned train-only stats accepted as-is; "
                            "consistency recompute skipped on the run path (user "
                            "decision 2026-09-05); set behavior_norm=recompute to audit")

    device_obj = torch.device(device)
    provenance = build_provenance(
        cfg, paths, cache_manifest_sha=cache.manifest_sha256,
        split_file_sha=sha256_file(paths.split_file),
        behavior_norm_sha256=stats.source_sha256,
        device=str(device_obj),
        run_stats={"n_train": len(train_records), "n_val": len(val_records),
                   "n_subjects": len(index.subject_table),
                   "max_recordings": cfg.data.max_recordings},
        rerun_overwrite=rerun_overwrite,
    )
    (run_dir / "run_provenance.json").write_text(
        json.dumps(_json_safe(provenance), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")

    sev_table_np, sev_weights = build_severity_weight_table(
        [r.bdi_score for r in train_records])
    weight_by_id = {r.sample_id: float(w) for r, w in zip(train_records, sev_table_np)}
    if cfg.identity.derange:
        rng = np.random.default_rng(cfg.run.seed)
        original = [r.subject for r in train_records]
        # subject-space bijection (fixed-point-free): every recording of one
        # subject gets the SAME new label; class count/frequency are preserved
        # exactly (audit R2-P1-2; plan L77).
        permuted = derange_subject_labels(
            original, rng, n_classes=len(index.subject_table))
        train_records = [dataclasses.replace(r, subject=int(s))
                         for r, s in zip(train_records, permuted)]

    model = DualStreamDI(
        cfg.model, input_mode=cfg.ablation.input_mode, n_subjects=len(index.subject_table),
        t1_enable=cfg.identity.t1.enable, t3_enable=cfg.identity.t3.enable,
        lambda1=cfg.identity.t1.lam, lambda3=cfg.identity.t3.lam,
    ).to(device_obj)
    # P0 (audit R2): heads are lazy; they MUST exist before the parameter
    # snapshot below, or AdamW/clip/zero_grad never see them and every DI arm
    # silently trains against a frozen random head.  No-op when both flags off.
    model.materialize_identity_heads()
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise EvaDiError("model exposes no trainable parameters")
    optimizer = torch.optim.AdamW(params, lr=cfg.training.optimizer["lr"],
                                  weight_decay=cfg.training.optimizer["weight_decay"])
    # bf16-mixed needs no loss scaling; the scaler stays disabled but keeps the
    # unscale_/clip/step sequence identical to a potential fp16 future.
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    # In-training multi-task gradient-conflict observer (EVA-DI-GRADAUDIT-v1).
    # Disabled by default -> zero behaviour change and no import at all.
    grad_auditor = None
    if cfg.identity.grad_audit.enable:
        from src.eva_di.grad_audit import GradientConflictAuditor

        grad_auditor = GradientConflictAuditor(
            model, every_n_steps=cfg.identity.grad_audit.every_n_steps,
            max_records=cfg.identity.grad_audit.max_records,
            log_dir=run_dir / "grad_audit")

    def make_batch(records):
        batch = collate(records, use_gap=cfg.model.use_gap, device=device_obj)
        weights = torch.tensor([weight_by_id[r.sample_id] for r in records],
                               dtype=torch.float32, device=device_obj)
        return batch, weights

    best_ccc, best_epoch = float("-inf"), 0
    best_record: dict | None = None
    patience_left = cfg.training.early_stop["patience"]
    history = []
    for epoch in range(1, cfg.training.epochs_max + 1):
        model.train()
        scale = min(1.0, epoch / cfg.identity.t1.warmup_epochs if cfg.identity.t1.warmup_epochs else 1.0)
        # SINGLE warmup (user decision 2026-09-05): the ramp w_k(e) is applied
        # EXACTLY ONCE -- on the CE weight inside losses.total_loss_terms
        # (validation plan Stage C w_k(e)=W_k*min(1,e/W)).  Design §5's per-epoch
        # set_lambda() hook stays wired but receives the CONSTANT lam; scaling it
        # as well (an earlier revision) made the reversal path quadratic in the
        # ramp (lam*w²).  scale remains recorded per epoch for provenance.
        model.set_lambda(lambda1=cfg.identity.t1.lam,
                         lambda3=cfg.identity.t3.lam)
        order = np.random.permutation(len(train_records))
        epoch_terms = {"total": 0.0, "reg": 0.0, "frame_ce": 0.0, "video_ce": 0.0}
        batches = 0
        for start in range(0, len(order), cfg.training.batch_recordings):
            chunk = [train_records[i] for i in order[start : start + cfg.training.batch_recordings]]
            batch, weights = make_batch(chunk)
            optimizer.zero_grad(set_to_none=True)
            def _loss_terms():
                out = model(batch)
                if cfg.training.reg_weighting == "plain":
                    reg = mse_norm(out.bdi_pred, batch.bdi)
                else:
                    reg = severity_weighted_l2(out.bdi_pred, batch.bdi, weights)
                frame_logits, video_logits = model.identity_logits(out, batch)
                frame_ce = (ce_masked(frame_logits, batch.subject, batch.frame_mask,
                                      cfg.identity.t1.label_smoothing)
                            if frame_logits is not None else out.bdi_pred.sum() * 0.0)
                video_ce = (ce_masked(video_logits, batch.subject, None,
                                      cfg.identity.t3.label_smoothing)
                            if video_logits is not None else out.bdi_pred.sum() * 0.0)
                return total_loss_terms(
                    reg_loss=reg, frame_ce=frame_ce, video_ce=video_ce, epoch=epoch,
                    w1=cfg.identity.t1.weight if cfg.identity.t1.enable else 0.0,
                    w3=cfg.identity.t3.weight if cfg.identity.t3.enable else 0.0,
                    warmup_epochs=cfg.identity.t1.warmup_epochs,
                )

            autocast = torch.autocast(device_type=device_obj.type,
                                      dtype=torch.bfloat16,
                                      enabled=(device_obj.type == "cuda"))
            # RNG snapshot for the GRADAUDIT replay (same dropout masks).
            rng_cpu = torch.random.get_rng_state()
            rng_gpu = (torch.cuda.get_rng_state()
                       if device_obj.type == "cuda" else None)
            with autocast:
                terms = _loss_terms()

            def _replay_raw_terms():
                """Replay with RNG restored and GRLs at -1 (raw task graph).
                The GRL Function snapshots lambda at forward time, so the
                ONLY way to get the un-reversed gradient of the SAME function
                evaluation is to re-forward under the restored RNG state."""
                with torch.random.fork_rng(
                        devices=([device_obj] if device_obj.type == "cuda" else [])):
                    torch.random.set_rng_state(rng_cpu)
                    if rng_gpu is not None:
                        torch.cuda.set_rng_state(rng_gpu)
                    saved = (model.grl_frame.lambda_, model.grl_video.lambda_)
                    model.grl_frame.lambda_ = -1.0
                    model.grl_video.lambda_ = -1.0
                    try:
                        with autocast:
                            raw = _loss_terms()
                    finally:
                        model.grl_frame.lambda_, model.grl_video.lambda_ = saved
                return raw

            if grad_auditor is not None:
                grad_auditor.capture(
                    terms,
                    [("frame_ce",
                      cfg.identity.t1.weight * scale if cfg.identity.t1.enable else 0.0,
                      cfg.identity.t1.lam),
                     ("video_ce",
                      cfg.identity.t3.weight * scale if cfg.identity.t3.enable else 0.0,
                      cfg.identity.t3.lam)],
                    _replay_raw_terms, epoch=epoch)
            scaler.scale(terms["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, cfg.training.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            for key in epoch_terms:
                epoch_terms[key] += float(terms[key].detach())
            batches += 1

        val_metrics = _evaluate(model, cfg, val_records, device_obj, index)
        record = {"epoch": epoch, "batches": batches,
                  "warmup_scale": float(scale),
                  "lambda1_eff": cfg.identity.t1.lam if cfg.identity.t1.enable else 0.0,
                  "lambda3_eff": cfg.identity.t3.lam if cfg.identity.t3.enable else 0.0,
                  **{f"train_{k}": epoch_terms[k] / max(batches, 1) for k in epoch_terms},
                  **val_metrics}
        history.append(record)
        (run_dir / "metrics.jsonl").open("a", encoding="utf-8").write(
            json.dumps(_json_safe(record)) + "\n")
        # last.pt mirrors trans B0 (behavior_b0.py:679): always checkpointed, so
        # a run whose val_ccc is undefined (constant predictions, tiny val set)
        # still yields a consumable artifact; best.pt remains the selection.
        torch.save({"model": model.state_dict(), "epoch": epoch,
                    "provenance": provenance}, run_dir / "checkpoints" / "last.pt")
        if val_metrics["val_ccc"] > best_ccc and not np.isnan(val_metrics["val_ccc"]):
            best_ccc, best_epoch = val_metrics["val_ccc"], epoch
            best_record = record
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "provenance": provenance}, run_dir / "checkpoints" / "best.pt")
            patience_left = cfg.training.early_stop["patience"]
        else:
            patience_left -= 1
        if patience_left <= 0:
            break

    grad_audit_note = None
    if grad_auditor is not None:
        from src.eva_di.grad_audit import summarize, write_report

        grad_audit_note = summarize(grad_auditor.records)
        grad_audit_note["report"] = str(
            write_report(grad_auditor.records, run_dir / "grad_audit"))

    summary = {"best_epoch": best_epoch, "best_val_ccc": best_ccc,
               "best_defined": best_record is not None,
               "epochs_run": len(history),
               "stopped_early": bool(patience_left <= 0 and len(history) < cfg.training.epochs_max),
               "val_n": (history[-1].get("val_n") if history else 0),
               "n_subjects": len(index.subject_table),
               "max_recordings": cfg.data.max_recordings,
               "final": history[-1] if history else {}, "run_dir": str(run_dir),
               "checkpoint": str(run_dir / "checkpoints" / ("best.pt" if best_record else "last.pt")),
               "behavior_stats_consistency": consistency_note,
               "grad_audit": grad_audit_note,
                   "reg_weighting": cfg.training.reg_weighting,
                   "severity_weights": sev_weights,
                   "reg_target_stats": {"mean": index.target_stats.mean,
                                        "std": index.target_stats.std}}
    (run_dir / "val_metrics.json").write_text(
        json.dumps(_json_safe({"best_epoch": best_epoch, "best_val_ccc": best_ccc,
                               "best_defined": best_record is not None,
                               "best_record": best_record,
                               "behavior_stats_consistency": consistency_note}),
                   ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")
    if grad_audit_note and grad_audit_note.get("grl_violations"):
        msg = (f"GRL reversal-scale check flagged {grad_audit_note['grl_violations']} "
               f"head-steps (max_abs_scale_dev={grad_audit_note.get('grl_max_abs_scale_dev')}; "
               f"replay_bitexact={grad_audit_note.get('replay_bitexact_all')}; "
               f"see {grad_audit_note['report']})")
        if cfg.identity.grad_audit.on_grl_violation == "fail":
            raise EvaDiError(msg + " -> on_grl_violation=fail")
        grad_audit_note["violation_note"] = msg + " -> on_grl_violation=report (training continued)"
    return summary


def _evaluate(model, cfg, records, device, index) -> dict:
    """FP32 (no autocast) validation pass; metrics on the clamped BDI scale.

    Val runs at fp32 on purpose (batch=1, negligible cost): bf16 quantization
    of the reg-head output moved val predictions by ~0.04 BDI steps (R2-P3).
    Dataset/collate guarantee every retained recording has >=1 valid frame, so
    no per-recording skip branch lives here anymore (R2-C5).
    """
    import torch

    from src.eva_di.batching import collate

    model.eval()
    y, y_hat = [], []
    with torch.no_grad():
        for rec in records:
            batch = collate([rec], use_gap=cfg.model.use_gap, device=device)
            out = model(batch)
            pred = float(out.bdi_pred[0].float().cpu())
            if not math.isfinite(pred):
                # CPython max(0.0, nan) == 0.0: a divergent epoch would
                # otherwise enter the metrics as a "legit 0" prediction and
                # could even win best-model selection (audit R2-C4).
                raise EvaDiError(f"non-finite val prediction for {rec.sample_id}")
            y.append(rec.bdi_score)
            # Stage C contract: metrics are computed on the BDI scale clamped to
            # the valid instrument range [0, 63].
            y_hat.append(min(63.0, max(0.0, pred * index.target_stats.std
                                       + index.target_stats.mean)))
    from src.eva_di.metrics import ccc as _ccc
    from src.eva_di.metrics import mae as _mae
    from src.eva_di.metrics import pcc as _pcc
    from src.eva_di.metrics import rmse as _rmse

    if not y:
        return {"val_ccc": float("nan"), "val_mae": float("nan"),
                "val_rmse": float("nan"), "val_pcc": float("nan"), "val_n": 0}
    ya, yha = np.asarray(y), np.asarray(y_hat)
    return {"val_ccc": _ccc(ya, yha), "val_mae": _mae(ya, yha),
            "val_rmse": _rmse(ya, yha), "val_pcc": _pcc(ya, yha), "val_n": len(y)}


def main(argv=None) -> int:  # pragma: no cover - runtime entry
    parser = argparse.ArgumentParser(description="Train the eva_di dual-stream DI model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if "envs/light/" not in __import__("sys").executable.replace("\\", "/"):
        raise EvaDiError(f"training must run in conda env 'light'; got {__import__('sys').executable}")
    cfg = load_config(args.config)
    if cfg.run.mode != "train":
        raise EvaDiError(f"config run.mode={cfg.run.mode!r} is not a train run")
    print(json.dumps(_json_safe(run_training(cfg, device=args.device)), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
