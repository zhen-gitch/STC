#!/usr/bin/env python
"""Offline subject-attacker accuracy audit for Stage B identity risk (E1/E3).

Loads a trained MTL-Lite checkpoint, runs the test split through the
identity-adversarial ``subject_id_head``, and reports top-1 / top-3 subject
classification accuracy on the seen-subject subset (test subjects present in
the train-only subject table), plus coverage = seen / total.

This is the offline counterpart of the training-time identity CE: it measures
whether a trained classifier recovers subject id from ``z_dep`` (the shared
features), a B5 identity-risk signal distinct from A1 retrieval.  A successful
adversarial defense drives attacker accuracy toward chance while preserving BDI
utility.

Only meaningful for E1/E3 (``identity_adversarial=True``); E0/E2 have no
``subject_id_head`` in their checkpoint and the script writes a ``skipped``
summary instead of erroring, so the diagnose chain and aggregator handle them
uniformly.

Checkpoint load order matters: ``subject_id_head`` is built lazily by
:meth:`MTLLiteDepressionModel.set_subject_index`, so the head module must exist
BEFORE ``load_state_dict`` or its weights are silently skipped (strict=False).
The order here is: construct model -> set_subject_index (build head) ->
load_state_dict (fill weights) -> eval.

Example::

    python scripts/audit_subject_attacker.py \\
        --run-dir <LOG_DIR>/stage_b/e1_identity_adversarial/version_0 \\
        --ckpt best --split test \\
        --output-dir <RUN_DIR>/diagnostics/test/subject_attacker
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Offline subject-attacker accuracy audit (Stage B identity risk)."
    )
    parser.add_argument("--run-dir", required=True, help="MTL-Lite CSVLogger version directory.")
    parser.add_argument("--ckpt", default="best", help="'best', 'last', or an explicit checkpoint path.")
    parser.add_argument("--split", default="test", help="Split to evaluate (default test).")
    parser.add_argument("--output-dir", required=True, help="Directory for tables and report.")
    parser.add_argument("--config", default=None, help="Resolved config YAML (default <run-dir>/resolved_config.yaml).")
    parser.add_argument("--base-config", default="configs/avec2014_base.yaml", help="Shared base YAML fallback.")
    parser.add_argument("--local-paths", default="configs/local_paths.yaml", help="Machine-local YAML fallback.")
    parser.add_argument("--override", action="append", default=[], help="Optional fallback override YAML.")
    parser.add_argument("--allow-missing-local-paths", action="store_true")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or a torch device string.")
    parser.add_argument("--batch-size", type=int, default=1, help="Diagnostic dataloader batch size.")
    return parser


# Reuse the layerwise probe's config/device/data helpers to stay consistent.
def _load_helpers():
    from scripts.audit_layerwise_identity_probe import (  # type: ignore
        build_data_module,
        get_split_loader,
        load_config,
        resolve_device,
    )
    return build_data_module, get_split_loader, load_config, resolve_device


def _build_train_subject_index(cfgs):
    """Reuse the runner's train-only subject table builder."""
    from src.trainers.mtl_lite_runner import build_train_subject_index

    return build_train_subject_index(cfgs)


def _load_model_with_head(cfgs, checkpoint_path, device, subject_index):
    """Construct model, build the attacker head, THEN load checkpoint weights.

    The head is built lazily by ``set_subject_index``; if ``load_state_dict``
    ran first, the head module would not exist and its weights would be skipped
    (strict=False).  Order: model -> set_subject_index -> load_state_dict -> eval.
    """
    import torch

    from src.models.mtl_lite import MTLLiteDepressionModel

    model = MTLLiteDepressionModel(cfgs)
    model.set_subject_index(subject_index)  # builds subject_id_head when identity on

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Detect whether the checkpoint actually trained an identity head.  When
    # identity_adversarial was off (E0/E2) the head was never built, so there is
    # no subject_id_head.* key in state_dict -- mark skipped.
    has_head_keys = any(k.startswith("subject_id_head.") for k in state_dict)
    model_has_head = getattr(model, "subject_id_head", None) is not None
    if not (has_head_keys and model_has_head):
        return None, False  # not attacker-evaluable (E0/E2)

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, True


def _evaluate(model, data_loader, device, subject_index):
    """Run the split through the head; return per-query rows (seen/unseen).

    Walks the dataloader order (one video per row).  For each video:
      - forward to get identity_logits (None when the head was not active);
      - map the batch's subject_ids to train indices; unseen subjects -> None;
      - argmax / top-3 vs the true subject index for seen videos.
    """
    import torch

    inv_index = {idx: sid for sid, idx in subject_index.items()}
    per_query = []
    num_classes = len(subject_index)

    with torch.no_grad():
        for video_tensor, mask, labels in data_loader:
            video_tensor = video_tensor.to(device)
            mask = mask.to(device)
            outputs = model(video_tensor, mask)
            logits = outputs.identity_logits  # None when head inactive
            subject_ids = [str(s) for s in labels["subject_id"]]
            video_ids = labels.get("video_id", labels["subject_id"])
            if isinstance(video_ids, str):
                video_ids = [video_ids]
            else:
                video_ids = [str(v) for v in video_ids]

            if logits is None:
                # Head inactive for this batch (shouldn't happen for E1/E3 post-
                # load, but degrade gracefully).
                for vid, sid in zip(video_ids, subject_ids):
                    per_query.append({
                        "video_id": vid, "subject_id": sid, "seen": False,
                        "true_subject_index": None, "pred_subject_index": None,
                        "pred_top3_subject_indices": None,
                        "correct_top1": False, "correct_top3": False,
                    })
                continue

            logits = logits.detach().cpu()
            top3 = torch.topk(logits, k=min(3, logits.shape[-1]), dim=-1).indices.tolist()
            pred = logits.argmax(dim=-1).tolist()

            for i, sid in enumerate(subject_ids):
                true_idx = subject_index.get(sid)
                seen = true_idx is not None
                p = pred[i]
                t3 = top3[i]
                correct1 = seen and (p == true_idx)
                correct3 = seen and (true_idx in t3)
                per_query.append({
                    "video_id": video_ids[i],
                    "subject_id": sid,
                    "seen": seen,
                    "true_subject_index": true_idx,
                    "pred_subject_index": p if seen else None,
                    "pred_top3_subject_indices": t3 if seen else None,
                    "correct_top1": bool(correct1),
                    "correct_top3": bool(correct3),
                })
    return per_query, num_classes


def main():
    args = build_parser().parse_args()
    from src.diagnostics.io import ensure_dir, find_checkpoint
    from src.diagnostics.subject_attacker import (
        compute_attacker_metrics,
        skipped_summary,
        write_subject_attacker_per_query,
        write_subject_attacker_report,
        write_subject_attacker_summary,
    )

    build_data_module, get_split_loader, load_config, resolve_device = _load_helpers()

    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = ensure_dir(Path(args.output_dir))
    tables_dir = ensure_dir(output_dir / "tables")
    reports_dir = ensure_dir(output_dir / "reports")

    cfgs = load_config(args)
    device = resolve_device(args.device)
    subject_index = _build_train_subject_index(cfgs)
    print(f"[SUBJECT-ATTACKER] train subject classes: {len(subject_index)}")

    checkpoint_path = find_checkpoint(run_dir, args.ckpt)

    model, ok = _load_model_with_head(cfgs, checkpoint_path, device, subject_index)
    if not ok:
        # E0/E2 (identity off) or checkpoint has no head -- write skipped summary.
        summary = skipped_summary("no identity head (identity_adversarial off)")
        write_subject_attacker_summary(summary, tables_dir / "subject_attacker_summary.csv")
        write_subject_attacker_per_query([], tables_dir / "subject_attacker_per_query.csv")
        write_subject_attacker_report(reports_dir / "subject_attacker_report.md", summary, [])
        print("[SUBJECT-ATTACKER] no identity head in checkpoint; wrote skipped summary.")
        return

    data_module = build_data_module(cfgs, args.batch_size)
    data_loader = get_split_loader(data_module, args.split)

    print(f"[SUBJECT-ATTACKER] evaluating {args.split} split ...")
    per_query, num_classes = _evaluate(model, data_loader, device, subject_index)
    summary = compute_attacker_metrics(per_query, num_subject_classes=num_classes)

    write_subject_attacker_summary(summary, tables_dir / "subject_attacker_summary.csv")
    write_subject_attacker_per_query(per_query, tables_dir / "subject_attacker_per_query.csv")
    write_subject_attacker_report(reports_dir / "subject_attacker_report.md", summary, per_query)

    print(f"[SUBJECT-ATTACKER] evaluated={summary['num_evaluated']} "
          f"seen={summary['num_seen']} coverage={summary['coverage']} "
          f"top1={summary['top1_accuracy']} top3={summary['top3_accuracy']} "
          f"chance={summary['chance_top1']}")
    print(f"[SUBJECT-ATTACKER] output: {output_dir}")


if __name__ == "__main__":
    main()
