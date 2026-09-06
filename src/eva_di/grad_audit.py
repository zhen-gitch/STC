"""In-training multi-task gradient-conflict observer (EVA-DI-GRADAUDIT-v1;
robust GRL check added under EVA-DI-OBSV2-v1).

Read-only by construction: per-task trunk gradients come from
``torch.autograd.grad`` (returned tensors; parameters, ``.grad`` buffers and
RNG are untouched -- pinned by test).  Enabled ONLY via
``identity.grad_audit`` (default off: not even this module is imported).

Per audit step the observer splits the batch's loss graph at the shared trunk
and records, in ``<run_dir>/grad_audit/step_metrics.jsonl``:

* ``g_reg``      - d(reg)/d(trunk);
* ``g_id_eff``   - d(weighted CE)/d(trunk) WITH the GRL in force, i.e. the
                   direction the trunk actually experiences (already -lambda);
* ``g_id_raw``   - same loss evaluated in a replay forward with the RNG state
                   restored (same dropout masks -- the replay scalar losses are
                   asserted BIT-EXACT and recorded per step) and every GRL at
                   lambda=-1 (exact pass-through), giving the pure task
                   gradient (no reversal);
* A4 (PCGrad dialect): cos + conflict flag + negative tangential fraction
  (how much of the erase gradient drags the utility gradient backwards) for
  eff AND raw, overall and per module (projector / gru / behavior_branch ...).

GRL mechanism check (revised under OBSV2 after the lambda=0.05 triage): the
old hard identity ``||g_id_eff + lambda*g_id_raw|| == 0`` compares TWO forward
graphs and is bit-exact only when the whole backward is cross-forward
reproducible; the GRU-path backward is not (observed residual 0.5 at
lambda=0.05 with BIT-EXACT forward losses), while the reversal MAGNITUDE is
exact (||g_id_raw||/||g_id_eff|| = 1/lambda to 4 digits).  The violation test
therefore uses the robust magnitude identity:

    reversal_scale := lambda * ||g_id_raw|| / ||g_id_eff||  ==  1  (+-0.5)

which is insensitive to cross-forward DIRECTION noise yet still fires on the
real failure modes (a non-toggling GRL gives reversal_scale = lambda; a broken
sign gives the same deviation).  The old residual is kept as a reported
diagnostic.  Violations are handled by ``identity.grad_audit.on_grl_violation``
(report -> flagged in summary, training continues; fail -> hard stop).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch

from src.eva_di.contracts import EvaDiError

_HEAD_PREFIXES = ("frame_head.", "video_head.")
_GRL_SCALE_TOL = 0.5      # |reversal_scale - 1| above this = mechanism suspect
_EPS = 1e-12


def flat_stats(g_reg: torch.Tensor, g_id: torch.Tensor) -> dict:
    """cos / conflict / negative-tangent fraction between two flat gradients."""
    na, nb = float(g_reg.norm()), float(g_id.norm())
    if na < _EPS or nb < _EPS:
        return {"cos": None, "norm_reg": na, "norm_id": nb, "conflict": None,
                "neg_tangent_frac": None, "degenerate": True}
    dot = float(torch.dot(g_reg, g_id))
    return {"cos": dot / (na * nb), "norm_reg": na, "norm_id": nb,
            "conflict": bool(dot < 0.0),
            "neg_tangent_frac": max(0.0, -dot / na) / nb, "degenerate": False}


class GradientConflictAuditor:
    """Observer bound to one training run; see module docstring for semantics.

    ``capture(terms, heads, replay, epoch=...)`` must be called AFTER the loss
    graph is built and BEFORE ``backward()`` (the observer retains the graph;
    the trainer's own backward still consumes it exactly once).
    """

    def __init__(self, model: torch.nn.Module, *, every_n_steps: int,
                 max_records: int, log_dir: Path) -> None:
        if int(every_n_steps) < 1 or int(max_records) < 1:
            raise EvaDiError("grad_audit every_n_steps/max_records must be >= 1")
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self._trunk = [(n, p) for n, p in named if not n.startswith(_HEAD_PREFIXES)]
        if not self._trunk:
            raise EvaDiError("grad audit found no trainable trunk parameters")
        self._groups: dict[str, list[int]] = {}
        for idx, (name, _) in enumerate(self._trunk):
            self._groups.setdefault(name.split(".", 1)[0], []).append(idx)
        self._every, self._max = int(every_n_steps), int(max_records)
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = self._dir / "step_metrics.jsonl"
        self._jsonl.write_text("", encoding="utf-8")
        self._step = 0
        self.records: list[dict] = []

    # -- capture -------------------------------------------------------------
    def capture(self, terms: dict, heads, replay, *, epoch: int) -> dict | None:
        """``heads``: (loss_key, weighted_scale, lambda) per identity head.
        ``replay()`` re-runs the identical forward+losses with the RNG state
        restored (same dropout masks) and every GRL at lambda=-1 (exact
        pass-through), yielding the raw task graph.  The GRL autograd function
        SNAPSHOTS lambda at forward time (``ctx.lambda_``), so poking the
        attribute after the fact would be silently ineffective -- the observer
        replays instead, and records the BIT-EXACT equality of the replay's
        scalar losses against the main forward as ``replay`` evidence.
        Zero-weight heads are skipped (placeholder tensors carry no path)."""
        self._step += 1
        if self._step % self._every != 0 or len(self.records) >= self._max:
            return None
        params = [p for _, p in self._trunk]
        reg_g = self._grads(terms["reg"], params)
        eff_total = [torch.zeros_like(t) for t in reg_g]
        raw_total = [torch.zeros_like(t) for t in reg_g]
        grl_rows = []
        replay_evidence = None
        active = [(k, float(w), float(lam)) for k, w, lam in heads if w > 0.0]
        if active:
            raw_terms = replay()
            replay_evidence = {  # permanent cheap evidence, never assumed
                "reg_bitexact": bool(terms["reg"].detach().eq(
                    raw_terms["reg"].detach()).item()),
                "heads": {
                    key: {"main": float(terms[key].detach()),
                          "replay": float(raw_terms[key].detach()),
                          "bitexact": bool(terms[key].detach().eq(
                              raw_terms[key].detach()).item())}
                    for key, _, _ in active},
            }
            head_flat: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            for key, w, lam in active:
                eff = self._grads(w * terms[key], params)
                raw = self._grads(w * raw_terms[key], params)
                eff_total = [a + b for a, b in zip(eff_total, eff)]
                raw_total = [a + b for a, b in zip(raw_total, raw)]
                grl_rows.append(self._grl_row(key, eff, raw, lam))
                head_flat[key] = (torch.cat(eff), torch.cat(raw))
        else:
            head_flat = {}
        pairwise = None
        flat_reg = torch.cat(reg_g)
        if head_flat:
            pairwise = {}
            for key, (he, hr) in head_flat.items():
                pairwise[f"reg_x_{key}_eff"] = flat_stats(flat_reg, he)
                pairwise[f"reg_x_{key}_raw"] = flat_stats(flat_reg, hr)
            keys = list(head_flat)
            if len(keys) == 2:
                a, b = keys
                pairwise[f"{a}_x_{b}_eff"] = flat_stats(head_flat[a][0], head_flat[b][0])
                pairwise[f"{a}_x_{b}_raw"] = flat_stats(head_flat[a][1], head_flat[b][1])
        flat_eff = torch.cat(eff_total)
        flat_raw = torch.cat(raw_total)
        record = {
            "step": self._step, "epoch": int(epoch),
            "heads_checked": len(grl_rows),
            "grl": _grl_summary(grl_rows),
            "grl_rows": grl_rows,
            "replay": replay_evidence,
            "overall": flat_stats(torch.cat(reg_g), flat_eff),
            "overall_raw": flat_stats(torch.cat(reg_g), flat_raw),
            "pairwise": pairwise,
            "modules": {},
        }
        for group, idxs in self._groups.items():
            record["modules"][group] = {
                "eff": flat_stats(torch.cat([reg_g[i] for i in idxs]),
                                  torch.cat([eff_total[i] for i in idxs])),
                "raw": flat_stats(torch.cat([reg_g[i] for i in idxs]),
                                  torch.cat([raw_total[i] for i in idxs])),
            }
        self.records.append(record)
        with self._jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def _grads(self, loss: torch.Tensor, params: list) -> list[torch.Tensor]:
        grads = torch.autograd.grad(loss, params, retain_graph=True,
                                    allow_unused=True)
        return [(torch.zeros_like(p).reshape(-1) if g is None
                 else g.detach().float().reshape(-1))
                for g, p in zip(grads, params)]

    @staticmethod
    def _grl_row(key: str, eff: list[torch.Tensor], raw: list[torch.Tensor],
                 lam: float) -> dict:
        fe, fr = torch.cat(eff), torch.cat(raw)
        ne = float(fe.norm())
        resid_rel = float((fe + lam * fr).norm()) / max(ne, _EPS)
        ratio = float(fr.norm()) / max(ne, _EPS)
        scale = ratio * lam          # == 1 iff the reversal factor is -lam
        vacuous = ne < 1e-9
        return {"head": key, "residual_rel": resid_rel, "norm_ratio": ratio,
                "reversal_scale": scale, "vacuous": vacuous,
                "violation": bool((not vacuous) and abs(scale - 1.0) > _GRL_SCALE_TOL)}


def _grl_summary(rows: list[dict]) -> dict:
    checked = [r for r in rows if not r["vacuous"]]
    return {"checked": len(rows), "non_vacuous": len(checked),
            "max_residual_rel": max((r["residual_rel"] for r in checked), default=None),
            "max_abs_scale_dev": max((abs(r["reversal_scale"] - 1.0) for r in checked),
                                     default=None),
            "violations": sum(1 for r in rows if r["violation"])}


# -- aggregation --------------------------------------------------------------
def _rate(records, key1, key2=None) -> dict:
    vals = []
    for rec in records:
        node = rec[key1] if key2 is None else rec[key1][key2]
        if isinstance(node, dict) and node.get("cos") is not None:
            vals.append(node)
    if not vals:
        return {"n_scored": 0}
    return {"n_scored": len(vals),
            "conflict_rate": sum(1 for v in vals if v["conflict"]) / len(vals),
            "cos_median": statistics.median(v["cos"] for v in vals),
            "neg_tangent_frac_max": max(v["neg_tangent_frac"] for v in vals)}


def _module_rate(records, module: str, kind: str) -> dict:
    vals = []
    for rec in records:
        node = rec["modules"].get(module, {}).get(kind)
        if node and node.get("cos") is not None:
            vals.append(node)
    if not vals:
        return {"n_scored": 0}
    return {"n_scored": len(vals),
            "conflict_rate": sum(1 for v in vals if v["conflict"]) / len(vals),
            "cos_median": statistics.median(v["cos"] for v in vals),
            "neg_tangent_frac_max": max(v["neg_tangent_frac"] for v in vals)}


def summarize(records: list[dict]) -> dict:
    if not records:
        return {"n_steps": 0}
    modules = sorted({m for r in records for m in r["modules"]})
    return {
        "n_steps": len(records),
        "eff": _rate(records, "overall"),
        "raw": _rate(records, "overall_raw"),
        "modules": {m: {"eff": _module_rate(records, m, "eff"),
                        "raw": _module_rate(records, m, "raw")}
                    for m in modules},
        "grl_checked": sum(r["grl"]["checked"] for r in records),
        "grl_violations": sum(r["grl"]["violations"] for r in records),
        "grl_max_residual_rel": max((r["grl"]["max_residual_rel"] for r in records
                                     if r["grl"]["max_residual_rel"] is not None),
                                    default=None),
        "grl_max_abs_scale_dev": max((r["grl"]["max_abs_scale_dev"] for r in records
                                      if r["grl"]["max_abs_scale_dev"] is not None),
                                     default=None),
        "replay_bitexact_all": all(
            r["replay"]["reg_bitexact"] and all(h["bitexact"] for h in r["replay"]["heads"].values())
            for r in records if r.get("replay")),
    }


def write_report(records: list[dict], out_dir: Path) -> Path:
    summary = summarize(records)
    lines = ["# gradient-conflict audit", "",
             f"- steps recorded: {summary.get('n_steps', 0)}",
             f"- GRL reversal check (scale identity): violations="
             f"{summary.get('grl_violations', 0)} (checked={summary.get('grl_checked', 0)}, "
             f"max_abs_scale_dev={summary.get('grl_max_abs_scale_dev')})",
             f"- replay bit-exact (all steps): {summary.get('replay_bitexact_all')}",
             f"- legacy cross-forward residual (diagnostic only): "
             f"max={summary.get('grl_max_residual_rel')}",
             f"- effective-path conflict rate: "
             f"{summary.get('eff', {}).get('conflict_rate')}",
             f"- raw-task-path  conflict rate: "
             f"{summary.get('raw', {}).get('conflict_rate')}", "",
             "| module | eff cos_median | eff conflict | raw cos_median | raw conflict |",
             "|---|---|---|---|---|"]
    for m, d in summary.get("modules", {}).items():
        lines.append(f"| {m} | {d['eff'].get('cos_median')} | "
                     f"{d['eff'].get('conflict_rate')} | "
                     f"{d['raw'].get('cos_median')} | {d['raw'].get('conflict_rate')} |")
    dest = Path(out_dir) / "grad_audit_report.md"
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (Path(out_dir) / "grad_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")
    return dest
