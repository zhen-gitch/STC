"""Ablation-matrix planner for the EVA-DI study (EVA-DI-MATRIX spec v1).

The matrix is a DATA file (``configs/eva_di/matrix_v1.yaml``) plus this
deterministic planner: it turns the spec into an ordered, leakage-safe run
plan (arm -> config -> run_id -> commands), flags each entry as done/ready/
blocked by inspecting existing run artifacts, and -- only when explicitly
launched -- executes trainings SEQUENTIALLY with stop-on-first-failure.  The
planner itself never trains anything; ``build_plan`` is pure and hermetic.

Command templates (per arm, sequential): train -> export exit-P/exit-V (val
and train) -> external identity metrics.  Export/identity reference the same
configs and the run's best/last checkpoint, so one launch reproduces the whole
evidence chain from git commit + config + seed (experiment rules, AGENTS.md).
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from src.eva_di.contracts import EvaDiError

SPEC_SCHEMA = "eva_di_matrix_v1"
PY = "{python}"


def load_matrix_spec(path: Path) -> dict:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or spec.get("schema_version") != SPEC_SCHEMA:
        raise EvaDiError(f"{path}: schema_version must be {SPEC_SCHEMA}")
    stage1 = spec.get("stage1") or {}
    if not isinstance(stage1.get("arms"), list) or not stage1["arms"]:
        raise EvaDiError(f"{path}: stage1.arms must be a non-empty list")
    if not isinstance(stage1.get("seed"), int):
        raise EvaDiError(f"{path}: stage1.seed must be an int")
    stage2 = spec.get("stage2") or {"seeds": [], "arms": []}
    if not isinstance(stage2.get("seeds"), list) or not isinstance(stage2.get("arms"), list):
        raise EvaDiError(f"{path}: stage2.seeds/arms must be lists")
    for arm in list(stage1["arms"]) + list(spec.get("ablations") or []):
        if not isinstance(arm, dict) or "name" not in arm or "config" not in arm:
            raise EvaDiError(f"{path}: every arm needs name+config: {arm!r}")
    return spec


def _read_yaml_light(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def build_plan(spec: dict, *, repo_root: Path, runs_root: Path,
               python: str = sys.executable,
               seeds: list[int] | None = None,
               arms: list[str] | None = None,
               include_ablations: bool = True) -> list[dict]:
    """Deterministic plan. stage1 (seed 42) unless ``seeds``/``arms`` override
    to a stage-2 paired-seed plan for surviving arms."""
    stage1 = spec["stage1"]
    if seeds is None and arms is None:
        entries = [(a["name"], a["config"], stage1["seed"]) for a in stage1["arms"]]
    else:
        use_seeds = seeds if seeds is not None else (spec.get("stage2") or {}).get("seeds", [])
        use_arms = arms if arms is not None else (spec.get("stage2") or {}).get("arms", [])
        known = {a["name"]: a["config"] for a in stage1["arms"]}
        entries = []
        for seed in use_seeds:
            for name in use_arms:
                if name not in known:
                    raise EvaDiError(f"stage-2 arm {name!r} not in stage1 arms")
                entries.append((name, known[name], int(seed)))
    if include_ablations and seeds is None and arms is None:
        entries += [(a["name"], a["config"], stage1["seed"])
                    for a in spec.get("ablations") or []]

    names: set[str] = set()
    plan: list[dict] = []
    for name, rel_cfg, seed in entries:
        cfg_path = Path(repo_root) / rel_cfg
        status = "ready"
        light: dict = {}
        if not cfg_path.is_file():
            status = f"blocked: config missing ({rel_cfg})"
        else:
            light = _read_yaml_light(cfg_path)
            cfg_seed = (light.get("run") or {}).get("seed", 42)
            cfg_run_id = (light.get("run") or {}).get("run_id")
            if int(cfg_seed) != int(seed):
                status = (f"blocked: {rel_cfg} run.seed={cfg_seed} != plan seed "
                          f"{seed}; a per-seed derived config is required")
            elif not cfg_run_id:
                status = f"blocked: {rel_cfg} has no explicit run.run_id"
            elif cfg_run_id in names:
                status = f"blocked: duplicate run_id {cfg_run_id!r}"
            else:
                run_dir = Path(runs_root) / str(cfg_run_id)
                if (run_dir / "summary.json").is_file():
                    status = "done"
                names.add(str(cfg_run_id))
        entry = {"arm": name, "seed": int(seed), "config": rel_cfg,
                 "run_id": (light.get("run") or {}).get("run_id")
                 if cfg_path.is_file() else None,
                 "status": status}
        rid = entry["run_id"]
        base = f"{PY} -m src.eva_di.train --config {rel_cfg}"
        ckpt = f"RUN_ROOT/{rid}/checkpoints/best.pt" if rid else None
        entry["commands"] = {
            "train": base,
            "export_val": (f"{PY} -m src.eva_di.export_embeddings --config {rel_cfg} "
                           f"--split val --checkpoint {ckpt}") if rid else None,
            "export_train": (f"{PY} -m src.eva_di.export_embeddings --config {rel_cfg} "
                             f"--split train --checkpoint {ckpt}") if rid else None,
            "identity": (f"{PY} -m src.eva_di.identity_metrics --npz "
                         f"RUN_ROOT/{rid}/embeddings/p_mean_val.npz "
                         f"RUN_ROOT/{rid}/embeddings/v_h0_val.npz "
                         f"--output-dir RUN_ROOT/{rid}/identity") if rid else None,
        }
        plan.append(entry)
    return plan


def plan_table(plan: list[dict]) -> str:
    lines = [f"{'arm':<12}{'seed':<6}{'status':<60}run_id"]
    for e in plan:
        lines.append(f"{e['arm']:<12}{e['seed']:<6}{e['status']:<60}{e['run_id']}")
    return "\n".join(lines)


def launch(plan: list[dict], *, repo_root: Path, python: str, device: str,
           dry_run: bool = True) -> int:
    """Sequential train launches, stop on first failure.  Train commands only
    -- export/identity are run after each training completes (their inputs do
    not exist until then).  ``dry_run`` prints and returns 0."""
    for entry in plan:
        if entry["status"] == "done":
            print(f"[matrix] skip {entry['arm']} ({entry['status']})")
            continue
        if entry["status"] != "ready":
            raise EvaDiError(f"cannot launch {entry['arm']}: {entry['status']}")
        cmd = entry["commands"]["train"].replace(PY, python)
        cmd += f" --device {device}"
        print(f"[matrix] $ {cmd}")
        if dry_run:
            continue
        rc = subprocess.run(shlex.split(cmd), cwd=str(repo_root)).returncode
        if rc != 0:
            print(f"[matrix] FAILED {entry['arm']} rc={rc}; stopping")
            return rc
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="EVA-DI ablation-matrix planner")
    parser.add_argument("--spec", default="configs/eva_di/matrix_v1.yaml")
    parser.add_argument("--runs-root", required=True,
                        help="log root containing <run_id>/summary.json")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="stage-2 override (paired seeds)")
    parser.add_argument("--arms", nargs="*", default=None,
                        help="stage-2 override (surviving arm names)")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--launch", action="store_true",
                        help="actually run trainings (never in planning runs)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    spec = load_matrix_spec(Path(args.spec))
    plan = build_plan(spec, repo_root=repo_root, runs_root=Path(args.runs_root),
                      python=args.python, seeds=args.seeds, arms=args.arms)
    if args.as_json:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    else:
        print(plan_table(plan))
    if args.launch:
        return launch(plan, repo_root=repo_root, python=args.python,
                      device=args.device, dry_run=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
