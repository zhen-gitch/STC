"""Training-curve visualisation for eva_di runs (EVA-DI-OBSV2-v1).

READ-ONLY by construction: consumes ``<run>/metrics.jsonl`` and (when present)
``<run>/grad_audit/step_metrics.jsonl``; writes PNGs only.  Missing series are
LEFT BLANK -- never drawn as zero (observation discipline).  Agg backend: no
display needed; matplotlib is already an environment dependency.

CLI::

    python -m src.eva_di.viz_training --runs-root <dir> [--output-dir <dir>]
    python -m src.eva_di.viz_training --run-dir <dir> [--run-dir <dir> ...]

Per run: ``<run>/viz/curves.png`` (or --output-dir/<run>__curves.png) with
panels  train losses | val CCC+PCC | val MAE+RMSE | grad-audit conflict curves.
Across runs: ``compare_<metric>.png`` for every metric present in >=1 run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.eva_di.contracts import EvaDiError  # noqa: E402

_PANELS = (
    (("train_total", "total"), ("train_reg", "reg"),
     ("train_frame_ce", "frame CE"), ("train_video_ce", "video CE"),
     "train losses (per-epoch mean)"),
    (("val_ccc", "CCC"), ("val_pcc", "PCC"), "val correlation"),
    (("val_mae", "MAE"), ("val_rmse", "RMSE"), "val error"),
)
_COMPARE = (("val_ccc", "ccc"), ("val_pcc", "pcc"),
            ("val_mae", "mae"), ("val_rmse", "rmse"))


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _points(rows: list[dict], key: str) -> tuple[list, list]:
    xs, ys = [], []
    for r in rows:
        v = r.get(key)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            xs.append(r["epoch"])
            ys.append(float(v))
    return xs, ys


def _blank(ax, note: str) -> None:
    ax.text(0.5, 0.5, f"{note}\n(blank by policy -- never drawn as 0)",
            ha="center", va="center", transform=ax.transAxes, fontsize=8)


def plot_run(run_dir: Path, output_dir: Path | None = None) -> Path:
    run_dir = Path(run_dir)
    rows = _jsonl(run_dir / "metrics.jsonl")
    if not rows:
        raise EvaDiError(f"no metrics.jsonl under {run_dir}")
    out = Path(output_dir) if output_dir else run_dir / "viz"
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    flat_axes = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]
    for ax, panel in zip(flat_axes, _PANELS):
        series, title = panel[:-1], panel[-1]
        drew = False
        for key, label in series:
            xs, ys = _points(rows, key)
            if xs:
                ax.plot(xs, ys, marker="o", ms=3, label=label)
                drew = True
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        if drew:
            ax.legend(fontsize=8)
        else:
            _blank(ax, title)
    ax = flat_axes[3]
    ga = _jsonl(run_dir / "grad_audit" / "step_metrics.jsonl")
    if ga:
        steps, cum, nt = [], [], []
        seen = conflicted = 0
        for r in ga:
            o = r.get("overall") or {}
            if isinstance(o.get("cos"), float):
                seen += 1
                conflicted += 1 if o.get("conflict") else 0
                steps.append(r["step"])
                cum.append(conflicted / seen)
            if isinstance(o.get("neg_tangent_frac"), (int, float)):
                nt.append((r["step"], float(o["neg_tangent_frac"])))
        if steps:
            ax.plot(steps, cum, marker=".", label="eff conflict rate (cumulative)")
        if nt:
            ax.plot([s for s, _ in nt], [v for _, v in nt], marker=".",
                    label="neg-tangent frac (per step)")
        ax.set_title("grad audit: conflict dynamics vs step")
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.3)
        if steps or nt:
            ax.legend(fontsize=8)
        else:
            _blank(ax, "grad audit")
    else:
        _blank(ax, "no grad_audit records")
    fig.suptitle(f"run {run_dir.name}  ({len(rows)} epochs)")
    fig.tight_layout()
    dest = out / "curves.png" if output_dir is None else out / f"{run_dir.name}__curves.png"
    fig.savefig(dest, dpi=110)
    plt.close(fig)
    return dest


def plot_compare(run_dirs: list[Path], output_dir: Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = [(Path(d).name, _jsonl(Path(d) / "metrics.jsonl")) for d in run_dirs]
    made: list[Path] = []
    for key, slug in _COMPARE:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        drew = False
        for name, rows in loaded:
            xs, ys = _points(rows, key)
            if xs:
                ax.plot(xs, ys, marker="o", ms=3, label=name)
                drew = True
        if not drew:
            plt.close(fig)
            continue
        ax.set_title(f"{key} across runs")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        dest = output_dir / f"compare_{slug}.png"
        fig.savefig(dest, dpi=110)
        plt.close(fig)
        made.append(dest)
    return made


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--run-dir", action="append", default=[])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    runs = [Path(p) for p in args.run_dir]
    if args.runs_root:
        root = Path(args.runs_root)
        if not root.is_dir():
            raise EvaDiError(f"runs root missing: {root}")
        runs += sorted(d for d in root.iterdir()
                       if d.is_dir() and (d / "metrics.jsonl").is_file())
    if not runs:
        raise EvaDiError("no run dirs given (--run-dir / --runs-root)")
    out = Path(args.output_dir) if args.output_dir else None
    for d in runs:
        print(plot_run(d, out))
    if len(runs) > 1:
        for p in plot_compare(runs, out or runs[0].parent / "viz_compare"):
            print(p)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
