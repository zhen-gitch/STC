"""Version-aware Stage B run discovery and sweep labeling.

Stage B trains a base config (E0-E3) plus lambda_id / POWER sweeps, each as a
new ``version_N`` directory under the experiment's run dir.  Aggregation needs
to compare the base configs *and* the sweeps in one table, but a version dir's
name does not encode which sweep it ran.  This module recovers the sweep
identity by reading each version's ``resolved_config.yaml`` (written by
:func:`src.trainers.mtl_lite_runner.save_resolved_config`) and labels the run
accordingly::

    base (lambda_id==0.05, power==0.5)          -> eN
    lambda sweep (lambda_id!=0.05, power==0.5)  -> eN_lambda<L>
    power sweep (power!=0.5, lambda_id==0.05)   -> eN_power<P>
    both non-base                               -> eN_lambda<L>_power<P>

Multiple versions sharing a sweep signature (e.g. re-runs) are de-duplicated to
the latest, so re-running a sweep does not duplicate rows in the aggregate
table.  The base-config signature collapses to the bare experiment id ``eN``,
so the E0-anchored baseline (``--baseline e0``) keeps working.
"""

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from omegaconf import OmegaConf

# Base-config sweep values (matching configs/stage_b/base_regression_only.yaml
# and the e1/e2/e3 overrides).  A version whose resolved config has these values
# is the canonical base run for its experiment and gets the bare ``eN`` label.
BASE_LAMBDA = 0.05
BASE_POWER = 0.5

# Experiment id -> EXPERIMENT_NAME (mirrors configs/stage_b/e*_*.yaml).
EXP_NAME = {
    "e0": "e0_rgb_mtl_lite",
    "e1": "e1_identity_adversarial",
    "e2": "e2_severity_balanced",
    "e3": "e3_identity_adversarial_severity_balanced",
}


def _fmt(value) -> str:
    """Format a sweep value for a label, preserving a decimal point (1.0 not 1)."""
    return str(float(value))


def label_for_sweep(exp: str, lambda_id, power) -> str:
    """Build the aggregate-table run label for one sweep signature.

    Args:
        exp: Experiment id (e0/e1/e2/e3).
        lambda_id: Resolved MODEL.IDENTITY_ADVERSARIAL.LAMBDA_ID (or base).
        power: Resolved MODEL.SEVERITY_BALANCED_REGRESSION.POWER (or base).

    Returns:
        ``eN`` for the base signature, otherwise ``eN_<sweep>``.
    """
    parts = []
    if abs(float(lambda_id) - BASE_LAMBDA) > 1e-9:
        parts.append(f"lambda{_fmt(lambda_id)}")
    if abs(float(power) - BASE_POWER) > 1e-9:
        parts.append(f"power{_fmt(power)}")
    return exp if not parts else f"{exp}_" + "_".join(parts)


def read_sweep_from_config(run_dir) -> Optional[Tuple[float, float]]:
    """Read the (lambda_id, power) sweep signature from a version's resolved config.

    Returns ``None`` when ``resolved_config.yaml`` is absent (e.g. an incomplete
    run dir).  When the identity/severity blocks are disabled or missing, the
    corresponding value defaults to the base-config value, so an E0/E2 run
    (identity off) and an E0/E1 run (severity off) both resolve to the base
    signature.
    """
    cfg_path = Path(run_dir) / "resolved_config.yaml"
    if not cfg_path.exists():
        return None
    cfg = OmegaConf.load(cfg_path)

    lam = BASE_LAMBDA
    pw = BASE_POWER
    try:
        ia = cfg.MODEL.IDENTITY_ADVERSARIAL
        if bool(getattr(ia, "ENABLE", False)):
            lam = float(getattr(ia, "LAMBDA_ID", BASE_LAMBDA))
    except (AttributeError, ValueError, TypeError):
        pass
    try:
        sbr = cfg.MODEL.SEVERITY_BALANCED_REGRESSION
        if bool(getattr(sbr, "ENABLE", False)):
            pw = float(getattr(sbr, "POWER", BASE_POWER))
    except (AttributeError, ValueError, TypeError):
        pass
    return lam, pw


def discover_versions(base_dir, exp: str, latest_only: bool = False) -> List[Tuple[str, Path]]:
    """Enumerate ``version_*`` dirs under ``base_dir`` and label each by sweep.

    Args:
        base_dir: The experiment run root (``<LOG_DIR>/<group>/<ename>``)
            containing ``version_0/``, ``version_1/``, ...
        exp: Experiment id (e0/e1/e2/e3) used as the label prefix.
        latest_only: If True, return only the single latest version labeled ``eN``
            (the pre-version-aware behavior).

    Returns:
        List of ``(label, run_dir)`` tuples in ascending version order, with at
        most one entry per sweep signature (the latest version for that
        signature).  Versions without ``resolved_config.yaml`` are skipped.
    """
    base = Path(base_dir)
    if not base.exists():
        return []
    versions = sorted(
        [p for p in base.glob("version_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not versions:
        return []
    if latest_only:
        return [(exp, versions[-1])]

    version_sweeps: List[Tuple[Path, Tuple[float, float]]] = []
    for v in versions:
        sw = read_sweep_from_config(v)
        if sw is None:
            continue
        version_sweeps.append((v, sw))

    # Keep only the latest version per sweep signature (last wins).
    latest_by_sig = {}
    for v, sw in version_sweeps:
        latest_by_sig[sw] = v

    seen = set()
    out: List[Tuple[str, Path]] = []
    for v, sw in version_sweeps:
        if latest_by_sig.get(sw) != v:
            continue
        label = label_for_sweep(exp, sw[0], sw[1])
        if label in seen:
            continue
        seen.add(label)
        out.append((label, v))
    return out


def _experiment_base_dir(name: str):
    """Resolve ``<LOG_DIR>/<EXPERIMENT_GROUP>/<EXPERIMENT_NAME>`` for an experiment.

    Mirrors the config merge in ``scripts/stage_b/aggregate_stage_b.sh``'s
    resolver: base + local_paths + base_regression_only + the eX override.
    """
    import glob

    from src.config import resolve_config_path, load_yaml_config, DEFAULT_BASE_CONFIG

    cfg = load_yaml_config(DEFAULT_BASE_CONFIG)
    lp = resolve_config_path("configs/local_paths.yaml")
    if lp.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(lp))
    cfg = OmegaConf.merge(cfg, OmegaConf.load("configs/stage_b/base_regression_only.yaml"))
    ov = glob.glob(f"configs/stage_b/{name}_*.yaml")
    if ov:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(ov[0]))

    log_dir = getattr(cfg, "LOG_DIR", None)
    if not log_dir:
        return None
    root = Path(log_dir)
    if not root.is_absolute():
        root = Path.cwd() / root
    ename = str(getattr(cfg, "EXPERIMENT_NAME", EXP_NAME.get(name, name)))
    group = str(getattr(cfg, "EXPERIMENT_GROUP", "default"))
    return root / group / ename


def resolve_stage_b_run_specs(
    names: Iterable[str], latest_only: bool = False
) -> List[Tuple[str, Path]]:
    """Resolve labeled run specs across multiple experiments.

    Args:
        names: Experiment ids (e0/e1/e2/e3).
        latest_only: If True, only the latest version per experiment (labeled ``eN``).

    Returns:
        List of ``(label, run_dir)`` tuples.  Experiments whose run dir is
        missing are silently skipped.
    """
    specs: List[Tuple[str, Path]] = []
    for name in names:
        if name not in EXP_NAME:
            continue
        base = _experiment_base_dir(name)
        if not base or not base.exists():
            continue
        specs.extend(discover_versions(base, name, latest_only=latest_only))
    return specs
