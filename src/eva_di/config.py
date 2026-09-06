"""Strict configuration loader for ``eva_di_config_v1``.

Rules (design doc section 6): unknown keys are an error; ``data.splits`` may
only draw from ``{train, val}`` -- the presence of ``test`` is refused at parse
time; matrix configs override base keys via ``extends`` with loop protection;
all ranges are validated before any consumer sees the object.

Hash semantics (audit R2-P1-4): ``source_sha256`` is the sha256 of the leaf
YAML file; ``resolved_sha256`` is the sha256 of the canonical YAML dump of the
fully merged ``extends`` payload, so a base-file drift changes the resolved
hash recorded in every run provenance.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.eva_di.contracts import (
    DECISION_SPLITS,
    EVA_BACKBONE,
    EVA_INPUT_SIZE,
    EVA_WEIGHT_SHA256,
    FRAME_BUDGET,
    PROJ_DIM,
    SELECTION_POLICY,
    EvaDiConfigError,
    EvaDiPathError,
    SCHEMA_CONFIG,
)

_VALID_INPUT_MODES = ("dual", "eva_only", "of3_only")
_VALID_MODES = ("train", "extract")


def _require_int(value: Any, where: str, *, lo: int = 1) -> int:
    """Positive-by-default int guard; bools are rejected explicitly.

    PyYAML 1.1 parses ``yes/no/on/off`` to bool and ``bool`` subclasses ``int``,
    so ``isinstance(True, int)`` must be refused (same rule the seed check
    always applied -- audit R2-P2-1).
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaDiConfigError(
            f"{where} must be an int (bool is not accepted); got {value!r}"
            f" of type {type(value).__name__}")
    if value < lo:
        raise EvaDiConfigError(f"{where} must be >= {lo}; got {value}")
    return value


def _require_num(value: Any, where: str, *, lo: float | None = None,
                 hi: float | None = None, hi_open: bool = False,
                 lo_open: bool = False) -> float:
    """Finite float guard with optional bounds; bool rejected (R2-P2-2)."""
    bad = f"got {value!r} of type {type(value).__name__} (write floats as 0.0003, not 3e-4)"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaDiConfigError(f"{where} must be a finite number; {bad}")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise EvaDiConfigError(f"{where} must be finite; {bad}")
    if lo is not None and (number < lo or (lo_open and number == lo)):
        raise EvaDiConfigError(f"{where} must be {'>' if lo_open else '>='} {lo}; got {number}")
    if hi is not None and (number > hi or (hi_open and number == hi)):
        raise EvaDiConfigError(f"{where} must be {'<' if hi_open else '<='} {hi}; got {number}")
    return number


def _require_keys(payload: dict[str, Any], required: set[str], where: str) -> None:
    missing = sorted(required - set(payload))
    if missing:
        raise EvaDiConfigError(f"{where}: missing required key(s) {missing}")


@dataclass(frozen=True)
class RunCfg:
    mode: str
    seed: int
    run_id: str | None


@dataclass(frozen=True)
class PathsCfg:
    overrides: dict[str, str | None] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractionCfg:
    backbone: str
    weight_sha256: str
    chunk: int
    autocast: str
    input_size: int
    selection_policy: str
    frame_budget: int


@dataclass(frozen=True)
class DataCfg:
    splits: tuple[str, ...]
    max_recordings: int | None
    behavior_norm: str = "reuse"


@dataclass(frozen=True)
class ModelCfg:
    use_gap: bool
    proj_dim: int
    gru_hidden: int
    gru_layers: int
    behavior_mlp_dim: int
    behavior_dropout: float
    proj_dropout: float


@dataclass(frozen=True)
class IdentityHeadCfg:
    enable: bool
    weight: float
    lam: float
    warmup_epochs: int
    label_smoothing: float


@dataclass(frozen=True)
class GradAuditCfg:
    enable: bool
    every_n_steps: int
    max_records: int
    on_grl_violation: str = "report"


@dataclass(frozen=True)
class IdentityCfg:
    t1: IdentityHeadCfg
    t3: IdentityHeadCfg
    derange: bool
    grad_audit: GradAuditCfg


@dataclass(frozen=True)
class AblationCfg:
    input_mode: str


@dataclass(frozen=True)
class TrainingCfg:
    precision: str
    optimizer: dict[str, Any]
    batch_recordings: int
    epochs_max: int
    early_stop: dict[str, Any]
    grad_clip_norm: float
    reg_weighting: str = "severity"


@dataclass(frozen=True)
class EvaDiConfig:
    run: RunCfg
    paths: PathsCfg
    extraction: ExtractionCfg
    data: DataCfg
    model: ModelCfg
    identity: IdentityCfg
    ablation: AblationCfg
    training: TrainingCfg
    source_sha256: str = ""
    resolved_sha256: str = ""
    config_name: str = ""
    resolved_payload: dict = field(default_factory=dict)


def _require_mapping(payload: Any, where: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise EvaDiConfigError(f"{where}: expected a mapping, got {type(payload).__name__}")
    return payload


def _check_keys(payload: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise EvaDiConfigError(f"{where}: unknown key(s) {unknown} (strict schema)")


def _load_raw(path: Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in _seen:
        raise EvaDiConfigError(f"config extends cycle: {[str(p) for p in _seen + (path,)]}")
    if not path.is_file():
        raise EvaDiPathError(f"config file missing: {path}")
    payload = _require_mapping(
        yaml.safe_load(path.read_text(encoding="utf-8")), str(path)
    )
    extends = payload.pop("extends", None)
    if extends is None:
        return payload
    if not isinstance(extends, str) or not extends.strip():
        raise EvaDiConfigError(
            f"{path}: 'extends' must be a non-empty path string, got {extends!r}"
            f" of type {type(extends).__name__}")
    base_path = path.parent / extends
    base = _load_raw(base_path, _seen + (path,))
    return _deep_merge(base, payload)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge dicts recursively; ANY non-dict value (including lists) replaces
    the parent value wholesale; ``null`` is a valid override; keys cannot be
    deleted by children (audit R2-P3-3: semantics documented here)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path) -> EvaDiConfig:
    from src.eva_di.paths import sha256_file

    path = Path(path)
    raw = _load_raw(path)
    source_sha = sha256_file(path)
    # R2-P1-4: the resolved hash must cover the MERGED payload (base included),
    # otherwise editing eva_di_base.yaml is invisible in every run provenance.
    resolved_sha = hashlib.sha256(
        yaml.safe_dump(raw, sort_keys=True).encode("utf-8")).hexdigest()

    if raw.get("schema_version") != SCHEMA_CONFIG:
        raise EvaDiConfigError(
            f"schema_version {raw.get('schema_version')!r} != {SCHEMA_CONFIG!r}"
        )
    _check_keys(raw, {"schema_version", "run", "paths", "extraction", "data",
                     "model", "identity", "ablation", "training"}, "config root")
    for section in ("run", "paths", "extraction", "data", "model", "identity",
                    "ablation", "training"):
        if section not in raw:
            raise EvaDiConfigError(f"missing required section {section!r}")

    run = _require_mapping(raw["run"], "run")
    _check_keys(run, {"mode", "seed", "run_id"}, "run")
    _require_keys(run, {"mode", "seed"}, "run")
    if run.get("mode") not in _VALID_MODES:
        raise EvaDiConfigError(f"run.mode must be one of {_VALID_MODES}")
    seed = run.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise EvaDiConfigError("run.seed must be a non-negative int")
    if seed >= 2**32:
        raise EvaDiConfigError(
            f"run.seed {seed} >= 2**32 exceeds numpy.random.seed's domain (R2-P3-4)")
    run_id = run.get("run_id")
    if run_id is not None:
        if (not isinstance(run_id, str) or not run_id.strip()
                or "/" in run_id or "\\" in run_id or run_id in (".", "..")):
            raise EvaDiConfigError(
                "run.run_id must be null or a non-empty path-safe string; got "
                f"{run_id!r} of type {type(run_id).__name__}")
        run_id = run_id.strip()
    run_cfg = RunCfg(mode=run["mode"], seed=seed, run_id=run_id)

    paths_raw = _require_mapping(raw["paths"], "paths")
    _check_keys(paths_raw, {"of3_root", "avec_root", "split_file", "label_dir",
                            "cache_root", "weight_path", "behavior_norm_root",
                            "log_root", "output_root"}, "paths")
    for key, value in paths_raw.items():
        if value is not None and not isinstance(value, str):
            raise EvaDiConfigError(
                f"paths.{key} must be null or a string; got {value!r} of type "
                f"{type(value).__name__}")
    paths_cfg = PathsCfg(
        overrides={k: (v if v not in (None, "") else None) for k, v in paths_raw.items()}
    )

    extraction = _require_mapping(raw["extraction"], "extraction")
    _check_keys(extraction, {"backbone", "weight_sha256", "chunk", "autocast",
                             "input_size", "selection_policy", "frame_budget"}, "extraction")
    _require_keys(extraction, {"backbone", "weight_sha256", "chunk", "autocast",
                               "input_size", "selection_policy", "frame_budget"},
                  "extraction")
    if extraction.get("backbone") != EVA_BACKBONE:
        raise EvaDiConfigError(f"extraction.backbone must be {EVA_BACKBONE!r}")
    if not isinstance(extraction.get("weight_sha256"), str):
        raise EvaDiConfigError(
            "extraction.weight_sha256 must be a quoted 64-hex string (R2-P3-8)")
    if extraction.get("weight_sha256") != EVA_WEIGHT_SHA256:
        raise EvaDiConfigError("extraction.weight_sha256 does not match the pinned constant")
    if extraction.get("selection_policy") != SELECTION_POLICY:
        raise EvaDiConfigError(f"extraction.selection_policy must be {SELECTION_POLICY!r}")
    chunk = _require_int(extraction.get("chunk"), "extraction.chunk")
    if extraction.get("autocast") not in ("bf16", "none"):
        raise EvaDiConfigError("extraction.autocast must be bf16|none")
    if extraction.get("input_size") != EVA_INPUT_SIZE:
        raise EvaDiConfigError(f"extraction.input_size is pinned to {EVA_INPUT_SIZE}")
    frame_budget = extraction.get("frame_budget")
    if frame_budget != FRAME_BUDGET:
        raise EvaDiConfigError(f"extraction.frame_budget is pinned to {FRAME_BUDGET}")
    extraction_cfg = ExtractionCfg(
        backbone=extraction["backbone"],
        weight_sha256=extraction["weight_sha256"],
        chunk=chunk,
        autocast=extraction["autocast"],
        input_size=extraction["input_size"],
        selection_policy=extraction["selection_policy"],
        frame_budget=frame_budget,
    )

    data = _require_mapping(raw["data"], "data")
    _check_keys(data, {"splits", "max_recordings", "behavior_norm"}, "data")
    _require_keys(data, {"splits"}, "data")
    splits = data.get("splits")
    if not isinstance(splits, list) or not splits:
        raise EvaDiConfigError("data.splits must be a non-empty list")
    if any(not isinstance(s, str) for s in splits):
        raise EvaDiConfigError(f"data.splits entries must be strings; got {splits!r}")
    if len(set(splits)) != len(splits):
        raise EvaDiConfigError(f"data.splits must not contain duplicates; got {splits!r}")
    if "test" in splits:
        raise EvaDiConfigError(
            "data.splits must never contain 'test' (locked split; decision paths are val-only)"
        )
    if not set(splits) <= set(DECISION_SPLITS):
        raise EvaDiConfigError(f"data.splits must be a subset of {DECISION_SPLITS}")
    if "train" not in splits:
        raise EvaDiConfigError("data.splits must contain 'train'")
    max_recordings = data.get("max_recordings")
    if max_recordings is not None:
        max_recordings = _require_int(max_recordings, "data.max_recordings")
    behavior_norm = data.get("behavior_norm", "reuse")
    if behavior_norm not in ("reuse", "recompute"):
        raise EvaDiConfigError("data.behavior_norm must be 'reuse' or 'recompute'")
    data_cfg = DataCfg(splits=tuple(splits), max_recordings=max_recordings,
                       behavior_norm=behavior_norm)

    model = _require_mapping(raw["model"], "model")
    _check_keys(model, {"use_gap", "proj_dim", "gru_hidden", "gru_layers",
                        "behavior_mlp_dim", "behavior_dropout", "proj_dropout"}, "model")
    _require_keys(model, {"use_gap", "proj_dim", "gru_hidden", "gru_layers",
                          "behavior_mlp_dim", "behavior_dropout", "proj_dropout"},
                  "model")
    for flag in ("use_gap",):
        if not isinstance(model.get(flag), bool):
            raise EvaDiConfigError(f"model.{flag} must be a bool")
    for key in ("proj_dim", "gru_hidden", "gru_layers", "behavior_mlp_dim"):
        _require_int(model.get(key), f"model.{key}")
    if model["proj_dim"] != PROJ_DIM:
        raise EvaDiConfigError(f"model.proj_dim is pinned to {PROJ_DIM}")
    if model["gru_hidden"] != PROJ_DIM:
        # model.py's masked mean/reg-head assumes hidden==proj; validate at
        # parse time so `python -O` cannot strip the runtime guard (R2-P2-#3).
        raise EvaDiConfigError(
            f"model.gru_hidden is pinned to proj_dim ({PROJ_DIM}); got {model['gru_hidden']}")
    for key in ("behavior_dropout", "proj_dropout"):
        value = _require_num(model.get(key), f"model.{key}", lo=0.0, hi=1.0, hi_open=True)
        model[key] = value
    model_cfg = ModelCfg(**{k: model[k] for k in (
        "use_gap", "proj_dim", "gru_hidden", "gru_layers",
        "behavior_mlp_dim", "behavior_dropout", "proj_dropout")})

    identity = _require_mapping(raw["identity"], "identity")
    _check_keys(identity, {"t1", "t3", "derange", "grad_audit"}, "identity")
    _require_keys(identity, {"t1", "t3", "derange"}, "identity")
    if not isinstance(identity.get("derange"), bool):
        raise EvaDiConfigError("identity.derange must be a bool")

    def _head(payload: Any, where: str) -> IdentityHeadCfg:
        head = _require_mapping(payload, where)
        _check_keys(head, {"enable", "weight", "lam", "warmup_epochs", "label_smoothing"}, where)
        _require_keys(head, {"enable", "weight", "lam", "warmup_epochs", "label_smoothing"},
                      where)
        if not isinstance(head.get("enable"), bool):
            raise EvaDiConfigError(f"{where}.enable must be a bool")
        weight = _require_num(head.get("weight"), f"{where}.weight", lo=0.0)
        lam = _require_num(head.get("lam"), f"{where}.lam", lo=0.0, hi=1.0, lo_open=True)
        warmup = _require_int(head.get("warmup_epochs"), f"{where}.warmup_epochs", lo=0)
        smooth = _require_num(head.get("label_smoothing"), f"{where}.label_smoothing",
                              lo=0.0, hi=1.0, hi_open=True)
        return IdentityHeadCfg(
            enable=head["enable"], weight=weight, lam=lam,
            warmup_epochs=warmup, label_smoothing=smooth,
        )

    ga = identity.get("grad_audit")
    if ga is None:
        grad_audit = GradAuditCfg(enable=False, every_n_steps=10, max_records=2000)
    else:
        ga = _require_mapping(ga, "identity.grad_audit")
        _check_keys(ga, {"enable", "every_n_steps", "max_records", "on_grl_violation"},
                    "identity.grad_audit")
        if not isinstance(ga.get("enable"), bool):
            raise EvaDiConfigError("identity.grad_audit.enable must be a bool")
        policy = ga.get("on_grl_violation", "report")
        if policy not in ("report", "fail"):
            raise EvaDiConfigError(
                "identity.grad_audit.on_grl_violation must be 'report' or 'fail'")
        grad_audit = GradAuditCfg(
            enable=ga["enable"],
            every_n_steps=_require_int(ga.get("every_n_steps", 10),
                                       "identity.grad_audit.every_n_steps", lo=1),
            max_records=_require_int(ga.get("max_records", 2000),
                                     "identity.grad_audit.max_records", lo=1),
            on_grl_violation=policy,
        )
    identity_cfg = IdentityCfg(
        t1=_head(identity["t1"], "identity.t1"),
        t3=_head(identity["t3"], "identity.t3"),
        derange=identity["derange"],
        grad_audit=grad_audit,
    )
    # Mechanism plan (DUAL_LEVEL_IDENTITY_ADVERSARIAL_PLAN.md L53) pins ONE
    # shared warmup schedule for both lambdas; train.py drives both from
    # t1.warmup_epochs.  Reject a divergent t3 value instead of silently
    # ignoring it (audit R2-P1-1: dead-key trap).
    if identity_cfg.grad_audit.enable and not (identity_cfg.t1.enable
                                               or identity_cfg.t3.enable):
        raise EvaDiConfigError(
            "identity.grad_audit.enable requires an enabled identity head (t1 or t3)")
    if identity_cfg.t1.warmup_epochs != identity_cfg.t3.warmup_epochs:
        raise EvaDiConfigError(
            "identity.t1.warmup_epochs and identity.t3.warmup_epochs must be "
            "equal (the plan pins one shared warmup schedule for both lambdas; "
            "t1's value drives the schedule)")
    if identity_cfg.derange and not (identity_cfg.t1.enable or identity_cfg.t3.enable):
        raise EvaDiConfigError(
            "identity.derange is a negative control for the identity heads and "
            "requires at least one of identity.t{1,3}.enable to be true")

    ablation = _require_mapping(raw["ablation"], "ablation")
    _check_keys(ablation, {"input_mode"}, "ablation")
    _require_keys(ablation, {"input_mode"}, "ablation")
    if ablation.get("input_mode") not in _VALID_INPUT_MODES:
        raise EvaDiConfigError(f"ablation.input_mode must be one of {_VALID_INPUT_MODES}")
    ablation_cfg = AblationCfg(input_mode=ablation["input_mode"])

    training = _require_mapping(raw["training"], "training")
    _check_keys(training, {"precision", "optimizer", "batch_recordings",
                           "epochs_max", "early_stop", "grad_clip_norm",
                           "reg_weighting"}, "training")
    if training.get("reg_weighting", "severity") not in ("severity", "plain"):
        raise EvaDiConfigError(
            "training.reg_weighting must be 'severity' (legacy weighted MSE) "
            "or 'plain' (unweighted MSE on the normalized target)")
    _require_keys(training, {"precision", "optimizer", "batch_recordings",
                             "epochs_max", "early_stop", "grad_clip_norm"}, "training")
    if training.get("precision") != "bf16-mixed":
        raise EvaDiConfigError("training.precision is pinned to 'bf16-mixed' (runner casts explicitly)")
    batch = _require_int(training.get("batch_recordings"), "training.batch_recordings")
    epochs = _require_int(training.get("epochs_max"), "training.epochs_max")
    optimizer = _require_mapping(training["optimizer"], "training.optimizer")
    _check_keys(optimizer, {"name", "lr", "weight_decay"}, "training.optimizer")
    _require_keys(optimizer, {"name", "lr", "weight_decay"}, "training.optimizer")
    if optimizer.get("name") != "adamw":
        raise EvaDiConfigError("training.optimizer.name is pinned to 'adamw'")
    _require_num(optimizer.get("lr"), "training.optimizer.lr", lo=0.0, lo_open=True)
    _require_num(optimizer.get("weight_decay"), "training.optimizer.weight_decay",
                 lo=0.0)
    early_stop = _require_mapping(training["early_stop"], "training.early_stop")
    _check_keys(early_stop, {"metric", "patience"}, "training.early_stop")
    _require_keys(early_stop, {"metric", "patience"}, "training.early_stop")
    if early_stop.get("metric") != "val_ccc":
        raise EvaDiConfigError("training.early_stop.metric is pinned to 'val_ccc' (val-only decisions)")
    _require_int(early_stop.get("patience"), "training.early_stop.patience")
    clip = _require_num(training.get("grad_clip_norm"), "training.grad_clip_norm",
                        lo=0.0, lo_open=True)
    training_cfg = TrainingCfg(
        precision=training["precision"], optimizer=optimizer, batch_recordings=batch,
        epochs_max=epochs, early_stop=early_stop, grad_clip_norm=float(clip),
        reg_weighting=training.get("reg_weighting", "severity"),
    )
    for head_name in ("t1", "t3"):
        warmup = getattr(identity_cfg, head_name).warmup_epochs
        if warmup and warmup > epochs:
            raise EvaDiConfigError(
                f"identity.{head_name}.warmup_epochs={warmup} exceeds "
                f"training.epochs_max={epochs}; the reversal would never reach "
                "full strength within the run")

    return EvaDiConfig(
        run=run_cfg, paths=paths_cfg, extraction=extraction_cfg, data=data_cfg,
        model=model_cfg, identity=identity_cfg, ablation=ablation_cfg,
        training=training_cfg, source_sha256=source_sha, resolved_sha256=resolved_sha,
        config_name=path.stem, resolved_payload=raw,
    )
