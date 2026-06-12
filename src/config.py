from pathlib import Path
from typing import Iterable, Optional, Union

from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_BASE_CONFIG = CONFIG_DIR / "avec2014_base.yaml"
DEFAULT_LOCAL_PATHS_CONFIG = CONFIG_DIR / "local_paths.yaml"
LEGACY_DEFAULT_CONFIG = CONFIG_DIR / "pre" / "default_config.yaml"

ConfigPath = Union[str, Path]


def resolve_project_path(path: ConfigPath) -> Path:
    """Resolve a project-relative path without requiring the file to exist."""
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_config_path(path: ConfigPath) -> Path:
    """Resolve config paths relative to the project root or configs directory."""
    path = Path(path)
    if path.is_absolute():
        return path

    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path

    return CONFIG_DIR / path


def load_yaml_config(config_path: ConfigPath) -> DictConfig:
    """Load one YAML config file with a clear missing-file error."""
    resolved_path = resolve_config_path(config_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_path}")
    return OmegaConf.load(resolved_path)


def _iter_override_paths(overrides: Optional[Iterable[ConfigPath]]) -> Iterable[Path]:
    if overrides is None:
        return []
    return [resolve_config_path(path) for path in overrides]


def load_experiment_config(
    base_config: ConfigPath = DEFAULT_BASE_CONFIG,
    local_paths_config: ConfigPath = DEFAULT_LOCAL_PATHS_CONFIG,
    overrides: Optional[Iterable[ConfigPath]] = None,
    require_local_paths: bool = True,
) -> DictConfig:
    """Load the canonical experiment config stack.

    Merge order is:
    1. shared base config
    2. machine-local paths config
    3. optional experiment/debug overrides
    """
    config_parts = [load_yaml_config(base_config)]

    local_paths = resolve_config_path(local_paths_config)
    if local_paths.exists():
        config_parts.append(OmegaConf.load(local_paths))
    elif require_local_paths:
        raise FileNotFoundError(
            "Local paths config is required but was not found: "
            f"{local_paths}. Create it from configs/local_paths.example.yaml."
        )

    for override_path in _iter_override_paths(overrides):
        if not override_path.exists():
            raise FileNotFoundError(f"Override config file not found: {override_path}")
        config_parts.append(OmegaConf.load(override_path))

    return OmegaConf.merge(*config_parts)


def load_legacy_default_config() -> DictConfig:
    """Load the historical complete config retained under configs/pre."""
    return load_yaml_config(LEGACY_DEFAULT_CONFIG)
