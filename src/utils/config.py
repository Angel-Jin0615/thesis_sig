from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def get_project_root() -> Path:
    """Return project root based on this file location."""
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load YAML config and resolve key paths relative to project root."""
    project_root = get_project_root()
    path = Path(config_path) if config_path else project_root / "configs" / "config.yaml"
    with path.open("r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    path_keys = [
        "raw_data_dir",
        "processed_data_dir",
        "cache_dir",
        "figures_dir",
        "tables_dir",
    ]
    for key in path_keys:
        if key in config:
            config[key] = str((project_root / config[key]).resolve())

    config["project_root"] = str(project_root)
    return config


def ensure_directories_from_config(config: Dict[str, Any]) -> None:
    """Create commonly used output directories from config."""
    dirs = [
        Path(config["raw_data_dir"]),
        Path(config["processed_data_dir"]),
        Path(config["cache_dir"]) / "signatures",
        Path(config["cache_dir"]) / "future_segments",
        Path(config["cache_dir"]) / "mean_signatures",
        Path(config["cache_dir"]) / "generated",
        Path(config["figures_dir"]),
        Path(config["tables_dir"]),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
