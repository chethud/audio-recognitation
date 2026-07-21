"""Resolve which config.yaml to load (lean Render profile vs local)."""
from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_config_path(base: Path | None = None) -> Path:
    """
    Prefer ALM_CONFIG if set.
    On Render / ALM_LOW_MEMORY=1, use config.render.yaml when present.
    Otherwise config.yaml.
    """
    root = base or project_root()
    override = (os.environ.get("ALM_CONFIG") or "").strip()
    if override:
        path = Path(override)
        return path if path.is_absolute() else (root / path)

    low = (os.environ.get("ALM_LOW_MEMORY") or "").strip().lower() in ("1", "true", "yes")
    render = (os.environ.get("RENDER") or "").strip().lower() in ("1", "true", "yes")
    if low or render:
        lean = root / "config.render.yaml"
        if lean.is_file():
            return lean
    return root / "config.yaml"


def low_memory_mode() -> bool:
    if (os.environ.get("ALM_LOW_MEMORY") or "").strip().lower() in ("1", "true", "yes"):
        return True
    return (os.environ.get("RENDER") or "").strip().lower() in ("1", "true", "yes")
