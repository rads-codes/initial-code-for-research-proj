#Should allow config_loader.load_config() in order to import configs.py for main
"""
src/ccir/config_loader.py

Responsibilities (per project outline):
- load_config() returns a RunConfig object backed by configs.py (DEFAULT_CONFIG).
- Support an optional module override for flexible "run presets".
- Validate the config before returning (fail fast on invalid settings).

This module should be import-safe:
- No filesystem writes
- No API key / env file loading (handled elsewhere)
"""

from __future__ import annotations

import importlib
import os
from typing import Optional, Any

from ccir.configs import RunConfig  # type: ignore


DEFAULT_CONFIG_MODULE = "ccir.configs"
ENV_CONFIG_MODULE = "CCIR_CONFIG_MODULE"


class ConfigLoadError(RuntimeError):
    """Raised when the config module/object cannot be loaded correctly."""


def _import_module(module_path: str):
    try:
        return importlib.import_module(module_path)
    except Exception as e:  # noqa: BLE001 (we want a precise wrapper error)
        raise ConfigLoadError(
            f"Failed to import config module '{module_path}'. "
            f"Set {ENV_CONFIG_MODULE} to a valid import path, e.g. '{DEFAULT_CONFIG_MODULE}'. "
            f"Original error: {type(e).__name__}: {e}"
        ) from e


def _resolve_config_obj(mod: Any) -> RunConfig:
    # Preferred convention: DEFAULT_CONFIG
    if hasattr(mod, "DEFAULT_CONFIG"):
        cfg = getattr(mod, "DEFAULT_CONFIG")
    # Fallback convention: get_config() -> RunConfig
    elif hasattr(mod, "get_config") and callable(getattr(mod, "get_config")):
        cfg = mod.get_config()
    else:
        raise ConfigLoadError(
            "Config module must define either:\n"
            "  - DEFAULT_CONFIG: RunConfig\n"
            "or\n"
            "  - get_config() -> RunConfig\n"
            f"Module loaded: {getattr(mod, '__name__', repr(mod))}"
        )

    if not isinstance(cfg, RunConfig):
        raise ConfigLoadError(
            "Loaded config object is not a RunConfig.\n"
            f"Got type: {type(cfg).__name__}\n"
            "Ensure DEFAULT_CONFIG (or get_config()) returns ccir.configs.RunConfig."
        )

    return cfg


def load_config(module_path: Optional[str] = None) -> RunConfig:
    """
    Load and validate a RunConfig.

    Resolution order:
    1) explicit argument `module_path` (if provided)
    2) env var CCIR_CONFIG_MODULE (if set)
    3) default module 'ccir.configs'

    Returns:
        RunConfig (validated)
    """
    chosen_module = (
        module_path
        or os.getenv(ENV_CONFIG_MODULE, "").strip()
        or DEFAULT_CONFIG_MODULE
    )

    mod = _import_module(chosen_module)
    cfg = _resolve_config_obj(mod)

    # Fail fast: enforce invariants like L <= K, levels in (0,1), etc.
    cfg.validate()

    return cfg