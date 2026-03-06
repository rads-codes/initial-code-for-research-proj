"""
src/ccir/utils/env_loader.py

Purpose
- Load API keys (and other env vars) from a local env file (e.g., api_keys.env)
  early in the pipeline (called by __main__.py).

Design
- Dependency-free (no python-dotenv required).
- Import-safe: does not read files or mutate environment on import.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class EnvLoadResult:
    """Result of loading an env file."""
    used_file: Optional[Path]
    loaded: Dict[str, str]          # keys actually written into os.environ
    parsed: Dict[str, str]          # all key/value pairs parsed from the file


def _infer_repo_root() -> Path:
    """
    Infer repo root from this file location.

    Expected layout:
      <repo>/src/ccir/utils/env_loader.py
    """
    # env_loader.py -> utils (0) -> ccir (1) -> src (2) -> repo (3)
    return Path(__file__).resolve().parents[3]


def _candidate_env_paths(repo_root: Path) -> Tuple[Path, ...]:
    """
    Default search order for env files.
    """
    return (
        repo_root / "api_keys.env",
        repo_root / ".env",
        Path.cwd() / "api_keys.env",
        Path.cwd() / ".env",
    )


def _strip_inline_comment(value: str) -> str:
    """
    Remove an inline comment from a value in a conservative way.

    We only treat '#' as a comment delimiter if it is preceded by whitespace,
    e.g.:
      KEY=value # comment   -> "value"
      KEY=value#notcomment  -> "value#notcomment"
    """
    s = value
    idx = s.find(" #")
    if idx != -1:
        return s[:idx].rstrip()
    return s


def _unquote(value: str) -> str:
    """
    Strip matching single or double quotes around a value.
    """
    v = value.strip()
    if len(v) >= 2 and ((v[0] == v[-1]) and v[0] in ("'", '"')):
        return v[1:-1]
    return v


def _parse_env_file(text: str) -> Dict[str, str]:
    """
    Parse dotenv-like KEY=VALUE lines.
    Supports:
      - blank lines
      - full-line comments starting with '#'
      - optional 'export ' prefix
      - quoted values
      - inline comments after whitespace: 'VALUE # comment'
    """
    out: Dict[str, str] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()
            if not line:
                continue

        if "=" not in line:
            # ignore malformed lines rather than crashing; caller can enforce required_keys
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        value = value.strip()
        value = _strip_inline_comment(value)
        value = _unquote(value)

        out[key] = value

    return out


def load_env_file_if_present(
    env_path: Optional[os.PathLike | str] = None,
    *,
    override: bool = False,
    required_keys: Optional[Iterable[str]] = None,
) -> EnvLoadResult:
    """
    Load environment variables from a file if present.

    Args:
      env_path:
        - If provided: load exactly this file.
        - If None: search common filenames in repo root (api_keys.env, .env),
          then fall back to cwd.

      override:
        - If False (default): do NOT overwrite keys that already exist in os.environ.
        - If True: file values overwrite os.environ.

      required_keys:
        - If provided, enforce that these keys exist in os.environ after loading
          (either pre-existing or loaded from file). Missing -> RuntimeError.

    Returns:
      EnvLoadResult with:
        - used_file: the file that was loaded (or None if none found)
        - parsed: all parsed key/value pairs from the file (empty if none found)
        - loaded: only the keys that were actually written to os.environ

    Raises:
      RuntimeError if required_keys are missing after load.
      FileNotFoundError if env_path is explicitly provided but does not exist.
    """
    repo_root = _infer_repo_root()

    chosen: Optional[Path]
    if env_path is not None:
        chosen = Path(env_path)
        if not chosen.exists():
            raise FileNotFoundError(str(chosen))
        if not chosen.is_file():
            raise ValueError(f"Not a file: {chosen}")
    else:
        chosen = None
        for p in _candidate_env_paths(repo_root):
            if p.exists() and p.is_file():
                chosen = p
                break

    if chosen is None:
        # Nothing to do; still enforce required keys if requested.
        if required_keys is not None:
            missing = [k for k in required_keys if not os.getenv(k)]
            if missing:
                raise RuntimeError(
                    "Missing required environment variables (no env file found): "
                    + ", ".join(missing)
                )
        return EnvLoadResult(used_file=None, loaded={}, parsed={})

    text = chosen.read_text(encoding="utf-8")
    parsed = _parse_env_file(text)

    loaded: Dict[str, str] = {}
    for k, v in parsed.items():
        if override or (k not in os.environ):
            os.environ[k] = v
            loaded[k] = v

    if required_keys is not None:
        missing = [k for k in required_keys if not os.getenv(k)]
        if missing:
            raise RuntimeError(
                "Missing required environment variables after loading "
                f"{chosen}: " + ", ".join(missing)
            )

    return EnvLoadResult(used_file=chosen, loaded=loaded, parsed=parsed)

#how to use
'''
in src/ccir/__main__.py, call it early
from ccir.utils.env_loader import load_env_file_if_present

# Example: require both keys for a run that uses SerpAPI + OpenRouter
load_env_file_if_present(required_keys=["SERPAPI_API_KEY", "OPENROUTER_API_KEY"])

if you want the file to override whatever is in you shell
load_env_file_if_present(override=True)

if you want to point to a custom file
load_env_file_if_present(env_path="path/to/api_keys.env", required_keys=[...])

in pipeline entrypoint:
load_env_file_if_present(required_keys=["OPENROUTER_API_KEY"])

import os
print("OPENROUTER_API_KEY loaded:", bool(os.getenv("OPENROUTER_API_KEY")))
OPENROUTER_API_KEY loaded: True
'''