from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, List, Dict, Union, Optional

'''
Should have the functions read_jsonl(), write_jsonl_atoms(), ensure_parent_dir(), read_text(), write_text_atomic(), read_jsonl(path), write_jsonl_atomic(path, rows), append_jsonl(path, row), ensure_parent_dir(path)
'''
"""
src/ccir/io_utils.py

Purpose
- Single, consistent place for reading/writing JSONL and text files.
- Provide atomic writes (write temp -> fsync -> os.replace) to avoid partial outputs.

Used by
- All step scripts (00-13) and utilities that read/write run artifacts.
"""

JsonObj = Dict[str, Any]
RowLike = Union[JsonObj, Any]  # dict or dataclass instance (validated at runtime)


def ensure_parent_dir(path: Path) -> None:
    """Ensure the parent directory for `path` exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def read_text(path: Path, *, encoding: str = "utf-8") -> str:
    """Read a UTF-8 text file."""
    return path.read_text(encoding=encoding)


def _atomic_replace(tmp_path: Path, final_path: Path) -> None:
    """
    Replace final_path with tmp_path atomically (best-effort cross-platform).
    tmp_path must be on the same filesystem as final_path.
    """
    os.replace(str(tmp_path), str(final_path))


def write_text_atomic(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    newline_eof: bool = False,
) -> None:
    """
    Atomically write text to `path`.
    - Writes to a temp file in the same directory, fsyncs, then os.replace().
    - If newline_eof=True, ensures the file ends with a newline.
    """
    ensure_parent_dir(path)

    if newline_eof and text and not text.endswith("\n"):
        text = text + "\n"

    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with open(tmp_path, "w", encoding=encoding, newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        _atomic_replace(tmp_path, path)
    finally:
        # Best-effort cleanup if something went wrong before replace()
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _normalize_row(row: RowLike) -> JsonObj:
    """
    Convert supported row types to a JSON object (dict).
    Supported:
    - dict
    - dataclass instance (converted via asdict)
    """
    if isinstance(row, dict):
        return row
    if is_dataclass(row) and not isinstance(row, type):
        return asdict(row)
    raise TypeError(
        f"JSONL row must be a dict or dataclass instance; got {type(row).__name__}"
    )


def read_jsonl(path: Path, *, encoding: str = "utf-8") -> List[JsonObj]:
    """
    Read JSON Lines file into a list of dicts.
    - Skips blank lines.
    - Raises ValueError on malformed JSON or non-object rows.
    """
    out: List[JsonObj] = []
    with open(path, "r", encoding=encoding) as f:
        for lineno, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON on line {lineno} in {path}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Expected JSON object per line in {path}, got {type(obj).__name__} on line {lineno}"
                )
            out.append(obj)
    return out


def write_jsonl_atomic(
    path: Path,
    rows: Iterable[RowLike],
    *,
    encoding: str = "utf-8",
    sort_keys: bool = False,
) -> None:
    """
    Atomically write rows to JSONL at `path`.
    - Each row must be a dict or dataclass instance.
    - Writes compact JSON by default; 1 object per line; always ends with newline.
    """
    ensure_parent_dir(path)

    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with open(tmp_path, "w", encoding=encoding, newline="\n") as f:
            for row in rows:
                obj = _normalize_row(row)
                line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=sort_keys)
                f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        _atomic_replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def append_jsonl(
    path: Path,
    row: RowLike,
    *,
    encoding: str = "utf-8",
    sort_keys: bool = False,
) -> None:
    """
    Append a single JSON object as one line to `path`.
    - Ensures parent dir exists.
    - Ensures there is a newline boundary before appending if file is non-empty.
    - fsyncs the append for durability.
    """
    ensure_parent_dir(path)
    obj = _normalize_row(row)
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=sort_keys)

    # Open in append+read mode so we can check last byte if needed.
    with open(path, "a+", encoding=encoding, newline="\n") as f:
        try:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size > 0:
                f.seek(size - 1, os.SEEK_SET)
                last = f.read(1)
                if last != "\n":
                    f.write("\n")
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        except OSError as e:
            raise RuntimeError(f"Failed appending JSONL row to {path}: {e}") from e


# Backwards-compat alias (outline typo)
write_jsonl_atoms = write_jsonl_atomic

#using this
'''
from pathlib import Path
from ccir.io_utils import read_jsonl, write_jsonl_atomic, append_jsonl, read_text, write_text_atomic

rows = read_jsonl(Path("data/processed/claims/all.jsonl"))

write_jsonl_atomic(Path("data/processed/claims/forLLMs.jsonl"), rows_subset)

append_jsonl(Path("data/processed/claims/report_01.jsonl"), {"event": "kept", "n": 123})

text = read_text(Path("some_doc.txt"))
write_text_atomic(Path("out.txt"), text, newline_eof=True)
'''