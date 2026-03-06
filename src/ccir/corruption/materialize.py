from __future__ import annotations

#Used by 08 to help make corruptions

"""
src/ccir/corruption/materialize.py

Purpose:
- Apply corruption choices to ordered sentence rows and return corrupted plaintext.
- This module does not write files itself; step 08 handles file output.

Supported materializers:
- materialize_drop(...)
- materialize_replace(...)

All outputs are newline-joined plaintext ending with a trailing newline.
"""

import random
from typing import Any, Dict, List, Sequence


def _validate_rng(rng: random.Random | None) -> random.Random:
    return rng if rng is not None else random.Random(12345)


def _sentence_text(row: Dict[str, Any]) -> str:
    for key in ("sentence_text", "sentence", "text", "sentenceText"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _clean_pool_item(text: str) -> str:
    return str(text).strip()


def materialize_drop(
    sentences: Sequence[Dict[str, Any]],
    drop_indices: Sequence[int],
) -> str:
    """
    Remove the selected sentences and return the remaining plaintext.
    """
    drop_set = set(drop_indices)
    kept: List[str] = []

    for i, row in enumerate(sentences):
        if i in drop_set:
            continue
        text = _sentence_text(row)
        if text:
            kept.append(text)

    if not kept:
        return ""

    return "\n".join(kept).strip() + "\n"


def apply_drop(
    sentences: Sequence[Dict[str, Any]],
    drop_indices: Sequence[int],
) -> str:
    return materialize_drop(sentences, drop_indices)


def render_drop(
    sentences: Sequence[Dict[str, Any]],
    drop_indices: Sequence[int],
) -> str:
    return materialize_drop(sentences, drop_indices)


def materialize_replace(
    sentences: Sequence[Dict[str, Any]],
    replace_indices: Sequence[int],
    replacement_pool: Sequence[str],
    rng: random.Random | None = None,
) -> str:
    """
    Replace selected sentence positions with random items from replacement_pool.

    If replacement_pool is empty, this degrades gracefully to dropping the selected
    sentences rather than failing.
    """
    rng = _validate_rng(rng)
    replace_set = set(replace_indices)

    if not replacement_pool:
        return materialize_drop(sentences, replace_indices)

    cleaned_pool = [_clean_pool_item(x) for x in replacement_pool if _clean_pool_item(x)]
    if not cleaned_pool:
        return materialize_drop(sentences, replace_indices)

    out: List[str] = []
    for i, row in enumerate(sentences):
        if i in replace_set:
            out.append(rng.choice(cleaned_pool))
        else:
            text = _sentence_text(row)
            if text:
                out.append(text)

    if not out:
        return ""

    return "\n".join(out).strip() + "\n"


def apply_replace(
    sentences: Sequence[Dict[str, Any]],
    replace_indices: Sequence[int],
    replacement_pool: Sequence[str],
    rng: random.Random | None = None,
) -> str:
    return materialize_replace(sentences, replace_indices, replacement_pool, rng)


def render_replace(
    sentences: Sequence[Dict[str, Any]],
    replace_indices: Sequence[int],
    replacement_pool: Sequence[str],
    rng: random.Random | None = None,
) -> str:
    return materialize_replace(sentences, replace_indices, replacement_pool, rng)