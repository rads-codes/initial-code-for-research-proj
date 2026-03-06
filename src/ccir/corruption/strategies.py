from __future__ import annotations

#Used by 08 to help make corruptions

"""
src/ccir/corruption/strategies.py

Purpose:
- Choose which sentence indices to modify for corruption step 08.
- Keep selection deterministic when a seeded random.Random is provided.
- Return sentence *indices* (0-based, sorted), not sentence text.

Supported strategies:
- random_drop
- targeted_drop
- replacement_mix
"""

import random
from typing import Any, Dict, List, Sequence


def _num_to_modify(n_sentences: int, pct: int) -> int:
    """
    Convert a percentage like 20 into a count of sentences to modify.

    Rules:
    - round to nearest integer
    - at least 1 if n_sentences > 0
    - at most n_sentences
    """
    if n_sentences <= 0:
        return 0
    k = int(round((pct / 100.0) * n_sentences))
    return max(1, min(n_sentences, k))


def _validate_rng(rng: random.Random | None) -> random.Random:
    return rng if rng is not None else random.Random(12345)


def _score_of(row: Dict[str, Any]) -> float:
    for key in ("embedding_score", "score", "cosine_similarity", "similarity"):
        val = row.get(key)
        if val is None:
            continue
        try:
            return float(val)
        except Exception:
            continue
    return 0.0


def pick_random_drop(
    sentences: Sequence[Dict[str, Any]],
    pct: int,
    rng: random.Random | None = None,
) -> List[int]:
    """
    Pick random sentence indices to delete.
    """
    rng = _validate_rng(rng)
    n = len(sentences)
    k = _num_to_modify(n, pct)
    if k == 0:
        return []
    return sorted(rng.sample(range(n), k))


def choose_random_drop(
    sentences: Sequence[Dict[str, Any]],
    pct: int,
    rng: random.Random | None = None,
) -> List[int]:
    return pick_random_drop(sentences, pct, rng)


def random_drop(
    sentences: Sequence[Dict[str, Any]],
    pct: int,
    rng: random.Random | None = None,
) -> List[int]:
    return pick_random_drop(sentences, pct, rng)


def pick_targeted_drop(
    sentences: Sequence[Dict[str, Any]],
    pct: int,
    rng: random.Random | None = None,
) -> List[int]:
    """
    Pick top-scoring sentence indices to delete.
    Higher embedding_score = more claim-relevant = removed first.
    Tie-breaker is original sentence order.
    """
    del rng  # unused, but accepted for signature compatibility

    n = len(sentences)
    k = _num_to_modify(n, pct)
    if k == 0:
        return []

    ranked = sorted(
        enumerate(sentences),
        key=lambda pair: (-_score_of(pair[1]), pair[0]),
    )
    return sorted(idx for idx, _row in ranked[:k])


def choose_targeted_drop(
    sentences: Sequence[Dict[str, Any]],
    pct: int,
    rng: random.Random | None = None,
) -> List[int]:
    return pick_targeted_drop(sentences, pct, rng)


def targeted_drop(
    sentences: Sequence[Dict[str, Any]],
    pct: int,
    rng: random.Random | None = None,
) -> List[int]:
    return pick_targeted_drop(sentences, pct, rng)


def pick_replacements(
    sentences: Sequence[Dict[str, Any]],
    pct: int,
    rng: random.Random | None = None,
) -> List[int]:
    """
    Pick random sentence indices to replace with filler content.
    """
    rng = _validate_rng(rng)
    n = len(sentences)
    k = _num_to_modify(n, pct)
    if k == 0:
        return []
    return sorted(rng.sample(range(n), k))


def choose_replacements(
    sentences: Sequence[Dict[str, Any]],
    pct: int,
    rng: random.Random | None = None,
) -> List[int]:
    return pick_replacements(sentences, pct, rng)


def replacement_mix(
    sentences: Sequence[Dict[str, Any]],
    pct: int,
    rng: random.Random | None = None,
) -> List[int]:
    return pick_replacements(sentences, pct, rng)