from __future__ import annotations

"""
src/ccir/retrieval/embeddings.py

Multilingual text embeddings for the CCIR pipeline.

Provides:
- embed_text(text) -> List[float]
- embed_texts(texts) -> List[List[float]]
- cosine_similarity(a, b) -> float

Design goals:
- import-safe: no model loads at import time
- multilingual by default
- deterministic inference
- simple JSON-serializable outputs
- minimal public API for step 07 and related retrieval code

Recommended default model:
- sentence-transformers/paraphrase-multilingual-mpnet-base-v2

You can override the model with:
- env var: CCIR_EMBEDDING_MODEL
"""

import os
from typing import Iterable, List, Optional, Sequence

import numpy as np


# -----------------------------
# Defaults
# -----------------------------

DEFAULT_EMBEDDING_MODEL = os.getenv(
    "CCIR_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)

# Lazy singleton cache
_MODEL = None
_MODEL_NAME: Optional[str] = None


# -----------------------------
# Internal helpers
# -----------------------------

def _require_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for ccir.retrieval.embeddings.\n"
            "Install it with:\n"
            "  pip install sentence-transformers\n"
            "You may also need torch installed depending on your environment."
        ) from e
    return SentenceTransformer


def _get_model(model_name: Optional[str] = None):
    """
    Lazily load and cache the embedding model.
    """
    global _MODEL, _MODEL_NAME

    chosen = model_name or DEFAULT_EMBEDDING_MODEL

    if _MODEL is not None and _MODEL_NAME == chosen:
        return _MODEL

    SentenceTransformer = _require_sentence_transformers()
    _MODEL = SentenceTransformer(chosen)
    _MODEL_NAME = chosen
    return _MODEL


def _normalize_input_text(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return " ".join(text.split()).strip()


def _normalize_vector(vec: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D vector, got shape {arr.shape}")
    return arr


# -----------------------------
# Public API
# -----------------------------

def embed_text(text: str, *, model_name: Optional[str] = None) -> List[float]:
    """
    Embed one text string and return a JSON-serializable vector.
    """
    cleaned = _normalize_input_text(text)
    model = _get_model(model_name)

    vec = model.encode(
        cleaned,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vec, dtype=np.float32).tolist()


def embed_texts(texts: Sequence[str], *, model_name: Optional[str] = None) -> List[List[float]]:
    """
    Embed a batch of texts and return JSON-serializable vectors.
    """
    if texts is None:
        raise ValueError("texts must not be None")

    cleaned_texts = [_normalize_input_text(t) for t in texts]
    model = _get_model(model_name)

    matrix = model.encode(
        cleaned_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    matrix = np.asarray(matrix, dtype=np.float32)

    if matrix.ndim == 1:
        # single item edge case from some backends
        return [matrix.tolist()]

    return [row.tolist() for row in matrix]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Cosine similarity between two vectors.

    Works with lists, tuples, or numpy arrays.
    Returns a Python float.
    """
    va = _normalize_vector(a)
    vb = _normalize_vector(b)

    if va.shape != vb.shape:
        raise ValueError(f"Vector shape mismatch: {va.shape} vs {vb.shape}")

    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0

    return float(np.dot(va, vb) / denom)


# -----------------------------
# Optional convenience helpers
# -----------------------------

def get_embedding_model_name() -> str:
    """
    Returns the currently configured embedding model name.
    """
    return _MODEL_NAME or DEFAULT_EMBEDDING_MODEL


def preload_model(model_name: Optional[str] = None) -> str:
    """
    Explicitly load the model early, if desired.
    Returns the loaded model name.
    """
    model = _get_model(model_name)
    _ = model  # silence linters
    return get_embedding_model_name()