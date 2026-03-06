#Used for BM25 ranking in 05.
from __future__ import annotations

"""
src/ccir/retrieval/bm25.py

Pure-Python BM25 ranking helpers for document retrieval.

Primary use in this project:
- Given a claim text and a list of candidate plaintext documents,
  compute BM25 scores and rank the documents.

Design goals:
- No external dependency required
- Deterministic behavior
- Simple API that step05_BM25_ranking.py can call
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


# -----------------------------
# Tokenization
# -----------------------------

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> List[str]:
    """
    Simple Unicode-aware tokenizer.

    Notes:
    - Lowercases text
    - Extracts word-like tokens with \\w+
    - Keeps digits if present in text
    """
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


# -----------------------------
# BM25 core
# -----------------------------

@dataclass(frozen=True)
class BM25Config:
    """
    Standard BM25 parameters.
    """
    k1: float = 1.5
    b: float = 0.75
    epsilon_idf: float = 0.25


class BM25Scorer:
    """
    BM25 scorer over a fixed corpus of documents.
    """

    def __init__(
        self,
        documents: Sequence[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon_idf: float = 0.25,
    ) -> None:
        self.documents: List[str] = list(documents)
        self.k1 = float(k1)
        self.b = float(b)
        self.epsilon_idf = float(epsilon_idf)

        self.tokenized_docs: List[List[str]] = [tokenize(doc) for doc in self.documents]
        self.doc_freqs: List[Counter[str]] = [Counter(doc) for doc in self.tokenized_docs]
        self.doc_lengths: List[int] = [len(doc) for doc in self.tokenized_docs]
        self.corpus_size: int = len(self.tokenized_docs)
        self.avgdl: float = (
            sum(self.doc_lengths) / self.corpus_size if self.corpus_size > 0 else 0.0
        )

        self.df: Dict[str, int] = self._compute_document_frequencies(self.tokenized_docs)
        self.idf: Dict[str, float] = self._compute_idf()

    @staticmethod
    def _compute_document_frequencies(tokenized_docs: Sequence[Sequence[str]]) -> Dict[str, int]:
        df: Dict[str, int] = {}
        for doc in tokenized_docs:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        return df

    def _compute_idf(self) -> Dict[str, float]:
        """
        Compute BM25 IDF.

        Uses the common Robertson-style variant:
          idf(t) = log(1 + (N - df + 0.5) / (df + 0.5))

        This stays positive and behaves well in small corpora.
        """
        idf: Dict[str, float] = {}
        n = self.corpus_size

        if n == 0:
            return idf

        for term, df_t in self.df.items():
            value = math.log(1.0 + (n - df_t + 0.5) / (df_t + 0.5))
            idf[term] = value

        return idf

    def score_tokens(self, query_tokens: Sequence[str], doc_index: int) -> float:
        """
        Score a single document for the given tokenized query.
        """
        if doc_index < 0 or doc_index >= self.corpus_size:
            raise IndexError(f"doc_index out of range: {doc_index}")

        if self.corpus_size == 0:
            return 0.0

        tf = self.doc_freqs[doc_index]
        dl = self.doc_lengths[doc_index]

        if dl == 0:
            return 0.0

        score = 0.0
        query_counts = Counter(query_tokens)

        for term, qf in query_counts.items():
            f = tf.get(term, 0)
            if f == 0:
                continue

            idf_t = self.idf.get(term)
            if idf_t is None:
                continue

            numerator = f * (self.k1 + 1.0)
            denominator = f + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl if self.avgdl > 0 else 0.0))
            score += idf_t * (numerator / denominator) * qf

        return float(score)

    def score(self, query: str, doc_index: int) -> float:
        """
        Score a single document for a raw query string.
        """
        return self.score_tokens(tokenize(query), doc_index)

    def score_all(self, query: str) -> List[float]:
        """
        Return BM25 score for every document in corpus order.
        """
        query_tokens = tokenize(query)
        return [self.score_tokens(query_tokens, i) for i in range(self.corpus_size)]

    def rank(self, query: str, top_k: int | None = None) -> List[Tuple[int, float]]:
        """
        Return ranked results as [(doc_index, score), ...], descending by score.
        """
        scores = self.score_all(query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        if top_k is not None:
            if top_k < 0:
                raise ValueError("top_k must be >= 0")
            ranked = ranked[:top_k]

        return [(int(i), float(score)) for i, score in ranked]


# -----------------------------
# Public helper APIs
# -----------------------------

def score_documents(
    query: str,
    documents: Sequence[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    """
    Return BM25 score per document in original corpus order.

    Output shape:
      [score_doc0, score_doc1, ...]
    """
    scorer = BM25Scorer(documents, k1=k1, b=b)
    return scorer.score_all(query)


def rank_documents(
    query: str,
    documents: Sequence[str],
    *,
    top_k: int | None = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[Tuple[int, float]]:
    """
    Return ranked BM25 results.

    Output shape:
      [(doc_index, score), ...]
    """
    scorer = BM25Scorer(documents, k1=k1, b=b)
    return scorer.rank(query, top_k=top_k)


# -----------------------------
# Compatibility aliases
# -----------------------------

def bm25_rank(
    query: str,
    documents: Sequence[str],
    *,
    top_k: int | None = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[Tuple[int, float]]:
    return rank_documents(query, documents, top_k=top_k, k1=k1, b=b)


def rank_corpus(
    query: str,
    corpus: Sequence[str],
    *,
    top_k: int | None = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[Tuple[int, float]]:
    return rank_documents(query, corpus, top_k=top_k, k1=k1, b=b)


def score_corpus(
    query: str,
    corpus: Sequence[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    return score_documents(query, corpus, k1=k1, b=b)


def rank(
    query: str,
    documents: Sequence[str],
    *,
    top_k: int | None = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[Tuple[int, float]]:
    return rank_documents(query, documents, top_k=top_k, k1=k1, b=b)