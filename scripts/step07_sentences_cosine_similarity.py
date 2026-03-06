from __future__ import annotations

"""
scripts/07_sentences_cosine_similarity.py

Inputs:
  - data/processed/runs/<run_id>/claims/forLLMs.jsonl
  - data/processed/runs/<run_id>/evidence/rankings/topKURLs.jsonl
  - plaintext files under:
      data/processed/runs/<run_id>/cache/gold/gold_docs/<claim_id>/<url_id>.txt

Process:
  - For each claim in topKURLs.jsonl, compute one embedding for the claim_text
  - For each selected URL for that claim:
      - read the gold plaintext article
      - split into sentences
      - compute one embedding per sentence
      - write per-article sentences.jsonl
      - compute cosine similarity(claim_embedding, sentence_embedding)
      - write per-article embeddings.jsonl

Outputs:
  - data/processed/runs/<run_id>/cache/gold/<claim_id>/<url_id>/sentences.jsonl
  - data/processed/runs/<run_id>/cache/gold/<claim_id>/<url_id>/embeddings.jsonl
"""

import argparse
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from tqdm import tqdm

from ccir.config_loader import load_config
from ccir.io_utils import ensure_parent_dir, read_jsonl, read_text, write_jsonl_atomic
from ccir.paths import Paths
from ccir.retrieval.embeddings import embed_text, embed_texts, cosine_similarity

try:
    from ccir.logging_utils import StepLogger
except Exception:
    StepLogger = None  # type: ignore

try:
    from ccir.schemas import utc_now_iso
except Exception:
    from datetime import datetime, timezone

    def utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


STEP_ID = "07"

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_RE = re.compile(r"\s+")


# -----------------------------
# Logger compatibility helpers
# -----------------------------

def _log_count(logger: Any, key: str, value: int = 1) -> None:
    if logger is None:
        return

    for name in ("count", "inc", "increment"):
        fn = getattr(logger, name, None)
        if callable(fn):
            try:
                fn(key, value)
                return
            except TypeError:
                try:
                    fn(key)
                    return
                except Exception:
                    pass

    fn = getattr(logger, "log_count", None)
    if callable(fn):
        try:
            fn(key, value)
            return
        except TypeError:
            try:
                fn(key)
                return
            except Exception:
                pass


def _log_metric(logger: Any, key: str, value: Any) -> None:
    if logger is None:
        return

    for name in ("metric", "log_metric", "set_metric"):
        fn = getattr(logger, name, None)
        if callable(fn):
            try:
                fn(key, value)
                return
            except Exception:
                pass


def _log_message(logger: Any, message: str, **extra: Any) -> None:
    if logger is None:
        if extra:
            print(f"{message} | {extra}")
        else:
            print(message)
        return

    for name in ("message", "log", "info", "event"):
        fn = getattr(logger, name, None)
        if callable(fn):
            try:
                fn(message, **extra)
                return
            except TypeError:
                try:
                    fn(message)
                    return
                except Exception:
                    pass

    if extra:
        print(f"{message} | {extra}")
    else:
        print(message)


# -----------------------------
# Helpers
# -----------------------------

def _code_version(config: Any = None, explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    if config is not None:
        cfg_cv = getattr(config, "code_version", None)
        if cfg_cv:
            return str(cfg_cv)
    return os.getenv("CCIR_CODE_VERSION", "dev")


def _normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip())


def _split_sentences(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    text = re.sub(r"\n+", " ", text)
    text = _normalize_text(text)

    if not text:
        return []

    parts = _SENTENCE_SPLIT_RE.split(text)
    out: List[str] = []
    for part in parts:
        sent = _normalize_text(part)
        if sent:
            out.append(sent)
    return out


def _sentence_id(i: int, sentence_text_norm: str) -> str:
    digest = hashlib.sha256(sentence_text_norm.encode("utf-8")).hexdigest()[:8]
    return f"s{i:04d}_{digest}"


def _to_jsonable_vector(vec: Any) -> List[float]:
    if vec is None:
        return []

    if hasattr(vec, "tolist"):
        vec = vec.tolist()

    if isinstance(vec, (list, tuple)):
        return [float(x) for x in vec]

    return [float(vec)]


def _cosine_to_float(a: Any, b: Any) -> float:
    score = cosine_similarity(a, b)
    if hasattr(score, "item"):
        return float(score.item())
    return float(score)


def _build_claim_text_map(rows: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in rows:
        claim_id = row.get("claim_id")
        claim_text = row.get("claim_text")
        if isinstance(claim_id, str) and isinstance(claim_text, str) and claim_text.strip():
            out[claim_id] = claim_text.strip()
    return out


def _extract_url_items(row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    candidate_keys = [
        "urls",
        "top_urls",
        "top_k_urls",
        "selected_urls",
        "ranked_urls",
    ]
    for key in candidate_keys:
        value = row.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    return []


def _extract_url_id(url_row: Mapping[str, Any]) -> Optional[str]:
    for key in ("url_id", "URL_ID", "id"):
        value = url_row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _article_output_dir(paths: Paths, claim_id: str, url_id: str) -> Path:
    return paths.cache_gold_dir / claim_id / url_id


def _sentences_jsonl_path(paths: Paths, claim_id: str, url_id: str) -> Path:
    return _article_output_dir(paths, claim_id, url_id) / "sentences.jsonl"


def _embeddings_jsonl_path(paths: Paths, claim_id: str, url_id: str) -> Path:
    return _article_output_dir(paths, claim_id, url_id) / "embeddings.jsonl"


# -----------------------------
# Core step
# -----------------------------

def run_step07(
    *,
    paths: Paths,
    config: Any = None,
    log: Any = None,
    logger: Any = None,
    code_version: Optional[str] = None,
    **_: Any,
) -> Dict[str, int]:
    active_logger = logger if logger is not None else log

    claims_path = paths.run_claims_for_llms_jsonl
    topk_path = paths.run_evidence_topk_urls_jsonl

    claim_rows = read_jsonl(claims_path)
    topk_rows = read_jsonl(topk_path)
    claim_text_by_id = _build_claim_text_map(claim_rows)

    cv = _code_version(config=config, explicit=code_version)

    counts: Dict[str, int] = {
        "claims_seen": 0,
        "claims_processed": 0,
        "claims_missing_text": 0,
        "claims_with_no_topk_urls": 0,
        "articles_seen": 0,
        "articles_missing_gold_doc": 0,
        "articles_empty_gold_doc": 0,
        "articles_no_sentences": 0,
        "articles_written": 0,
        "sentences_written": 0,
        "embeddings_written": 0,
    }

    _log_message(
        active_logger,
        "step07_start",
        claims_path=str(claims_path),
        topk_path=str(topk_path),
        num_claim_rows=len(claim_rows),
        num_topk_rows=len(topk_rows),
    )

    claim_bar = tqdm(
        topk_rows,
        desc="Step 07 claims",
        unit="claim",
        dynamic_ncols=True,
    )

    for topk_row in claim_bar:
        claim_id = topk_row.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id.strip():
            continue
        claim_id = claim_id.strip()

        counts["claims_seen"] += 1
        _log_count(active_logger, "claims_seen", 1)

        claim_bar.set_postfix_str(f"claim_id={claim_id}")

        claim_text = claim_text_by_id.get(claim_id)
        if not claim_text:
            counts["claims_missing_text"] += 1
            _log_count(active_logger, "claims_missing_text", 1)
            _log_message(active_logger, "missing_claim_text", claim_id=claim_id)
            continue

        url_items = _extract_url_items(topk_row)
        if not url_items:
            counts["claims_with_no_topk_urls"] += 1
            _log_count(active_logger, "claims_with_no_topk_urls", 1)
            _log_message(active_logger, "no_topk_urls", claim_id=claim_id)
            continue

        claim_embedding_raw = embed_text(claim_text)
        claim_embedding_json = _to_jsonable_vector(claim_embedding_raw)

        article_bar = tqdm(
            url_items,
            desc=f"Articles for {claim_id}",
            unit="article",
            leave=False,
            dynamic_ncols=True,
        )

        for url_row in article_bar:
            url_id = _extract_url_id(url_row)
            if not url_id:
                _log_message(active_logger, "missing_url_id", claim_id=claim_id)
                continue

            article_bar.set_postfix_str(f"url_id={url_id}")

            counts["articles_seen"] += 1
            _log_count(active_logger, "articles_seen", 1)

            gold_doc_path = paths.gold_doc_path(claim_id, url_id)
            if not gold_doc_path.exists():
                counts["articles_missing_gold_doc"] += 1
                _log_count(active_logger, "articles_missing_gold_doc", 1)
                _log_message(
                    active_logger,
                    "missing_gold_doc",
                    claim_id=claim_id,
                    url_id=url_id,
                    path=str(gold_doc_path),
                )
                continue

            article_text = read_text(gold_doc_path).strip()
            if not article_text:
                counts["articles_empty_gold_doc"] += 1
                _log_count(active_logger, "articles_empty_gold_doc", 1)
                _log_message(
                    active_logger,
                    "empty_gold_doc",
                    claim_id=claim_id,
                    url_id=url_id,
                    path=str(gold_doc_path),
                )
                continue

            sentences = _split_sentences(article_text)
            if not sentences:
                counts["articles_no_sentences"] += 1
                _log_count(active_logger, "articles_no_sentences", 1)
                _log_message(
                    active_logger,
                    "no_sentences_extracted",
                    claim_id=claim_id,
                    url_id=url_id,
                )
                continue

            sentence_embeddings_raw = embed_texts(sentences)
            if len(sentence_embeddings_raw) != len(sentences):
                raise RuntimeError(
                    f"embed_texts returned {len(sentence_embeddings_raw)} embeddings "
                    f"for {len(sentences)} sentences "
                    f"(claim_id={claim_id}, url_id={url_id})"
                )

            sentences_out = _sentences_jsonl_path(paths, claim_id, url_id)
            embeddings_out = _embeddings_jsonl_path(paths, claim_id, url_id)

            sentence_rows: List[Dict[str, Any]] = []
            embedding_rows: List[Dict[str, Any]] = []

            for i, (sentence_text, sentence_embedding_raw) in enumerate(zip(sentences, sentence_embeddings_raw)):
                sentence_text_norm = _normalize_text(sentence_text)
                sentence_id = _sentence_id(i, sentence_text_norm)
                sentence_embedding_json = _to_jsonable_vector(sentence_embedding_raw)
                embedding_score = _cosine_to_float(claim_embedding_raw, sentence_embedding_raw)

                base_row = {
                    "run_id": paths.run_id,
                    "created_utc": utc_now_iso(),
                    "code_version": cv,
                    "claim_id": claim_id,
                    "claim_text_embedding": claim_embedding_json,
                    "url_id": url_id,
                    "sentence_id": sentence_id,
                    "sentence_text": sentence_text,
                    "sentence_embedding": sentence_embedding_json,
                }

                sentence_rows.append(base_row)

                scored_row = dict(base_row)
                scored_row["embedding_score"] = embedding_score
                embedding_rows.append(scored_row)

            ensure_parent_dir(sentences_out)
            ensure_parent_dir(embeddings_out)
            write_jsonl_atomic(sentences_out, sentence_rows)
            write_jsonl_atomic(embeddings_out, embedding_rows)

            counts["articles_written"] += 1
            counts["sentences_written"] += len(sentence_rows)
            counts["embeddings_written"] += len(embedding_rows)

            _log_count(active_logger, "articles_written", 1)
            _log_count(active_logger, "sentences_written", len(sentence_rows))
            _log_count(active_logger, "embeddings_written", len(embedding_rows))

        counts["claims_processed"] += 1
        _log_count(active_logger, "claims_processed", 1)

    for key, value in counts.items():
        _log_metric(active_logger, key, value)

    _log_message(active_logger, "step07_done", **counts)
    return counts


# -----------------------------
# Standalone CLI
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 07: sentence embeddings + cosine similarity")
    p.add_argument("--run-id", required=True, help="Run id, e.g. pilot1")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)

    config = load_config()
    paths = Paths(run_id=args.run_id)
    paths.ensure_run_dirs()

    local_logger = None
    if StepLogger is not None:
        try:
            local_logger = StepLogger(paths=paths, step=STEP_ID)
        except Exception:
            local_logger = None

    run_step07(
        paths=paths,
        config=config,
        logger=local_logger,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())