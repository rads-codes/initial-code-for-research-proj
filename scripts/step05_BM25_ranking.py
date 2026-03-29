from __future__ import annotations

"""
purpose: ranks using BM25, select top L for claim evidence pool
inputs: runs/<run_id>/evidence/URLs.jsonl, plaintext files for each URL under runs/<run_id>/cache/plaintext/<claim_id>/<url_id>.txt, runs/<run_id>/claims/forLLMs.jsonl (for claim_text)
output: runs/<run_id>/evidence/rankings/topKURLs.jsonl
"""

import argparse
import inspect
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from ccir.config_loader import load_config
from ccir.io_utils import read_jsonl, read_text, write_jsonl_atomic
from ccir.logging_utils import StepLogger
from ccir.paths import Paths
from ccir.retrieval import BM25 as bm25_module
from ccir.schemas import TopKURL, URLItem, to_dict, utc_now_iso


#small utilities
def _code_version() -> str:
    return os.getenv("CCIR_CODE_VERSION", "dev")


def _log_count(logger: Any, key: str, n: int = 1) -> None:
    """
    Compatibility helper for different logger APIs.
    """
    if logger is None:
        return

    if hasattr(logger, "count"):
        logger.count(key, n)
        return
    if hasattr(logger, "increment"):
        logger.increment(key, n)
        return
    if hasattr(logger, "add_count"):
        logger.add_count(key, n)
        return

    if hasattr(logger, "log"):
        logger.log({"type": "count", "key": key, "n": n})


def _safe_log(logger: Any, message: str, **kwargs: Any) -> None:
    if logger is None:
        return
    if hasattr(logger, "log"):
        logger.log(message, **kwargs)
    elif hasattr(logger, "info"):
        logger.info(message, **kwargs)


def _get_config_top_l(cfg: Any) -> int:
    """
    Read BM25 top-L from config.
    Preferred shape: cfg.bm25.top_l
    """
    if hasattr(cfg, "bm25") and hasattr(cfg.bm25, "top_l"):
        return int(cfg.bm25.top_l)
    if hasattr(cfg, "retrieval") and hasattr(cfg.retrieval, "top_l"):
        return int(cfg.retrieval.top_l)
    if hasattr(cfg, "top_l"):
        return int(cfg.top_l)
    if hasattr(cfg, "bm25_top_l"):
        return int(cfg.bm25_top_l)

    raise AttributeError(
        "Could not find BM25 top-L in config. Expected cfg.bm25.top_l or similar."
    )


def _claim_text_map(claim_rows: List[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in claim_rows:
        claim_id = str(row.get("claim_id", "")).strip()
        claim_text = str(row.get("claim_text", "")).strip()
        if claim_id and claim_text:
            out[claim_id] = claim_text
    return out


def _get_url_items(url_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expected primary field: row["urls"] from ClaimWithURLs.
    """
    value = url_row.get("urls")
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    return []


def _get_url_id(url_item: Dict[str, Any]) -> str:
    value = url_item.get("url_id")
    return str(value).strip() if value else ""


def _urlitem_from_dict(url_item: Dict[str, Any]) -> URLItem:
    """
    Convert a URL dict into a schema-valid URLItem.
    Ignores any extra keys such as temporary bm25_score.
    """
    return URLItem(
        url_id=str(url_item["url_id"]),
        url=str(url_item["url"]),
        title=url_item.get("title"),
        snippet=url_item.get("snippet"),
        source=url_item.get("source"),
        rank=url_item.get("rank"),
    )


#BM25
def _pick_bm25_callable(module: Any) -> Callable[..., Any]:
    """
    Try a few plausible function names from ccir.retrieval.bm25.py.
    """
    candidates = [
        "rank_documents",
        "rank_corpus",
        "bm25_rank",
        "score_documents",
        "score_corpus",
        "rank",
    ]
    for name in candidates:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn

    raise AttributeError(
        "Could not find a callable BM25 ranking function in ccir.retrieval.bm25. "
        f"Tried: {candidates}"
    )


def _normalize_rank_result(result: Any) -> List[Tuple[int, float]]:
    """
    Normalize common BM25 return shapes to [(doc_index, score), ...].
    """
    if isinstance(result, list) and all(isinstance(x, (int, float)) for x in result):
        return [(i, float(score)) for i, score in enumerate(result)]

    if isinstance(result, list) and all(
        isinstance(x, (tuple, list)) and len(x) >= 2 for x in result
    ):
        out: List[Tuple[int, float]] = []
        for item in result:
            out.append((int(item[0]), float(item[1])))
        return out

    if isinstance(result, list) and all(isinstance(x, dict) for x in result):
        out: List[Tuple[int, float]] = []
        for item in result:
            idx = item.get("index", item.get("idx", item.get("doc_index")))
            score = item.get("score", item.get("bm25_score"))
            if idx is None or score is None:
                continue
            out.append((int(idx), float(score)))
        if out:
            return out

    raise TypeError(
        "Unsupported BM25 return format. "
        "Expected list[float], list[(idx, score)], or list[dict]."
    )


def _call_bm25(
    bm25_fn: Callable[..., Any],
    *,
    query: str,
    documents: List[str],
) -> List[Tuple[int, float]]:
    """
    Call the BM25 function using introspection so the script tolerates
    small API differences in retrieval/bm25.py.
    """
    sig = inspect.signature(bm25_fn)
    kwargs: Dict[str, Any] = {}

    for name in sig.parameters:
        lname = name.lower()
        if lname in {"query", "claim", "claim_text", "text_query"}:
            kwargs[name] = query
        elif lname in {"documents", "docs", "corpus", "texts"}:
            kwargs[name] = documents
        elif lname in {"top_k", "k"}:
            kwargs[name] = len(documents)

    result = bm25_fn(**kwargs)
    return _normalize_rank_result(result)


#core step logic
def build_topk_rows(
    *,
    paths: Paths,
    top_l: int,
    logger: Any,
) -> List[Dict[str, Any]]:
    url_rows = read_jsonl(paths.evidence_urls)
    claim_rows = read_jsonl(paths.claims_for_llms)
    claim_text_by_id = _claim_text_map(url_rows=claim_rows) if False else _claim_text_map(claim_rows)

    bm25_fn = _pick_bm25_callable(bm25_module)

    out_rows: List[Dict[str, Any]] = []

    _log_count(logger, "claims_total", len(url_rows))

    for row in url_rows:
        claim_id = str(row.get("claim_id", "")).strip()
        if not claim_id:
            _log_count(logger, "claims_missing_id", 1)
            continue

        claim_text = claim_text_by_id.get(claim_id, "").strip()
        if not claim_text:
            _log_count(logger, "claims_with_no_claim_text", 1)
            _safe_log(logger, f"Skipping claim {claim_id}: no claim_text found")
            continue

        url_items = _get_url_items(row)
        if not url_items:
            _log_count(logger, "claims_with_no_url_items", 1)
            out_rows.append(
                to_dict(
                    TopKURL(
                        claim_id=claim_id,
                        top_urls=[],
                        run_id=paths.run_id,
                        created_utc=utc_now_iso(),
                        code_version=_code_version(),
                    )
                )
            )
            continue

        candidates: List[Dict[str, Any]] = []
        documents: List[str] = []

        _log_count(logger, "docs_candidates_total", len(url_items))

        for item in url_items:
            url_id = _get_url_id(item)
            if not url_id:
                _log_count(logger, "docs_missing_url_id", 1)
                continue

            txt_path = paths.plaintext_path(claim_id, url_id)
            if not txt_path.exists():
                _log_count(logger, "docs_missing_plaintext", 1)
                continue

            text = read_text(txt_path).strip()
            if not text:
                _log_count(logger, "docs_empty_plaintext", 1)
                continue

            candidates.append(item)
            documents.append(text)
            _log_count(logger, "docs_with_plaintext", 1)

        if not candidates:
            _log_count(logger, "claims_with_no_candidates", 1)
            _safe_log(logger, f"Claim {claim_id}: no usable plaintext docs")
            out_rows.append(
                to_dict(
                    TopKURL(
                        claim_id=claim_id,
                        top_urls=[],
                        run_id=paths.run_id,
                        created_utc=utc_now_iso(),
                        code_version=_code_version(),
                    )
                )
            )
            continue

        ranked = _call_bm25(
            bm25_fn,
            query=claim_text,
            documents=documents,
        )
        ranked_sorted = sorted(ranked, key=lambda x: x[1], reverse=True)

        selected_items: List[URLItem] = []
        seen: set[str] = set()

        for doc_idx, _score in ranked_sorted:
            if doc_idx < 0 or doc_idx >= len(candidates):
                continue

            item_dict = candidates[doc_idx]
            url_id = _get_url_id(item_dict)
            if not url_id or url_id in seen:
                continue

            try:
                selected_items.append(_urlitem_from_dict(item_dict))
                seen.add(url_id)
            except KeyError:
                _log_count(logger, "docs_invalid_url_item", 1)
                continue

            if len(selected_items) >= top_l:
                break

        _log_count(logger, "claims_ranked", 1)
        _log_count(logger, "docs_selected_total", len(selected_items))

        out_rows.append(
            to_dict(
                TopKURL(
                    claim_id=claim_id,
                    top_urls=selected_items,
                    run_id=paths.run_id,
                    created_utc=utc_now_iso(),
                    code_version=_code_version(),
                )
            )
        )

    return out_rows


def run_step05(
    *,
    paths: Paths,
    config: Any | None = None,
    logger: Any | None = None,
    log: Any | None = None,
    **_: Any,
) -> Path:
    active_logger = logger if logger is not None else log
    cfg = load_config()
    top_l = _get_config_top_l(cfg)

    rows = build_topk_rows(
        paths=paths,
        top_l=top_l,
        logger=logger,
    )
    write_jsonl_atomic(paths.evidence_topk_urls, rows)
    return paths.evidence_topk_urls


#CLI
def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Step 05: BM25 rank cached plaintext documents")
    ap.add_argument("run_id", help="Run ID, e.g. pilot1")
    return ap


def main() -> None:
    args = _build_argparser().parse_args()
    paths = Paths(run_id=args.run_id)
    paths.ensure_run_dirs()

    logger = StepLogger(
        step="05",
        report_path=paths.report_jsonl(5),
    )

    out_path = run_step(paths, logger=logger)
    _safe_log(logger, f"Wrote BM25 rankings to {out_path}")


if __name__ == "__main__":
    main()
