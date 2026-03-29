from __future__ import annotations

'''
purpose: formatting plaintext files for each URL in top K
input: plaintext files for each URL under data/processed/plaintext/<claim_id>, data/processed/evidence/rankings/topKURLs.jsonl
outputs: plaintext files for each URL in topKURLs.jsonl saved under data/processed/runs/<run_id>/cache/gold/gold_docs/<claim_id>/<URL_id>
'''

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ccir.io_utils import read_jsonl, read_text, write_text_atomic
from ccir.paths import Paths


#help logging
def _log_count(logger: Any, key: str, inc: int = 1) -> None:
    if logger is None:
        return

    if hasattr(logger, "count") and callable(getattr(logger, "count")):
        logger.count(key, inc)
        return

    if hasattr(logger, "increment") and callable(getattr(logger, "increment")):
        logger.increment(key, inc)
        return

    if hasattr(logger, "counts") and isinstance(getattr(logger, "counts"), dict):
        logger.counts[key] = logger.counts.get(key, 0) + inc


def _log_event(logger: Any, message: str, **fields: Any) -> None:
    if logger is None:
        return

    if hasattr(logger, "log") and callable(getattr(logger, "log")):
        logger.log(message, **fields)
        return

    if hasattr(logger, "info") and callable(getattr(logger, "info")):
        logger.info(message, **fields)
        return

    if hasattr(logger, "append") and callable(getattr(logger, "append")):
        logger.append({"message": message, **fields})


#parse rows
def _extract_topk_items(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("top_k_urls", "top_urls", "urls", "selected_urls", "ranked_urls"):
        value = row.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    return []


def _extract_url_id(url_row: Dict[str, Any]) -> Optional[str]:
    for key in ("url_id", "URL_ID", "id"):
        value = url_row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


#core logic
def run_step06(
    *,
    paths: Paths,
    config: Any = None,   # accepted for __main__.py compatibility; unused here
    logger: Any = None,
    log: Any = None,
    step_logger: Any = None,
    run_id: Optional[str] = None,
    code_version: Optional[str] = None,
    config_hash: Optional[str] = None,
) -> Dict[str, int]:
    active_logger = logger or log or step_logger

    topk_path = paths.run_evidence_topk_urls_jsonl
    counts: Dict[str, int] = {
        "rows_seen": 0,
        "urls_requested": 0,
        "copied": 0,
        "missing_source": 0,
        "skipped_bad_row": 0,
        "skipped_bad_url": 0,
    }

    if not topk_path.exists():
        raise FileNotFoundError(f"topKURLs.jsonl not found: {topk_path}")

    rows = read_jsonl(topk_path)

    _log_event(
        active_logger,
        "step06_build_gold_start",
        run_id=paths.run_id,
        topk_path=str(topk_path),
        gold_docs_dir=str(paths.cache_gold_docs_dir),
    )

    for row in rows:
        counts["rows_seen"] += 1
        _log_count(active_logger, "rows_seen", 1)

        if not isinstance(row, dict):
            counts["skipped_bad_row"] += 1
            _log_count(active_logger, "skipped_bad_row", 1)
            continue

        claim_id = row.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id.strip():
            counts["skipped_bad_row"] += 1
            _log_count(active_logger, "skipped_bad_row", 1)
            _log_event(active_logger, "step06_skip_bad_row_missing_claim_id", row=row)
            continue

        claim_id = claim_id.strip()
        topk_items = _extract_topk_items(row)

        if not topk_items:
            _log_event(
                active_logger,
                "step06_no_topk_items_for_claim",
                claim_id=claim_id,
            )
            continue

        for item in topk_items:
            counts["urls_requested"] += 1
            _log_count(active_logger, "urls_requested", 1)

            url_id = _extract_url_id(item)
            if not url_id:
                counts["skipped_bad_url"] += 1
                _log_count(active_logger, "skipped_bad_url", 1)
                _log_event(
                    active_logger,
                    "step06_skip_bad_url_missing_url_id",
                    claim_id=claim_id,
                    item=item,
                )
                continue

            src = paths.plaintext_path(claim_id, url_id)
            dst = paths.gold_doc_path(claim_id, url_id)

            if not src.exists():
                counts["missing_source"] += 1
                _log_count(active_logger, "missing_source", 1)
                _log_event(
                    active_logger,
                    "step06_missing_source_plaintext",
                    claim_id=claim_id,
                    url_id=url_id,
                    source=str(src),
                )
                continue

            text = read_text(src)
            write_text_atomic(dst, text)

            counts["copied"] += 1
            _log_count(active_logger, "copied", 1)

    _log_event(active_logger, "step06_build_gold_end", **counts)

    if counts["urls_requested"] > 0 and counts["copied"] == 0:
        raise RuntimeError(
            "Step 06 found ranked URLs but copied 0 documents. "
            "Check that step 04 produced plaintext files under cache/plaintext/<claim_id>/<url_id>.txt."
        )

    return counts


def run(**kwargs: Any) -> Dict[str, int]:
    return run_step06(**kwargs)


#CLI
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Copy top-L plaintext docs into cache/gold/gold_docs.")
    p.add_argument("run_id", help="Run id, e.g. pilot1")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    paths = Paths(run_id=args.run_id)
    paths.ensure_run_dirs()
    counts = run_step06(paths=paths)
    print(
        "step06_build_gold completed:",
        f"rows_seen={counts['rows_seen']},",
        f"urls_requested={counts['urls_requested']},",
        f"copied={counts['copied']},",
        f"missing_source={counts['missing_source']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
