from __future__ import annotations

"""
purpose: cleans text, checks character limit
input: runs/<run_id>/evidence/URLs.jsonl
output: data/processed/runs/<run_id>/cache/plaintext/<claim_id>/<url_id>.txt
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from ccir.config_loader import load_config
from ccir.io_utils import ensure_parent_dir, read_jsonl, write_text_atomic
from ccir.logging_utils import StepLogger
from ccir.paths import Paths


#helps logging
def _log_count(logger: Any, key: str, n: int = 1) -> None:
    if logger is None:
        return

    fn = getattr(logger, "count", None)
    if callable(fn):
        fn(key, n)
        return

    fn = getattr(logger, "counts", None)
    if callable(fn):
        try:
            fn({key: n})
            return
        except TypeError:
            pass
        try:
            fn(key, n)
            return
        except TypeError:
            pass


def _log_event(logger: Any, event: str, payload: Dict[str, Any]) -> None:
    fn = getattr(logger, "log_event", None)
    if callable(fn):
        fn(event, payload)


#small utilities
def _get_nested(obj: Any, path: str, default: Any = None) -> Any:
    """Read dotted attributes safely, e.g. 'retrieval.min_chars'."""
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur


def _resolve_clean_fn() -> Callable[[str], str]:
    """Resolve a clean_text function from common module/name variants."""
    for mod_name in ("ccir.clean_text", "ccir.cleanText"):
        try:
            mod = __import__(mod_name, fromlist=["*"])
            for fn_name in ("clean_text", "cleanText", "clean"):
                fn = getattr(mod, fn_name, None)
                if callable(fn):
                    return fn
        except Exception:
            continue

    raise ImportError(
        "Could not find a text cleaning function. "
        "Expected ccir.clean_text.clean_text(...) (preferred)."
    )


def _resolve_fetch_fn() -> Callable[..., Optional[str]]:
    """Resolve a fetch function from ccir.web.fetch."""
    try:
        mod = __import__("ccir.web.fetch", fromlist=["*"])
    except Exception as e:
        raise ImportError("Could not import ccir.web.fetch") from e

    candidate_names = (
        "fetch_url_text",
        "fetch_and_extract",
        "fetch_extract_text",
        "fetch_text",
        "fetch",
    )
    for name in candidate_names:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn

    for name in dir(mod):
        if "fetch" in name.lower():
            fn = getattr(mod, name, None)
            if callable(fn):
                return fn

    raise ImportError(
        "Could not find a fetch function in ccir.web.fetch. "
        "Define one of: fetch_url_text, fetch_and_extract, fetch_extract_text, fetch_text, fetch."
    )


def _iter_urls_from_claim_row(row: Dict[str, Any]) -> Iterable[Tuple[str, str, str]]:
    """Yield (claim_id, url_id, url) from a ClaimWithURLs-like row."""
    claim_id = row.get("claim_id") or row.get("id") or row.get("claimId")
    if not claim_id:
        return

    urls = row.get("urls") or row.get("URLS") or row.get("evidence_urls") or []
    if not isinstance(urls, list):
        return

    for u in urls:
        if not isinstance(u, dict):
            continue
        url_id = u.get("url_id") or u.get("URL_ID") or u.get("id") or u.get("urlId")
        url = u.get("canonical_url") or u.get("url") or u.get("URL") or u.get("link")
        if claim_id and url_id and url:
            yield str(claim_id), str(url_id), str(url)


# -----------------------------
# Progress helpers
# -----------------------------
class _SimpleProgress:
    """
    progress bar
    """

    def __init__(self, total: int, desc: str = "step04") -> None:
        self.total = max(int(total), 0)
        self.desc = desc
        self.current = 0
        self.start_time = time.time()
        self._last_print_time = 0.0
        self._postfix: Dict[str, Any] = {}

        if self.total == 0:
            print(f"{self.desc}: 0/0 | no urls found", flush=True)

    def set_postfix(self, stats: Dict[str, Any], refresh: bool = False) -> None:
        self._postfix = dict(stats)
        if refresh:
            self._print(force=True)

    def update(self, n: int = 1) -> None:
        self.current += n
        self._print(force=(self.current >= self.total))

    def _print(self, force: bool = False) -> None:
        now = time.time()
        should_print = force or (now - self._last_print_time >= 0.5)
        if not should_print:
            return

        self._last_print_time = now
        elapsed = now - self.start_time
        rate = (self.current / elapsed) if elapsed > 0 else 0.0
        remaining = self.total - self.current
        eta = (remaining / rate) if rate > 0 else 0.0

        postfix_str = ""
        if self._postfix:
            ordered = ["written", "skipped", "too_short", "empty", "errors"]
            parts = [f"{k}={self._postfix[k]}" for k in ordered if k in self._postfix]
            postfix_str = " | " + " ".join(parts)

        msg = (
            f"\r{self.desc}: {self.current}/{self.total}"
            f" | elapsed {elapsed:.1f}s | eta {eta:.1f}s{postfix_str}"
        )
        sys.stdout.write(msg)
        sys.stdout.flush()

        if self.current >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()


def _progress_iter(
    items: List[Tuple[str, str, str]],
    enabled: bool,
    desc: str = "step04",
):
    if not enabled:
        return iter(items)

    try:
        from tqdm import tqdm  # type: ignore
        return tqdm(items, total=len(items), desc=desc, unit="url", dynamic_ncols=True)
    except Exception:
        progress = _SimpleProgress(total=len(items), desc=desc)

        class _WrappedIterator:
            def __iter__(self) -> Iterator[Tuple[str, str, str]]:
                for item in items:
                    yield item
                    progress.update(1)

            def set_postfix(self, stats: Dict[str, Any], refresh: bool = False) -> None:
                progress.set_postfix(stats, refresh=refresh)

        return _WrappedIterator()


def _print_summary(
    *,
    urls_jsonl: Path,
    plaintext_root: Path,
    total_rows: int,
    total_urls: int,
    stats: Dict[str, int],
) -> None:
    print("\nSTEP04 SUMMARY", flush=True)
    print(f"input_jsonl={urls_jsonl}", flush=True)
    print(f"output_dir={plaintext_root}", flush=True)
    print(f"claim_rows={total_rows}", flush=True)
    print(f"urls_total={total_urls}", flush=True)
    print(f"written={stats['written']}", flush=True)
    print(f"skipped={stats['skipped']}", flush=True)
    print(f"too_short={stats['too_short']}", flush=True)
    print(f"empty={stats['empty']}", flush=True)
    print(f"errors={stats['errors']}", flush=True)


#step runner
def run_step04_cache_url_content(
    *,
    paths: Paths,
    config: Any,
    logger: Any,
    overwrite: bool = False,
    show_progress: bool = True,
    verbose_errors: bool = True,
) -> None:
    min_chars = int(_get_nested(config, "retrieval.min_chars", 1200))
    max_chars = int(_get_nested(config, "retrieval.max_chars", 10000))
    timeout_s = float(_get_nested(config, "retrieval.fetch_timeout_s", 25))
    retries = int(_get_nested(config, "retrieval.fetch_retries", 2))

    clean_fn = _resolve_clean_fn()
    fetch_fn = _resolve_fetch_fn()

    urls_jsonl = paths.run_evidence_urls_jsonl
    plaintext_root = paths.cache_plaintext_dir

    if not urls_jsonl.exists():
        raise FileNotFoundError(f"Step04 input not found: {urls_jsonl}")

    rows = read_jsonl(urls_jsonl)

    all_triplets: List[Tuple[str, str, str]] = []
    for r in rows:
        if isinstance(r, dict):
            all_triplets.extend(list(_iter_urls_from_claim_row(r)))

    _log_count(logger, "claims_rows", len(rows))
    _log_count(logger, "urls_total", len(all_triplets))

    print(f"STEP04 input: {urls_jsonl}", flush=True)
    print(f"STEP04 output dir: {plaintext_root}", flush=True)
    print(f"STEP04 claim rows: {len(rows)}", flush=True)
    print(f"STEP04 total urls: {len(all_triplets)}", flush=True)
    print(
        f"STEP04 config: min_chars={min_chars}, max_chars={max_chars}, timeout_s={timeout_s}, retries={retries}",
        flush=True,
    )

    if not all_triplets:
        stats = {
            "written": 0,
            "skipped": 0,
            "too_short": 0,
            "empty": 0,
            "errors": 0,
        }
        _print_summary(
            urls_jsonl=urls_jsonl,
            plaintext_root=plaintext_root,
            total_rows=len(rows),
            total_urls=0,
            stats=stats,
        )
        return

    iterable = _progress_iter(
        all_triplets,
        enabled=show_progress,
        desc="step04",
    )

    stats = {
        "written": 0,
        "skipped": 0,
        "too_short": 0,
        "empty": 0,
        "errors": 0,
    }

    for claim_id, url_id, url in iterable:
        out_path = plaintext_root / claim_id / f"{url_id}.txt"
        ensure_parent_dir(out_path)

        if out_path.exists() and not overwrite:
            _log_count(logger, "skipped_existing", 1)
            stats["skipped"] += 1
            if hasattr(iterable, "set_postfix"):
                iterable.set_postfix(stats, refresh=False) 
            continue

        _log_count(logger, "fetch_attempted", 1)

        try:
            text: Optional[str] = None

            try:
                text = fetch_fn(url, timeout_s=timeout_s, retries=retries)  
            except TypeError:
                try:
                    text = fetch_fn(url, timeout=timeout_s, retries=retries) 
                except TypeError:
                    try:
                        text = fetch_fn(url, timeout_s=timeout_s) 
                    except TypeError:
                        try:
                            text = fetch_fn(url, timeout=timeout_s)  
                        except TypeError:
                            text = fetch_fn(url) 

            if not text or not isinstance(text, str) or not text.strip():
                _log_count(logger, "dropped_empty_or_none", 1)
                stats["empty"] += 1
                if hasattr(iterable, "set_postfix"):
                    iterable.set_postfix(stats, refresh=False)
                continue

            cleaned = clean_fn(text)
            if not cleaned or not cleaned.strip():
                _log_count(logger, "dropped_empty_after_clean", 1)
                stats["empty"] += 1
                if hasattr(iterable, "set_postfix"):
                    iterable.set_postfix(stats, refresh=False) 
                continue

            cleaned = cleaned.strip()

            if len(cleaned) < min_chars:
                _log_count(logger, "dropped_too_short", 1)
                stats["too_short"] += 1
                if hasattr(iterable, "set_postfix"):
                    iterable.set_postfix(stats, refresh=False) 
                continue

            if len(cleaned) > max_chars:
                cleaned = cleaned[:max_chars]
                _log_count(logger, "truncated", 1)

            write_text_atomic(out_path, cleaned)
            _log_count(logger, "written", 1)
            stats["written"] += 1

            if hasattr(iterable, "set_postfix"):
                iterable.set_postfix(stats, refresh=False)  

        except Exception as e:
            _log_count(logger, "fetch_errors", 1)
            stats["errors"] += 1

            if hasattr(iterable, "set_postfix"):
                iterable.set_postfix(stats, refresh=False) 

            payload = {
                "claim_id": claim_id,
                "url_id": url_id,
                "url": url,
                "error": type(e).__name__,
                "message": str(e),
            }
            _log_event(logger, "url_error", payload)

            if verbose_errors:
                print(
                    f"\n[step04 error] claim_id={claim_id} url_id={url_id} "
                    f"error={type(e).__name__}: {e}",
                    flush=True,
                )

    _print_summary(
        urls_jsonl=urls_jsonl,
        plaintext_root=plaintext_root,
        total_rows=len(rows),
        total_urls=len(all_triplets),
        stats=stats,
    )


def run_step04(
    *,
    paths: Paths,
    config: Any,
    logger: Any,
    overwrite: bool = False,
    no_progress: bool = False,
    verbose_errors: bool = True,
    **_: Any,
) -> None:
    """Orchestrator entrypoint. Must NOT parse argv."""
    print("RUN_STEP04 ENTRYPOINT HIT", flush=True)
    run_step04_cache_url_content(
        paths=paths,
        config=config,
        logger=logger,
        overwrite=overwrite,
        show_progress=(not no_progress),
        verbose_errors=verbose_errors,
    )


#CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="Step 04: cache URL content into run plaintext cache")
    ap.add_argument("--run-id", required=True, help="Run id (runs/<run_id>/...)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing cached plaintext files")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress display")
    ap.add_argument("--quiet-errors", action="store_true", help="Do not print per-URL errors to terminal")
    ap.add_argument("--config-module", default=None, help="Optional config module override")
    args = ap.parse_args()

    if args.config_module:
        import os
        os.environ["CCIR_CONFIG_MODULE"] = args.config_module

    config = load_config()
    paths = Paths(run_id=args.run_id)

    step_logger = StepLogger(step_id="04", run_id=args.run_id, paths=paths)

    with step_logger.step_scope(
        inputs=[str(paths.run_evidence_urls_jsonl)],
        outputs=[str(paths.cache_plaintext_dir)],
    ):
        run_step04_cache_url_content(
            paths=paths,
            config=config,
            logger=step_logger,
            overwrite=args.overwrite,
            show_progress=(not args.no_progress),
            verbose_errors=(not args.quiet_errors),
        )


if __name__ == "__main__":
    main()
