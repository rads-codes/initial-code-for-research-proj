from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests

from ccir.config_loader import load_config
from ccir.io_utils import read_jsonl, write_jsonl_atomic
from ccir.logging_utils import StepLogger
from ccir.paths import Paths
from ccir.schemas import ClaimWithURLs, URLItem, to_dict, utc_now_iso
from ccir.utils.hashing import sha256_hex


#defaults
DEFAULT_PAYWALL_KEYWORDS = [
    "subscribe",
    "subscription",
    "sign in to continue",
    "sign in to read",
    "already a subscriber",
    "register to continue",
    "to continue reading",
    "unlock this article",
    "read more with a subscription",
    "premium",
    "members-only",
    "member exclusive",
    "trial",
    "paywall",
]

DEFAULT_FACTCHECK_DOMAIN_SUBSTRINGS = [
    #global fact-checkers
    "snopes.com",
    "factcheck.org",
    "politifact.com",
    "fullfact.org",
    "leadstories.com",
    "checkyourfact.com",
    "logicallyfacts.com",

    #news agency fact-checkers
    "reuters.com/fact-check",
    "apnews.com/hub/fact-check",
    "factcheck.afp.com",
    "factuel.afp.com",
    "afpfactcheck.com",

    #US
    "usatoday.com/story/news/factcheck",
    "washingtonpost.com/news/fact-checker",
    "bbc.com/news/reality_check",
    "bbc.co.uk/news/reality_check",

    #German
    "correctiv.org/faktencheck",
    "dpa-factchecking.com",
    "faktenfuchs",
    "tagesschau.de/faktenfinder",
    "mimikama.org",
    "volksverpetzer.de/faktencheck",
    "dw.com/de/faktencheck",
    "dw.com/en/fact-check",

    #Greek
    "ellinikahoaxes.gr",

    #Romanian
    "factual.ro",
    "veridica.ro",
    "context.ro",

    #Spanish
    "maldita.es",
    "maldita.es/malditobulo",
    "newtral.es/fact-check",
    "chequeado.com",
    "efe.com/efe/verifica",

    #Italian
    "facta.news",
    "pagellapolitica.it/fact-checking",
    "open.online",
    "open.online/fact-check",
    "bufale.net",

    #French
    "liberation.fr/checknews",
    "lesdecodeurs.lemonde.fr",
    "france24.com/fr/info-intox",
    "20minutes.fr/societe/fake-off",

    #Central and Eastern Europe
    "demagog.org.pl",
    "fakenews.pl",
    "pravda-or-not.com",

    #Turkey
    "teyit.org",

    #Caucasus
    "mythdetector.ge",

    #Ukraine
    "stopfake.org",

    #South Asia
    "boomlive.in",
    "altnews.in",
    "thequint.com/news/webqoof",
    "newschecker.in",
    "newschecker.in/fact-check",
    "factcrescendo.com",
    "newsmeter.in/fact-check",

    #Africa 
    "africacheck.org",
    "dubawa.org",
    "pesacheck.org",

    #Australia
    "aap.com.au/factcheck",
]

YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "www.youtu.be",
}

_TRACKING_PARAM_PREFIXES = ("utm_",)
_TRACKING_PARAMS_EXACT = {"gclid", "fbclid", "mc_cid", "mc_eid", "igshid"}


@dataclass(frozen=True)
class Step03Params:
    max_claims: int = 0
    seed: int = 12345


# ----------------------------
# SerpAPI caching + resume helpers
# ----------------------------

def _serp_cache_dir(paths: Paths) -> Path:
    d = paths.cache_dir / "serpapi"  #runs/<run_id>/cache/serpapi/
    d.mkdir(parents=True, exist_ok=True)
    return d


def _serp_cache_path(paths: Paths, claim_id: str) -> Path:
    safe = str(claim_id).replace("/", "_").replace("\\", "_")
    return _serp_cache_dir(paths) / f"{safe}.json"


def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _write_json_file_atomic(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    tmp.replace(path)


def _resume_existing(out_urls: Path) -> Tuple[List[Dict[str, Any]], set[str]]:
    if not out_urls.exists():
        return [], set()

    try:
        rows = [r for r in read_jsonl(out_urls) if isinstance(r, dict)]
    except Exception:
        return [], set()

    done: set[str] = set()
    for r in rows:
        cid = r.get("claim_id")
        if isinstance(cid, str) and cid:
            done.add(cid)

    return rows, done


def _merge_by_claim_id(existing_rows: List[Dict[str, Any]], new_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for r in existing_rows:
        cid = r.get("claim_id")
        if isinstance(cid, str) and cid:
            merged[cid] = r

    for r in new_rows:
        cid = r.get("claim_id")
        if isinstance(cid, str) and cid:
            merged[cid] = r

    return [merged[cid] for cid in sorted(merged.keys())]


#progress bar :)
def _fmt_time(seconds: float) -> str:
    s = max(0, int(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    if h:
        return f"{h:d}:{m:02d}:{ss:02d}"
    return f"{m:d}:{ss:02d}"


def _progress_update(done: int, total: int, start_t: float) -> None:
    if total <= 0:
        return
    now = time.time()
    elapsed = now - start_t
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0.0
    width = 28
    frac = min(1.0, max(0.0, done / total))
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)

    msg = f"[{bar}] {done}/{total} ({frac*100:5.1f}%) | elapsed {_fmt_time(elapsed)} | eta {_fmt_time(eta)}"
    print("\r" + msg, end="", flush=True)
    if done >= total:
        print("", flush=True)


#url helpers
def canonicalize_url(url: str) -> str:
    try:
        raw = url.strip()
        p = urlparse(raw)

        if not p.netloc and p.path and "://" not in raw:
            p = urlparse("https://" + raw)

        scheme = (p.scheme or "https").lower()
        netloc = (p.netloc or "").lower()
        path = p.path or "/"

        query_pairs = parse_qsl(p.query, keep_blank_values=True)
        filtered_pairs = []
        for k, v in query_pairs:
            lk = k.lower()
            if lk in _TRACKING_PARAMS_EXACT:
                continue
            if any(lk.startswith(pref) for pref in _TRACKING_PARAM_PREFIXES):
                continue
            filtered_pairs.append((k, v))

        query = urlencode(filtered_pairs, doseq=True)
        return urlunparse((scheme, netloc, path, "", query, ""))
    except Exception:
        return url.strip()


def is_youtube(url: str) -> bool:
    try:
        host = (urlparse(url).netloc or "").lower()
        return host in YOUTUBE_HOSTS or host.endswith(".youtube.com")
    except Exception:
        return False

COMMON_FACTCHECK_PATH_SUBSTRINGS = [
    "/fact-check",
    "/factcheck",
    "/fact-checks",
    "/fact-checking",
    "/fact-checker",
    "/factchecking",

    "/faktencheck",
    "/faktenfinder",

    "/reality_check",
    "/reality-check",

    "/fake-off",
    "/fake-news-check",

    "/debunk",
    "/debunked",
    "/debunking",

    "/verifica",
    "/verificare",
    "/verificado",
    "/verificato",

    "/malditobulo",

    "/checknews",
    "/decodeurs",

    "/truth-or-fake",
]

def is_factcheck_domain(url: str, domain_substrings: List[str]) -> bool:
    u = url.lower()
    return (
        any(s.lower() in u for s in domain_substrings)
        or any(p in u for p in COMMON_FACTCHECK_PATH_SUBSTRINGS)
    )

def looks_paywalled(text: str, paywall_keywords: List[str]) -> bool:
    t = (text or "").lower()
    return any(k.lower() in t for k in paywall_keywords)


def looks_like_pdf_url(url: str) -> bool:
    try:
        u = url.strip()
        p = urlparse(u)
        path = (p.path or "").lower()
        query = (p.query or "").lower()

        if path.endswith(".pdf"):
            return True
        if path.endswith("/pdf") or path.endswith("/pdf/"):
            return True
        if "format=pdf" in query or "output=pdf" in query or "download=pdf" in query:
            return True
        return False
    except Exception:
        return url.lower().endswith(".pdf")


#serpapi
class SerpAPIError(RuntimeError):
    pass


def claim_date_to_mmddyyyy(claim_date: Optional[str]) -> Optional[str]:
    if not claim_date:
        return None
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", claim_date.strip())
    if not m:
        return None
    yyyy, mm, dd = m.group(1), m.group(2), m.group(3)
    return f"{mm}/{dd}/{yyyy}"


def serpapi_search(
    *,
    api_key: str,
    query: str,
    num_results: int,
    cd_max_mmddyyyy: Optional[str],
    timeout_s: int,
    retries: int,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "engine": "google",
        "q": query,
        "num": max(10, min(num_results, 100)),
        "api_key": api_key,
    }
    if cd_max_mmddyyyy:
        params["tbs"] = f"cdr:1,cd_max:{cd_max_mmddyyyy}"

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            r = requests.get("https://serpapi.com/search.json", params=params, timeout=timeout_s)
            if r.status_code != 200:
                raise SerpAPIError(f"SerpAPI HTTP {r.status_code}: {r.text[:200]}")
            return r.json()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.25 * (attempt + 1))
                continue
            raise SerpAPIError(str(last_err)) from last_err


#steps
def run_step03(
    *,
    paths: Paths,
    run_id: Optional[str] = None,
    code_version: str = "dev",
    config: Any = None,
    cfg: Any = None,
    params: Optional[Step03Params] = None,
    logger: Any = None,
    log: Any = None,
    step_logger: Any = None,
    **_: Any,
) -> None:
    cfg_obj = cfg if cfg is not None else config
    if cfg_obj is None:
        cfg_obj = load_config()

    rid = run_id or paths.run_id
    p = params or Step03Params()

    report_path = paths.report_jsonl(3)
    if step_logger is not None:
        slog = step_logger
    elif log is not None:
        slog = log
    elif logger is not None:
        slog = logger
    else:
        slog = StepLogger(step="03", report_path=report_path, run_id=rid, code_version=code_version)

    api_key = os.environ.get("SERPAPI_API_KEY") or os.environ.get("SERPAPI_KEY")
    if not api_key:
        raise RuntimeError("Missing SERPAPI_API_KEY (or SERPAPI_KEY) in environment.")
    api_key_backup = os.environ.get("SERPAPI_API_KEY_BACKUP") or ""
    api_keys: List[str] = [k for k in [api_key, api_key_backup] if k]

    retrieval_cfg = getattr(cfg_obj, "retrieval", cfg_obj)
    k = int(getattr(retrieval_cfg, "serpapi_k", 10))
    timeout_s = int(getattr(retrieval_cfg, "fetch_timeout_s", 25))
    retries = int(getattr(retrieval_cfg, "fetch_retries", 2))
    drop_youtube = bool(getattr(retrieval_cfg, "drop_youtube", True))
    drop_factcheck = bool(getattr(retrieval_cfg, "drop_factcheck_sites", True))
    drop_paywalled = bool(getattr(retrieval_cfg, "drop_paywalled", True))

    max_workers = int(getattr(retrieval_cfg, "step03_max_workers", 6))
    max_workers = max(1, min(max_workers, 16))

    paywall_keywords = list(getattr(retrieval_cfg, "paywall_keywords", DEFAULT_PAYWALL_KEYWORDS))
    factcheck_domains = list(getattr(retrieval_cfg, "factcheck_domains", DEFAULT_FACTCHECK_DOMAIN_SUBSTRINGS))

    in_for_llms = paths.claims_for_llms
    in_for_scoring = paths.claims_for_scoring
    out_urls = paths.evidence_urls

    rows_llm = read_jsonl(in_for_llms)
    rows_score = read_jsonl(in_for_scoring)

    llm_ids = {r.get("claim_id") for r in rows_llm if isinstance(r, dict)}
    score_ids = {r.get("claim_id") for r in rows_score if isinstance(r, dict)}
    mismatch = len(llm_ids.symmetric_difference(score_ids))

    if hasattr(slog, "counts") and isinstance(slog.counts, dict):
        slog.counts["claims_forLLMs_rows"] = len(rows_llm)
        slog.counts["claims_forScoring_rows"] = len(rows_score)
        slog.counts["claim_id_mismatch_count"] = mismatch

    random.seed(p.seed)
    claims = [r for r in rows_llm if isinstance(r, dict)]
    claims.sort(key=lambda r: str(r.get("claim_id", "")))
    if p.max_claims and p.max_claims > 0:
        claims = claims[: p.max_claims]

    existing_rows, done_ids = _resume_existing(out_urls)
    if done_ids:
        claims = [c for c in claims if str(c.get("claim_id") or "") not in done_ids]

    if hasattr(slog, "counts") and isinstance(slog.counts, dict):
        slog.counts["resume_existing_rows"] = len(existing_rows)
        slog.counts["resume_done_claim_ids"] = len(done_ids)
        slog.counts["resume_claims_remaining"] = len(claims)

    cv = getattr(cfg_obj, "code_version", None) or code_version or "dev"

    def _event(name: str, payload: Dict[str, Any]) -> None:
        if hasattr(slog, "event"):
            slog.event(name, payload)

    def process_claim(
        claim: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], Counter, List[Tuple[str, Dict[str, Any]]]]:
        c = Counter()
        events: List[Tuple[str, Dict[str, Any]]] = []

        claim_id = claim.get("claim_id")
        claim_text = claim.get("claim_text") or ""
        claim_date = claim.get("claim_date")

        if not claim_id or not claim_text:
            c["dropped_missing_claim_fields"] += 1
            return None, c, events

        cd_max = claim_date_to_mmddyyyy(claim_date)
        url_items: List[URLItem] = []
        seen_canon: set[str] = set()

        requested = min(max(k * 3, 15), 100)

        cache_path = _serp_cache_path(paths, str(claim_id))
        resp: Dict[str, Any] = {}

        cached = _read_json_file(cache_path) if cache_path.exists() else None
        if cached is not None:
            c["serpapi_cache_hit"] += 1
            resp = cached
        else:
            c["serpapi_calls"] += 1
            last_exc: Optional[Exception] = None
            resp = {}
            for key_idx, current_key in enumerate(api_keys):
                try:
                    resp = serpapi_search(
                        api_key=current_key,
                        query=str(claim_text),
                        num_results=requested,
                        cd_max_mmddyyyy=cd_max,
                        timeout_s=timeout_s,
                        retries=0,  #one attempt per key
                    )
                    if key_idx > 0:
                        c["serpapi_backup_key_used"] += 1
                    last_exc = None
                    break  
                except Exception as e:
                    last_exc = e
                    if key_idx < len(api_keys) - 1 and "429" in str(e):
                        events.append((
                            "serpapi_key_fallback",
                            {"claim_id": str(claim_id), "key_index": key_idx, "error": str(e)[:200]},
                        ))
                    else:
                        break 

            if last_exc is not None:
                c["serpapi_errors"] += 1
                events.append(("serpapi_error", {"claim_id": str(claim_id), "error": str(last_exc)[:400]}))
                resp = {}

            if isinstance(resp, dict) and resp:
                _write_json_file_atomic(cache_path, resp)
                c["serpapi_cache_write"] += 1

        organic = resp.get("organic_results") or []
        if not isinstance(organic, list):
            organic = []

        for idx, item in enumerate(organic, start=1):
            if len(url_items) >= k:
                break
            if not isinstance(item, dict):
                continue

            link = item.get("link") or item.get("url")
            if not link or not isinstance(link, str):
                continue

            title = item.get("title") or None
            snippet = item.get("snippet") or None

            if drop_youtube and is_youtube(link):
                c["dropped_youtube"] += 1
                continue

            if drop_paywalled and (
                looks_paywalled(snippet or "", paywall_keywords)
                or looks_paywalled(title or "", paywall_keywords)
            ):
                c["dropped_paywall"] += 1
                continue

            if drop_factcheck and is_factcheck_domain(link, factcheck_domains):
                c["dropped_factcheck"] += 1
                continue

            canon = canonicalize_url(link)
            if canon in seen_canon:
                c["dropped_duplicate"] += 1
                continue

            if looks_like_pdf_url(link):
                try:
                    _ = requests.head(
                        link,
                        timeout=6,
                        allow_redirects=True,
                        headers={"User-Agent": "ccir/step03"},
                    )
                except Exception:
                    pass

            url_id = "u_" + sha256_hex(canon)[:16]
            try:
                source = urlparse(link).netloc.lower() or None
            except Exception:
                source = None

            url_items.append(
                URLItem(
                    url_id=url_id,
                    url=link,
                    title=title,
                    snippet=snippet,
                    source=source,
                    rank=idx,
                )
            )
            seen_canon.add(canon)

        c["kept_urls_total"] += len(url_items)

        out_row = to_dict(
            ClaimWithURLs(
                run_id=rid,
                created_utc=utc_now_iso(),
                code_version=cv,
                claim_id=str(claim_id),
                urls=url_items,
            )
        )
        return out_row, c, events

    #progress bar
    total_claims = len(claims)
    start_t = time.time()
    done = 0
    if total_claims:
        _progress_update(0, total_claims, start_t)

    indexed_claims = list(enumerate(claims))
    results_by_idx: Dict[int, Dict[str, Any]] = {}
    total = Counter()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {ex.submit(process_claim, claim): idx for idx, claim in indexed_claims}
        for fut in as_completed(fut_map):
            idx = fut_map[fut]
            out_row, counters, events = fut.result()

            total.update(counters)
            for name, payload in events:
                _event(name, payload)

            if out_row is not None:
                results_by_idx[idx] = out_row

            done += 1
            if total_claims:
                _progress_update(done, total_claims, start_t)

    new_rows = [results_by_idx[i] for i in sorted(results_by_idx.keys())]
    outputs = _merge_by_claim_id(existing_rows, new_rows)
    write_jsonl_atomic(out_urls, outputs)

    if hasattr(slog, "counts") and isinstance(slog.counts, dict):
        slog.counts["step03_max_workers"] = max_workers

        slog.counts["serpapi_calls"] = int(total["serpapi_calls"])
        slog.counts["serpapi_errors"] = int(total["serpapi_errors"])
        slog.counts["serpapi_cache_hit"] = int(total["serpapi_cache_hit"])
        slog.counts["serpapi_cache_write"] = int(total["serpapi_cache_write"])
        slog.counts["serpapi_backup_key_used"] = int(total["serpapi_backup_key_used"])
        slog.counts["serpapi_keys_available"] = len(api_keys)

        slog.counts["claims_written"] = len(outputs)
        slog.counts["kept_urls_total"] = int(total["kept_urls_total"])
        slog.counts["dropped_paywall"] = int(total["dropped_paywall"])
        slog.counts["dropped_factcheck"] = int(total["dropped_factcheck"])
        slog.counts["dropped_youtube"] = int(total["dropped_youtube"])
        slog.counts["dropped_duplicate"] = int(total["dropped_duplicate"])
        slog.counts["dropped_missing_claim_fields"] = int(total["dropped_missing_claim_fields"])
        slog.counts["dropped_pdf_unparseable"] = 0

        slog.counts["new_claims_processed"] = len(new_rows)

    if hasattr(slog, "flush"):
        slog.flush(summary={"output": str(out_urls)})


#cli wrapper
def cli_main() -> int:
    ap = argparse.ArgumentParser(description="Step 03: collect evidence URLs via SerpAPI.")
    ap.add_argument("--run-id", required=True, help="Run id (used for Paths + lineage).")
    ap.add_argument("--max-claims", type=int, default=0, help="If >0, cap number of claims.")
    ap.add_argument("--seed", type=int, default=12345, help="Deterministic ordering.")
    args = ap.parse_args()

    cfg = load_config()
    paths = Paths(run_id=args.run_id)
    code_version = getattr(cfg, "code_version", "dev")

    run_step03(
        paths=paths,
        run_id=args.run_id,
        code_version=code_version,
        cfg=cfg,
        params=Step03Params(max_claims=args.max_claims, seed=args.seed),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
