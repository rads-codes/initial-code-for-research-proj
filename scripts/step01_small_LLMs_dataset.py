from __future__ import annotations

"""
scripts/step01_small_LLMs_dataset.py

Run-scoped by default (matches Paths design):
  Input:  paths.run_claims_all_jsonl          (runs/<run_id>/claims/all.jsonl)
  Output: paths.run_claims_dir/forLLMs.jsonl  (runs/<run_id>/claims/forLLMs.jsonl)

Optional fallback:
  If you set CCIR_ALLOW_SHARED_CLAIMS=1, will fall back to shared
  paths.claims_all_jsonl -> data/processed/claims/all.jsonl
"""

import argparse
import inspect
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Sequence

from ccir.config_loader import load_config
from ccir.io_utils import ensure_parent_dir, read_jsonl, write_jsonl_atomic
from ccir.paths import Paths
from ccir.schemas import LLMClaimRow, utc_now_iso

STEP_ID = "01"
STEP_NAME = "step01_small_LLMs_dataset"


def _get_dataset_settings(config: Any) -> tuple[list[str], int, dict[str, int]]:
    dataset = getattr(config, "dataset", None)
    if dataset is None:
        raise RuntimeError("Config missing 'dataset' section.")

    languages = list(getattr(dataset, "languages", []))
    if not languages:
        raise RuntimeError("Config dataset.languages is empty.")

    default_n = int(getattr(dataset, "claims_per_language_default", 0))
    if default_n <= 0:
        raise RuntimeError("Config dataset.claims_per_language_default must be > 0.")

    override = dict(getattr(dataset, "claims_per_language_override", {}) or {})
    override = {str(k): int(v) for k, v in override.items()}
    return languages, default_n, override


def _row_get(row: Dict[str, Any], key: str) -> Any:
    return row.get(key, None)


def _is_valid_all_claims_row(row: Dict[str, Any]) -> bool:
    required = ["claim_id", "lang", "claim_text", "claim_date"]
    return all(_row_get(row, k) not in (None, "") for k in required)


def _to_llm_claim_row(*, src: Dict[str, Any], run_id: str, code_version: Optional[str]) -> LLMClaimRow:
    base_kwargs: Dict[str, Any] = dict(
        claim_id=_row_get(src, "claim_id"),
        lang=_row_get(src, "lang"),
        claim_text=_row_get(src, "claim_text"),
        claim_date=_row_get(src, "claim_date"),
    )

    lineage_kwargs = {
        "run_id": run_id,
        "created_utc": utc_now_iso(),
        "code_version": code_version,
    }

    sig = inspect.signature(LLMClaimRow)
    accepted = set(sig.parameters.keys())

    kwargs = dict(base_kwargs)
    for k, v in lineage_kwargs.items():
        if k in accepted:
            kwargs[k] = v

    return LLMClaimRow(**kwargs)  # type: ignore[arg-type]


def _emit_log_summary(log_obj: Any, counters: Dict[str, Any]) -> None:
    if log_obj is None:
        return
    try:
        if callable(log_obj):
            log_obj("step_summary", counters)
            return
        if hasattr(log_obj, "log") and callable(getattr(log_obj, "log")):
            log_obj.log(counters)
            return
        if hasattr(log_obj, "write") and callable(getattr(log_obj, "write")):
            log_obj.write(counters)
            return
    except Exception:
        return


def _resolve_in_out_paths(paths: Paths, run_id: str) -> tuple[Any, Any]:
    """
    Use run-scoped paths by default.
    Optional fallback to shared paths if CCIR_ALLOW_SHARED_CLAIMS=1.
    """
    in_path = paths.run_claims_all_jsonl
    out_path = paths.run_claims_dir / "forLLMs.jsonl"

    if in_path.exists():
        return in_path, out_path

    allow_shared = os.getenv("CCIR_ALLOW_SHARED_CLAIMS", "").strip() in {"1", "true", "TRUE", "yes", "YES"}
    if allow_shared:
        shared_in = getattr(paths, "claims_all_jsonl", None)
        shared_out = getattr(paths, "claims_for_llms_jsonl", None)
        if shared_in is not None and shared_in.exists():
            return shared_in, shared_out

    raise FileNotFoundError(
        f"Missing input claims file for run '{run_id}': {in_path}\n"
        f"Run step00 first for this run_id, e.g.:\n"
        f"  python -m ccir {run_id} --steps 00,01\n"
    )


def run_step01(
    *,
    paths: Paths,
    config: Any,
    run_id: str,
    code_version: Optional[str] = None,
    log: Any = None,
    logger: Any = None,
    step_logger: Any = None,
    config_hash: Optional[str] = None,
) -> Dict[str, Any]:
    languages, default_n, override = _get_dataset_settings(config)

    in_path, out_path = _resolve_in_out_paths(paths, run_id)

    rows = read_jsonl(in_path)

    counters: Dict[str, Any] = {
        "step": STEP_NAME,
        "step_id": STEP_ID,
        "run_id": run_id,
        "code_version": code_version,
        "config_hash": config_hash,
        "input_path": str(in_path),
        "output_path": str(out_path),
        "read_total": len(rows),
        "skipped_invalid": 0,
        "skipped_lang": 0,
        "skipped_duplicate_claim_id": 0,
        "requested_by_lang": {},
        "available_by_lang": {},
        "kept_by_lang": {},
        "kept_total": 0,
        "warnings": [],
    }

    requested_by_lang = {lang: int(override.get(lang, default_n)) for lang in languages}
    counters["requested_by_lang"] = requested_by_lang

    buckets: Dict[str, List[Dict[str, Any]]] = {lang: [] for lang in languages}
    seen_claim_ids: set[str] = set()

    for r in rows:
        if not isinstance(r, dict):
            counters["skipped_invalid"] += 1
            continue
        if not _is_valid_all_claims_row(r):
            counters["skipped_invalid"] += 1
            continue

        lang = str(_row_get(r, "lang"))
        if lang not in buckets:
            counters["skipped_lang"] += 1
            continue

        claim_id = str(_row_get(r, "claim_id"))
        if claim_id in seen_claim_ids:
            counters["skipped_duplicate_claim_id"] += 1
            continue
        seen_claim_ids.add(claim_id)

        buckets[lang].append(r)

    counters["available_by_lang"] = {lang: len(buckets[lang]) for lang in languages}

    selected: List[LLMClaimRow] = []
    for lang in languages:
        n = requested_by_lang[lang]
        avail = buckets[lang]
        if len(avail) < n:
            counters["warnings"].append(
                f"Language '{lang}': requested {n} but only {len(avail)} available; using all available."
            )

        take = avail[:n]
        counters["kept_by_lang"][lang] = len(take)
        for r in take:
            selected.append(_to_llm_claim_row(src=r, run_id=run_id, code_version=code_version))

    counters["kept_total"] = len(selected)

    ensure_parent_dir(out_path)

    serializable: List[Dict[str, Any]] = []
    for obj in selected:
        serializable.append(asdict(obj) if is_dataclass(obj) else obj)  # type: ignore[arg-type]

    write_jsonl_atomic(out_path, serializable)

    log_obj = log or logger or step_logger
    _emit_log_summary(log_obj, counters)
    return counters


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=STEP_NAME)
    p.add_argument("--run-id", default="manual")
    p.add_argument("--code-version", default=None)
    args = p.parse_args(list(argv) if argv is not None else None)

    config = load_config()
    paths = Paths(run_id=args.run_id)

    counters = run_step01(
        paths=paths,
        config=config,
        run_id=args.run_id,
        code_version=args.code_version,
        log=None,
        config_hash=None,
    )

    print(
        f"[{STEP_ID}] wrote {counters['kept_total']} rows to {counters['output_path']} "
        f"(by_lang={counters['kept_by_lang']})"
    )
    for w in counters["warnings"]:
        print(f"[{STEP_ID}] WARN: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())