from __future__ import annotations

"""
purpose: translate romanian to english
inputs: runs/<run_id>/claims/forLLMs.jsonl, runs/<run_id>/claims/forScoring.jsonl
outputs: runs/<run_id>/claims/forLLMs.jsonl appends, runs/<run_id>/claims/forScoring.jsonl appends, runs/<run_id>/cache/translation_cache.jsonl (log)
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ccir.config_loader import load_config
from ccir.io_utils import append_jsonl, ensure_parent_dir, read_jsonl, write_jsonl_atomic
from ccir.paths import Paths
from ccir.schemas import utc_now_iso

STEP_ID = "02b"
STEP_NAME = "step02b_translate_claims"

#appended suffix to differentiate translated vs not translated claims
MT_SUFFIX = "_mt"

#translation prompt:

_TRANSLATION_PROMPT_TEMPLATE = """\
Translate the following Romanian claim to English.

Rules:
- Translate as literally as possible, preserving the original meaning exactly.
- Preserve all named entities (people, places, organizations), numbers, dates, \
percentages, and quantities exactly as they appear in the original.
- Do not paraphrase, simplify, add commentary, or explain the claim.
- Return only the translated claim text. Do not include any prefix, label, or \
surrounding explanation.

Romanian claim:
{claim_text}

English translation:"""


def _build_translation_prompt(claim_text: str) -> str:
    return _TRANSLATION_PROMPT_TEMPLATE.format(claim_text=claim_text.strip())


#LLM call

def _call_translation_llm(
    *,
    claim_text: str,
    translation_cfg: Any,
) -> str:
    """
    call the configured translation model and return the raw translated string
    """
    from types import SimpleNamespace

    from ccir.models.runner import run_small_llm

    model_spec = SimpleNamespace(
        provider=getattr(translation_cfg, "provider", "openrouter"),
        name=getattr(translation_cfg, "model_name", ""),
    )

    if not model_spec.name:
        raise RuntimeError("translation.model_name is empty; set it in configs.py")

    prompt = _build_translation_prompt(claim_text)

    result = run_small_llm(
        model_spec=model_spec,
        prompt_text=prompt,
        temperature=float(getattr(translation_cfg, "temperature", 0.0)),
        max_tokens=int(getattr(translation_cfg, "max_tokens", 512)),
        expect_json=False,
    )

    raw = result.get("raw_text", "").strip()
    return raw


#translation validation

def _validate_translation(original: str, translated: str) -> List[str]:
    """
    sanity check validation strings
    """
    warnings: List[str] = []

    if not translated or not translated.strip():
        warnings.append("translated output is empty")
        return warnings

    #if model dropped something
    numbers = re.findall(r"(?<!\w)\d[\d.,]*(?!\w)", original)
    for num in numbers:
        #compares
        canonical = re.sub(r"[,.]", "", num)
        translated_stripped = re.sub(r"[,.]", "", translated)
        if canonical and canonical not in translated_stripped:
            warnings.append(f"number '{num}' from original not found in translation")

    return warnings


#translation helpers
def _load_translation_cache(cache_path: Path) -> Dict[str, str]:
    """
    load existing cached translations, return dictionary
    """
    if not cache_path.exists():
        return {}

    cache: Dict[str, str] = {}
    try:
        rows = read_jsonl(cache_path)
        for r in rows:
            if isinstance(r, dict):
                mid = r.get("mt_claim_id")
                txt = r.get("translated_text")
                if mid and txt:
                    cache[str(mid)] = str(txt)
    except Exception:
        pass
    return cache


def _append_translation_log(
    cache_path: Path,
    *,
    source_claim_id: str,
    mt_claim_id: str,
    original_text: str,
    translated_text: str,
    model_name: str,
    created_utc: str,
    warnings: List[str],
) -> None:
    """Append one log entry to the translation cache file."""
    ensure_parent_dir(cache_path)
    entry: Dict[str, Any] = {
        "mt_claim_id": mt_claim_id,
        "source_claim_id": source_claim_id,
        "original_text": original_text,
        "translated_text": translated_text,
        "model_name": model_name,
        "created_utc": created_utc,
        "warnings": warnings,
    }
    append_jsonl(cache_path, entry)


#core logic
def run_step02b(
    *,
    paths: Paths,
    config: Any,
    run_id: Optional[str] = None,
    code_version: Optional[str] = None,
    log: Any = None,
    logger: Any = None,
    step_logger: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    rid = run_id or paths.run_id
    cv = code_version or "dev"

    #check ro_mt_en
    translation_cfg = getattr(config, "translation", None)
    dataset_languages = list(getattr(getattr(config, "dataset", None), "languages", []))
    if "ro_mt_en" not in dataset_languages:
        _log(log or logger or step_logger, f"[{STEP_ID}] 'ro_mt_en' not in dataset.languages; skipping translation.")
        return {
            "step": STEP_NAME,
            "step_id": STEP_ID,
            "run_id": rid,
            "skipped": True,
            "reason": "'ro_mt_en' not in dataset.languages",
        }

    model_name = str(getattr(translation_cfg, "model_name", ""))

    #paths
    for_llms_path = paths.claims_for_llms
    for_scoring_path = paths.claims_for_scoring
    cache_path = paths.cache_translation_log_jsonl

    if not for_llms_path.exists():
        raise FileNotFoundError(
            f"[{STEP_ID}] forLLMs.jsonl not found at {for_llms_path}. Run step01 first."
        )
    if not for_scoring_path.exists():
        raise FileNotFoundError(
            f"[{STEP_ID}] forScoring.jsonl not found at {for_scoring_path}. Run step02 first."
        )

    #read existing rows
    llm_rows: List[Dict[str, Any]] = [
        r for r in read_jsonl(for_llms_path) if isinstance(r, dict)
    ]
    scoring_rows: List[Dict[str, Any]] = [
        r for r in read_jsonl(for_scoring_path) if isinstance(r, dict)
    ]

    #build sets of claim IDs
    existing_llm_ids = {str(r.get("claim_id", "")) for r in llm_rows}
    existing_scoring_ids = {str(r.get("claim_id", "")) for r in scoring_rows}

    #rating lookup
    scoring_map: Dict[str, str] = {}
    for r in scoring_rows:
        cid = r.get("claim_id")
        rating = r.get("rating")
        if cid and rating:
            scoring_map[str(cid)] = str(rating)

    #romanian rows to translate
    ro_rows = [r for r in llm_rows if str(r.get("lang", "")) == "ro"]

    counters: Dict[str, Any] = {
        "step": STEP_NAME,
        "step_id": STEP_ID,
        "run_id": rid,
        "code_version": cv,
        "model_name": model_name,
        "ro_source_rows": len(ro_rows),
        "already_translated": 0,
        "translated_ok": 0,
        "translation_errors": 0,
        "translation_warnings": 0,
        "scoring_rows_added": 0,
        "warnings": [],
    }

    #load translation cache
    cache = _load_translation_cache(cache_path)

    #translate
    new_llm_rows: List[Dict[str, Any]] = []
    new_scoring_rows: List[Dict[str, Any]] = []

    for src_row in ro_rows:
        source_id = str(src_row.get("claim_id", ""))
        if not source_id:
            continue

        mt_claim_id = source_id + MT_SUFFIX

        #skip if alr there
        if mt_claim_id in existing_llm_ids:
            counters["already_translated"] += 1
            continue

        original_text = str(src_row.get("claim_text", "")).strip()
        if not original_text:
            counters["warnings"].append(f"Skipping {source_id}: empty claim_text")
            continue

        #get translation
        created_utc = utc_now_iso()
        translation_warnings: List[str] = []

        if mt_claim_id in cache:
            translated_text = cache[mt_claim_id]
        else:
            try:
                translated_text = _call_translation_llm(
                    claim_text=original_text,
                    translation_cfg=translation_cfg,
                )
            except Exception as exc:
                msg = f"Translation API error for {source_id}: {exc}"
                counters["warnings"].append(msg)
                counters["translation_errors"] += 1
                _log(log or logger or step_logger, f"[{STEP_ID}] ERROR: {msg}")
                continue

            #validate
            translation_warnings = _validate_translation(original_text, translated_text)
            for w in translation_warnings:
                counters["warnings"].append(f"{source_id}: {w}")
                counters["translation_warnings"] += 1

            if not translated_text or not translated_text.strip():
                msg = f"Empty translation for {source_id}; skipping"
                counters["warnings"].append(msg)
                counters["translation_errors"] += 1
                _log(log or logger or step_logger, f"[{STEP_ID}] WARN: {msg}")
                continue

            #save to cache
            cache[mt_claim_id] = translated_text
            _append_translation_log(
                cache_path,
                source_claim_id=source_id,
                mt_claim_id=mt_claim_id,
                original_text=original_text,
                translated_text=translated_text,
                model_name=model_name,
                created_utc=created_utc,
                warnings=translation_warnings,
            )

        #build translated rows
        mt_llm_row: Dict[str, Any] = {
            "run_id": rid,
            "created_utc": created_utc,
            "code_version": cv,
            "claim_id": mt_claim_id,
            "lang": "ro_mt_en",
            "claim_text": translated_text,
            "claim_date": src_row.get("claim_date", ""),
        }
        new_llm_rows.append(mt_llm_row)
        counters["translated_ok"] += 1

        #build matching scoring row
        if mt_claim_id not in existing_scoring_ids:
            rating = scoring_map.get(source_id, "")
            if rating:
                mt_scoring_row: Dict[str, Any] = {
                    "run_id": rid,
                    "created_utc": created_utc,
                    "code_version": cv,
                    "claim_id": mt_claim_id,
                    "rating": rating,
                }
                new_scoring_rows.append(mt_scoring_row)
                counters["scoring_rows_added"] += 1
            else:
                counters["warnings"].append(
                    f"No scoring row found for source claim {source_id}; "
                    f"ro_mt_en row {mt_claim_id} will have no rating."
                )

    #write updated files
    if new_llm_rows:
        write_jsonl_atomic(for_llms_path, llm_rows + new_llm_rows)

    if new_scoring_rows:
        write_jsonl_atomic(for_scoring_path, scoring_rows + new_scoring_rows)

    #log
    _log(
        log or logger or step_logger,
        f"[{STEP_ID}] Translation complete: "
        f"{counters['translated_ok']} translated, "
        f"{counters['already_translated']} already cached in forLLMs.jsonl, "
        f"{counters['translation_errors']} errors, "
        f"{counters['translation_warnings']} validation warnings.",
    )
    for w in counters["warnings"]:
        _log(log or logger or step_logger, f"[{STEP_ID}] WARN: {w}")

    return counters


#help log
def _log(log_obj: Any, message: str) -> None:
    if log_obj is None:
        print(message)
        return
    for meth in ("info", "log_info", "log"):
        fn = getattr(log_obj, meth, None)
        if callable(fn):
            try:
                fn({"message": message})
                return
            except Exception:
                pass
    print(message)


#CLI wrapper
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=STEP_NAME)
    ap.add_argument("--run-id", required=True, help="Run id (used for Paths).")
    ap.add_argument("--code-version", default=None, help="Manual code version string.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    config = load_config()
    paths = Paths(run_id=args.run_id)

    counters = run_step02b(
        paths=paths,
        config=config,
        run_id=args.run_id,
        code_version=args.code_version,
    )

    if counters.get("skipped"):
        print(f"[{STEP_ID}] Skipped: {counters.get('reason')}")
    else:
        print(
            f"[{STEP_ID}] Done: {counters['translated_ok']} translated, "
            f"{counters['already_translated']} already done, "
            f"{counters['translation_errors']} errors."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
