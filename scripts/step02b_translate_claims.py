from __future__ import annotations

"""
scripts/step02b_translate_claims.py

Translate Romanian claims to English for the ro_mt_en experimental condition.

Must run after step01 (forLLMs.jsonl) and step02 (forScoring.jsonl), before step03.

Inputs
------
- runs/<run_id>/claims/forLLMs.jsonl      (written by step01)
- runs/<run_id>/claims/forScoring.jsonl   (written by step02)

Outputs
-------
- runs/<run_id>/claims/forLLMs.jsonl      (appended with ro_mt_en rows)
- runs/<run_id>/claims/forScoring.jsonl   (appended with ro_mt_en scoring rows)
- runs/<run_id>/cache/translation_cache.jsonl  (per-claim translation log/cache)

Behavior
--------
- If config.translation.enabled is False, this step is a no-op (exits immediately).
- Only rows with lang == "ro" are translated.
- Translated rows have:
    claim_id  = <original_claim_id> + "_mt"
    lang      = "ro_mt_en"
    claim_text = English translation of the original Romanian claim
    claim_date, run_id, created_utc, code_version = same as original
- Scoring rows mirror the same claim_id mapping with the same rating.
- Already-translated claim_ids (ending in "_mt") found in forLLMs.jsonl are skipped
  so the step is safe to re-run (idempotent).
- Translation is cached in translation_cache.jsonl; cached translations are reused
  on re-runs without calling the API again.

claim_id naming convention
--------------------------
  mt_claim_id = source_claim_id + "_mt"
  Step11 can derive source_claim_id = mt_claim_id[:-3] (strip "_mt") to join
  ro_mt_en rows back to their original ro counterparts.
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

# Suffix appended to the original claim_id to form the translated claim_id.
# Step11 uses this convention to join ro_mt_en rows back to their ro source.
MT_SUFFIX = "_mt"

# ---------------------------------------------------------------------------
# Translation prompt
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# LLM call (reuses ccir.models.runner infrastructure)
# ---------------------------------------------------------------------------

def _call_translation_llm(
    *,
    claim_text: str,
    translation_cfg: Any,
) -> str:
    """
    Call the configured translation model and return the raw translated string.
    Reuses the existing run_small_llm() runner with expect_json=False.
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


# ---------------------------------------------------------------------------
# Translation validation
# ---------------------------------------------------------------------------

def _validate_translation(original: str, translated: str) -> List[str]:
    """
    Basic sanity-check of a translated claim.
    Returns a (possibly empty) list of warning strings.
    An empty list means no warnings.
    """
    warnings: List[str] = []

    if not translated or not translated.strip():
        warnings.append("translated output is empty")
        return warnings

    # Check that standalone numbers from the original survive in the translation.
    # This catches cases where the model dropped a key quantity.
    numbers = re.findall(r"(?<!\w)\d[\d.,]*(?!\w)", original)
    for num in numbers:
        # Strip punctuation-style separators for comparison
        canonical = re.sub(r"[,.]", "", num)
        translated_stripped = re.sub(r"[,.]", "", translated)
        if canonical and canonical not in translated_stripped:
            warnings.append(f"number '{num}' from original not found in translation")

    return warnings


# ---------------------------------------------------------------------------
# Translation cache helpers
# ---------------------------------------------------------------------------

def _load_translation_cache(cache_path: Path) -> Dict[str, str]:
    """
    Load existing cached translations.
    Returns dict: mt_claim_id -> translated_text
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


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Check if ro_mt_en condition is active
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Read existing rows
    # ------------------------------------------------------------------
    llm_rows: List[Dict[str, Any]] = [
        r for r in read_jsonl(for_llms_path) if isinstance(r, dict)
    ]
    scoring_rows: List[Dict[str, Any]] = [
        r for r in read_jsonl(for_scoring_path) if isinstance(r, dict)
    ]

    # Build sets of claim_ids already present (to support idempotent re-runs)
    existing_llm_ids = {str(r.get("claim_id", "")) for r in llm_rows}
    existing_scoring_ids = {str(r.get("claim_id", "")) for r in scoring_rows}

    # Build scoring lookup: original claim_id -> rating
    scoring_map: Dict[str, str] = {}
    for r in scoring_rows:
        cid = r.get("claim_id")
        rating = r.get("rating")
        if cid and rating:
            scoring_map[str(cid)] = str(rating)

    # Identify Romanian source rows to translate
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

    # ------------------------------------------------------------------
    # Load translation cache
    # ------------------------------------------------------------------
    cache = _load_translation_cache(cache_path)

    # ------------------------------------------------------------------
    # Translate
    # ------------------------------------------------------------------
    new_llm_rows: List[Dict[str, Any]] = []
    new_scoring_rows: List[Dict[str, Any]] = []

    for src_row in ro_rows:
        source_id = str(src_row.get("claim_id", ""))
        if not source_id:
            continue

        mt_claim_id = source_id + MT_SUFFIX

        # Skip if already in forLLMs.jsonl
        if mt_claim_id in existing_llm_ids:
            counters["already_translated"] += 1
            continue

        original_text = str(src_row.get("claim_text", "")).strip()
        if not original_text:
            counters["warnings"].append(f"Skipping {source_id}: empty claim_text")
            continue

        # ------------------------------------------------------------------
        # Get translation (from cache or API)
        # ------------------------------------------------------------------
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

            # Validate
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

            # Save to cache
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

        # ------------------------------------------------------------------
        # Build the translated LLM row
        # Schema matches LLMClaimRow exactly (no extra fields):
        #   claim_id, lang, claim_text, claim_date, run_id, created_utc, code_version
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Build the matching scoring row (same rating as the source ro claim)
        # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Write updated files (atomic)
    # ------------------------------------------------------------------
    if new_llm_rows:
        write_jsonl_atomic(for_llms_path, llm_rows + new_llm_rows)

    if new_scoring_rows:
        write_jsonl_atomic(for_scoring_path, scoring_rows + new_scoring_rows)

    # ------------------------------------------------------------------
    # Log summary
    # ------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------

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
