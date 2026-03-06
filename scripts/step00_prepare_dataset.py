from __future__ import annotations

"""
scripts/step00_prepare_dataset.py

Step 00: Prepare EuroVerdict into stable claims/all.jsonl + verdict mapping doc.

Key behaviors (per spec):
- Extract: ID, Language, Claim, Date, Rating (fallback to Verdict)
- Normalize dates to YYYY-MM-DD
- Map EuroVerdict Rating/Verdict to 4 AVeriTeC labels:
    Supported, Refuted, Not Enough Evidence, Conflicting Evidence
  (Cherry Picking is not its own label; map to Conflicting Evidence.)
- Discard rows missing: ID, Language, Claim, Date, and (Rating or Verdict)
- Write:
    <run_root>/claims/all.jsonl           (AllClaimsFormat JSONL)
    <run_root>/verdicts/mapping.jsonl    (single-row JSONL documenting mapping + stats)
  and report_00.jsonl via logging_utils.step_logger

This module is safe to import; no work runs at import time.
"""

import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ccir.io_utils import read_jsonl, read_text, write_jsonl_atomic
from ccir.logging_utils import step_logger
from ccir.paths import Paths
from ccir.schemas import AllClaimsFormat, utc_now_iso

JsonObj = Dict[str, Any]

_WS_RE = re.compile(r"\s+")

AV_LABEL_SUPPORTED = "Supported"
AV_LABEL_REFUTED = "Refuted"
AV_LABEL_NEE = "Not Enough Evidence"
AV_LABEL_CONFLICT = "Conflicting Evidence"


def _clean_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        return s or None
    return str(x).strip() or None


def _strip_diacritics(s: str) -> str:
    # NFKD splits base chars + combining marks; drop combining marks
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def _norm_text(x: str) -> str:
    """
    Normalize label-ish text for robust matching:
    - lowercase
    - strip diacritics (critical for Greek + accented Latin)
    - collapse whitespace
    - remove punctuation (replace with spaces)
    """
    s = x.strip().lower()
    s = _strip_diacritics(s)
    s = _WS_RE.sub(" ", s)
    # keep letters/numbers/spaces across Latin/Greek/Cyrillic; replace punctuation with spaces
    s = re.sub(r"[^0-9a-z\u00C0-\u024F\u0370-\u03FF\u0400-\u04FF\s]", " ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _parse_date_iso(date_raw: str) -> Optional[str]:
    """
    Normalize dates to YYYY-MM-DD.

    EuroVerdict.json is overwhelmingly YYYY-MM-DD, but we support a few alternates
    + ISO timestamps. We keep strict behavior: if unparsable, the row is dropped.
    """
    s = date_raw.strip()
    if not s:
        return None

    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d.%m.%Y",
        "%d-%m-%Y",
        "%Y.%m.%d",
    ):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass

    # ISO timestamp fallback (e.g., 2024-01-15T00:00:00Z)
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.date().isoformat()
    except Exception:
        return None


def _contains_any(norm: str, needles: List[str]) -> bool:
    return any(n in norm for n in needles)


def _map_to_averitec_label(rating_or_verdict: str) -> Tuple[Optional[str], JsonObj]:
    """
    Map EuroVerdict Rating/Verdict strings to AVeriTeC 4-way labels.

    Tuned to the actual labels observed in your mapping.jsonl:
      - Greek: Παραπληροφόρηση, Λείπει θεματικό περιεχόμενο, Συνωμοσιολογία,
               Απάτη, Κινδυνολογία, Μίξη γεγονότων και παραποιήσεων,
               composites like "Παραπληροφόρηση, ψεύτικη εικόνα", "Σάτιρα, ..."
      - German: Fehlender Kontext., Falscher Kontext., Teilweise falsch., Falsch., Manipuliert.
      - French: Faux, Partiellement faux, Manque de contexte, Contexte manquant,
                typo: Manque de context
      - Italian: Notizia falsa, Fuori contesto
      - Romanian: Fals, Context lipsă, Parțial fals
      - English: False, Missing context, Partly false
    """
    raw = rating_or_verdict
    norm = _norm_text(raw)
    debug: JsonObj = {"raw": raw, "norm": norm}

    # Cherry-picking collapses into Conflicting Evidence
    if "cherry" in norm and "pick" in norm:
        debug["cherry_picking_removed"] = True
        return AV_LABEL_CONFLICT, debug

    # --- Not Enough Evidence ---
    # Greek bucket in your dataset
    if "λειπει θεματικο περιεχομενο" in norm:
        return AV_LABEL_NEE, debug

    # --- Conflicting Evidence / Missing context / Mixed / Satire ---
    # Missing/out-of-context buckets (multi-lingual + observed typo)
    if _contains_any(
        norm,
        [
            # English
            "missing context",
            "out of context",
            "misleading",
            "partly false",
            "partially false",
            # Italian
            "fuori contesto",
            # German
            "falscher kontext",
            "fehlender kontext",
            "kontext ist falsch",
            "der kontext ist falsch",
            "teilweise falsch",
            # French
            "manque de contexte",
            "contexte manquant",
            "manque de context",  # observed typo
            "partiellement faux",
            # Romanian
            "context lipsa",
            "parțial fals",
            "partial fals",
            "parcial fals",
        ],
    ):
        return AV_LABEL_CONFLICT, debug

    # Greek: "Παραπλανητικό" (misleading)
    if "παραπλανητικο" in norm:
        return AV_LABEL_CONFLICT, debug

    # Greek "mixture of facts and distortions" -> conflict
    if "μιξη γεγονοτων" in norm or "παραποιησεων" in norm:
        return AV_LABEL_CONFLICT, debug

    # Satire often appears mixed with other tags -> conflict
    if "σατιρα" in norm:
        return AV_LABEL_CONFLICT, debug

    # "Like farming" bucket (observed) -> conflict (engagement manipulation)
    if "like farming" in norm:
        return AV_LABEL_CONFLICT, debug

    # --- Refuted / False / Misinformation / Conspiracy / Fraud / Alarmism ---
    # Greek broad buckets: treat as Refuted (most consistent with "misinformation" categorization)
    if _contains_any(
        norm,
        [
            "παραπληροφορηση",
            "συνωμοσιολογια",
            "απατη",
            "κινδυνολογια",
            "ψευδ",  # stem covers ψευδες/ψευδης/ψευδης ισχυρισμος
            "ψευδοεπιστημη",
        ],
    ):
        return AV_LABEL_REFUTED, debug

    # Standard false-ish buckets
    if _contains_any(
        norm,
        [
            # English
            "false",
            "incorrect",
            "wrong",
            "refuted",
            "debunked",
            "hoax",
            "fake",
            # Italian
            "notizia falsa",
            # German
            "falsch",
            "grosstenteils falsch",
            "großtenteils falsch",
            "manipuliert",
            # French
            "faux",
            # Romanian
            "fals",
        ],
    ):
        return AV_LABEL_REFUTED, debug

    # --- Supported / True (rare in your file) ---
    if _contains_any(norm, ["true", "correct", "accurate", "verified", "supported"]) and "not true" not in norm:
        return AV_LABEL_SUPPORTED, debug

    return None, debug


def _to_jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    return x


def _path_attr_or_fallback(paths: Paths, attr: str, fallback: Path) -> Path:
    p = getattr(paths, attr, None)
    if isinstance(p, Path):
        return p
    return fallback


def run_step_00_prepare_dataset(*, paths: Paths, run_id: str, code_version: str) -> None:
    raw_in = _path_attr_or_fallback(paths, "raw_euroverdict_jsonl", paths.raw_root / "EuroVerdict.json")

    # Prefer run-scoped output under runs/<run_id>/...
    run_root = (
        getattr(paths, "run_root", None)
        or getattr(paths, "run_dir", None)
        or (paths.processed_root / "runs" / run_id)
    )

    out_claims = _path_attr_or_fallback(paths, "run_claims_all_jsonl", paths.run_root / "claims" / "all.jsonl")
    out_mapping = _path_attr_or_fallback(paths, "run_verdicts_mapping_jsonl",
                                         paths.run_root / "verdicts" / "mapping.jsonl")

    step_name = "00_prepare_dataset"

    with step_logger(
        paths.report_jsonl(0),
        run_id=run_id,
        code_version=code_version,
        step=step_name,
        start_message="Preparing EuroVerdict claims dataset",
        start_fields={"input": str(raw_in), "output_claims": str(out_claims), "output_mapping": str(out_mapping)},
    ) as log:
        # Load rows (JSON array preferred; JSONL fallback supported)
        rows: List[JsonObj] = []
        try:
            with log.timer("read_jsonl"):
                rows = list(read_jsonl(raw_in))
        except Exception:
            with log.timer("read_json_array_fallback"):
                txt = read_text(raw_in)
                parsed = json.loads(txt)
                if isinstance(parsed, list):
                    rows = [r for r in parsed if isinstance(r, dict)]
                else:
                    raise RuntimeError(f"Unsupported EuroVerdict input format in {raw_in}: expected JSONL or JSON list")

        log.incr("rows_read", len(rows))

        kept: List[AllClaimsFormat] = []
        dropped_examples: Dict[str, List[JsonObj]] = defaultdict(list)

        mapped_label_counts: Counter[str] = Counter()
        unmapped_label_counts: Counter[str] = Counter()
        kept_lang_counts: Counter[str] = Counter()
        dropped_lang_missing_date: Counter[str] = Counter()

        with log.timer("normalize_rows"):
            for r in rows:
                claim_id = _clean_str(r.get("ID"))
                lang = _clean_str(r.get("Language"))
                claim_text = _clean_str(r.get("Claim"))
                date_s = _clean_str(r.get("Date"))
                rating_or_verdict = _clean_str(r.get("Rating")) or _clean_str(r.get("Verdict"))

                # Required fields (per spec)
                if claim_id is None:
                    log.incr("dropped_missing_id")
                    if len(dropped_examples["missing_id"]) < 5:
                        dropped_examples["missing_id"].append({"row": r})
                    continue
                if lang is None:
                    log.incr("dropped_missing_lang")
                    if len(dropped_examples["missing_lang"]) < 5:
                        dropped_examples["missing_lang"].append({"claim_id": claim_id, "row": r})
                    continue
                if claim_text is None:
                    log.incr("dropped_missing_claim_text")
                    if len(dropped_examples["missing_claim_text"]) < 5:
                        dropped_examples["missing_claim_text"].append({"claim_id": claim_id, "row": r})
                    continue
                if date_s is None:
                    log.incr("dropped_missing_date")
                    dropped_lang_missing_date[lang] += 1
                    if len(dropped_examples["missing_date"]) < 10:
                        dropped_examples["missing_date"].append({"claim_id": claim_id, "lang": lang})
                    continue
                if rating_or_verdict is None:
                    log.incr("dropped_missing_rating_and_verdict")
                    if len(dropped_examples["missing_rating_and_verdict"]) < 5:
                        dropped_examples["missing_rating_and_verdict"].append({"claim_id": claim_id, "row": r})
                    continue

                date_norm = _parse_date_iso(date_s)
                if date_norm is None:
                    log.incr("dropped_bad_date")
                    if len(dropped_examples["bad_date"]) < 10:
                        dropped_examples["bad_date"].append({"claim_id": claim_id, "date_raw": date_s})
                    continue

                mapped_label, debug = _map_to_averitec_label(rating_or_verdict)
                if mapped_label is None:
                    unmapped_label_counts[rating_or_verdict] += 1
                    log.incr("dropped_unmapped_label")
                    if len(dropped_examples["unmapped_label"]) < 50:
                        dropped_examples["unmapped_label"].append({"claim_id": claim_id, "lang": lang, **debug})
                    continue

                mapped_label_counts[mapped_label] += 1
                kept_lang_counts[lang] += 1

                kept.append(
                    AllClaimsFormat(
                        run_id=run_id,
                        created_utc=utc_now_iso(),
                        code_version=code_version,
                        claim_id=claim_id,
                        lang=lang,
                        claim_text=claim_text,
                        claim_date=date_norm,
                        rating=mapped_label,
                    )
                )

        log.incr("kept", len(kept))
        log.incr("dropped_total", len(rows) - len(kept))

        with log.timer("write_claims_all"):
            write_jsonl_atomic(out_claims, [_to_jsonable(x) for x in kept])

        mapping_doc: JsonObj = {
            "run_id": run_id,
            "created_utc": utc_now_iso(),
            "code_version": code_version,
            "step": step_name,
            "input": str(raw_in),
            "output_claims": str(out_claims),
            "output_mapping": str(out_mapping),
            "mapping_policy": {
                "labels": [AV_LABEL_SUPPORTED, AV_LABEL_REFUTED, AV_LABEL_NEE, AV_LABEL_CONFLICT],
                "notes": [
                    "Used Rating if present else fell back to Verdict.",
                    "Cherry-picking is not emitted as a standalone label; it maps to Conflicting Evidence.",
                    "Rows missing ID/Language/Claim/Date/(Rating or Verdict) are dropped.",
                    "In the provided EuroVerdict.json, es/pl have Date=null for all rows; those are dropped by design.",
                    "Greek diacritics are stripped during normalization for robust matching.",
                ],
            },
            "counts": {
                "rows_read": len(rows),
                "kept": len(kept),
                "dropped": len(rows) - len(kept),
                "kept_by_lang": dict(kept_lang_counts),
                "dropped_missing_date_by_lang": dict(dropped_lang_missing_date),
                "mapped_label_counts": dict(mapped_label_counts),
                "unmapped_label_top": unmapped_label_counts.most_common(100),
            },
            "dropped_examples": dict(dropped_examples),
        }

        with log.timer("write_verdict_mapping_doc"):
            write_jsonl_atomic(out_mapping, [mapping_doc])

        keep_rate = (len(kept) / len(rows)) if rows else 0.0
        log.set_metric("keep_rate", keep_rate, emit=True)


def run(
    *,
    paths: Paths,
    config: Any | None = None,
    run_id: str | None = None,
    code_version: str | None = None,
    config_hash: str | None = None,
    log: Any | None = None,
    logger: Any | None = None,
) -> None:
    """
    Entrypoint used by src/ccir/__main__.py.
    Accepts a superset of args; forwards only what Step 00 needs.
    """
    rid = run_id or getattr(paths, "run_id", None)
    if rid is None:
        raise ValueError("run_id is required (or paths.run_id must exist).")
    cv = code_version or "dev"
    run_step_00_prepare_dataset(paths=paths, run_id=rid, code_version=cv)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--code-version", default="dev")
    args = ap.parse_args()

    p = Paths(run_id=args.run_id)
    if hasattr(p, "ensure_shared_dirs"):
        p.ensure_shared_dirs()
    if hasattr(p, "ensure_run_dirs"):
        p.ensure_run_dirs()

    run_step_00_prepare_dataset(paths=p, run_id=args.run_id, code_version=args.code_version)