from __future__ import annotations

"""
purpose: prepare euroverdict into claims/all.jsonl and a verdict mapping doc
 - extract ID, language, claim, date, rating/verdict
 - normalize dates to YYYY-MM-DD
 - map multilingual euroverdict rating to supported/refuted/not enough evidence/conflicting evidence
 - discard rows without ID, language, claim, date, and rating/verdict
 - write claims/all.jsonl and verdicts/mapping.jsonl
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
    #drop combining unnecessary punctuation
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def _norm_text(x: str) -> str:
    """
    normalize labels
    """
    s = x.strip().lower()
    s = _strip_diacritics(s)
    s = _WS_RE.sub(" ", s)
    #replace punctuation with spaces
    s = re.sub(r"[^0-9a-z\u00C0-\u024F\u0370-\u03FF\u0400-\u04FF\s]", " ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _parse_date_iso(date_raw: str) -> Optional[str]:
    """
    normalize dates
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

    #fallback for timestamp
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.date().isoformat()
    except Exception:
        return None


def _contains_any(norm: str, needles: List[str]) -> bool:
    return any(n in norm for n in needles)


def _map_to_averitec_label(rating_or_verdict: str) -> Tuple[Optional[str], JsonObj]:
    """
    euroverdict label mapping to 4-way labels
    """
    raw = rating_or_verdict
    norm = _norm_text(raw)
    debug: JsonObj = {"raw": raw, "norm": norm}

    #cherry picking evidence isn't used
    if "cherry" in norm and "pick" in norm:
        debug["cherry_picking_removed"] = True
        return AV_LABEL_CONFLICT, debug

    #greek not enough evidence
    if "λειπει θεματικο περιεχομενο" in norm:
        return AV_LABEL_NEE, debug

    #conflicting evidence
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

    #greek misleading
    if "παραπλανητικο" in norm:
        return AV_LABEL_CONFLICT, debug

    #greek conflicting
    if "μιξη γεγονοτων" in norm or "παραποιησεων" in norm:
        return AV_LABEL_CONFLICT, debug

    #satire
    if "σατιρα" in norm:
        return AV_LABEL_CONFLICT, debug

    #conflict (random label)
    if "like farming" in norm:
        return AV_LABEL_CONFLICT, debug

    #refuted greek
    if _contains_any(
        norm,
        [
            "παραπληροφορηση",
            "συνωμοσιολογια",
            "απατη",
            "κινδυνολογια",
            "ψευδ",  #covers ψευδες/ψευδης/ψευδης ισχυρισμος
            "ψευδοεπιστημη",
        ],
    ):
        return AV_LABEL_REFUTED, debug

    #refuted
    if _contains_any(
        norm,
        [
            # english
            "false",
            "incorrect",
            "wrong",
            "refuted",
            "debunked",
            "hoax",
            "fake",
            # italian
            "notizia falsa",
            # german
            "falsch",
            "grosstenteils falsch",
            "großtenteils falsch",
            "manipuliert",
            #french (not necessary)
            "faux",
            #romanian
            "fals",
        ],
    ):
        return AV_LABEL_REFUTED, debug

    #supported
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

    #output under runs/<run_id>/
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
        #load rows
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

                #req fields
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
