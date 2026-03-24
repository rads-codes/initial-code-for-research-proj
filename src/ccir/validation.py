from __future__ import annotations

"""
src/ccir/validation.py

STRICT RUN-SCOPED validation.

This file expects all pipeline artifacts to live under:
  data/processed/runs/<run_id>/...

It validates:
- File exists + non-empty
- JSONL parseable
- Rows match dataclass schemas (shallow type checks)
- Expected row counts based on config.dataset selection

Called by __main__.py at:
- stage="pre" (validate inputs exist before running step)
- stage="post" (validate outputs exist after running step)

Import-safe: no filesystem writes at import time.
"""

from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, get_args, get_origin

from ccir.io_utils import read_jsonl
from ccir.schemas import (
    AllClaimsFormat,
    ClaimWithURLs,
    LLMClaimRow,
    ScoringClaimRow,
    TopKURL,
)


class ValidationError(RuntimeError):
    """Raised when validation fails."""


# -----------------------------
# Typing helpers
# -----------------------------

def _is_optional_type(tp: Any) -> bool:
    origin = get_origin(tp)
    return origin is Union and type(None) in get_args(tp)


def _shallow_type_ok(value: Any, expected_type: Any) -> bool:
    """
    Shallow runtime checks for common cases.
    We keep this intentionally lightweight (dataclasses don't enforce types).
    """
    if expected_type is Any:
        return True

    # Optional[T]
    if _is_optional_type(expected_type):
        if value is None:
            return True
        non_none = tuple(t for t in get_args(expected_type) if t is not type(None))
        return any(_shallow_type_ok(value, t) for t in non_none) if non_none else True

    origin = get_origin(expected_type)

    if origin in (list, List):
        return isinstance(value, list)
    if origin in (dict, Dict):
        return isinstance(value, dict)
    if origin in (tuple, Tuple):
        return isinstance(value, tuple)

    if expected_type in (str, int, float, bool):
        return isinstance(value, expected_type)

    if expected_type is Path:
        return isinstance(value, (str, Path))

    # Don't over-reject unknown typing constructs
    return True


# -----------------------------
# Filesystem checks
# -----------------------------

def validate_file_exists(path: Path) -> None:
    if not path.exists():
        raise ValidationError(f"Missing required file: {path}")


def validate_file_nonempty(path: Path) -> None:
    validate_file_exists(path)
    if path.stat().st_size <= 0:
        raise ValidationError(f"File exists but is empty: {path}")


def validate_jsonl_parseable(path: Path) -> List[Dict[str, Any]]:
    """
    Schema-less JSONL:
    - file exists
    - non-empty
    - read_jsonl returns dict rows
    """
    validate_file_nonempty(path)
    rows_any = read_jsonl(path)
    rows = list(rows_any) if not isinstance(rows_any, list) else rows_any
    if not isinstance(rows, list):
        raise ValidationError(f"{path}: expected read_jsonl() to return list/iterable, got {type(rows)}")
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise ValidationError(f"{path}: row {i} expected dict/object, got {type(r)}")
    return rows


# -----------------------------
# Dataclass schema validation
# -----------------------------

def _required_field_names(cls: Type[Any]) -> set[str]:
    req: set[str] = set()
    for f in fields(cls):
        has_default = not (f.default is MISSING and f.default_factory is MISSING)  # type: ignore[attr-defined]
        if has_default:
            continue
        if _is_optional_type(f.type):
            continue
        req.add(f.name)
    return req


def validate_rows_match_dataclass(
    rows: Sequence[Dict[str, Any]],
    cls: Type[Any],
    *,
    allow_extra: bool = False,
    max_errors: int = 20,
) -> None:
    if not is_dataclass(cls):
        raise ValidationError(f"Target schema {cls} is not a dataclass.")

    allowed = {f.name for f in fields(cls)}
    required = _required_field_names(cls)

    errors: List[str] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"Row {i}: expected dict, got {type(row)}")
            if len(errors) >= max_errors:
                break
            continue

        row_keys = set(row.keys())
        missing = sorted(required - row_keys)
        extra = sorted(row_keys - allowed)

        if missing:
            errors.append(f"Row {i}: missing required keys {missing}")
        if (not allow_extra) and extra:
            errors.append(f"Row {i}: unexpected extra keys {extra}")

        for f in fields(cls):
            if f.name not in row:
                continue
            if not _shallow_type_ok(row[f.name], f.type):
                errors.append(f"Row {i}: field '{f.name}' expected {f.type}, got {type(row[f.name])}")

        if len(errors) >= max_errors:
            break

    if errors:
        raise ValidationError(f"Schema validation failed for {cls.__name__}:\n" + "\n".join(errors))


def validate_row_count(
    path: Path,
    rows: Sequence[Any],
    *,
    expected_exact: Optional[int] = None,
    min_rows: Optional[int] = None,
) -> None:
    n = len(rows)
    if expected_exact is not None and n != expected_exact:
        raise ValidationError(f"{path}: expected exactly {expected_exact} rows, found {n}")
    if min_rows is not None and n < min_rows:
        raise ValidationError(f"{path}: expected at least {min_rows} rows, found {n}")


def expected_selected_claim_count(config: Any) -> int:
    """
    Count based on configs.py structure:
      config.dataset.languages: list[str]
      config.dataset.claims_per_language_default: int
      config.dataset.claims_per_language_override: dict[str,int] (optional)

    Synthetic languages (e.g. "ro_mt_en") are excluded from this count because they are
    generated by step02b rather than selected from the raw dataset by step01.
    """
    # Languages that are generated synthetically by later steps, not sourced from raw data.
    _SYNTHETIC_LANGUAGES = {"ro_mt_en"}

    try:
        langs: List[str] = list(config.dataset.languages)
        default_n: int = int(config.dataset.claims_per_language_default)
        overrides: Dict[str, int] = dict(getattr(config.dataset, "claims_per_language_override", {}) or {})
    except Exception as e:
        raise ValidationError(
            "Config missing dataset selection fields. Expected config.dataset.languages, "
            "claims_per_language_default, and optionally claims_per_language_override."
        ) from e

    return sum(int(overrides.get(lang, default_n)) for lang in langs if lang not in _SYNTHETIC_LANGUAGES)


def _read_and_validate_jsonl(
    path: Path,
    cls: Type[Any],
    *,
    allow_extra: bool,
    expected_exact: Optional[int] = None,
    min_rows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    validate_file_nonempty(path)
    rows_any = read_jsonl(path)
    rows = list(rows_any) if not isinstance(rows_any, list) else rows_any
    validate_rows_match_dataclass(rows, cls, allow_extra=allow_extra)
    validate_row_count(path, rows, expected_exact=expected_exact, min_rows=min_rows)
    return rows


# -----------------------------
# Main validation entrypoint
# -----------------------------

def validate_step(
    *,
    stage: str,
    step: str,
    paths: Any,
    config: Any,
    allow_extra: bool = False,
) -> None:
    """
    Validate known inputs/outputs for a step.

    Parameters
    - stage: "pre" or "post" (also accepts heavy aliases like "pre_heavy"/"post_heavy")
    - step: "00", "01", ..., "11"
    - paths: Paths object
    - config: RunConfig
    """
    # Normalize stage aliases used by __main__.py
    stage_norm_map = {
        "pre": "pre",
        "post": "post",
        # heavy aliases (your __main__.py uses "pre_heavy")
        "pre_heavy": "pre",
        "post_heavy": "post",
        # optional alternates (your __main__ has a fallback for these)
        "heavy_pre": "pre",
        "heavy_post": "post",
    }
    stage_norm = stage_norm_map.get(stage)
    if stage_norm is None:
        raise ValidationError(
            "stage must be one of "
            f"{sorted(stage_norm_map.keys())}, got {stage!r}"
        )

    selected_n = expected_selected_claim_count(config)

    # --- Run-scoped canonical paths (required) ---
    claims_all = getattr(paths, "run_claims_all_jsonl", None)
    claims_for_llms = getattr(paths, "run_claims_for_llms_jsonl", None)
    claims_for_scoring = getattr(paths, "run_claims_for_scoring_jsonl", None)
    verdicts_mapping = getattr(paths, "run_verdicts_mapping_jsonl", None)

    evidence_urls = getattr(paths, "run_evidence_urls_jsonl", None)
    evidence_topk = getattr(paths, "run_evidence_topk_urls_jsonl", None)

    for name, p in [
        ("run_claims_all_jsonl", claims_all),
        ("run_claims_for_llms_jsonl", claims_for_llms),
        ("run_claims_for_scoring_jsonl", claims_for_scoring),
        ("run_verdicts_mapping_jsonl", verdicts_mapping),
        ("run_evidence_urls_jsonl", evidence_urls),
        ("run_evidence_topk_urls_jsonl", evidence_topk),
    ]:
        if not isinstance(p, Path):
            raise ValidationError(f"Paths missing {name} (expected Path)")

    # Convention: schema_cls is None => schema-less JSONL (parseable + non-empty only)
    step_outputs: Dict[str, List[Tuple[Path, Optional[Type[Any]], Dict[str, Optional[int]]]]] = {
        "00": [
            (claims_all, AllClaimsFormat, {"min_rows": 1, "expected_exact": None}),
            (verdicts_mapping, None, {"min_rows": 1, "expected_exact": None}),
        ],
        "01": [
            (claims_for_llms, LLMClaimRow, {"expected_exact": selected_n, "min_rows": None}),
        ],
        "02": [
            (claims_for_scoring, ScoringClaimRow, {"expected_exact": selected_n, "min_rows": None}),
        ],
        "03": [
            # Use min_rows (not expected_exact) because step02b may have added ro_mt_en
            # rows to evidence_urls, making the total exceed the base selected_n count.
            (evidence_urls, ClaimWithURLs, {"min_rows": selected_n, "expected_exact": None}),
        ],
        "05": [
            (evidence_topk, TopKURL, {"min_rows": selected_n, "expected_exact": None}),
        ],
    }

    step_inputs: Dict[str, List[Tuple[Path, Optional[Type[Any]], Dict[str, Optional[int]]]]] = {
        "01": [
            (claims_all, AllClaimsFormat, {"min_rows": selected_n, "expected_exact": None}),
        ],
        "02": [
            (claims_all, AllClaimsFormat, {"min_rows": selected_n, "expected_exact": None}),
            (claims_for_llms, LLMClaimRow, {"expected_exact": selected_n, "min_rows": None}),
        ],
        "03": [
            # After step02b, forLLMs and forScoring may have more rows than selected_n
            # (the added ro_mt_en rows). Use min_rows so validation still passes.
            (claims_for_llms, LLMClaimRow, {"min_rows": selected_n, "expected_exact": None}),
            (claims_for_scoring, ScoringClaimRow, {"min_rows": selected_n, "expected_exact": None}),
        ],
        "05": [
            (evidence_urls, ClaimWithURLs, {"min_rows": selected_n, "expected_exact": None}),
        ],
        # Step 04 is "heavy" but its *inputs* are the same as Step 05’s input: URLs
        "04": [
            (evidence_urls, ClaimWithURLs, {"min_rows": selected_n, "expected_exact": None}),
        ],
    }

    def _validate_item(path: Path, schema_cls: Optional[Type[Any]], cnt: Dict[str, Optional[int]]) -> None:
        if schema_cls is None:
            rows = validate_jsonl_parseable(path)
            validate_row_count(
                path,
                rows,
                expected_exact=cnt.get("expected_exact"),
                min_rows=cnt.get("min_rows"),
            )
        else:
            _read_and_validate_jsonl(
                path,
                schema_cls,
                allow_extra=allow_extra,
                expected_exact=cnt.get("expected_exact"),
                min_rows=cnt.get("min_rows"),
            )

    if stage_norm == "pre":
        for path, cls, cnt in step_inputs.get(step, []):
            _validate_item(path, cls, cnt)

    if stage_norm == "post":
        for path, cls, cnt in step_outputs.get(step, []):
            _validate_item(path, cls, cnt)