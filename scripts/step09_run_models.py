from __future__ import annotations

"""
scripts/step09_run_models.py

Build SmallLLMPrompts.jsonl and run configured small fact-checking models.
"""

import argparse
import dataclasses
import importlib
import inspect
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ccir.config_loader import load_config
from ccir.io_utils import append_jsonl, read_jsonl, write_jsonl_atomic
from ccir.paths import Paths

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# Optional schema imports
# -----------------------------------------------------------------------------

try:
    from ccir.schemas import SmallLLMprompt, LLMOutput, to_dict, utc_now_iso
except Exception:
    SmallLLMprompt = None
    LLMOutput = None

    def to_dict(x: Any) -> Dict[str, Any]:
        if dataclasses.is_dataclass(x):
            return dataclasses.asdict(x)
        if isinstance(x, dict):
            return dict(x)
        raise TypeError(f"Unsupported row type for to_dict(): {type(x)!r}")

    def utc_now_iso() -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

STEP_ID = "09"
DEFAULT_PROMPT_ID = "small_factcheck_v1"
VALID_VERDICTS = {
    "Supported",
    "Refuted",
    "Conflicting_Evidence",
    "Not_Enough_Evidence",
}

# Set to False later if you want to re-enable ccir.prompts.prompts.
FORCE_FALLBACK_PROMPT = True


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------

def _log_info(log: Any, message: str, **fields: Any) -> None:
    if log is None:
        return
    for meth in ("info", "log_info"):
        if hasattr(log, meth):
            getattr(log, meth)(message, **fields)
            return
    if hasattr(log, "log"):
        payload = {"message": message}
        payload.update(fields)
        log.log(payload)


def _log_count(log: Any, key: str, delta: int = 1) -> None:
    if log is None:
        return
    for meth in ("count", "increment", "incr", "add_count"):
        if hasattr(log, meth):
            getattr(log, meth)(key, delta)
            return
    if hasattr(log, "log"):
        log.log({"kind": "count", "key": key, "delta": delta})


def _maybe_tqdm(iterable: Any, **kwargs: Any) -> Any:
    if tqdm is None:
        return iterable

    defaults = {
        "dynamic_ncols": True,
        "ascii": True,
        "leave": True,
        "mininterval": 0.5,
    }

    for k, v in defaults.items():
        kwargs.setdefault(k, v)

    return tqdm(iterable, **kwargs)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _field_names(schema_cls: Any) -> Optional[set[str]]:
    if schema_cls is None:
        return None
    if dataclasses.is_dataclass(schema_cls):
        return {f.name for f in dataclasses.fields(schema_cls)}
    return None


SMALL_PROMPT_FIELDS = _field_names(SmallLLMprompt)
LLM_OUTPUT_FIELDS = _field_names(LLMOutput)


def _filter_to_schema(payload: Dict[str, Any], allowed: Optional[set[str]]) -> Dict[str, Any]:
    if not allowed:
        return payload
    return {k: v for k, v in payload.items() if k in allowed}


def _make_schema_row(schema_cls: Any, payload: Dict[str, Any], allowed: Optional[set[str]]) -> Dict[str, Any]:
    filtered = _filter_to_schema(payload, allowed)
    if schema_cls is None:
        return filtered
    try:
        return to_dict(schema_cls(**filtered))
    except Exception:
        return filtered


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _claim_id(row: Dict[str, Any]) -> str:
    for key in ("claim_id", "id", "claimId"):
        if key in row and row[key] is not None:
            return str(row[key])
    raise KeyError(f"Could not find claim_id in row keys={list(row.keys())}")


def _claim_text(row: Dict[str, Any]) -> str:
    for key in ("claim_text", "claim", "text"):
        if key in row and row[key] is not None:
            return str(row[key]).strip()
    return ""


def _claim_lang(row: Dict[str, Any]) -> Optional[str]:
    for key in ("lang", "language", "claim_lang"):
        if key in row and row[key] is not None:
            return str(row[key])
    return None


def _claim_date(row: Dict[str, Any]) -> Optional[str]:
    for key in ("claim_date", "date", "claimDate"):
        if key in row and row[key] is not None:
            return str(row[key])
    return None


def _extract_topk_urls(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("top_urls", "urls", "topKURLs", "selected_urls", "top_k_urls"):
        value = row.get(key)
        if isinstance(value, list):
            return [dict(x) for x in value if isinstance(x, dict)]
    return []


def _url_id(item: Dict[str, Any]) -> Optional[str]:
    for key in ("url_id", "URL_ID", "id"):
        if key in item and item[key] is not None:
            return str(item[key])
    return None


def _url_source(item: Dict[str, Any]) -> Optional[str]:
    for key in ("url", "canonical_url", "source", "source_url"):
        if key in item and item[key]:
            return str(item[key])
    return None


def _normalize_verdict(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "Not_Enough_Evidence"

    key = raw.lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "supported": "Supported",
        "refuted": "Refuted",
        "conflicting_evidence": "Conflicting_Evidence",
        "conflicting": "Conflicting_Evidence",
        "not_enough_evidence": "Not_Enough_Evidence",
        "insufficient_evidence": "Not_Enough_Evidence",
        "nee": "Not_Enough_Evidence",
    }
    verdict = mapping.get(key, raw)
    if verdict not in VALID_VERDICTS:
        return "Not_Enough_Evidence"
    return verdict


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            obj = json.loads(fenced.group(1))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass

    return {}


# -----------------------------------------------------------------------------
# Prompt rendering
# -----------------------------------------------------------------------------

def _default_prompt_text(claim_text: str, evidence_docs: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = [
        "Determine whether the given claim is supported or refuted based on only the evidence provided below.",
        "",
        "The possible verdict labels are Supported (when the evidence clearly supports the claim), "
        "Refuted (when the evidence clearly contradicts the claim), "
        "Conflicting_Evidence (when the evidence has both supporting and contradicting statements), "
        "or Not_Enough_Evidence (when the evidence doesn’t provide enough information to evaluate the claim).",
        "",
        "Read the claim, carefully review the evidence documents, decide which verdict label best applies, "
        "and briefly explain your reasoning using specific evidence quotes.",
        "",
        "Claim:",
        claim_text,
        "",
    ]

    for i, doc in enumerate(evidence_docs, start=1):
        src = doc.get("source") or doc.get("url_id") or f"doc_{i}"
        txt = doc.get("url_text", "")
        parts.extend(
            [
                f"Evidence Document {i}:",
                f"[source: {src}]",
                txt,
                "",
            ]
        )

    parts.extend(
        [
            "Return your answer as a valid JSON in the following format:",
            "",
            "{",
            '  "verdict": "<one of Supported, Refuted, Conflicting_Evidence, or Not_Enough_Evidence>",',
            '  "explanation": "<brief reasoning referencing the evidence>"',
            "}",
        ]
    )
    return "\n".join(parts)


def _format_evidence_for_template(evidence_docs: Sequence[Dict[str, Any]]) -> str:
    chunks: List[str] = []
    for i, doc in enumerate(evidence_docs, start=1):
        src = doc.get("source") or doc.get("url_id") or f"doc_{i}"
        txt = doc.get("url_text", "")
        chunks.append(f"Evidence Document {i}:\n[source: {src}]\n{txt}")
    return "\n\n".join(chunks)


def _render_prompt(claim_text: str, evidence_docs: Sequence[Dict[str, Any]]) -> Tuple[str, str]:
    if FORCE_FALLBACK_PROMPT:
        return DEFAULT_PROMPT_ID, _default_prompt_text(claim_text, evidence_docs)

    try:
        mod = importlib.import_module("ccir.prompts.prompts")
    except Exception:
        return DEFAULT_PROMPT_ID, _default_prompt_text(claim_text, evidence_docs)

    for name in (
        "render_small_llm_prompt",
        "build_small_llm_prompt",
        "make_small_llm_prompt",
        "render_prompt",
        "build_prompt",
        "make_prompt",
        "small_llm_prompt",
        "SMALL_LLM_PROMPT",
    ):
        if not hasattr(mod, name):
            continue

        fn = getattr(mod, name)

        try:
            if isinstance(fn, str):
                return DEFAULT_PROMPT_ID, fn.format(
                    claim_text=claim_text,
                    evidence=_format_evidence_for_template(evidence_docs),
                )

            sig = inspect.signature(fn)
            kwargs: Dict[str, Any] = {}

            if "claim_text" in sig.parameters:
                kwargs["claim_text"] = claim_text
            if "evidence_docs" in sig.parameters:
                kwargs["evidence_docs"] = list(evidence_docs)
            if "evidence" in sig.parameters:
                kwargs["evidence"] = list(evidence_docs)
            if "docs" in sig.parameters:
                kwargs["docs"] = list(evidence_docs)

            result = fn(**kwargs)

            if isinstance(result, tuple) and len(result) == 2:
                return str(result[0]), str(result[1])

            if isinstance(result, dict):
                pid = str(result.get("prompt_id", DEFAULT_PROMPT_ID))
                ptxt = str(result.get("prompt_text") or result.get("prompt") or "")
                if ptxt:
                    return pid, ptxt

            if isinstance(result, str) and result.strip():
                return DEFAULT_PROMPT_ID, result

        except Exception:
            continue

    return DEFAULT_PROMPT_ID, _default_prompt_text(claim_text, evidence_docs)


# -----------------------------------------------------------------------------
# Evidence loading
# -----------------------------------------------------------------------------

def _gold_doc_path(paths: Paths, claim_id: str, url_id: str) -> Path:
    return paths.gold_doc_path(claim_id, url_id)


def _corrupted_doc_path(paths: Paths, variant_name: str, claim_id: str, url_id: str) -> Path:
    return paths.corrupted_doc_path(variant_name, claim_id, url_id)


def _load_evidence_docs_for_condition(
    *,
    paths: Paths,
    claim_id: str,
    url_items: Sequence[Dict[str, Any]],
    variant_name: str,
) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []

    for item in url_items:
        uid = _url_id(item)
        if not uid:
            continue

        if variant_name == "gold":
            p = _gold_doc_path(paths, claim_id, uid)
        else:
            p = _corrupted_doc_path(paths, variant_name, claim_id, uid)

        if not p.exists():
            continue

        text = _read_text(p)
        if not text:
            continue

        docs.append(
            {
                "url_id": uid,
                "url_text": text,
                "source": _url_source(item),
                "path": str(p),
            }
        )

    return docs


def _available_variant_names(paths: Paths, config: Any, claim_id: str) -> List[str]:
    variant_names: List[str] = []

    corruption_cfg = getattr(config, "corruption", None)
    if corruption_cfg is not None and hasattr(corruption_cfg, "variant_names"):
        try:
            variant_names = list(corruption_cfg.variant_names())
        except Exception:
            variant_names = []

    if not variant_names and paths.cache_corrupted_dir.exists():
        variant_names = [p.name for p in sorted(paths.cache_corrupted_dir.iterdir()) if p.is_dir()]

    out: List[str] = []
    for name in variant_names:
        claim_dir = paths.cache_corrupted_dir / name / claim_id
        if claim_dir.exists() and claim_dir.is_dir():
            out.append(name)
    return out


# -----------------------------------------------------------------------------
# Prompt row construction
# -----------------------------------------------------------------------------

def _build_prompt_row(
    *,
    claim_id: str,
    claim_text: str,
    lang: Optional[str],
    claim_date: Optional[str],
    evidence_docs: Sequence[Dict[str, Any]],
    variant_name: str,
    run_id: str,
    code_version: str,
) -> Dict[str, Any]:
    prompt_id, prompt_text = _render_prompt(claim_text, evidence_docs)

    doc_rows = [
        {
            "url_id": d.get("url_id"),
            "URL_ID": d.get("url_id"),
            "url_text": d.get("url_text"),
            "URL_text": d.get("url_text"),
            "source": d.get("source"),
        }
        for d in evidence_docs
    ]

    payload: Dict[str, Any] = {
        "claim_id": claim_id,
        "claim_text": claim_text,
        "lang": lang,
        "claim_date": claim_date,
        "variant_name": variant_name,
        "condition": variant_name,
        "urls": doc_rows,
        "evidence": doc_rows,
        "documents": doc_rows,
        "prompt_id": prompt_id,
        "prompt_text": prompt_text,
        "prompt": prompt_text,
        "run_id": run_id,
        "created_utc": utc_now_iso(),
        "code_version": code_version,
    }
    return _make_schema_row(SmallLLMprompt, payload, SMALL_PROMPT_FIELDS)


def _build_all_prompt_rows(
    *,
    paths: Paths,
    config: Any,
    code_version: str,
    log: Any = None,
) -> List[Dict[str, Any]]:
    claims_rows = read_jsonl(paths.claims_for_llms)
    topk_rows = read_jsonl(paths.evidence_topk_urls)

    claim_map: Dict[str, Dict[str, Any]] = {_claim_id(r): r for r in claims_rows}
    topk_map: Dict[str, Dict[str, Any]] = {_claim_id(r): r for r in topk_rows}

    prompt_rows: List[Dict[str, Any]] = []

    claim_items = list(claim_map.items())
    claim_iter = _maybe_tqdm(
        claim_items,
        desc="Building prompts",
        unit="claim",
        leave=True,
        dynamic_ncols=True,
        ascii=True,
    )

    for claim_id, claim_row in claim_iter:
        claim_text = _claim_text(claim_row)
        if not claim_text:
            _log_count(log, "claims_missing_text")
            continue

        lang = _claim_lang(claim_row)
        claim_date = _claim_date(claim_row)

        topk_row = topk_map.get(claim_id)
        if not topk_row:
            _log_count(log, "claims_missing_topk")
            continue

        url_items = _extract_topk_urls(topk_row)
        if not url_items:
            _log_count(log, "claims_empty_topk")
            continue

        gold_docs = _load_evidence_docs_for_condition(
            paths=paths,
            claim_id=claim_id,
            url_items=url_items,
            variant_name="gold",
        )
        if gold_docs:
            prompt_rows.append(
                _build_prompt_row(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    lang=lang,
                    claim_date=claim_date,
                    evidence_docs=gold_docs,
                    variant_name="gold",
                    run_id=paths.run_id,
                    code_version=code_version,
                )
            )
            _log_count(log, "gold_prompts_built")
        else:
            _log_count(log, "gold_prompts_missing_evidence")

        for variant_name in _available_variant_names(paths, config, claim_id):
            docs = _load_evidence_docs_for_condition(
                paths=paths,
                claim_id=claim_id,
                url_items=url_items,
                variant_name=variant_name,
            )
            if not docs:
                _log_count(log, "variant_prompts_missing_evidence")
                continue

            prompt_rows.append(
                _build_prompt_row(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    lang=lang,
                    claim_date=claim_date,
                    evidence_docs=docs,
                    variant_name=variant_name,
                    run_id=paths.run_id,
                    code_version=code_version,
                )
            )
            _log_count(log, "corrupted_prompts_built")

        if tqdm is not None and hasattr(claim_iter, "set_postfix"):
            claim_iter.set_postfix({"prompts": len(prompt_rows)})

    return prompt_rows


# -----------------------------------------------------------------------------
# Model runner integration
# -----------------------------------------------------------------------------

def _call_runner(model_spec: Any, prompt_text: str, *, claim_id: str, variant_name: str) -> Dict[str, Any]:
    mod = importlib.import_module("ccir.models.runner")
    last_exc: Optional[BaseException] = None

    for name in ("run_small_llm", "run_model", "run_prompt", "generate", "infer", "run"):
        if not hasattr(mod, name):
            continue

        fn = getattr(mod, name)
        try:
            sig = inspect.signature(fn)
            kwargs: Dict[str, Any] = {}

            if "model_spec" in sig.parameters:
                kwargs["model_spec"] = model_spec
            if "model" in sig.parameters and "model_spec" not in kwargs:
                kwargs["model"] = model_spec
            if "model_name" in sig.parameters:
                kwargs["model_name"] = getattr(model_spec, "name", None) or getattr(model_spec, "model_id", None)
            if "provider" in sig.parameters:
                provider = getattr(model_spec, "provider", None)
                kwargs["provider"] = provider.value if hasattr(provider, "value") else provider

            if "prompt_text" in sig.parameters:
                kwargs["prompt_text"] = prompt_text
            elif "prompt" in sig.parameters:
                kwargs["prompt"] = prompt_text
            elif "text" in sig.parameters:
                kwargs["text"] = prompt_text
            elif "input_text" in sig.parameters:
                kwargs["input_text"] = prompt_text

            if "temperature" in sig.parameters:
                kwargs["temperature"] = getattr(model_spec, "temperature", 0.0)
            if "top_p" in sig.parameters:
                kwargs["top_p"] = getattr(model_spec, "top_p", 1.0)
            if "max_tokens" in sig.parameters:
                kwargs["max_tokens"] = getattr(model_spec, "max_tokens", 512)
            if "expect_json" in sig.parameters:
                kwargs["expect_json"] = True
            if "claim_id" in sig.parameters:
                kwargs["claim_id"] = claim_id
            if "variant_name" in sig.parameters:
                kwargs["variant_name"] = variant_name

            result = fn(**kwargs)

            if isinstance(result, dict):
                raw_text = str(result.get("raw_text") or result.get("text") or result.get("output") or "")
                parsed = result.get("parsed")
                if not isinstance(parsed, dict):
                    parsed = _extract_json_object(raw_text)
                return {"raw_text": raw_text, "parsed": parsed}

            if isinstance(result, str):
                return {"raw_text": result, "parsed": _extract_json_object(result)}

        except Exception as exc:
            last_exc = exc
            continue

    if last_exc is not None:
        raise RuntimeError(
            f"Runner call failed for model_id={getattr(model_spec, 'model_id', 'unknown')!r}: {type(last_exc).__name__}: {last_exc}"
        ) from last_exc

    raise RuntimeError(
        f"Could not find a compatible callable in ccir.models.runner for model_id={getattr(model_spec, 'model_id', 'unknown')!r}"
    )


# -----------------------------------------------------------------------------
# Output row construction
# -----------------------------------------------------------------------------

def _prompt_row_docs(prompt_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_docs = (
        prompt_row.get("evidence")
        or prompt_row.get("urls")
        or prompt_row.get("documents")
        or []
    )

    docs: List[Dict[str, Any]] = []
    for item in raw_docs:
        if not isinstance(item, dict):
            continue
        docs.append(
            {
                "url_id": item.get("url_id") or item.get("URL_ID"),
                "url_text": item.get("url_text") or item.get("URL_text") or "",
                "source": item.get("source") or item.get("url"),
            }
        )
    return docs


def _build_output_row(
    *,
    prompt_row: Dict[str, Any],
    model_spec: Any,
    runner_result: Dict[str, Any],
    run_id: str,
    code_version: str,
) -> Dict[str, Any]:
    parsed = runner_result.get("parsed") or {}
    raw_text = str(runner_result.get("raw_text") or "")

    verdict = _normalize_verdict(parsed.get("verdict"))
    explanation = str(parsed.get("explanation") or "").strip()

    if not explanation:
        explanation = raw_text.strip()

    payload: Dict[str, Any] = {
        "claim_id": prompt_row.get("claim_id"),
        "claim_text": prompt_row.get("claim_text"),
        "lang": prompt_row.get("lang"),
        "claim_date": prompt_row.get("claim_date"),
        "variant_name": prompt_row.get("variant_name") or prompt_row.get("condition"),
        "condition": prompt_row.get("variant_name") or prompt_row.get("condition"),
        "model_id": getattr(model_spec, "model_id", None),
        "model_name": getattr(model_spec, "name", None),
        "provider": getattr(getattr(model_spec, "provider", None), "value", getattr(model_spec, "provider", None)),
        "verdict": verdict,
        "explanation": explanation,
        "prompt_id": prompt_row.get("prompt_id", DEFAULT_PROMPT_ID),
        "raw_output": raw_text,
        "run_id": run_id,
        "created_utc": utc_now_iso(),
        "code_version": code_version,
    }
    return _make_schema_row(LLMOutput, payload, LLM_OUTPUT_FIELDS)


def _build_error_row(
    *,
    prompt_row: Dict[str, Any],
    model_spec: Any,
    exc: Exception,
    run_id: str,
    code_version: str,
) -> Dict[str, Any]:
    return {
        "claim_id": prompt_row.get("claim_id"),
        "claim_text": prompt_row.get("claim_text"),
        "lang": prompt_row.get("lang"),
        "claim_date": prompt_row.get("claim_date"),
        "variant_name": prompt_row.get("variant_name") or prompt_row.get("condition"),
        "condition": prompt_row.get("variant_name") or prompt_row.get("condition"),
        "model_id": getattr(model_spec, "model_id", None),
        "model_name": getattr(model_spec, "name", None),
        "provider": getattr(getattr(model_spec, "provider", None), "value", getattr(model_spec, "provider", None)),
        "prompt_id": prompt_row.get("prompt_id", DEFAULT_PROMPT_ID),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "run_id": run_id,
        "created_utc": utc_now_iso(),
        "code_version": code_version,
    }


# -----------------------------------------------------------------------------
# Public entrypoint used by __main__.py
# -----------------------------------------------------------------------------

def run_step09(
    *,
    paths: Paths,
    config: Any,
    run_id: Optional[str] = None,
    code_version: Optional[str] = None,
    log: Any = None,
    logger: Any = None,
    dry_run: bool = False,
    **_: Any,
) -> Dict[str, Any]:
    active_log = logger or log
    actual_run_id = run_id or paths.run_id
    actual_code_version = code_version or getattr(config, "code_version", None) or "dev"

    paths.ensure_shared_dirs()
    paths.ensure_run_dirs()

    prompt_rows = _build_all_prompt_rows(
        paths=paths,
        config=config,
        code_version=actual_code_version,
        log=active_log,
    )

    write_jsonl_atomic(paths.small_llm_prompts, prompt_rows)
    _log_info(
        active_log,
        "wrote small llm prompts",
        path=str(paths.small_llm_prompts),
        row_count=len(prompt_rows),
    )

    if dry_run:
        return {
            "prompt_count": len(prompt_rows),
            "prompts_path": str(paths.small_llm_prompts),
            "responses_written": 0,
            "errors_written": 0,
        }

    models = list(getattr(config, "models", []))
    if not models:
        raise ValueError("config.models is empty; step 09 requires at least one ModelSpec.")

    total_outputs = 0
    total_errors = 0
    written_files: List[str] = []
    error_files: List[str] = []

    for model_spec in models:
        model_id = getattr(model_spec, "model_id", None)
        if not model_id:
            raise ValueError(f"ModelSpec is missing model_id: {model_spec!r}")

        out_path = paths.small_llm_responses_jsonl(model_id)
        err_path = out_path.with_name(f"SmallLLMErrors{model_id}.jsonl")

        if out_path.exists():
            out_path.unlink()
        if err_path.exists():
            err_path.unlink()

        output_rows: List[Dict[str, Any]] = []
        error_rows: List[Dict[str, Any]] = []
        error_count = 0

        prompt_iter = _maybe_tqdm(
            prompt_rows,
            desc=f"Running {model_id}",
            unit="prompt",
            leave=True,
            dynamic_ncols=True,
            ascii=True,
        )

        for prompt_row in prompt_iter:
            prompt_text = str(prompt_row.get("prompt_text") or "")
            if not prompt_text:
                docs = _prompt_row_docs(prompt_row)
                _, prompt_text = _render_prompt(str(prompt_row.get("claim_text") or ""), docs)

            try:
                runner_result = _call_runner(
                    model_spec,
                    prompt_text,
                    claim_id=str(prompt_row.get("claim_id")),
                    variant_name=str(prompt_row.get("variant_name") or "unknown"),
                )
                out_row = _build_output_row(
                    prompt_row=prompt_row,
                    model_spec=model_spec,
                    runner_result=runner_result,
                    run_id=actual_run_id,
                    code_version=actual_code_version,
                )
                output_rows.append(out_row)
                append_jsonl(out_path, out_row)
                total_outputs += 1
                _log_count(active_log, f"responses_written_{model_id}")
            except Exception as exc:
                error_count += 1
                total_errors += 1
                _log_count(active_log, f"errors_{model_id}")
                _log_info(
                    active_log,
                    "model invocation failed",
                    model_id=model_id,
                    claim_id=str(prompt_row.get("claim_id")),
                    variant_name=str(prompt_row.get("variant_name")),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )

                err_row = _build_error_row(
                    prompt_row=prompt_row,
                    model_spec=model_spec,
                    exc=exc,
                    run_id=actual_run_id,
                    code_version=actual_code_version,
                )
                error_rows.append(err_row)
                append_jsonl(err_path, err_row)

            if tqdm is not None and hasattr(prompt_iter, "set_postfix"):
                prompt_iter.set_postfix(
                    {
                        "written": len(output_rows),
                        "errors": error_count,
                    }
                )

        written_files.append(str(out_path))
        error_files.append(str(err_path))

        _log_info(
            active_log,
            "wrote small llm responses",
            model_id=model_id,
            path=str(out_path),
            row_count=len(output_rows),
        )
        _log_info(
            active_log,
            "wrote small llm errors",
            model_id=model_id,
            path=str(err_path),
            row_count=len(error_rows),
        )

    return {
        "prompt_count": len(prompt_rows),
        "responses_written": total_outputs,
        "errors_written": total_errors,
        "written_files": written_files,
        "error_files": error_files,
    }


# -----------------------------------------------------------------------------
# Optional direct CLI use
# -----------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run step 09: build prompts and run models.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config()
    if hasattr(config, "validate") and callable(getattr(config, "validate")):
        config.validate()

    paths = Paths(run_id=args.run_id)
    run_step09(paths=paths, config=config, run_id=args.run_id, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())