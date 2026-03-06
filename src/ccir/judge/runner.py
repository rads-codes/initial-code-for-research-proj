from __future__ import annotations

"""
src/ccir/judge/runner.py

Robust judge runner for step 10.

Features
--------
- supports OpenRouter, OpenAI, and Ollama
- retries with backoff for transient API failures
- surfaces real error messages
- extracts JSON from messy judge outputs
- normalizes judge score fields into the expected schema
- compatible with scripts/step10_run_judge.py
"""

import json
import os
import random
import re
import time
from typing import Any, Dict, Optional

import requests


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OLLAMA_URL = "http://localhost:11434/api/generate"

REQUEST_TIMEOUT_S = 45
MAX_RETRIES = 2
BACKOFF_BASE_S = 1.5

_SCORE_KEYS = (
    "political_bias",
    "sociocultural_bias",
    "linguistic_bias",
    "logic_of_reasoning",
    "evidence_usage",
)


# -----------------------------------------------------------------------------
# Public entrypoints
# -----------------------------------------------------------------------------

def run_judge(
    *,
    judge_spec: Any,
    prompt_text: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 900,
    expect_json: bool = True,
    **_: Any,
) -> Dict[str, Any]:
    """
    Preferred entrypoint expected by step10.

    Returns:
      {
        "raw_text": <model text>,
        "parsed": <dict or {}>,
      }
    """
    provider = _provider_value(getattr(judge_spec, "provider", None))
    model_name = getattr(judge_spec, "name", None)

    if not model_name:
        raise RuntimeError("judge_spec.name is missing")

    if provider == "openrouter":
        raw_text = _call_openrouter(
            model_name=model_name,
            prompt_text=prompt_text,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    elif provider == "openai":
        raw_text = _call_openai(
            model_name=model_name,
            prompt_text=prompt_text,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    elif provider == "ollama":
        raw_text = _call_ollama(
            model_name=model_name,
            prompt_text=prompt_text,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    else:
        raise RuntimeError(
            f"Unsupported provider={provider!r} for judge={getattr(judge_spec, 'judge_id', None)!r}"
        )

    parsed = parse_judge_json(raw_text) if expect_json else {}
    return {
        "raw_text": raw_text,
        "parsed": parsed,
    }


# Compatibility aliases for step10 probing.
def run_model(*, judge_spec: Any = None, model_spec: Any = None, prompt_text: str, **kwargs: Any) -> Dict[str, Any]:
    spec = judge_spec if judge_spec is not None else model_spec
    return run_judge(judge_spec=spec, prompt_text=prompt_text, **kwargs)


def judge_one(*, judge_spec: Any = None, model_spec: Any = None, prompt_text: str, **kwargs: Any) -> Dict[str, Any]:
    spec = judge_spec if judge_spec is not None else model_spec
    return run_judge(judge_spec=spec, prompt_text=prompt_text, **kwargs)


def run(*, judge_spec: Any = None, model_spec: Any = None, prompt_text: str, **kwargs: Any) -> Dict[str, Any]:
    spec = judge_spec if judge_spec is not None else model_spec
    return run_judge(judge_spec=spec, prompt_text=prompt_text, **kwargs)


# -----------------------------------------------------------------------------
# JSON parsing / normalization
# -----------------------------------------------------------------------------

def parse_judge_json(raw_text: str) -> Dict[str, Any]:
    """
    Best-effort extraction of the expected judge schema:
      {
        "evidence_summary": str,
        "reasoning_analysis": str,
        "scores": {
          "political_bias": int,
          "sociocultural_bias": int,
          "linguistic_bias": int,
          "logic_of_reasoning": int,
          "evidence_usage": int
        },
        "overall_score": int,
        "brief_explanation": str
      }
    """
    obj = _extract_json_object(raw_text)
    return _normalize_judge_payload(obj, raw_text=raw_text)


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}

    # direct parse
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    # fenced json
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            obj = json.loads(fenced.group(1))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass

    # largest object substring
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end + 1]
        try:
            obj = json.loads(snippet)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass

    return {}


def _normalize_judge_payload(obj: Dict[str, Any], *, raw_text: str) -> Dict[str, Any]:
    scores_in = obj.get("scores")
    if not isinstance(scores_in, dict):
        scores_in = {}

    # Allow some fallback aliases if a model drifts slightly.
    alias_lookup = {
        "political_bias": ("political_bias", "political", "political_neutrality"),
        "sociocultural_bias": ("sociocultural_bias", "sociocultural", "cultural_bias", "social_bias"),
        "linguistic_bias": ("linguistic_bias", "linguistic", "language_bias"),
        "logic_of_reasoning": ("logic_of_reasoning", "logic", "reasoning", "logical_reasoning"),
        "evidence_usage": ("evidence_usage", "evidence", "use_of_evidence"),
    }

    scores: Dict[str, int] = {}
    for canonical_key, aliases in alias_lookup.items():
        raw_val = None
        for key in aliases:
            if key in scores_in:
                raw_val = scores_in.get(key)
                break
            if key in obj:
                raw_val = obj.get(key)
                break
        scores[canonical_key] = _coerce_score(raw_val)

    overall_score = obj.get("overall_score")
    try:
        overall_score = int(overall_score)
    except Exception:
        overall_score = sum(scores.values())

    evidence_summary = str(obj.get("evidence_summary") or "").strip()
    reasoning_analysis = str(obj.get("reasoning_analysis") or "").strip()
    brief_explanation = str(obj.get("brief_explanation") or "").strip()

    # Sensible fallback if the judge returned malformed but non-empty text.
    if not evidence_summary and not reasoning_analysis and not brief_explanation and raw_text.strip():
        brief_explanation = raw_text.strip()[:2000]

    return {
        "evidence_summary": evidence_summary,
        "reasoning_analysis": reasoning_analysis,
        "scores": scores,
        "overall_score": overall_score,
        "brief_explanation": brief_explanation,
    }


def _coerce_score(value: Any, default: int = 3) -> int:
    try:
        n = int(value)
    except Exception:
        try:
            n = int(float(value))
        except Exception:
            n = default
    return max(1, min(5, n))


# -----------------------------------------------------------------------------
# Provider calls
# -----------------------------------------------------------------------------

def _call_openrouter(
    *,
    model_name: str,
    prompt_text: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    site_url = os.getenv("OPENROUTER_SITE_URL")
    app_name = os.getenv("OPENROUTER_APP_NAME")
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt_text},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    data = _post_json_with_retries(
        url=OPENROUTER_URL,
        headers=headers,
        payload=payload,
        provider_name="OpenRouter",
    )

    try:
        return str(data["choices"][0]["message"]["content"])
    except Exception as exc:
        raise RuntimeError(f"Unexpected OpenRouter response shape: {json.dumps(data)[:1500]}") from exc


def _call_openai(
    *,
    model_name: str,
    prompt_text: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt_text},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    data = _post_json_with_retries(
        url=OPENAI_URL,
        headers=headers,
        payload=payload,
        provider_name="OpenAI",
    )

    try:
        return str(data["choices"][0]["message"]["content"])
    except Exception as exc:
        raise RuntimeError(f"Unexpected OpenAI response shape: {json.dumps(data)[:1500]}") from exc


def _call_ollama(
    *,
    model_name: str,
    prompt_text: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    payload = {
        "model": model_name,
        "prompt": prompt_text,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
        },
    }

    data = _post_json_with_retries(
        url=OLLAMA_URL,
        headers=None,
        payload=payload,
        provider_name="Ollama",
    )

    try:
        return str(data["response"])
    except Exception as exc:
        raise RuntimeError(f"Unexpected Ollama response shape: {json.dumps(data)[:1500]}") from exc


# -----------------------------------------------------------------------------
# HTTP / retry helpers
# -----------------------------------------------------------------------------

def _post_json_with_retries(
    *,
    url: str,
    headers: Optional[Dict[str, str]],
    payload: Dict[str, Any],
    provider_name: str,
) -> Dict[str, Any]:
    last_exc: Optional[BaseException] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT_S,
            )

            # slight throttle to reduce rate-limit bursts

            if 200 <= resp.status_code < 300:
                try:
                    return resp.json()
                except Exception as exc:
                    raise RuntimeError(
                        f"{provider_name} returned non-JSON response: {resp.text[:1500]}"
                    ) from exc

            body = _safe_body(resp)

            # Retryable statuses
            if resp.status_code in (408, 409, 425, 429, 500, 502, 503, 504):
                if attempt < MAX_RETRIES:
                    _sleep_with_backoff(attempt)
                    continue
                raise RuntimeError(
                    f"{provider_name} request failed after retries: status={resp.status_code}, body={body}"
                )

            # Non-retryable statuses
            raise RuntimeError(
                f"{provider_name} request failed: status={resp.status_code}, body={body}"
            )

        except requests.RequestException as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                _sleep_with_backoff(attempt)
                continue
            raise RuntimeError(
                f"{provider_name} network error after retries: {type(exc).__name__}: {exc}"
            ) from exc

    if last_exc is not None:
        raise RuntimeError(
            f"{provider_name} failed after retries: {type(last_exc).__name__}: {last_exc}"
        ) from last_exc

    raise RuntimeError(f"{provider_name} failed for unknown reason")


def _sleep_with_backoff(attempt: int) -> None:
    jitter = random.uniform(0.0, 0.75)
    #time.sleep(BACKOFF_BASE_S * attempt + jitter)


def _safe_body(resp: requests.Response) -> str:
    try:
        return resp.text[:1500]
    except Exception:
        return "<unavailable>"


def _provider_value(provider: Any) -> str:
    if provider is None:
        return ""
    if hasattr(provider, "value"):
        return str(provider.value)
    return str(provider)