from __future__ import annotations

"""
src/ccir/models/runner.py

Robust model runner for step 09.

Features
--------
- supports OpenRouter, OpenAI, and Ollama
- retries with backoff for transient API failures
- surfaces real error messages
- extracts JSON from messy model outputs
- compatible with scripts/step09_run_models.py
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

REQUEST_TIMEOUT_S = 30
MAX_RETRIES = 1
BACKOFF_BASE_S = 1.0

VALID_VERDICTS = {
    "Supported",
    "Refuted",
    "Conflicting_Evidence",
    "Not_Enough_Evidence",
}


# -----------------------------------------------------------------------------
# Public entrypoints
# -----------------------------------------------------------------------------

def run_small_llm(
    *,
    model_spec: Any,
    prompt_text: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    expect_json: bool = True,
    **_: Any,
) -> Dict[str, Any]:
    """
    Preferred entrypoint expected by step09.

    Returns:
      {
        "raw_text": <model text>,
        "parsed": <dict or {}>,
      }
    """
    provider = _provider_value(getattr(model_spec, "provider", None))
    model_name = getattr(model_spec, "name", None)

    if not model_name:
        raise RuntimeError("model_spec.name is missing")

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
            f"Unsupported provider={provider!r} for model={getattr(model_spec, 'model_id', None)!r}"
        )

    parsed = parse_model_json(raw_text) if expect_json else {}
    return {
        "raw_text": raw_text,
        "parsed": parsed,
    }


# Compatibility aliases for step09 probing.
def run_model(*, model_spec: Any, prompt_text: str, **kwargs: Any) -> Dict[str, Any]:
    return run_small_llm(model_spec=model_spec, prompt_text=prompt_text, **kwargs)


def run_prompt(*, model_spec: Any, prompt_text: str, **kwargs: Any) -> Dict[str, Any]:
    return run_small_llm(model_spec=model_spec, prompt_text=prompt_text, **kwargs)


def generate(*, model_spec: Any, prompt_text: str, **kwargs: Any) -> Dict[str, Any]:
    return run_small_llm(model_spec=model_spec, prompt_text=prompt_text, **kwargs)


def infer(*, model_spec: Any, prompt_text: str, **kwargs: Any) -> Dict[str, Any]:
    return run_small_llm(model_spec=model_spec, prompt_text=prompt_text, **kwargs)


def run(*, model_spec: Any, prompt_text: str, **kwargs: Any) -> Dict[str, Any]:
    return run_small_llm(model_spec=model_spec, prompt_text=prompt_text, **kwargs)


# -----------------------------------------------------------------------------
# JSON parsing
# -----------------------------------------------------------------------------

def parse_model_json(raw_text: str) -> Dict[str, str]:
    """
    Best-effort extraction of:
      {"verdict": ..., "explanation": ...}
    """
    obj = _extract_json_object(raw_text)

    verdict = _normalize_verdict(obj.get("verdict"))
    explanation = str(obj.get("explanation") or "").strip()

    if not explanation:
        explanation = raw_text.strip()

    return {
        "verdict": verdict,
        "explanation": explanation,
    }


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
        snippet = text[start:end + 1]
        try:
            obj = json.loads(snippet)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass

    return {}


def _normalize_verdict(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "Not_Enough_Evidence"

    key = raw.lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "supported": "Supported",
        "support": "Supported",
        "refuted": "Refuted",
        "refute": "Refuted",
        "conflicting_evidence": "Conflicting_Evidence",
        "conflicting": "Conflicting_Evidence",
        "conflict": "Conflicting_Evidence",
        "not_enough_evidence": "Not_Enough_Evidence",
        "insufficient_evidence": "Not_Enough_Evidence",
        "nee": "Not_Enough_Evidence",
    }
    verdict = mapping.get(key, raw)
    if verdict not in VALID_VERDICTS:
        return "Not_Enough_Evidence"
    return verdict


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
            time.sleep(8)

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
    jitter = random.uniform(0.0, 0.5)
    time.sleep(BACKOFF_BASE_S * attempt + jitter)


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