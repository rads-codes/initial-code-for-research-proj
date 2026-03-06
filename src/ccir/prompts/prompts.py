from __future__ import annotations

"""
src/ccir/prompts/prompts.py

Prompt templates for step 09.

Responsibilities
----------------
- provide a stable prompt_id for step 09
- render the exact fact-check prompt shape described in the project outline
- stay import-safe (no I/O, no env reads)

Used by:
- scripts/step09_run_models.py
"""

from typing import Any, Dict, List, Sequence, Tuple


SMALL_LLM_PROMPT_ID = "small_factcheck_v1"

VALID_VERDICTS = (
    "Supported",
    "Refuted",
    "Conflicting_Evidence",
    "Not_Enough_Evidence",
)


def _doc_source(doc: Dict[str, Any], idx: int) -> str:
    for key in ("source", "url", "source_url", "canonical_url", "url_id", "URL_ID"):
        value = doc.get(key)
        if value:
            return str(value)
    return f"doc_{idx}"


def _doc_text(doc: Dict[str, Any]) -> str:
    for key in ("url_text", "URL_text", "text", "plaintext"):
        value = doc.get(key)
        if value is not None:
            return str(value).strip()
    return ""


def format_evidence_docs(evidence_docs: Sequence[Dict[str, Any]]) -> str:
    """
    Format evidence documents for insertion into the step-09 fact-check prompt.
    """
    chunks: List[str] = []
    for idx, doc in enumerate(evidence_docs, start=1):
        src = _doc_source(doc, idx)
        txt = _doc_text(doc)
        chunks.append(
            "\n".join(
                [
                    f"Evidence Document {idx}:",
                    f"[source: {src}]",
                    txt,
                ]
            )
        )
    return "\n\n".join(chunks)


def build_small_llm_prompt(claim_text: str, evidence_docs: Sequence[Dict[str, Any]]) -> str:
    """
    Render the exact step-09 prompt described in the project outline.
    """
    evidence_block = format_evidence_docs(evidence_docs)

    return (
        "Determine whether the given claim is supported or refuted based on only the evidence provided below.\n\n"
        "The possible verdict labels are Supported (when the evidence clearly supports the claim), "
        "Refuted (when the evidence clearly contradicts the claim), Conflicting_Evidence (when the evidence has both supporting "
        "and contradicting statements), or Not_Enough_Evidence (when the evidence doesn’t provide enough information to evaluate the claim).\n\n"
        "Read the claim, carefully review the evidence documents, decide which verdict label best applies, "
        "and briefly explain your reasoning using specific evidence quotes.\n\n"
        "Claim:\n"
        f"{claim_text.strip()}\n\n"
        f"{evidence_block}\n\n"
        "Return your answer as a valid JSON in the following format:\n\n"
        "{\n"
        '  "verdict": "<one of Supported, Refuted, Conflicting_Evidence, or Not_Enough_Evidence>",\n'
        '  "explanation": "<brief reasoning referencing the evidence>"\n'
        "}"
    )


def render_small_llm_prompt(claim_text: str, evidence_docs: Sequence[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Preferred interface for step09:
      returns (prompt_id, prompt_text)
    """
    return SMALL_LLM_PROMPT_ID, build_small_llm_prompt(claim_text, evidence_docs)


def make_small_llm_prompt(claim_text: str, evidence_docs: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    """
    Alternate interface if a caller wants a dict instead of a tuple.
    """
    prompt_text = build_small_llm_prompt(claim_text, evidence_docs)
    return {
        "prompt_id": SMALL_LLM_PROMPT_ID,
        "prompt_text": prompt_text,
    }