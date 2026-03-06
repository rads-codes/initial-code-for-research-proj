from __future__ import annotations  #postponed evaluation of annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, Union

#all objects here should also have the lineage fields run_id, created_utc, and code_version in addition to what they have
'''
object AllClaimsFormat, contains claim_id, lang, claim_text, claim_date, rating
object LLMClaimRow, contains claim_id, lang, claim_text, claim_date
object ScoringClaimRow, contains claim_id, rating.
object ClaimWithURLs, contains claim_id with a list of up to K URLs with an ID for each URL
object topKURL, contains claim_id with {up to L URLs with each URL’s ID}
object sentenceID, contains claim_id, claim_text_embedding, {URL_ID, sentence_ID, sentence_embedding, … for each URL}
object sentenceIDEmbedding, contains claim_id, claim_text_embedding, {URL_ID, object sentence ID, sentence_embedding, and embedding_score,... for each URL}
object SmallLLMprompt, contains claim_id, claim_text, {URL_ID, URL_text,... for all URLs}
object LLMOutput, contains claim_id, claim_text, verdict, explanation
object JudgeLLMprompt, contains claim_id, claim_text, {URL_ID, URL_text,... for all URLs}, verdict, explanation
object JudgeLLMResponses, contains evidence_summary, reasoning_analysis, scores (political_bias, sociocultural_bias, linguistic_bias, logic_of_reasoning, evidence_usage under scores), overall_score, brief_explanation
object MetadataEvents, contains event (run_start, step_start, step_end, step_error, run_end), created_utc, step (for step events), config_hash (on run_start) if there, code_version (manual version string, optional), inputs [] if there, outputs [] if there, counts{} if there or metrics{} if there, error{type, message, trace} if any
'''

"""
src/ccir/schemas.py

defines JSONL row schemas, every JSON object written to a JSONL file must match one of these dataclasses
all objects/rows include lineage fields: run_id, created_utc, code_version.
(these are used for reproducibility and for joining artifacts across runs.)
"""

# -----------------------------
# Shared helpers / conventions
# -----------------------------

T = TypeVar("T")


def utc_now_iso() -> str:
    """
    Return a UTC timestamp string in a single canonical format.

    Canonical format used in this codebase:
      YYYY-MM-DDTHH:MM:SSZ
    Example:
      2026-03-04T19:34:12Z
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _require_keys(d: Dict[str, Any], keys: List[str], context: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"{context}: missing required keys: {missing}")


def _is_list_of_floats(x: Any) -> bool:
    return isinstance(x, list) and all(isinstance(v, (int, float)) for v in x)


# Embeddings are stored as JSON arrays (list[float]).
Embedding = List[float]


# -----------------------------
# Lineage base (required)
# -----------------------------

@dataclass(frozen=True)
class Lineage:
    """
    Required lineage fields for *every* JSONL row.

    - run_id: unique identifier for a run. Used as the root folder name under
      data/processed/runs/<run_id>/...
    - created_utc: when this row was created (UTC ISO string).
    - code_version: a manual version string for code state (you set this).
    """
    run_id: str
    created_utc: str
    code_version: str


# -----------------------------
# Claims schemas (00-02)
# -----------------------------

@dataclass(frozen=True)
class AllClaimsFormat(Lineage):
    """
    data/processed/claims/all.jsonl
    One row per claim after normalizing EuroVerdict.

    Fields required by outline:
    - claim_id, lang, claim_text, claim_date, rating
    """
    claim_id: str
    lang: str
    claim_text: str
    claim_date: str  # normalized date string; keep consistent across pipeline
    rating: str      # mapped to AVeriTeC-style labels in step 00


@dataclass(frozen=True)
class LLMClaimRow(Lineage):
    """
    data/processed/claims/forLLMs.jsonl
    Subset of claims to run through model pipelines.
    """
    claim_id: str
    lang: str
    claim_text: str
    claim_date: str


@dataclass(frozen=True)
class ScoringClaimRow(Lineage):
    """
    data/processed/claims/forScoring.jsonl
    Ground truth rating for claims used in scoring/results.
    """
    claim_id: str
    rating: str


# -----------------------------
# Evidence URL schemas (03-06)
# -----------------------------

@dataclass(frozen=True)
class URLItem:
    """
    A single URL candidate for a claim.

    ID rule (implemented elsewhere, documented here):
      url_id = "u_" + sha256(canonical_url).hexdigest()[:16]
    """
    url_id: str
    url: str

    # Optional metadata captured from search provider (e.g., SerpAPI)
    title: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None  # e.g., publisher domain
    rank: Optional[int] = None    # search rank (1..K), if you keep it


@dataclass(frozen=True)
class ClaimWithURLs(Lineage):
    """
    data/processed/evidence/URLs.jsonl
    One row per claim with up to K URL candidates.
    """
    claim_id: str
    urls: List[URLItem] = field(default_factory=list)


@dataclass(frozen=True)
class TopKURL(Lineage):
    """
    data/processed/evidence/rankings/topKURLs.jsonl
    One row per claim with the chosen top L URLs after ranking (BM25, etc).
    """
    claim_id: str
    top_urls: List[URLItem] = field(default_factory=list)


# -----------------------------
# Sentence + embedding schemas (07)
# -----------------------------

@dataclass(frozen=True)
class SentenceID(Lineage):
    """
    Sentence embeddings row (sentences.jsonl in your outline).

    This is a *flat* row: one row per (claim_id, url_id, sentence_id).

    The outline says to store:
      claim_id, claim_text_embedding, url_id, sentence_id, sentence_embedding

    Sentence ID rule (implemented elsewhere, documented here):
      sentence_id = f"s{i:04d}_{sha256(sentence_text_norm).hexdigest()[:8]}"
    """
    claim_id: str
    url_id: str
    sentence_id: str

    claim_text_embedding: Embedding
    sentence_embedding: Embedding

    # Helpful to keep for debugging; optional to store to reduce size.
    sentence_text: Optional[str] = None


@dataclass(frozen=True)
class SentenceIDEmbedding(Lineage):
    """
    Sentence similarity/scoring row (embeddings.jsonl in your outline).

    One row per (claim_id, url_id, sentence_id) with an embedding score.
    """
    claim_id: str
    url_id: str
    sentence_id: str

    claim_text_embedding: Embedding
    sentence_embedding: Embedding
    embedding_score: float  # cosine similarity


# -----------------------------
# Prompt schemas (09-10)
# -----------------------------

@dataclass(frozen=True)
class EvidenceText:
    """
    Evidence text to include inside an LLM prompt.

    url_text is the cleaned plaintext content for that URL (possibly truncated).
    """
    url_id: str
    url_text: str


@dataclass(frozen=True)
class SmallLLMprompt(Lineage):
    """
    data/processed/LLMprompts/SmallLLMPrompts.jsonl

    One row per claim prompt, including evidence texts for selected URLs.
    """
    claim_id: str
    claim_text: str
    evidence: List[EvidenceText] = field(default_factory=list)

    # Optional prompt metadata (useful when you iterate prompts):
    prompt_id: Optional[str] = None
    variant_name: Optional[str] = None  # e.g., "gold", "mild_20_drop", etc.


@dataclass(frozen=True)
class JudgeLLMprompt(Lineage):
    """
    data/processed/LLMprompts/JudgeLLMPrompts.jsonl

    One row per claim prompt for the judge, including:
    - claim text
    - evidence texts (same as model saw, for that condition)
    - model verdict + explanation being judged
    """
    claim_id: str
    claim_text: str
    evidence: List[EvidenceText] = field(default_factory=list)

    verdict: str = ""        # model output label
    explanation: str = ""    # model output explanation

    prompt_id: Optional[str] = None
    variant_name: Optional[str] = None
    model_name: Optional[str] = None


# -----------------------------
# Model output schemas (09)
# -----------------------------

@dataclass(frozen=True)
class LLMOutput(Lineage):
    """
    data/processed/runs/<run_id>/smallLLMResponses/SmallLLMResponses<model>.jsonl

    One row per claim response from a small LLM.
    """
    claim_id: str
    claim_text: str
    verdict: str
    explanation: str

    # Optional join keys / debugging aids
    lang: Optional[str] = None
    claim_date: Optional[str] = None
    model_name: Optional[str] = None
    variant_name: Optional[str] = None
    prompt_id: Optional[str] = None


# -----------------------------
# Judge output schemas (10)
# -----------------------------

@dataclass(frozen=True)
class JudgeScores:
    """
    Subscores for judge rubric.

    Required keys listed in your spec:
      political_bias, sociocultural_bias, linguistic_bias,
      logic_of_reasoning, evidence_usage
    """
    political_bias: float
    sociocultural_bias: float
    linguistic_bias: float
    logic_of_reasoning: float
    evidence_usage: float


@dataclass(frozen=True)
class JudgeLLMResponses(Lineage):
    """
    data/processed/runs/<run_id>/LLMJudgeResponses/JudgeLLMResponses<model>.jsonl

    Your list specifies the *content* fields:
      evidence_summary, reasoning_analysis, scores{...}, overall_score, brief_explanation

    Practical assumption for joinability:
    - include claim_id (and optional model/variant) in the row too.
    """
    claim_id: str

    evidence_summary: str
    reasoning_analysis: str
    scores: JudgeScores
    overall_score: float
    brief_explanation: str

    # Optional join keys
    judge_model_name: Optional[str] = None
    model_name: Optional[str] = None
    variant_name: Optional[str] = None
    prompt_id: Optional[str] = None


# -----------------------------
# Run metadata schema (run_metadata.jsonl)
# -----------------------------

MetadataEventType = Literal["run_start", "step_start", "step_end", "step_error", "run_end"]


@dataclass(frozen=True)
class MetadataError:
    """
    Structured error information for step_error events.
    """
    type: str
    message: str
    trace: str


@dataclass(frozen=True)
class MetadataEvents(Lineage):
    """
    data/processed/runs/<run_id>/run_metadata.jsonl
    One row per event, written by src/ccir/__main__.py via utils/run_metadata.py.

    Fields required/mentioned by outline:
    - event: run_start | step_start | step_end | step_error | run_end
    - created_utc: (already in Lineage; keep consistent)
    - step: for step events
    - config_hash: on run_start (if there)
    - code_version: in Lineage (manual string; you set it)
    - inputs: list of touched input paths (if there)
    - outputs: list of touched output paths (if there)
    - counts or metrics (if there)
    - error: {type,message,trace} (if any)
    """
    event: MetadataEventType

    # Only meaningful for step_* events. Keep None for run_start/run_end.
    step: Optional[str] = None

    # Only meaningful on run_start (but harmless to include elsewhere if you want).
    config_hash: Optional[str] = None

    # Paths are stored as strings (relative or absolute, but pick one convention).
    inputs: Optional[List[str]] = None
    outputs: Optional[List[str]] = None

    # "counts" for integer counters; "metrics" for float-ish metrics.
    counts: Optional[Dict[str, int]] = None
    metrics: Optional[Dict[str, float]] = None

    # Optional structured error for step_error.
    error: Optional[MetadataError] = None


# -----------------------------
# Serialization helpers
# -----------------------------

def to_dict(obj: Any) -> Dict[str, Any]:
    """
    Convert a dataclass to a plain JSON-serializable dict.

    Notes:
    - Dataclasses + nested dataclasses are handled by asdict().
    - Embeddings must already be list[float].
    """
    return asdict(obj)


def from_dict(cls: Type[T], d: Dict[str, Any]) -> T:
    """
    Construct a dataclass from a dict with minimal validation.

    This is intended for validation.py to call when checking JSONL row shapes.
    """
    if not isinstance(d, dict):
        raise ValueError(f"{cls.__name__}: expected dict, got {type(d)}")

    # Every schema in this file ultimately includes Lineage, except nested helpers.
    lineage_required = {"run_id", "created_utc", "code_version"}
    if issubclass(cls, Lineage) or cls is MetadataEvents:
        missing = [k for k in lineage_required if k not in d]
        if missing:
            raise ValueError(f"{cls.__name__}: missing lineage keys: {missing}")

    # Lightweight embedding checks for the embedding-bearing schemas
    if cls in (SentenceID, SentenceIDEmbedding):
        if not _is_list_of_floats(d.get("claim_text_embedding")):
            raise ValueError(f"{cls.__name__}: claim_text_embedding must be list[float]")
        if not _is_list_of_floats(d.get("sentence_embedding")):
            raise ValueError(f"{cls.__name__}: sentence_embedding must be list[float]")

    # Handle nested dataclasses that won't auto-construct from plain dicts:
    if cls is ClaimWithURLs:
        urls = [URLItem(**u) for u in d.get("urls", [])]
        return ClaimWithURLs(
            run_id=d["run_id"],
            created_utc=d["created_utc"],
            code_version=d["code_version"],
            claim_id=d["claim_id"],
            urls=urls,
        )  # type: ignore[return-value]

    if cls is TopKURL:
        _require_keys(d, ["run_id", "created_utc", "code_version", "claim_id"], "TopKURL")
        top_urls = [URLItem(**u) for u in d.get("top_urls", [])]
        return TopKURL(
            run_id=d["run_id"],
            created_utc=d["created_utc"],
            code_version=d["code_version"],
            claim_id=d["claim_id"],
            top_urls=top_urls,
        )  # type: ignore[return-value]

    if cls is SmallLLMprompt:
        evidence = [EvidenceText(**e) for e in d.get("evidence", [])]
        return SmallLLMprompt(
            run_id=d["run_id"],
            created_utc=d["created_utc"],
            code_version=d["code_version"],
            claim_id=d["claim_id"],
            claim_text=d["claim_text"],
            evidence=evidence,
            prompt_id=d.get("prompt_id"),
            variant_name=d.get("variant_name"),
        )  # type: ignore[return-value]

    if cls is JudgeLLMprompt:
        evidence = [EvidenceText(**e) for e in d.get("evidence", [])]
        return JudgeLLMprompt(
            run_id=d["run_id"],
            created_utc=d["created_utc"],
            code_version=d["code_version"],
            claim_id=d["claim_id"],
            claim_text=d["claim_text"],
            evidence=evidence,
            verdict=d.get("verdict", ""),
            explanation=d.get("explanation", ""),
            prompt_id=d.get("prompt_id"),
            variant_name=d.get("variant_name"),
            model_name=d.get("model_name"),
        )  # type: ignore[return-value]

    if cls is JudgeLLMResponses:
        scores_dict = d.get("scores")
        if not isinstance(scores_dict, dict):
            raise ValueError("JudgeLLMResponses: scores must be an object/dict")
        _require_keys(
            scores_dict,
            ["political_bias", "sociocultural_bias", "linguistic_bias", "logic_of_reasoning", "evidence_usage"],
            "JudgeScores",
        )
        scores = JudgeScores(**scores_dict)

        base_kwargs = dict(d)
        base_kwargs["scores"] = scores
        return cls(**base_kwargs)  # type: ignore[arg-type,return-value]

    if cls is MetadataEvents:
        err = d.get("error")
        error_obj = MetadataError(**err) if isinstance(err, dict) else None

        base_kwargs = dict(d)
        base_kwargs["error"] = error_obj
        return cls(**base_kwargs)  # type: ignore[arg-type,return-value]

    # Default: assumes no nested dataclasses in fields.
    return cls(**d)  # type: ignore[arg-type,return-value]