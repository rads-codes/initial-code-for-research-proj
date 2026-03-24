from __future__ import annotations

"""
src/ccir/corruption/misleading_edit.py

Implements the 'misleading_edit' corruption strategy: injects small, plausible
but incorrect factual changes into highly relevant sentences of gold documents.

Supported entity types:
  DATE    – year/date perturbed ±1-3 years (format-preserving)
  NUMBER  – numeric value perturbed by small amount (format-preserving)
  PERCENT – percentage perturbed ±1-5 points (format-preserving)
  PERSON  – person name swapped from per-language pool
  ORG     – organisation name swapped from per-language pool
  GPE     – geopolitical entity (city / country) swapped from per-language pool
  LOC     – location swapped from per-language pool

Pipeline (per document):
  1. Split into sentences; use existing embedding_score from step 07.
  2. For each sentence extract entity spans via NER + regex.
  3. For each span generate type-matched replacement candidates:
       DATE/NUMBER/PERCENT  →  numeric perturbation
       PERSON/ORG/GPE/LOC   →  sample from per-language replacement pool
  4. Rank pool-based replacements by embedding similarity to original span;
     sample from top-K (default 5).
  5. Validate each candidate: differs from original, exactly one span changed,
     sentence stays well-formed, format preserved, type-consistent.
  6. Rank all valid candidates by sentence relevance score (descending);
     select top K% of sentences with at most one edit per sentence.
  7. Apply selected edits; return corrupted text and edit-log rows.

NER strategy (tiered, graceful degradation):
  Primary  – transformers.pipeline("ner", aggregation_strategy="simple")
             Model: env CCIR_NER_MODEL or 'Babelscape/wikineural-multilingual-ner'
  Fallback – regex heuristic: capitalized-phrase detection for PERSON/ORG/GPE
             (always available, language-agnostic, slightly noisier)

Pool building:
  Run NER over all gold doc sentences present in paths.cache_gold_dir and all
  cached plaintext under paths.cache_plaintext_dir.
  Collect PERSON, ORG, GPE, LOC entities; group by claim language.
"""

import os
import re
import random
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENTITY_TYPES_NUMERIC = {"DATE", "NUMBER", "PERCENT"}
ENTITY_TYPES_NAMED = {"PERSON", "ORG", "GPE", "LOC"}
ALL_ENTITY_TYPES = ENTITY_TYPES_NUMERIC | ENTITY_TYPES_NAMED

DEFAULT_TOP_K = 5

# Maximum pool size to embed when ranking named-entity candidates.
# Randomly subsample before batch-embedding to keep cost bounded.
# embed_texts(21 texts) ≈ 50 ms vs embed_texts(101 texts) ≈ 1300 ms on CPU.
_MAX_EMBED_CANDIDATES = 20

# Maximum number of sentences per document to run NER on.
# Sentences are selected in descending relevance-score order so the most
# relevant content is always covered.  Remaining sentences fall back to the
# fast regex+heuristic path.  This caps the NER batch size regardless of
# document length.
_MAX_NER_SENTENCES = 15

# Ranking priority for edit type: lower value = higher priority.
# DATE/NUMBER/PERCENT edits are most reliable (purely numeric, format-preserving);
# GPE/PERSON/LOC pool-based edits are next; ORG names are more ambiguous.
_TYPE_PRIORITY: Dict[str, int] = {
    "DATE": 0, "NUMBER": 0, "PERCENT": 0,
    "GPE": 1, "PERSON": 1, "LOC": 1,
    "ORG": 2,
}

# Key-term type priority used when extracting and ranking corruption targets.
# Lower value = higher priority; named entities are always preferred over numeric spans.
_KEY_TERM_TYPE_PRIORITY: Dict[str, int] = {
    "PERSON": 0, "GPE": 1, "LOC": 1, "ORG": 2,
    "DATE": 3, "PERCENT": 4, "NUMBER": 5,
}

# Structural scope / qualifier flips applied only at the stronger (pct≥50) level.
# Each entry: (regex_pattern, replacement_text, category_label).
# Matched case-insensitively; at most ONE flip is applied per document.
_STRUCTURAL_FLIPS: List[Tuple[str, str, str]] = [
    (r"\bdistrict(?:[-\s]wide)?\b",
     "statewide", "geographic_scope"),
    (r"\blocal\s+(?:law|order|regulation|ordinance|rule)\b",
     "state law", "geographic_scope"),
    (r"\bonly\s+with\s+(?:written\s+)?(?:consent|permission)\b",
     "without consent", "permission"),
    (r"\b(?:written\s+)?(?:consent|permission|authorization)\s+(?:is\s+)?required\b",
     "no authorization required", "permission"),
    (r"\bprohibited\b",  "permitted",  "policy_flip"),
    (r"\billegal\b",     "legal",      "policy_flip"),
    (r"\bbanned\b",      "allowed",    "policy_flip"),
    (r"\bmandatory\b",   "voluntary",  "policy_flip"),
    (r"\bnationwide\b",  "locally",    "geographic_scope"),
    (r"\bfederally\b",   "locally",    "geographic_scope"),
]

_NER_MODEL_ENV = "CCIR_NER_MODEL"
_DEFAULT_NER_MODEL = "Babelscape/wikineural-multilingual-ner"

# Entity types for which embedding-based nearest-neighbour pool replacement is used.
# DATE, NUMBER, PERCENT remain purely rule-based; LOC uses random selection.
ENTITY_TYPES_NN_POOL: Set[str] = {"PERSON", "GPE", "ORG"}

# Maximum pool entries per (language, entity_type) to precompute embeddings for.
# Bounds build-time embedding cost while still covering the most frequent entities.
_MAX_POOL_EMBED: int = 500

# HuggingFace NER label → our canonical type
_HF_LABEL_MAP: Dict[str, str] = {
    "PER": "PERSON", "PERSON": "PERSON",
    "ORG": "ORG",
    "LOC": "LOC", "GPE": "GPE",
    "MISC": "",          # discard
    "DATE": "DATE",
    "TIME": "",
    "PERCENT": "PERCENT",
    "MONEY": "NUMBER",
    "CARDINAL": "NUMBER",
    "ORDINAL": "",
    "QUANTITY": "NUMBER",
}

# Capitalized word / phrase heuristic – skip common function words
_STOP_WORDS: Set[str] = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "shall",
    "this", "that", "these", "those", "it", "its", "he", "she", "they",
    "we", "you", "i", "me", "him", "her", "us", "them",
    # German common words
    "der", "die", "das", "des", "dem", "den", "ein", "eine", "eines",
    "einer", "einem", "einen", "und", "oder", "ist", "sind", "war",
    "waren", "hat", "haben", "hatte", "hatten", "wird", "werden",
    "wurde", "wurden", "auch", "nicht", "mit", "als", "bei", "nach",
    "aus", "für", "auf", "von", "zu", "im", "in", "an",
    # Romanian
    "și", "sau", "că", "se", "la", "de", "în", "cu", "pe",
    "pentru", "este", "sunt", "era", "au", "o", "un", "unei",
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EntitySpan:
    text: str
    start: int
    end: int
    entity_type: str  # DATE, NUMBER, PERCENT, PERSON, ORG, GPE, LOC


@dataclass
class EditCandidate:
    sentence_idx: int
    original_sentence: str
    edited_sentence: str
    edit_type: str
    old_span: str
    new_span: str
    relevance_score: float


@dataclass
class KeyTerm:
    """A deduplicated key term extracted from a document for corruption targeting."""
    text: str                    # canonical surface text (first occurrence wins)
    normalized: str              # lowercased, punctuation-stripped form for dedup
    entity_type: str             # PERSON, GPE, LOC, ORG, DATE, NUMBER, PERCENT
    occurrences: List[int]       # sorted sentence indices where this term appears
    max_relevance_score: float   # max embedding_score among containing sentences


@dataclass
class PoolEntry:
    """A single entity replacement candidate stored in a per-language, per-type pool."""
    text: str
    entity_type: str        # PERSON, GPE, ORG, LOC
    language: str
    descriptor: str         # context-aware string used for precomputed embedding
    embedding: List[float]  # precomputed; empty list when unavailable
    granularity: str = ""   # GPE: "country" / "state" / "city" / "unknown"
    role_hint: str = ""     # PERSON: "politician" / "journalist" / "executive" / "unknown"
    org_subtype: str = ""   # ORG: "media" / "government" / "business" / "unknown"


# ---------------------------------------------------------------------------
# NER pipeline (lazy-loaded, optional)
# ---------------------------------------------------------------------------

_ner_pipeline_cache: Dict[str, Optional[Any]] = {}


def _load_ner_pipeline() -> Optional[Any]:
    """
    Attempt to load a multilingual NER pipeline once; cache the result.
    Returns None on any failure (graceful degradation).
    """
    cache_key = "__pipeline__"
    if cache_key in _ner_pipeline_cache:
        return _ner_pipeline_cache[cache_key]

    model_name = os.getenv(_NER_MODEL_ENV, _DEFAULT_NER_MODEL)
    try:
        from transformers import pipeline as hf_pipeline
        pipe = hf_pipeline(
            "ner",
            model=model_name,
            aggregation_strategy="simple",
            device=-1,  # CPU
        )
        _ner_pipeline_cache[cache_key] = pipe
        return pipe
    except Exception:
        _ner_pipeline_cache[cache_key] = None
        return None


def _ner_spans(sentence: str) -> List[EntitySpan]:
    """
    Extract named-entity spans using the transformers NER pipeline.
    Returns empty list if pipeline unavailable.
    """
    pipe = _load_ner_pipeline()
    if pipe is None:
        return []

    try:
        results = pipe(sentence)
    except Exception:
        return []

    return _parse_ner_results(sentence, results)


def _parse_ner_results(sentence: str, results: List[Dict[str, Any]]) -> List[EntitySpan]:
    """Convert raw HuggingFace NER results for one sentence into EntitySpan objects."""
    spans: List[EntitySpan] = []
    for ent in results:
        raw_label: str = ent.get("entity_group", ent.get("entity", ""))
        canonical = _HF_LABEL_MAP.get(raw_label.upper(), "")
        if not canonical or canonical not in ALL_ENTITY_TYPES:
            continue
        word = ent.get("word", "").strip()
        # strip subword artifacts
        word = re.sub(r"^##", "", word).strip()
        if not word:
            continue
        start = ent.get("start", 0)
        end = ent.get("end", start + len(word))
        # Align span text to actual sentence slice
        span_text = sentence[start:end].strip()
        if not span_text:
            span_text = word
        spans.append(EntitySpan(text=span_text, start=start, end=end, entity_type=canonical))
    return spans


def _ner_spans_batch(sentences: List[str]) -> List[List[EntitySpan]]:
    """
    Run NER on a list of sentences in a single batched pipeline call.
    Returns one List[EntitySpan] per input sentence.
    Falls back to empty lists per sentence if pipeline unavailable.
    """
    if not sentences:
        return []

    pipe = _load_ner_pipeline()
    if pipe is None:
        return [[] for _ in sentences]

    try:
        batch_results = pipe(sentences)
    except Exception:
        return [[] for _ in sentences]

    return [
        _parse_ner_results(sent, ents)
        for sent, ents in zip(sentences, batch_results)
    ]


# ---------------------------------------------------------------------------
# Regex-based extraction: DATE, NUMBER, PERCENT (always runs)
# ---------------------------------------------------------------------------

# Compiled patterns
_RE_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")

_RE_DATE_FULL = re.compile(
    r"\b(?:0?[1-9]|[12]\d|3[01])[./-](?:0?[1-9]|1[012])[./-](?:(?:19|20)?\d{2})\b"
)

_MONTH_NAMES = (
    r"January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
    r"|Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember"
    r"|Ianuarie|Februarie|Martie|Aprilie|Mai|Iunie|Iulie|August|Septembrie|Octombrie|Noiembrie|Decembrie"
    r"|Ιανουάριος|Φεβρουάριος|Μάρτιος|Απρίλιος|Μάιος|Ιούνιος|Ιούλιος|Αύγουστος"
    r"|Σεπτέμβριος|Οκτώβριος|Νοέμβριος|Δεκέμβριος"
)
_RE_DATE_WRITTEN = re.compile(
    rf"\b(?:{_MONTH_NAMES})\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{{2}}\b",
    re.IGNORECASE,
)

_RE_PERCENT = re.compile(
    r"\b(\d+(?:[.,]\d+)?)\s*(%|percent|Prozent|prozent|τοις εκατό|la sută)(?=\s|$|[,;.!?)\]])",
    re.IGNORECASE,
)

_RE_NUMBER = re.compile(
    r"\b(\d{1,3}(?:[,. ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)\b"
)

# Words that should not be treated as standalone numbers
_NUMBER_BLACKLIST_RE = re.compile(r"^(?:19|20)\d{2}$")  # years handled separately


def _extract_regex_spans(sentence: str) -> List[EntitySpan]:
    """
    Extract DATE, NUMBER, PERCENT spans from a sentence using regex.
    Yields non-overlapping spans in document order.
    """
    claimed: List[Tuple[int, int, str, str]] = []  # (start, end, type, text)

    # --- PERCENT (before NUMBER so "45%" is not also a NUMBER) ---
    for m in _RE_PERCENT.finditer(sentence):
        claimed.append((m.start(), m.end(), "PERCENT", m.group(0)))

    # --- DATE (full written form first) ---
    for m in _RE_DATE_WRITTEN.finditer(sentence):
        claimed.append((m.start(), m.end(), "DATE", m.group(0)))

    # --- DATE (numeric form: dd/mm/yyyy) ---
    for m in _RE_DATE_FULL.finditer(sentence):
        claimed.append((m.start(), m.end(), "DATE", m.group(0)))

    # --- YEAR (standalone 4-digit year) ---
    for m in _RE_YEAR.finditer(sentence):
        claimed.append((m.start(), m.end(), "DATE", m.group(0)))

    # --- NUMBER ---
    for m in _RE_NUMBER.finditer(sentence):
        raw = m.group(0)
        # Skip if looks like a year
        digits_only = re.sub(r"[,. ]", "", raw)
        if _NUMBER_BLACKLIST_RE.match(digits_only):
            continue
        claimed.append((m.start(), m.end(), "NUMBER", raw))

    # Deduplicate/resolve overlaps (keep earlier, longer span)
    claimed.sort(key=lambda t: (t[0], -(t[1] - t[0])))
    filtered: List[Tuple[int, int, str, str]] = []
    occupied: Set[int] = set()
    for start, end, etype, text in claimed:
        positions = set(range(start, end))
        if positions & occupied:
            continue
        occupied |= positions
        filtered.append((start, end, etype, text))

    return [
        EntitySpan(text=text, start=start, end=end, entity_type=etype)
        for start, end, etype, text in filtered
    ]


# ---------------------------------------------------------------------------
# Heuristic named-entity detection (fallback for PERSON/ORG/GPE/LOC)
# ---------------------------------------------------------------------------

_RE_CAP_PHRASE = re.compile(
    r"\b([A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞ\u0370-\u03FF\u0400-\u04FF]"
    r"[a-zàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþ\u0370-\u03FF\u0400-\u04FF]+"
    r"(?:[-'\u2019][A-ZÀ-Öa-zà-ö]+)*"
    r"(?:\s+[A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞ\u0370-\u03FF\u0400-\u04FF]"
    r"[a-zàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþ\u0370-\u03FF\u0400-\u04FF]+"
    r"(?:[-'\u2019][A-ZÀ-Öa-zà-ö]+)*){0,3})\b"
)


def _heuristic_named_spans(sentence: str) -> List[EntitySpan]:
    """
    Detect capitalized word sequences as PERSON/ORG/GPE candidates.
    Skips sentence-initial position and common stop-words.
    Labels all as PERSON (cannot distinguish type without NER).
    """
    spans: List[EntitySpan] = []
    for m in _RE_CAP_PHRASE.finditer(sentence):
        phrase = m.group(0)
        # Skip if at sentence start (might just be normal capitalisation)
        if m.start() == 0:
            continue
        # Skip if all words are stop-words
        words = phrase.split()
        if all(w.lower() in _STOP_WORDS for w in words):
            continue
        # Skip single very short words that are likely not named entities
        if len(words) == 1 and len(phrase) <= 2:
            continue
        spans.append(EntitySpan(
            text=phrase,
            start=m.start(),
            end=m.end(),
            entity_type="PERSON",  # coarse label; pool lookup uses same pool
        ))
    return spans


# ---------------------------------------------------------------------------
# Span extraction: combined NER + regex
# ---------------------------------------------------------------------------

def extract_all_spans(
    sentence: str,
    ner_spans_override: Optional[List[EntitySpan]] = None,
) -> List[EntitySpan]:
    """
    Extract all candidate entity spans from a sentence.
    Merges NER output with regex detections; resolves overlaps.

    Args:
        sentence:           The sentence text to analyse.
        ner_spans_override: Pre-computed NER spans (e.g. from a batch call).
                            When provided, the NER pipeline is not called again.
    """
    regex_spans = _extract_regex_spans(sentence)
    ner_spans: List[EntitySpan] = (
        ner_spans_override if ner_spans_override is not None else _ner_spans(sentence)
    )

    # If NER unavailable or returned nothing, use heuristic for PERSON/ORG/GPE
    if not any(s.entity_type in ENTITY_TYPES_NAMED for s in ner_spans):
        heuristic = _heuristic_named_spans(sentence)
        ner_spans = list(ner_spans) + heuristic

    all_spans = regex_spans + ner_spans

    # Resolve overlaps: keep highest-priority span (NER > regex > heuristic)
    all_spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
    occupied: Set[int] = set()
    merged: List[EntitySpan] = []
    for span in all_spans:
        positions = set(range(span.start, span.end))
        if positions & occupied:
            continue
        occupied |= positions
        merged.append(span)

    merged.sort(key=lambda s: s.start)
    return merged


# ---------------------------------------------------------------------------
# Numeric perturbation helpers
# ---------------------------------------------------------------------------

def _perturb_year(year_str: str, rng: random.Random) -> str:
    """Shift a 4-digit year by ±1-3 years."""
    try:
        year = int(year_str)
    except ValueError:
        return year_str
    delta = rng.choice([-3, -2, -1, 1, 2, 3])
    new_year = year + delta
    # Keep within plausible range
    new_year = max(1900, min(2099, new_year))
    if new_year == year:
        new_year = year + (1 if delta <= 0 else -1)
    return str(new_year)


def _perturb_numeric_date(date_str: str, rng: random.Random) -> Optional[str]:
    """Perturb a numeric date string (dd/mm/yyyy or similar)."""
    sep_match = re.search(r"([./-])", date_str)
    if not sep_match:
        return None
    sep = sep_match.group(1)
    parts = date_str.split(sep)
    if len(parts) != 3:
        return None
    # year is the last part if 4 digits, else first
    try:
        if len(parts[2]) == 4:
            yr_idx = 2
        elif len(parts[0]) == 4:
            yr_idx = 0
        else:
            return None
        year = int(parts[yr_idx])
        delta = rng.choice([-3, -2, -1, 1, 2, 3])
        parts[yr_idx] = str(year + delta)
        return sep.join(parts)
    except (ValueError, IndexError):
        return None


def _detect_number_format(num_str: str) -> Tuple[bool, bool]:
    """
    Return (uses_comma_decimal, uses_dot_decimal).
    Distinguishes:
      European decimal: "3,14" (comma before 1-2 digits at end)
      Anglo thousands:  "1,200" (comma before exactly 3 digits)
    """
    # Comma followed by 1-2 digits at end → decimal separator (European)
    if re.search(r",\d{1,2}$", num_str):
        return True, False
    # Dot followed by 1-2 digits at end → decimal separator (Anglo)
    if re.search(r"\.\d{1,2}$", num_str):
        return False, True
    # Dot followed by 3 digits → thousands separator (European), comma is decimal
    if re.search(r"\.\d{3}", num_str) and re.search(r",\d{1,2}$", num_str):
        return True, False
    return False, False


def _perturb_number_value(num_str: str, rng: random.Random) -> Optional[str]:
    """
    Perturb a numeric string by ~10-30%, preserving format
    (decimal separator, thousands separator, digit count).
    """
    uses_comma_decimal, uses_dot_decimal = _detect_number_format(num_str)

    # Normalise to float
    normalised = num_str.replace(" ", "")
    if uses_comma_decimal:
        # European format: 1.234,56 → 1234.56
        normalised = normalised.replace(".", "").replace(",", ".")
    else:
        # Anglo format: 1,234.56 or 1,234 → 1234.56 or 1234
        normalised = normalised.replace(",", "")

    try:
        value = float(normalised)
    except ValueError:
        return None

    if value == 0:
        return None

    # Determine decimal places in original
    if uses_comma_decimal or uses_dot_decimal:
        dp_match = re.search(r"[.,](\d+)$", num_str)
        decimal_places = len(dp_match.group(1)) if dp_match else 0
    else:
        decimal_places = 0

    # Perturbation: 10-30% of value, but at least 1 unit
    pct = rng.uniform(0.10, 0.30)
    delta = max(1, abs(value) * pct)
    if rng.random() < 0.5:
        delta = -delta
    new_value = value + delta

    # Preserve sign
    if value > 0 and new_value <= 0:
        new_value = value * rng.uniform(1.10, 1.30)

    # Format back
    if decimal_places == 0:
        int_val = int(round(new_value))
        # Re-apply thousands separator if original used one
        if abs(int_val) >= 1000 and ("," in num_str or "." in num_str):
            if uses_comma_decimal:
                # European: thousands as dot
                formatted = f"{int_val:,}".replace(",", ".")
            else:
                # Anglo: thousands as comma
                formatted = f"{int_val:,}"
        else:
            formatted = str(int_val)
    else:
        if uses_comma_decimal:
            formatted = f"{new_value:.{decimal_places}f}".replace(".", ",")
        else:
            formatted = f"{new_value:.{decimal_places}f}"

    return formatted if formatted != num_str else None


def _perturb_percent_value(pct_str: str, rng: random.Random) -> Optional[str]:
    """Perturb the numeric part of a percent string ±1-5 percentage points."""
    m = re.search(r"(\d+(?:[.,]\d+)?)", pct_str)
    if not m:
        return None
    num_part = m.group(1)
    try:
        val = float(num_part.replace(",", "."))
    except ValueError:
        return None

    delta = rng.uniform(1.0, 5.0)
    if rng.random() < 0.5:
        delta = -delta
    new_val = val + delta
    new_val = max(0.1, min(99.9, new_val))
    if abs(new_val - val) < 0.5:
        new_val = val + (2.0 if delta >= 0 else -2.0)
        new_val = max(0.1, min(99.9, new_val))

    if "." in num_part or "," in num_part:
        dp = len(re.search(r"[.,](\d+)", num_part).group(1))
        new_str = f"{new_val:.{dp}f}"
        if "," in num_part:
            new_str = new_str.replace(".", ",")
    else:
        new_str = str(int(round(new_val)))

    if new_str == num_part:
        return None

    return pct_str[:m.start()] + new_str + pct_str[m.end():]


def generate_numeric_replacement(span: EntitySpan, rng: random.Random) -> Optional[str]:
    """
    Generate a perturbed replacement for DATE, NUMBER, or PERCENT spans.
    Returns None if no valid perturbation found.
    """
    etype = span.entity_type
    text = span.text

    if etype == "DATE":
        # Standalone year?
        if re.fullmatch(r"(?:19|20)\d{2}", text.strip()):
            result = _perturb_year(text.strip(), rng)
            return result if result != text else None
        # Numeric date?
        if re.search(r"\d[./-]\d", text):
            return _perturb_numeric_date(text, rng)
        # Written date – try to shift year within it
        year_m = _RE_YEAR.search(text)
        if year_m:
            new_year = _perturb_year(year_m.group(0), rng)
            return text[:year_m.start()] + new_year + text[year_m.end():]
        return None

    if etype == "NUMBER":
        return _perturb_number_value(text, rng)

    if etype == "PERCENT":
        return _perturb_percent_value(text, rng)

    return None


# ---------------------------------------------------------------------------
# Entity descriptor, granularity, role, and subtype inference helpers
# ---------------------------------------------------------------------------

# GPE geographic-granularity keywords
_GPE_COUNTRY_MARKERS: Set[str] = {
    "country", "nation", "national", "federal", "republic", "kingdom",
    "democracy", "monarchy",
}
_GPE_STATE_MARKERS: Set[str] = {
    "state", "province", "pradesh", "region", "oblast", "canton",
    "prefecture", "territory",
}
_GPE_CITY_MARKERS: Set[str] = {
    "city", "town", "village", "municipality", "district", "capital",
    "metropolitan", "suburb",
}
# Sub-strings that appear inside state/province-level entity names
_GPE_STATE_ENTITY_FRAGMENTS: Set[str] = {
    "pradesh", "province", "oblast", "canton", "prefecture", "shire",
}


def _infer_gpe_granularity(entity_text: str, sentence: str) -> str:
    """Infer GPE geographic granularity: 'country', 'state', 'city', or 'unknown'."""
    entity_lower = entity_text.lower()
    ctx_lower = sentence.lower()
    if any(frag in entity_lower for frag in _GPE_STATE_ENTITY_FRAGMENTS):
        return "state"
    if any(kw in ctx_lower for kw in _GPE_COUNTRY_MARKERS):
        return "country"
    if any(kw in ctx_lower for kw in _GPE_STATE_MARKERS):
        return "state"
    if any(kw in ctx_lower for kw in _GPE_CITY_MARKERS):
        return "city"
    return "unknown"


# PERSON role keywords
_PERSON_ROLE_POLITICIAN: Set[str] = {
    "minister", "president", "prime", "senator", "governor", "mayor",
    "secretary", "chancellor", "politician", "legislator", "councillor",
    "parliamentarian", "chief minister", "mp", "mla",
}
_PERSON_ROLE_JOURNALIST: Set[str] = {
    "journalist", "reporter", "editor", "correspondent", "anchor",
    "columnist", "presenter", "broadcaster",
}
_PERSON_ROLE_EXECUTIVE: Set[str] = {
    "ceo", "cfo", "cto", "director", "chairman", "chairwoman",
    "executive", "officer",
}


def _infer_person_role(entity_text: str, sentence: str) -> str:
    """Infer PERSON role: 'politician', 'journalist', 'executive', or 'unknown'."""
    ctx_lower = sentence.lower()
    if any(kw in ctx_lower for kw in _PERSON_ROLE_POLITICIAN):
        return "politician"
    if any(kw in ctx_lower for kw in _PERSON_ROLE_JOURNALIST):
        return "journalist"
    if any(kw in ctx_lower for kw in _PERSON_ROLE_EXECUTIVE):
        return "executive"
    return "unknown"


# ORG subtype keywords
_ORG_MEDIA: Set[str] = {
    "newspaper", "news", "tv", "television", "channel", "media",
    "magazine", "radio", "broadcasting", "press", "publication",
}
_ORG_GOVT: Set[str] = {
    "ministry", "department", "government", "commission", "authority",
    "agency", "bureau", "court", "tribunal", "committee", "council",
    "parliament", "legislature", "senate", "assembly",
}
_ORG_BUSINESS: Set[str] = {
    "company", "corporation", "corp", "firm", "ltd", "limited", "inc",
    "pvt", "group", "holdings", "industries", "enterprises",
}


def _infer_org_subtype(entity_text: str, sentence: str) -> str:
    """Infer ORG subtype: 'media', 'government', 'business', or 'unknown'."""
    combined = (entity_text + " " + sentence).lower()
    if any(kw in combined for kw in _ORG_MEDIA):
        return "media"
    if any(kw in combined for kw in _ORG_GOVT):
        return "government"
    if any(kw in combined for kw in _ORG_BUSINESS):
        return "business"
    return "unknown"


def _build_entity_descriptor(
    entity_type: str,
    entity_text: str,
    sentence: str,
    max_context_words: int = 5,
) -> str:
    """
    Build a context-aware descriptor string for embedding.

    Format: "TYPE | entity_text | ctx_word1 | ctx_word2 | ..."

    Context words are extracted from the sentence, excluding the entity text
    itself and stop-words, up to max_context_words tokens.
    """
    entity_lower = entity_text.lower()
    words = re.split(r"\W+", sentence)
    ctx_words: List[str] = []
    for w in words:
        if not w or w.lower() in _STOP_WORDS:
            continue
        if w.lower() in entity_lower or entity_lower in w.lower():
            continue
        if len(w) < 3:
            continue
        ctx_words.append(w)
        if len(ctx_words) >= max_context_words:
            break
    return " | ".join([entity_type, entity_text] + ctx_words)


def _normalize_entity_text(text: str) -> str:
    """Normalised form for near-duplicate detection (lowercase, strip punctuation)."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Replacement pool building
# ---------------------------------------------------------------------------

def _run_ner_on_text(text: str) -> List[Tuple[str, str]]:
    """
    Return (entity_text, entity_type) pairs from NER + heuristic on text.
    """
    results: List[Tuple[str, str]] = []
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    for sent in sentences:
        if not sent.strip():
            continue
        ner_spans = _ner_spans(sent)
        if not ner_spans:
            ner_spans = _heuristic_named_spans(sent)
        for span in ner_spans:
            if span.entity_type in ENTITY_TYPES_NAMED:
                results.append((span.text, span.entity_type))
    return results


def build_entity_pools(
    paths: Any,
    claims_lang_map: Dict[str, str],
    *,
    max_per_type: int = 2000,
) -> Dict[str, Dict[str, List[PoolEntry]]]:
    """
    Build per-language entity replacement pools by running NER over:
      - All sentence texts in cache/gold/*/*/embeddings.jsonl
      - All plaintext files in cache/plaintext/ (if present)

    Returns:
      pools[lang][entity_type] = [PoolEntry, ...]

    For PERSON, GPE, and ORG: entries include context-aware descriptor strings
    and precomputed embeddings (up to _MAX_POOL_EMBED entries per bucket).
    For LOC: entries are stored without embeddings; random selection is used.
    Each list is deduplicated by entity text and capped at max_per_type items.
    """
    from ccir.io_utils import read_jsonl

    # pool_raw[lang][etype][entity_text] = source_sentence (first occurrence wins)
    pool_raw: Dict[str, Dict[str, Dict[str, str]]] = {}

    def _add(lang: str, etype: str, text: str, sentence: str) -> None:
        text = text.strip()
        if len(text) < 2:
            return
        lang_map = pool_raw.setdefault(lang, {})
        type_map = lang_map.setdefault(etype, {})
        if text not in type_map:   # first occurrence wins
            type_map[text] = sentence

    def _ingest_ner_batch(lang: str, sents: List[str]) -> None:
        """Run batched NER on a list of sentences and add named entities to pool."""
        if not sents:
            return
        ner_batch = _ner_spans_batch(sents)
        for sent, ner_spans in zip(sents, ner_batch):
            named = [s for s in ner_spans if s.entity_type in ENTITY_TYPES_NAMED]
            if not named:
                named = _heuristic_named_spans(sent)
            for span in named:
                _add(lang, span.entity_type, span.text, sent)

    gold_dir: Path = paths.cache_gold_dir
    if gold_dir.exists():
        for emb_path in sorted(gold_dir.glob("*/*/embeddings.jsonl")):
            parts = emb_path.relative_to(gold_dir).parts
            if len(parts) != 3:
                continue
            claim_id = parts[0]
            lang = claims_lang_map.get(claim_id, "en")
            try:
                rows = list(read_jsonl(emb_path))
            except Exception:
                continue
            sents = [r.get("sentence_text", "").strip() for r in rows if isinstance(r, dict)]
            sents = [s for s in sents if s]
            _ingest_ner_batch(lang, sents)

    # Also scan cached plaintext files (broader coverage)
    plaintext_dir: Path = paths.cache_plaintext_dir
    if plaintext_dir.exists():
        for txt_path in sorted(plaintext_dir.glob("*/*.txt")):
            parts = txt_path.relative_to(plaintext_dir).parts
            if len(parts) < 2:
                continue
            claim_id = parts[0]
            lang = claims_lang_map.get(claim_id, "en")
            try:
                content = txt_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            # Process only first 5000 chars per file to bound cost
            sents = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+|\n+", content[:5000])
                if s.strip()
            ]
            _ingest_ner_batch(lang, sents)

    # Build PoolEntry objects; precompute embeddings for PERSON, GPE, ORG.
    pools: Dict[str, Dict[str, List[PoolEntry]]] = {}
    for lang, type_map in pool_raw.items():
        pools[lang] = {}
        for etype, text_sent_map in type_map.items():
            # Cap pool size; sort by text for determinism
            items: List[Tuple[str, str]] = sorted(text_sent_map.items())[:max_per_type]

            if etype in ENTITY_TYPES_NN_POOL:
                # Build entries with context-aware descriptors and structural hints
                entries: List[PoolEntry] = []
                for text, sentence in items:
                    descriptor = _build_entity_descriptor(etype, text, sentence)
                    granularity = _infer_gpe_granularity(text, sentence) if etype == "GPE" else ""
                    role_hint = _infer_person_role(text, sentence) if etype == "PERSON" else ""
                    org_subtype = _infer_org_subtype(text, sentence) if etype == "ORG" else ""
                    entries.append(PoolEntry(
                        text=text,
                        entity_type=etype,
                        language=lang,
                        descriptor=descriptor,
                        embedding=[],
                        granularity=granularity,
                        role_hint=role_hint,
                        org_subtype=org_subtype,
                    ))

                # Precompute embeddings in one batch (bounded by _MAX_POOL_EMBED)
                to_embed = entries[:_MAX_POOL_EMBED]
                descriptors = [e.descriptor for e in to_embed]
                try:
                    from ccir.retrieval.embeddings import embed_texts
                    vecs = embed_texts(descriptors)
                    for entry, vec in zip(to_embed, vecs):
                        entry.embedding = vec
                except Exception:
                    pass  # entries keep empty embeddings; runtime falls back to random

                pools[lang][etype] = entries
            else:
                # LOC and other types: store without embeddings; random selection used
                pools[lang][etype] = [
                    PoolEntry(
                        text=text,
                        entity_type=etype,
                        language=lang,
                        descriptor="",
                        embedding=[],
                    )
                    for text, _ in items
                ]

    return pools


# ---------------------------------------------------------------------------
# Embedding-based nearest-neighbour ranking over precomputed pool entries
# ---------------------------------------------------------------------------

def _get_pool_entries_for_type(
    lang: str,
    entity_type: str,
    entity_pools: Dict[str, Dict[str, List[PoolEntry]]],
) -> List[PoolEntry]:
    """
    Return pool entries for (lang, entity_type).
    Falls back to 'en' pool if the language-specific pool is absent.
    For PERSON: also searches ORG/GPE/LOC pools when the PERSON pool is empty
    (covers heuristic-labelled spans that are all tagged PERSON).
    """
    lang_pool = entity_pools.get(lang, entity_pools.get("en", {}))
    entries: List[PoolEntry] = list(lang_pool.get(entity_type, []))
    if entity_type == "PERSON" and not entries:
        for alt in ("ORG", "GPE", "LOC"):
            entries.extend(lang_pool.get(alt, []))
    return entries


def _structural_filter(
    entries: List[PoolEntry],
    entity_type: str,
    query_granularity: str,
    query_role: str,
    query_subtype: str,
    min_after_filter: int = 3,
) -> List[PoolEntry]:
    """
    Narrow pool entries using type-specific structural constraints.

    GPE  – require matching geographic granularity (country/state/city).
    PERSON – prefer candidates with the same role context (politician/journalist/…).
    ORG  – prefer candidates with the same subtype (media/government/business).

    Falls back to the full list when fewer than min_after_filter entries survive.
    """
    if entity_type == "GPE" and query_granularity not in ("", "unknown"):
        filtered = [
            e for e in entries
            if e.granularity in ("", "unknown", query_granularity)
        ]
        if len(filtered) >= min_after_filter:
            return filtered

    elif entity_type == "PERSON" and query_role not in ("", "unknown"):
        filtered = [
            e for e in entries
            if e.role_hint in ("", "unknown", query_role)
        ]
        if len(filtered) >= min_after_filter:
            return filtered

    elif entity_type == "ORG" and query_subtype not in ("", "unknown"):
        filtered = [
            e for e in entries
            if e.org_subtype in ("", "unknown", query_subtype)
        ]
        if len(filtered) >= min_after_filter:
            return filtered

    return entries


def _rank_pool_entries(
    query_descriptor: str,
    candidates: List[PoolEntry],
    rng: random.Random,
    top_k: int,
) -> List[PoolEntry]:
    """
    Rank PoolEntry candidates by cosine similarity to the query descriptor.

    Uses precomputed embeddings stored on each PoolEntry; only one embedding
    call is made at runtime (for the query descriptor itself).
    Falls back to a random shuffle when no candidates have precomputed embeddings
    or when the embedding call fails.

    Returns up to top_k entries.
    """
    if not candidates:
        return []

    if len(candidates) <= top_k:
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        return shuffled

    # Only rank entries that have precomputed embeddings
    embedded = [e for e in candidates if e.embedding]
    if not embedded:
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        return shuffled[:top_k]

    try:
        from ccir.retrieval.embeddings import embed_texts
        query_vec = np.asarray(embed_texts([query_descriptor])[0], dtype=np.float32)
        q_norm = float(np.linalg.norm(query_vec))
        if q_norm == 0:
            raise ValueError("zero-norm query vector")

        scored: List[Tuple[float, PoolEntry]] = []
        for entry in embedded:
            cand_vec = np.asarray(entry.embedding, dtype=np.float32)
            c_norm = float(np.linalg.norm(cand_vec))
            sim = float(np.dot(query_vec, cand_vec) / (q_norm * c_norm)) if c_norm > 0 else 0.0
            scored.append((sim, entry))

        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:top_k]]
    except Exception:
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        return shuffled[:top_k]


# ---------------------------------------------------------------------------
# Replacement generation: named entities
# ---------------------------------------------------------------------------

def generate_named_replacement(
    span: EntitySpan,
    sentence: str,
    lang: str,
    entity_pools: Dict[str, Dict[str, List[PoolEntry]]],
    rng: random.Random,
    top_k: int,
    rank_by_embedding: bool = True,
) -> Optional[str]:
    """
    Choose a replacement for a PERSON/ORG/GPE/LOC span from the pool.

    For PERSON, GPE, and ORG:
      - Applies structural pre-filters (geographic granularity, role context,
        org subtype) before ranking.
      - Excludes candidates identical to, near-duplicate of, or already present
        in the sentence.
      - When rank_by_embedding=True: builds a context-aware query descriptor,
        embeds it once, and ranks filtered candidates by cosine similarity
        against their precomputed embeddings; samples from the top-K neighbours.
      - When rank_by_embedding=False (fast collection pass): applies structural
        filters only and selects randomly (defers the embedding call).

    For LOC (and other types): uses random selection from the pool without
    structural filters or embedding ranking.
    """
    entries = _get_pool_entries_for_type(lang, span.entity_type, entity_pools)
    if not entries:
        return None

    # Near-duplicate normalisation for exclusion
    orig_norm = _normalize_entity_text(span.text)
    sentence_lower = sentence.lower()

    filtered: List[PoolEntry] = [
        e for e in entries
        if _normalize_entity_text(e.text) != orig_norm
        and e.text.lower() not in sentence_lower
        and e.text.lower() != span.text.lower()
    ]
    if not filtered:
        return None

    if span.entity_type in ENTITY_TYPES_NN_POOL:
        # Infer structural attributes of the query entity
        query_granularity = (
            _infer_gpe_granularity(span.text, sentence) if span.entity_type == "GPE" else ""
        )
        query_role = (
            _infer_person_role(span.text, sentence) if span.entity_type == "PERSON" else ""
        )
        query_subtype = (
            _infer_org_subtype(span.text, sentence) if span.entity_type == "ORG" else ""
        )

        # Apply structural pre-filter
        filtered = _structural_filter(
            filtered, span.entity_type, query_granularity, query_role, query_subtype
        )
        if not filtered:
            return None

        if not rank_by_embedding:
            # Fast path: structural filter only, random choice
            return rng.choice(filtered).text

        # Build query descriptor and rank by NN cosine similarity
        query_descriptor = _build_entity_descriptor(span.entity_type, span.text, sentence)
        top_candidates = _rank_pool_entries(query_descriptor, filtered, rng, top_k)
        if not top_candidates:
            return None
        return rng.choice(top_candidates).text

    else:
        # LOC and other types: random selection without structural filter or NN ranking
        if not rank_by_embedding:
            return rng.choice(filtered).text
        rng.shuffle(filtered)
        return filtered[0].text if filtered else None


# ---------------------------------------------------------------------------
# Edit validation
# ---------------------------------------------------------------------------

_MULTI_SPACE_RE = re.compile(r"  +")


def _validate_edit(
    original: str,
    edited: str,
    old_span: str,
    new_span: str,
) -> bool:
    """
    Return True only if the edit is valid:
      1. new_span differs from old_span
      2. Exactly one contiguous span changed
      3. Sentence stays well-formed (length plausible, no broken punctuation)
      4. Format preserved (surrounding punctuation not moved)
      5. Type-consistent (guaranteed by caller, checked implicitly)
    """
    if not edited or not original:
        return False

    if old_span == new_span:
        return False

    if old_span not in original:
        return False

    # Exactly one occurrence of old_span replaced
    count_orig = original.count(old_span)
    count_edited = edited.count(new_span)
    if count_orig == 0:
        return False
    # edited should have new_span appear (at least) count_orig times and
    # old_span appear count_orig-1 times
    if old_span not in edited.replace(new_span, old_span, 1):
        pass  # Allow (this checks substitution occurred)

    # Verify exactly one replacement happened
    reconstructed = original.replace(old_span, new_span, 1)
    if reconstructed != edited:
        return False

    # Length sanity: edited length within ±70% of original
    len_ratio = len(edited) / max(1, len(original))
    if not (0.3 <= len_ratio <= 3.0):
        return False

    # No run-on double spaces introduced
    if _MULTI_SPACE_RE.search(edited):
        return False

    # Sentence must still start with the same character type
    if bool(original[0].isupper()) != bool(edited[0].isupper()):
        return False

    # Ensure new_span is not empty
    if not new_span.strip():
        return False

    return True


# ---------------------------------------------------------------------------
# Candidate edit generation for one sentence
# ---------------------------------------------------------------------------

def _make_edits_for_sentence(
    sentence_idx: int,
    sentence: str,
    relevance_score: float,
    lang: str,
    entity_pools: Dict[str, Dict[str, List[str]]],
    rng: random.Random,
    top_k: int,
    ner_spans_override: Optional[List[EntitySpan]] = None,
) -> List[EditCandidate]:
    """
    Generate all valid candidate edits for a single sentence.

    Args:
        ner_spans_override: Pre-computed NER spans from a batch call; avoids
                            an extra per-sentence pipeline invocation.
    """
    spans = extract_all_spans(sentence, ner_spans_override=ner_spans_override)
    candidates: List[EditCandidate] = []

    for span in spans:
        if span.entity_type in ENTITY_TYPES_NUMERIC:
            replacement = generate_numeric_replacement(span, rng)
        else:
            # rank_by_embedding=False: defer expensive embedding calls to the
            # post-selection refinement step in run_misleading_edit so that
            # we don't embed pool candidates for every sentence in the doc.
            replacement = generate_named_replacement(
                span, sentence, lang, entity_pools, rng, top_k,
                rank_by_embedding=False,
            )

        if replacement is None or replacement == span.text:
            continue

        # Apply the replacement (exactly one occurrence)
        edited = sentence.replace(span.text, replacement, 1)

        if not _validate_edit(sentence, edited, span.text, replacement):
            continue

        candidates.append(EditCandidate(
            sentence_idx=sentence_idx,
            original_sentence=sentence,
            edited_sentence=edited,
            edit_type=span.entity_type,
            old_span=span.text,
            new_span=replacement,
            relevance_score=relevance_score,
        ))

    return candidates


# ---------------------------------------------------------------------------
# Key-term extraction, selection, and structural corruption (new pipeline)
# ---------------------------------------------------------------------------

def extract_key_terms(
    sentences: Sequence[Dict[str, Any]],
    sentence_texts: List[str],
    ner_by_idx: Dict[int, List[EntitySpan]],
) -> List[KeyTerm]:
    """
    Extract and deduplicate key terms from all sentences in the document.

    - Spans from every sentence are collected via extract_all_spans (NER +
      regex + heuristic fallback).
    - Repeated surface forms are merged into one KeyTerm; the first
      occurrence's text is kept as the canonical form.
    - Returns terms sorted by descending max_relevance_score, then by
      _KEY_TERM_TYPE_PRIORITY so named entities rank above numeric spans.
    """
    seen: Dict[str, KeyTerm] = {}

    for i, row in enumerate(sentences):
        sentence = sentence_texts[i]
        if not sentence:
            continue
        score = float(row.get("embedding_score", 0.0))
        spans = extract_all_spans(sentence, ner_spans_override=ner_by_idx.get(i, []))

        for span in spans:
            norm = _normalize_entity_text(span.text)
            if not norm or len(norm) < 2:
                continue
            if norm in seen:
                kt = seen[norm]
                if i not in kt.occurrences:
                    kt.occurrences.append(i)
                if score > kt.max_relevance_score:
                    kt.max_relevance_score = score
                # Upgrade to higher-priority entity type if newly seen type is better
                if (_KEY_TERM_TYPE_PRIORITY.get(span.entity_type, 99) <
                        _KEY_TERM_TYPE_PRIORITY.get(kt.entity_type, 99)):
                    kt.entity_type = span.entity_type
            else:
                seen[norm] = KeyTerm(
                    text=span.text,
                    normalized=norm,
                    entity_type=span.entity_type,
                    occurrences=[i],
                    max_relevance_score=score,
                )

    return sorted(
        seen.values(),
        key=lambda kt: (
            -kt.max_relevance_score,
            _KEY_TERM_TYPE_PRIORITY.get(kt.entity_type, 99),
        ),
    )


def _num_key_terms_to_corrupt(n_terms: int, pct: int) -> int:
    """
    Number of unique key terms to corrupt for a given percentage level.

    - Ceiling rounding so even small documents get proportional coverage.
    - Minimum 1 whenever any terms are available.
    - For pct≥50: guarantee at least 2 when ≥2 terms exist so the stronger
      level always corrupts more terms than pct=20 on short documents.
    """
    if n_terms <= 0:
        return 0
    k = max(1, math.ceil(n_terms * pct / 100))
    if pct >= 50 and k < 2 and n_terms >= 2:
        k = 2
    return min(k, n_terms)


def apply_structural_corruption(
    sentence: str,
    rng: random.Random,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Attempt to apply one structural scope/qualifier flip to a single sentence.

    Scans _STRUCTURAL_FLIPS patterns (case-insensitive).  When multiple
    patterns match, one is chosen deterministically via *rng*.

    Returns:
        (modified_sentence, change_record)
        change_record is None when no pattern matched (sentence unchanged).
    """
    matches: List[Tuple[int, int, str, str, str]] = []
    for pattern, replacement, category in _STRUCTURAL_FLIPS:
        for m in re.finditer(pattern, sentence, re.IGNORECASE):
            matches.append((m.start(), m.end(), m.group(0), replacement, category))

    if not matches:
        return sentence, None

    # Sort by position for determinism before the rng choice
    matches.sort(key=lambda x: x[0])
    chosen_start, chosen_end, old_str, replacement, category = rng.choice(matches)

    modified = sentence[:chosen_start] + replacement + sentence[chosen_end:]
    return modified, {
        "category": category,
        "old_text": old_str,
        "new_text": replacement,
        "sentence_position": chosen_start,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_misleading_edit(
    sentences: Sequence[Dict[str, Any]],
    pct: int,
    lang: str,
    entity_pools: Dict[str, Dict[str, List[Any]]],
    rng: random.Random,
    top_k: int = DEFAULT_TOP_K,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Apply misleading edits to a document using sentence-based selection.

    Strategy:
      1. Run batched NER on the top-_MAX_NER_SENTENCES sentences (by
         embedding_score); remaining sentences use regex + heuristic fallback.
      2. For every non-empty sentence call _make_edits_for_sentence to
         collect all valid single-entity edit candidates.  A sentence is
         "eligible" when at least one candidate exists.
      3. Sentence budget: ceil(pct/100 * n_sentences).  Select up to that
         many eligible sentences, ranked by descending embedding_score
         (most claim-relevant sentences are corrupted first).  When fewer
         eligible sentences exist than the budget, all are used.
      4. For each selected sentence apply exactly ONE edit — the highest-
         priority candidate by _TYPE_PRIORITY (DATE/NUMBER/PERCENT first,
         then GPE/PERSON/LOC, then ORG).  The edit uses .replace(old, new, 1)
         so only the first occurrence of the span in that sentence is changed.
      5. For pct≥50: scan sentences in relevance order and apply at most
         one structural scope/qualifier flip (e.g. "prohibited"→"permitted").
      6. Return corrupted text and a structured JSONL edit log.

    Args:
        sentences:    Ordered sentence dicts from embeddings.jsonl
                      (must have 'sentence_text' and 'embedding_score').
        pct:          Integer corruption percentage (e.g., 20, 50).
        lang:         ISO language code for pool selection.
        entity_pools: Pre-built per-language entity replacement pools.
        rng:          Seeded random instance — deterministic from run seed.
        top_k:        Pool candidates to consider during embedding ranking.

    Returns:
        (corrupted_text, edit_log_rows)
        edit_log_rows always contains at least one row:
          - record_type="corruption_meta": document-level statistics
          - record_type="key_term_edit":   one row per edited sentence
          - record_type="structural_edit": one row if a structural flip applied
    """
    if not sentences:
        return "", []

    sentence_texts: List[str] = [
        row.get("sentence_text", "").strip() for row in sentences
    ]

    # ------------------------------------------------------------------
    # Step 1 – batched NER on the top-_MAX_NER_SENTENCES sentences.
    # Non-target sentences fall back to regex + heuristic (zero NER cost).
    # ------------------------------------------------------------------
    scored_indices = sorted(
        (i for i, t in enumerate(sentence_texts) if t),
        key=lambda i: -float(sentences[i].get("embedding_score", 0.0)),
    )
    ner_target_set = set(scored_indices[:_MAX_NER_SENTENCES])
    ner_texts_ordered = [sentence_texts[i] for i in sorted(ner_target_set)]
    ner_batch = _ner_spans_batch(ner_texts_ordered)

    ner_by_idx: Dict[int, List[EntitySpan]] = {
        i: [] for i, t in enumerate(sentence_texts) if t
    }
    for pos, i in enumerate(sorted(ner_target_set)):
        ner_by_idx[i] = ner_batch[pos]

    # ------------------------------------------------------------------
    # Step 2 – identify eligible sentences (≥1 valid edit candidate).
    # Iterate in descending relevance order so the list is already sorted
    # for the budget selection in Step 3.
    # ------------------------------------------------------------------
    eligible: List[Tuple[int, List[EditCandidate]]] = []
    for sent_idx in scored_indices:
        text = sentence_texts[sent_idx]
        if not text:
            continue
        relevance = float(sentences[sent_idx].get("embedding_score", 0.0))
        candidates = _make_edits_for_sentence(
            sentence_idx=sent_idx,
            sentence=text,
            relevance_score=relevance,
            lang=lang,
            entity_pools=entity_pools,
            rng=rng,
            top_k=top_k,
            ner_spans_override=ner_by_idx.get(sent_idx, []),
        )
        if candidates:
            eligible.append((sent_idx, candidates))

    # ------------------------------------------------------------------
    # Step 3 – select up to ceil(K% * n_sentences) eligible sentences.
    # If fewer eligible sentences exist than the budget, use all of them.
    # ------------------------------------------------------------------
    n_sentences_total = len(scored_indices)
    sentence_budget = max(1, math.ceil(pct / 100 * n_sentences_total))
    selected = eligible[:sentence_budget]

    # ------------------------------------------------------------------
    # Step 4 – for each selected sentence apply exactly ONE edit.
    # Pick the highest-priority candidate (_TYPE_PRIORITY); ties broken by
    # position in sentence (earlier span first) for determinism.
    # _make_edits_for_sentence already uses .replace(old, new, 1) so only
    # the first occurrence of the span is changed.
    # ------------------------------------------------------------------
    modified_sentences: List[str] = list(sentence_texts)
    replacement_records: List[Dict[str, Any]] = []

    for sent_idx, candidates in selected:
        best = min(
            candidates,
            key=lambda c: (_TYPE_PRIORITY.get(c.edit_type, 99), c.sentence_idx),
        )
        modified_sentences[sent_idx] = best.edited_sentence
        replacement_records.append({
            "sentence_idx": sent_idx,
            "key_term": best.old_span,
            "entity_type": best.edit_type,
            "replacement": best.new_span,
            "relevance_score": best.relevance_score,
            "affected_sentence_indices": [sent_idx],
        })

    # ------------------------------------------------------------------
    # Step 5 – structural corruption (pct≥50 only).
    # Scan sentences in relevance order and apply at most one flip.
    # ------------------------------------------------------------------
    structural_changes: List[Dict[str, Any]] = []
    if pct >= 50:
        for sent_idx in scored_indices:
            sentence = modified_sentences[sent_idx]
            if not sentence:
                continue
            modified, change = apply_structural_corruption(sentence, rng)
            if change is not None:
                modified_sentences[sent_idx] = modified
                structural_changes.append({"sentence_idx": sent_idx, **change})
                break  # at most one structural flip per document

    # ------------------------------------------------------------------
    # Step 6 – materialise corrupted text.
    # ------------------------------------------------------------------
    out_lines = [s for s in modified_sentences if s.strip()]
    corrupted_text = "\n".join(out_lines).strip() + "\n" if out_lines else ""

    # ------------------------------------------------------------------
    # Step 7 – build structured edit log.
    # ------------------------------------------------------------------
    edit_log: List[Dict[str, Any]] = [{
        "record_type": "corruption_meta",
        "corruption_level_pct": pct,
        "total_sentences": n_sentences_total,
        "eligible_sentences": len(eligible),
        "sentence_budget": sentence_budget,
        "sentences_edited": len(replacement_records),
        "structural_corruption_applied": len(structural_changes) > 0,
    }]

    for rec in replacement_records:
        edit_log.append({"record_type": "key_term_edit", **rec})

    for change in structural_changes:
        edit_log.append({"record_type": "structural_edit", **change})

    return corrupted_text, edit_log
