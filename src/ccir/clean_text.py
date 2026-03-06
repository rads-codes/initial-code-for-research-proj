from __future__ import annotations

import re
import unicodedata
from typing import Iterable


# -----------------------------
# Public API
# -----------------------------
def clean_text(text: str) -> str:
    """
    Clean extracted plaintext from HTML/PDF into a stable form for caching.

    Intended properties:
    - Deterministic: same input => same output.
    - Conservative: remove obvious noise while preserving content.
    - Plaintext-friendly: normalized whitespace and Unicode.

    This function does NOT truncate to min/max lengths; Step 04 handles that.
    """
    if not text:
        return ""

    s = _normalize_unicode(text)
    s = _strip_nulls_and_controls(s)
    s = _normalize_line_endings(s)

    # Remove obvious HTML artifacts if the extractor leaked them
    s = _remove_html_leftovers(s)

    # Remove some common boilerplate lines (conservative)
    s = _drop_boilerplate_lines(s)

    # Fix common PDF/HTML whitespace issues
    s = _fix_hard_wrapped_paragraphs(s)
    s = _fix_hyphenation_across_linebreaks(s)

    # Collapse spacing
    s = _collapse_spaces(s)
    s = _collapse_blank_lines(s)

    # Final trim
    return s.strip()


# -----------------------------
# Unicode + control cleaning
# -----------------------------
def _normalize_unicode(s: str) -> str:
    # NFKC normalizes things like full-width chars, compatibility forms, etc.
    s = unicodedata.normalize("NFKC", s)
    # Normalize common “smart punctuation” variants
    s = s.replace("\u2013", "-").replace("\u2014", "-")  # en/em dashes
    s = s.replace("\u2212", "-")  # minus sign
    s = s.replace("\u2018", "'").replace("\u2019", "'")  # curly apostrophes
    s = s.replace("\u201c", '"').replace("\u201d", '"')  # curly quotes
    s = s.replace("\u00a0", " ")  # NBSP
    s = s.replace("\u200b", "")   # zero-width space
    return s


def _strip_nulls_and_controls(s: str) -> str:
    # Remove NULs and most control chars; keep \n and \t for structure.
    out_chars = []
    for ch in s:
        code = ord(ch)
        if ch in ("\n", "\t", "\r"):
            out_chars.append(ch)
            continue
        # C0/C1 controls
        if code < 32 or (127 <= code <= 159):
            continue
        out_chars.append(ch)
    return "".join(out_chars)


def _normalize_line_endings(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


# -----------------------------
# HTML-ish cleanup
# -----------------------------
_TAG_RE = re.compile(r"</?[^>\n]+>")
_ENTITY_REPLACEMENTS = {
    "&nbsp;": " ",
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&#39;": "'",
}


def _remove_html_leftovers(s: str) -> str:
    # Replace some common entities (extractors usually already do this, but not always)
    for k, v in _ENTITY_REPLACEMENTS.items():
        s = s.replace(k, v)

    # Drop tags if they slipped through (best-effort; not an HTML parser)
    s = _TAG_RE.sub("", s)

    # Remove obvious script/style residue if present as plaintext blocks
    # (Conservative: only remove if the line looks like code-y junk.)
    lines = s.split("\n")
    cleaned_lines = []
    for line in lines:
        if _looks_like_script_junk(line):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _looks_like_script_junk(line: str) -> bool:
    t = line.strip()
    if not t:
        return False
    # Very short lines are rarely useful if they look like code tokens
    if len(t) <= 3 and any(c in t for c in "{};<>"):
        return True
    # Common JS/CSS markers
    junk_markers = ("function(", "var ", "let ", "const ", "=>", "{", "}", "</script", "<script", "</style", "<style")
    if any(m in t.lower() for m in junk_markers):
        # Avoid deleting legitimate prose lines that contain braces rarely.
        # Require some density of symbols.
        sym_count = sum(t.count(c) for c in "{}<>;=()[]")
        if sym_count >= 3:
            return True
    return False


# -----------------------------
# Boilerplate removal (conservative)
# -----------------------------
# Keep this intentionally small to avoid dropping real content.
_BOILERPLATE_PATTERNS: Iterable[re.Pattern[str]] = [
    re.compile(r"^\s*cookie(s)?\b.*(accept|consent|preferences|policy)\b", re.I),
    re.compile(r"^\s*privacy\s+policy\b", re.I),
    re.compile(r"^\s*terms\s+of\s+service\b", re.I),
    re.compile(r"^\s*subscribe\b", re.I),
    re.compile(r"^\s*sign\s+in\b", re.I),
    re.compile(r"^\s*register\b", re.I),
    re.compile(r"^\s*all\s+rights\s+reserved\b", re.I),
    re.compile(r"^\s*©\s*\d{4}", re.I),
]


def _drop_boilerplate_lines(s: str) -> str:
    lines = s.split("\n")
    kept: list[str] = []
    for line in lines:
        t = line.strip()
        if not t:
            kept.append("")
            continue
        if any(p.search(t) for p in _BOILERPLATE_PATTERNS):
            continue
        kept.append(line)
    return "\n".join(kept)


# -----------------------------
# Wrapping and hyphenation fixes
# -----------------------------
def _fix_hyphenation_across_linebreaks(s: str) -> str:
    """
    Join word-\nwrap -> wordwrap (common in PDFs).
    We only join when the hyphen is at end of line and both sides look like letters.
    """
    return re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", s)


def _fix_hard_wrapped_paragraphs(s: str) -> str:
    """
    Convert single newlines inside paragraphs to spaces, while preserving paragraph breaks.
    Heuristic:
      - Treat >=2 newlines as paragraph break and preserve them.
      - Within a paragraph, replace remaining newlines with spaces.
    """
    parts = re.split(r"\n{2,}", s)
    fixed_parts = []
    for p in parts:
        # Replace remaining newlines/tabs with spaces inside the paragraph
        p2 = p.replace("\t", " ")
        p2 = re.sub(r"\n+", " ", p2)
        fixed_parts.append(p2.strip())
    return "\n\n".join([p for p in fixed_parts if p is not None])


# -----------------------------
# Whitespace normalization
# -----------------------------
def _collapse_spaces(s: str) -> str:
    # Collapse multiple spaces (but keep newlines)
    s = re.sub(r"[ \f\v]+", " ", s)
    # Trim spaces around newlines
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n[ \t]+", "\n", s)
    return s


def _collapse_blank_lines(s: str) -> str:
    # Keep at most 2 consecutive newlines
    return re.sub(r"\n{3,}", "\n\n", s)


# -----------------------------
# Optional convenience alias
# -----------------------------
# If some scripts import cleanText.cleanText(...), you can keep a compatibility alias.
def cleanText(text: str) -> str:  # noqa: N802
    return clean_text(text)