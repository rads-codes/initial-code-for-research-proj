"""
src/ccir/utils/hashing.py

Purpose (per project outline):
- sha256 helpers for URL_ID, sentence IDs, and lineage inputs_sha256. :contentReference[oaicite:7]{index=7}

Doc-defined ID rules:
- url_id = "u_" + sha256(canonical_url).hexdigest()[:16] :contentReference[oaicite:8]{index=8}
- sentence_id = f"s{i:04d}_{sha256(sentence_text_norm).hexdigest()[:8]}" :contentReference[oaicite:9]{index=9}

This module is import-safe: no filesystem writes.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence, Union
from urllib.parse import parse_qsl, urlsplit, urlunsplit, urlencode


PathLike = Union[str, Path]


_WHITESPACE_RE = re.compile(r"\s+")


def sha256_hex_bytes(data: bytes) -> str:
    """Return sha256 hex digest for raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_hex_text(text: str, *, encoding: str = "utf-8") -> str:
    """Return sha256 hex digest for a string (utf-8 by default)."""
    return sha256_hex_bytes(text.encode(encoding))


def canonicalize_url(url: str) -> str:
    """
    Canonicalize URL into a stable string for hashing.

    Assumptions (docs do not define canonical_url):
    - strip whitespace
    - lowercase scheme + hostname
    - drop fragment (#...)
    - remove default ports (:80 for http, :443 for https)
    - normalize empty path to "/"
    - sort query params for stability
    """
    u = (url or "").strip()
    if not u:
        return ""

    parts = urlsplit(u)

    scheme = (parts.scheme or "").lower()
    netloc = parts.netloc

    # If urlsplit couldn't find a netloc but there's a "path" that looks like netloc,
    # keep behavior simple and just return stripped URL (still hashable).
    if not netloc and not scheme:
        return u

    # Split host:port (basic; does not handle every exotic netloc case)
    host = netloc
    port = ""
    if "@" in host:
        # userinfo present; keep it as-is but normalize host portion if possible
        userinfo, hostport = host.rsplit("@", 1)
    else:
        userinfo, hostport = "", host

    if ":" in hostport:
        h, p = hostport.rsplit(":", 1)
        if p.isdigit():
            host, port = h, p
        else:
            host, port = hostport, ""
    else:
        host, port = hostport, ""

    host = host.lower()

    # Drop default ports
    if (scheme == "http" and port == "80") or (scheme == "https" and port == "443"):
        port = ""

    rebuilt_hostport = host + (f":{port}" if port else "")
    rebuilt_netloc = f"{userinfo}@{rebuilt_hostport}" if userinfo else rebuilt_hostport

    path = parts.path or "/"

    # Normalize query ordering
    if parts.query:
        q = parse_qsl(parts.query, keep_blank_values=True)
        q.sort()
        query = urlencode(q, doseq=True)
    else:
        query = ""

    # Drop fragment
    fragment = ""

    return urlunsplit((scheme, rebuilt_netloc, path, query, fragment))


def url_id(url: str) -> str:
    """
    Compute the URL_ID used throughout the pipeline.

    Rule from docs:
    url_id = "u_" + sha256(canonical_url).hexdigest()[:16] :contentReference[oaicite:10]{index=10}
    """
    canon = canonicalize_url(url)
    h = sha256_hex_text(canon)
    return "u_" + h[:16]


def normalize_sentence_text(sentence: str) -> str:
    """
    Normalize sentence text for stable hashing.

    Assumptions (docs do not define sentence_text_norm):
    - Unicode normalize NFKC
    - strip
    - collapse whitespace runs to single spaces
    """
    s = unicodedata.normalize("NFKC", sentence or "")
    s = s.strip()
    s = _WHITESPACE_RE.sub(" ", s)
    return s


def sentence_id(i: int, sentence_text: str) -> str:
    """
    Compute sentence_ID for sentence i in an article.

    Rule from docs:
    sentence_ID = f"s{i:04d}_{sha256(sentence_text_norm).hexdigest()[:8]}" :contentReference[oaicite:11]{index=11}
    """
    if i < 0:
        raise ValueError(f"sentence index must be >= 0, got {i}")
    norm = normalize_sentence_text(sentence_text)
    h = sha256_hex_text(norm)
    return f"s{i:04d}_{h[:8]}"


def file_sha256(path: PathLike, *, chunk_size: int = 1024 * 1024) -> str:
    """
    Streaming sha256 for a file's contents.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if not p.is_file():
        raise ValueError(f"Not a file: {p}")

    hasher = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def inputs_sha256(paths: Sequence[PathLike]) -> str:
    """
    Compute a deterministic combined sha256 for lineage 'inputs_sha256'.

    Assumptions (docs do not define exact algorithm):
    - sort input paths by their string value for determinism
    - include both the path string and the file content hash in the combined digest
    - raise if any input is missing

    Output: hex sha256 of a manifest-like concatenation.
    """
    normalized: list[Path] = [Path(p) for p in paths]
    normalized_sorted = sorted(normalized, key=lambda x: str(x))

    manifest_hasher = hashlib.sha256()
    for p in normalized_sorted:
        p_str = str(p).encode("utf-8")
        h = file_sha256(p)
        # Include path + delimiter + content hash so changes to either affect the result.
        manifest_hasher.update(p_str)
        manifest_hasher.update(b"\0")
        manifest_hasher.update(h.encode("ascii"))
        manifest_hasher.update(b"\n")

    return manifest_hasher.hexdigest()


def sha256_hex(text: str) -> str:
    """
    Return sha256 hex digest of a string.
    Used for deterministic IDs like URL_ID.

    Example:
        sha256_hex("https://example.com")
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
'''
When collecting URLs (step 03):
u_id = url_id(raw_url) to produce "u_" + ...[:16]. 


When producing sentence embeddings rows (step 07):
sid = sentence_id(i, sentence_text) to produce s0000_deadbeef-style IDs. 

When writing lineage fields (any step):
inp_hash = inputs_sha256([paths.claims_all_jsonl(), paths.urls_jsonl(), ...])

Store that string into your row’s lineage field (wherever you decided to keep inputs_sha256).
'''