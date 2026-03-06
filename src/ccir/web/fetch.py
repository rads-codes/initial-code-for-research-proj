from __future__ import annotations

from typing import Optional
import time

import requests
import trafilatura


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}


def _download_html(url: str, timeout_s: float) -> Optional[str]:
    """
    Download raw HTML with requests.
    """
    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout_s)
        r.raise_for_status()
        return r.text
    except Exception:
        return None


def _extract_text(html: str) -> Optional[str]:
    """
    Extract clean article text using trafilatura.
    """
    try:
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
            deduplicate=True,
        )
        return text
    except Exception:
        return None


def fetch_url_text(
    url: str,
    timeout_s: float = 25,
    retries: int = 2,
) -> Optional[str]:
    """
    Fetch and extract article text.

    Returns:
        Clean plaintext string or None if extraction fails.

    Designed to be fast enough for large pipelines.
    """

    for attempt in range(retries + 1):

        html = _download_html(url, timeout_s)
        if not html:
            if attempt < retries:
                time.sleep(1)
                continue
            return None

        text = _extract_text(html)

        if text and text.strip():
            return text.strip()

        if attempt < retries:
            time.sleep(1)

    return None