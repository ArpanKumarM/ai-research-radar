"""Download an arXiv PDF and extract its main text with PyMuPDF."""

from __future__ import annotations

import logging
import re

import httpx2 as httpx
import pymupdf

from .paper import Paper

log = logging.getLogger(__name__)

# The reference list starts the back matter; appendices usually follow it.
_BACK_MATTER_RE = re.compile(r"\n\s*(?:\d+\s*|[A-Z]\.?\s*)?(References|Bibliography|REFERENCES)\s*\n")


def fetch_fulltext(paper: Paper, max_chars: int) -> tuple[str, str]:
    """Return (text, note). `note` says what was omitted so the report can be honest about it.

    Falls back to the abstract if the PDF can't be fetched or parsed.
    """
    try:
        resp = httpx.get(paper.pdf_url, timeout=90.0, follow_redirects=True,
                         headers={"User-Agent": "ai-research-radar/0.1"})
        resp.raise_for_status()
        with pymupdf.open(stream=resp.content, filetype="pdf") as doc:
            text = "\n".join(page.get_text() for page in doc)
    except Exception as e:  # network errors, corrupt PDFs, etc.
        log.warning("full text unavailable for %s: %s", paper.arxiv_id, e)
        return paper.abstract, "Full text unavailable; this summary is based on the abstract only."

    notes = []
    m = _BACK_MATTER_RE.search(text)
    if m and m.start() > len(text) * 0.3:
        text = text[: m.start()]
        notes.append("references and appendices were not read")
    if len(text) > max_chars:
        log.warning("%s main text is %d chars; reading first %d", paper.arxiv_id, len(text), max_chars)
        text = text[:max_chars]
        notes.append(f"only the first {max_chars:,} characters of the main text were read")
    note = ("Based on the full paper text; " + "; ".join(notes) + ".") if notes else "Based on the full paper text."
    return text, note
