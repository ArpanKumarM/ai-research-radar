"""The Paper record shared by every pipeline stage."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

# Venue names we recognise in arXiv "comments" / "journal_ref" fields.
VENUES = [
    "NeurIPS", "ICML", "ICLR", "CVPR", "ICCV", "ECCV", "ACL", "EMNLP", "NAACL", "COLING",
    "AAAI", "IJCAI", "KDD", "WWW", "SIGIR", "MICCAI", "MIDL", "ISBI", "COLM", "CoRL",
    "RSS", "UAI", "AISTATS", "TMLR", "JMLR", "TPAMI", "Nature", "Science",
]
_VENUE_RE = re.compile(r"\b(" + "|".join(re.escape(v) for v in VENUES) + r")\b(?:\s*'?(\d{2,4}))?")
_ACCEPTED_RE = re.compile(r"\b(accepted|to appear|published|camera[- ]ready|oral|spotlight|poster)\b", re.I)
_SUBMITTED_RE = re.compile(r"\b(submitted|under review|in submission)\b", re.I)
_GITHUB_RE = re.compile(r"https?://(?:www\.)?(?:github\.com|huggingface\.co|gitlab\.com)/[\w.\-/]+[\w/]")
_URL_RE = re.compile(r"https?://[^\s,;)\]}]+")


@dataclass
class Paper:
    arxiv_id: str              # version-less id, e.g. "2609.01234"
    version: int
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    primary_category: str
    published: datetime
    updated: datetime
    comment: str = ""
    journal_ref: str = ""
    # Filled in by later stages.
    prefilter_score: float = 0.0
    area: str = ""
    relevance: float = 0.0
    importance: float = 0.0
    triage_note: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def abs_url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"

    @property
    def pdf_url(self) -> str:
        return f"https://arxiv.org/pdf/{self.arxiv_id}v{self.version}"

    @property
    def triage_score(self) -> float:
        return 0.6 * self.relevance + 0.4 * self.importance

    @property
    def venue_status(self) -> str:
        """Best-effort publication status from author-supplied metadata.

        This is what the *authors* say on arXiv, not a verified proceedings lookup.
        """
        text = f"{self.journal_ref} {self.comment}"
        m = _VENUE_RE.search(text)
        if not m:
            return "Preprint"
        venue = m.group(1) + (f" {m.group(2)}" if m.group(2) else "")
        if self.journal_ref or _ACCEPTED_RE.search(text):
            return f"Accepted at {venue} (per authors)"
        if _SUBMITTED_RE.search(text):
            return f"Submitted to {venue} (per authors)"
        return f"Mentions {venue} (status unclear)"

    @property
    def links(self) -> list[str]:
        """Code / project / dataset links the authors put in the abstract or comments."""
        text = f"{self.abstract} {self.comment}"
        seen: dict[str, None] = {}
        for url in _GITHUB_RE.findall(text) + _URL_RE.findall(text):
            url = url.rstrip(".")
            if "arxiv.org" not in url:
                seen.setdefault(url, None)
        return list(seen)
