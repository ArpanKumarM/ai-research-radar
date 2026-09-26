"""Collect recent submissions from the arXiv API (Atom feed)."""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import httpx2 as httpx

from .config import ArxivConfig
from .paper import Paper

log = logging.getLogger(__name__)

API_URL = "https://export.arxiv.org/api/query"
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}
_ID_RE = re.compile(r"arxiv\.org/abs/(.+?)(?:v(\d+))?$")
_WS_RE = re.compile(r"\s+")


def _clean(text: str | None) -> str:
    return _WS_RE.sub(" ", text or "").strip()


def _parse_dt(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def parse_feed(xml_text: str) -> tuple[list[Paper], int]:
    """Parse one Atom page. Returns (papers, total_results)."""
    root = ET.fromstring(xml_text)
    total = int(root.findtext("opensearch:totalResults", "0", NS))
    papers = []
    for entry in root.findall("atom:entry", NS):
        m = _ID_RE.search(entry.findtext("atom:id", "", NS))
        if not m:
            continue  # arXiv returns an error pseudo-entry on bad queries
        primary = entry.find("arxiv:primary_category", NS)
        papers.append(
            Paper(
                arxiv_id=m.group(1),
                version=int(m.group(2) or 1),
                title=_clean(entry.findtext("atom:title", "", NS)),
                abstract=_clean(entry.findtext("atom:summary", "", NS)),
                authors=[_clean(a.findtext("atom:name", "", NS)) for a in entry.findall("atom:author", NS)],
                categories=[c.get("term", "") for c in entry.findall("atom:category", NS)],
                primary_category=primary.get("term", "") if primary is not None else "",
                published=_parse_dt(entry.findtext("atom:published", "", NS)),
                updated=_parse_dt(entry.findtext("atom:updated", "", NS)),
                comment=_clean(entry.findtext("arxiv:comment", "", NS)),
                journal_ref=_clean(entry.findtext("arxiv:journal_ref", "", NS)),
            )
        )
    return papers, total


def build_query(categories: list[str], start: datetime, end: datetime) -> str:
    cats = " OR ".join(f"cat:{c}" for c in categories)
    fmt = "%Y%m%d%H%M"
    return f"({cats}) AND submittedDate:[{start.strftime(fmt)} TO {end.strftime(fmt)}]"


def fetch_recent(cfg: ArxivConfig, end: datetime | None = None) -> list[Paper]:
    """Fetch papers first submitted in the lookback window, newest first, deduplicated."""
    end = end or datetime.now(timezone.utc)
    start = end - timedelta(days=cfg.lookback_days)
    query = build_query(cfg.categories, start, end)
    log.info("arXiv query: %s", query)

    papers: dict[str, Paper] = {}
    offset = 0
    with httpx.Client(timeout=60.0, headers={"User-Agent": "ai-research-radar/0.1"}) as client:
        while offset < cfg.max_results:
            params = {
                "search_query": query,
                "start": offset,
                "max_results": min(cfg.page_size, cfg.max_results - offset),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            page, total = _get_page(client, params, cfg.request_delay_s)
            for p in page:
                # Keep the newest version if a paper appears twice.
                if p.arxiv_id not in papers or p.version > papers[p.arxiv_id].version:
                    papers[p.arxiv_id] = p
            log.info("fetched %d/%d (unique so far: %d)", offset + len(page), total, len(papers))
            offset += len(page)
            if not page or offset >= total:
                break
            time.sleep(cfg.request_delay_s)
    return list(papers.values())


def _get_page(client: httpx.Client, params: dict, delay: float, attempts: int = 4) -> tuple[list[Paper], int]:
    # arXiv occasionally returns an empty page mid-pagination; retrying usually fixes it.
    for attempt in range(1, attempts + 1):
        try:
            resp = client.get(API_URL, params=params)
            resp.raise_for_status()
            page, total = parse_feed(resp.text)
            if page or params["start"] >= total:
                return page, total
            log.warning("empty arXiv page at offset %s (attempt %d)", params["start"], attempt)
        except (httpx.HTTPError, ET.ParseError) as e:
            log.warning("arXiv request failed (attempt %d): %s", attempt, e)
        time.sleep(delay * attempt)
    return [], 0
