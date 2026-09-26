"""Free, local relevance prefilter: keyword hits per interest area.

Cuts ~1,500 abstracts down to a few hundred before any paid model call.
"""

from __future__ import annotations

import math
import re
from functools import cache

from .config import Area
from .paper import Paper


@cache
def _pattern(keyword: str) -> re.Pattern:
    return re.compile(r"(?<![\w-])" + re.escape(keyword.lower()) + r"(?![\w-])")


def area_scores(paper: Paper, areas: dict[str, Area]) -> dict[str, float]:
    """Keyword score per area. Title hits count triple; repeated hits saturate."""
    title = paper.title.lower()
    body = paper.abstract.lower()
    scores = {}
    for name, area in areas.items():
        score = 0.0
        for kw in area.keywords:
            pat = _pattern(kw)
            hits = 3 * len(pat.findall(title)) + len(pat.findall(body))
            score += math.log1p(hits)
        scores[name] = score
    return scores


def prefilter(papers: list[Paper], areas: dict[str, Area], keep: int) -> list[Paper]:
    """Score, assign each paper its best area, and keep the top `keep` papers.

    Slots are split by area share so a flood of LLM papers can't crowd out pathology.
    """
    assign_areas(papers, areas)
    by_area: dict[str, list[Paper]] = {name: [] for name in areas}
    for p in papers:
        if p.prefilter_score > 0:
            by_area[p.area].append(p)

    return fill_by_share(by_area, areas, keep, key=lambda p: p.prefilter_score)


def assign_areas(papers: list[Paper], areas: dict[str, Area]) -> None:
    """Give every paper its best keyword area and score (the model may override the area later)."""
    for p in papers:
        scores = area_scores(p, areas)
        p.area = max(scores, key=scores.get)
        p.prefilter_score = scores[p.area]


def fill_by_share(by_area: dict[str, list[Paper]], areas: dict[str, Area], total: int, key) -> list[Paper]:
    """Pick `total` papers honouring area shares; unused quota flows to the best leftovers."""
    chosen: list[Paper] = []
    leftovers: list[Paper] = []
    for name, items in by_area.items():
        items = sorted(items, key=key, reverse=True)
        quota = round(total * areas[name].share)
        chosen += items[:quota]
        leftovers += items[quota:]
    leftovers.sort(key=key, reverse=True)
    chosen += leftovers[: max(0, total - len(chosen))]
    return sorted(chosen[:total], key=key, reverse=True)
