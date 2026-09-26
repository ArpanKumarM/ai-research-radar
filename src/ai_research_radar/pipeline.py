"""The weekly run: collect → dedupe → prefilter → triage → deep dives → verify → editorial → report."""

from __future__ import annotations

import logging
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from . import llm
from .arxiv import fetch_recent
from .config import Config
from .fulltext import fetch_fulltext
from .paper import Paper
from .prefilter import assign_areas, fill_by_share, prefilter
from .report import render
from .store import Store

log = logging.getLogger(__name__)


@dataclass
class DeepItem:
    paper: Paper
    dive: llm.DeepDive | None = None
    issues: list[llm.VerifierIssue] = field(default_factory=list)
    source_note: str = ""


def run(cfg: Config, *, dry_run: bool = False, date: str | None = None) -> dict:
    date = date or datetime.now(ZoneInfo(cfg.output.timezone)).date().isoformat()
    store = Store(cfg.output.db_path)
    stats: dict = {"collected": 0, "screened": 0, "shortlisted": 0, "cost_usd": None}

    # 1. Collect and dedupe (arxiv.fetch_recent keeps only the newest version of each id).
    papers = fetch_recent(cfg.arxiv)
    store.upsert_papers(papers)
    stats["collected"] = len(papers)
    if cfg.pipeline.exclude_already_reported:
        seen = store.reported_ids(before_date=date)
        papers = [p for p in papers if p.arxiv_id not in seen]
    log.info("collected %d papers (%d new to reports)", stats["collected"], len(papers))

    claude = None if dry_run else llm.Claude(cfg)
    n_deep, n_skim = cfg.pipeline.deep_dives, cfg.pipeline.skim
    by_area = lambda ps: {name: [p for p in ps if p.area == name] for name in cfg.areas}  # noqa: E731

    if claude:
        # 2. Cheap model scores every abstract (or a keyword-prefiltered subset if prefilter_keep > 0).
        keep = cfg.pipeline.prefilter_keep
        candidates = prefilter(papers, cfg.areas, keep) if keep > 0 else papers
        assign_areas(candidates, cfg.areas)
        random.Random(date).shuffle(candidates)  # mix topics within each batch to steady the scoring
        llm.triage(claude, candidates)
        candidates = [p for p in candidates if p.extra.get("triaged")]
        stats["screened"] = len(candidates)

        # 3. Stronger model compares the best of them side by side and picks the shortlist.
        pool = fill_by_share(by_area(candidates), cfg.areas, cfg.pipeline.rank_pool, key=lambda p: p.triage_score)
        try:
            shortlist = llm.rank(claude, pool, cfg.pipeline.shortlist)
        except Exception as e:
            log.error("ranking failed, falling back to screening scores: %s", e)
            shortlist = pool[: cfg.pipeline.shortlist]
    else:
        # Dry run: free keyword ranking only.
        candidates = prefilter(papers, cfg.areas, cfg.pipeline.prefilter_keep or 250)
        stats["screened"] = len(candidates)
        shortlist = fill_by_share(by_area(candidates), cfg.areas, cfg.pipeline.shortlist,
                                  key=lambda p: p.prefilter_score)
    stats["shortlisted"] = len(shortlist)

    # Deep dives and skims honour the area shares; within an area, shortlist order decides.
    position = {p.arxiv_id: i for i, p in enumerate(shortlist)}
    by_rank = lambda p: -position[p.arxiv_id]  # noqa: E731
    deep_papers = fill_by_share(by_area(shortlist), cfg.areas, n_deep, key=by_rank)
    rest = [p for p in shortlist if p not in deep_papers]
    skim = fill_by_share(by_area(rest), cfg.areas, n_skim, key=by_rank)

    # 4. Full-text deep dives + verification.
    deep = [DeepItem(p) for p in deep_papers]
    if claude:
        with ThreadPoolExecutor(4) as pool:
            list(pool.map(lambda item: _explain(claude, cfg, item), deep))
        deep = [d for d in deep if d.dive] + [d for d in deep if not d.dive]

    # 5. Editorial overview.
    editorial = None
    if claude and any(d.dive for d in deep):
        try:
            editorial = llm.editorial(claude, [(d.paper, d.dive) for d in deep if d.dive], skim)
        except Exception as e:
            log.error("editorial failed: %s", e)

    conference = [p for p in shortlist + candidates if p.venue_status != "Preprint"]
    conference = list({p.arxiv_id: p for p in conference}.values())[:25]

    if claude:
        stats["cost_usd"] = claude.cost_usd
        log.info("cost by model: %s", claude.cost_breakdown())
        stats["input_tokens"] = sum(u[0] for u in claude.usage.values())
        stats["output_tokens"] = sum(u[1] for u in claude.usage.values())

    md_path, html_path = render(cfg, date, {
        "stats": stats, "dry_run": dry_run, "editorial": editorial,
        "deep": deep, "skim": skim, "conference": conference,
    })
    if not dry_run:
        store.save_report(date, [(d.paper, "deep", d.dive.model_dump() if d.dive else None) for d in deep]
                          + [(p, "skim", None) for p in skim])
        store.save_run(date, stats)
    log.info("wrote %s and %s", md_path, html_path)
    return {**stats, "markdown": str(md_path), "html": str(html_path)}


def _explain(claude: llm.Claude, cfg: Config, item: DeepItem) -> None:
    p = item.paper
    text, item.source_note = fetch_fulltext(p, cfg.pipeline.fulltext_max_chars)
    try:
        item.dive = llm.deep_dive(claude, p, text)
    except Exception as e:
        log.error("deep dive failed for %s: %s", p.arxiv_id, e)
        item.source_note = "Detailed summary failed; showing the triage summary."
        return
    try:
        item.issues = llm.verify(claude, text, item.dive).issues
    except Exception as e:
        log.error("verification failed for %s: %s", p.arxiv_id, e)
        item.source_note += " Automated fact-check did not run for this paper."
    log.info("explained %s (%d verifier flags)", p.arxiv_id, len(item.issues))
