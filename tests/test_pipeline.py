from datetime import datetime, timezone

import pytest

from ai_research_radar import llm, pipeline
from ai_research_radar.arxiv import parse_feed
from ai_research_radar.config import load_config
from ai_research_radar.paper import Paper
from ai_research_radar.prefilter import prefilter

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>2</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2609.00001v2</id>
    <updated>2026-09-22T10:00:00Z</updated>
    <published>2026-09-21T10:00:00Z</published>
    <title>Agentic RAG for
      Whole-Slide Pathology</title>
    <summary>We build an agent with retrieval-augmented generation for whole slide images.
      Code: https://github.com/example/wsi-agent.</summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <arxiv:comment>Accepted at MICCAI 2026</arxiv:comment>
    <arxiv:primary_category term="cs.CV"/>
    <category term="cs.CV"/><category term="cs.AI"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2609.00002v1</id>
    <updated>2026-09-21T10:00:00Z</updated>
    <published>2026-09-21T10:00:00Z</published>
    <title>Speculative Decoding at Scale</title>
    <summary>Faster inference for a large language model with speculative decoding.</summary>
    <author><name>Grace Hopper</name></author>
    <arxiv:comment>Under review at ICLR 2027</arxiv:comment>
    <arxiv:primary_category term="cs.LG"/>
    <category term="cs.LG"/>
  </entry>
</feed>"""


def test_parse_feed():
    papers, total = parse_feed(FEED)
    assert total == 2
    a, b = papers
    assert (a.arxiv_id, a.version) == ("2609.00001", 2)
    assert a.title == "Agentic RAG for Whole-Slide Pathology"
    assert a.authors == ["Ada Lovelace", "Alan Turing"]
    assert a.venue_status == "Accepted at MICCAI 2026 (per authors)"
    assert a.links == ["https://github.com/example/wsi-agent"]
    assert b.venue_status == "Submitted to ICLR 2027 (per authors)"
    assert a.pdf_url == "https://arxiv.org/pdf/2609.00001v2"


def _paper(i: int, title: str, abstract: str = "") -> Paper:
    now = datetime(2026, 9, 21, tzinfo=timezone.utc)
    return Paper(f"2609.{i:05d}", 1, title, abstract, ["A"], ["cs.AI"], "cs.AI", now, now)


def test_prefilter_respects_area_shares():
    cfg = load_config()
    papers = [_paper(i, f"Large language model reasoning {i}") for i in range(50)]
    papers += [_paper(100 + i, f"Histopathology whole-slide segmentation {i}") for i in range(5)]
    papers.append(_paper(999, "A study of bird migration"))
    kept = prefilter(papers, cfg.areas, keep=20)
    assert len(kept) == 20
    # All 5 pathology papers survive despite being outnumbered 10:1.
    assert sum(p.area == "vision_multimodal_bio" for p in kept) == 5
    assert all(p.arxiv_id != "2609.00999" for p in kept)


def test_deep_dives_follow_area_shares(tmp_path, monkeypatch):
    """Even if the shortlist is grouped by area, deep dives are spread across areas."""
    cfg = load_config()
    cfg.output.reports_dir, cfg.output.db_path = str(tmp_path / "r"), str(tmp_path / "db")
    cfg.pipeline.deep_dives, cfg.pipeline.skim = 5, 0
    papers = [_paper(i, f"agent benchmark {i}") for i in range(10)]
    papers += [_paper(50 + i, f"histopathology segmentation {i}") for i in range(10)]
    monkeypatch.setattr(pipeline, "fetch_recent", lambda _: papers)
    pipeline.run(cfg, dry_run=True, date="2026-09-27")
    md = open(tmp_path / "r" / "2026-09-27.md").read()
    explained = md.split("## Papers explained")[1].split("## Worth")[0]
    assert explained.count("histopathology") >= 1 and explained.count("agent benchmark") >= 2


def test_full_run_with_fake_claude(tmp_path, monkeypatch):
    """Exercise every stage and the full report template without network or API calls."""
    cfg = load_config()
    cfg.output.reports_dir = str(tmp_path / "reports")
    cfg.output.db_path = str(tmp_path / "radar.db")
    cfg.pipeline.deep_dives, cfg.pipeline.skim = 2, 2
    papers, _ = parse_feed(FEED)
    papers += [_paper(10 + i, f"Tool use agents benchmark {i}", "An agent evaluation benchmark.") for i in range(4)]

    class FakeClaude:
        def __init__(self, cfg):
            self.cfg, self.usage, self.cost_usd = cfg, {"m": [1000, 200]}, 0.42

        def cost_breakdown(self):
            return {"m": self.cost_usd}

    def fake_triage(claude, cands):
        for i, p in enumerate(cands):
            p.relevance, p.importance = 9 - i, 5
            p.extra["triaged"] = True

    def fake_rank(claude, pool, n):
        for p in pool:
            p.triage_note = f"Gist of {p.title}."
        return pool[:n]

    dive = llm.DeepDive(evidence_strength="moderate", tldr="T", problem="P", contribution="C", method="M", evidence="E",
                        limitations=["L1 (authors)"], prior_work="W", why_it_matters="Y", try_this="X")
    monkeypatch.setattr(pipeline, "fetch_recent", lambda _: papers)
    monkeypatch.setattr(pipeline, "fetch_fulltext", lambda p, n: ("full text", "Based on the full paper text."))
    monkeypatch.setattr(llm, "Claude", FakeClaude)
    monkeypatch.setattr(llm, "triage", fake_triage)
    monkeypatch.setattr(llm, "rank", fake_rank)
    monkeypatch.setattr(llm, "deep_dive", lambda c, p, t: dive)
    monkeypatch.setattr(llm, "verify", lambda c, t, d: llm.Verification(issues=[llm.VerifierIssue(
        section="evidence", statement="E", verdict="overstated", explanation="Paper hedges this.")]))
    monkeypatch.setattr(llm, "editorial", lambda c, deep, skim: llm.Editorial(
        overview="Big week.", headlines=[llm.Headline(title="H", summary="S", arxiv_ids=[deep[0][0].arxiv_id])],
        themes=["agents"]))

    result = pipeline.run(cfg, date="2026-09-27")
    md = open(result["markdown"]).read()
    assert "## This week in 10 minutes" in md and "Big week." in md
    assert "**Why this matters to Arpan.** Y" in md
    assert "evidence: **moderate**" in md
    assert "Gist of" in md
    assert "*overstated* (evidence)" in md
    assert "Model cost this run: $0.42" in md
    assert "Accepted at MICCAI 2026 (per authors)" in md
    html = open(result["html"]).read()
    assert '<blockquote class="warn">' in html

    # Papers already reported are excluded next week.
    result2 = pipeline.run(cfg, date="2026-10-04")
    assert result2["shortlisted"] == len(papers) - 4


@pytest.mark.parametrize("comment,expected", [
    ("", "Preprint"),
    ("10 pages, NeurIPS 2026 camera-ready", "Accepted at NeurIPS 2026 (per authors)"),
    ("Workshop paper, ICML", "Mentions ICML (status unclear)"),
])
def test_venue_status(comment, expected):
    p = _paper(1, "x")
    p.comment = comment
    assert p.venue_status == expected
