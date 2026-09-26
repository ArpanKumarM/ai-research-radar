"""All Claude calls: abstract triage, full-text deep dives, claim verification, editor overview."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Literal, TypeVar

import anthropic
from pydantic import BaseModel, Field

from .config import Config
from .paper import Paper

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

# USD per million tokens (input, output).
PRICES = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-fable-5-1": (10.0, 50.0),
}
# Models that support server-side refusal fallbacks (fallbacks: "default").
FALLBACK_MODELS = {"claude-opus-5", "claude-opus-5-5", "claude-fable-5-1"}
# Models that take adaptive thinking + effort (Haiku 4.5 does not).
ADAPTIVE_MODELS = {"claude-sonnet-5", "claude-opus-5", "claude-opus-5-5", "claude-fable-5-1"}


# ---------------------------------------------------------------- output schemas

class TriageItem(BaseModel):
    arxiv_id: str
    area: str = Field(description="The best-fitting interest area key from the list given.")
    relevance: int = Field(description="0-10: how relevant to the reader's interests")
    importance: int = Field(description="0-10: likely significance to the field, independent of the reader")


class TriageBatch(BaseModel):
    items: list[TriageItem]


class RankedPick(BaseModel):
    arxiv_id: str
    gist: str = Field(description="Two plain-English sentences: what the paper does and its headline result, "
                                  "phrased as the authors' claim.")


class Ranking(BaseModel):
    picks: list[RankedPick] = Field(description="The chosen papers, best first.")


class DeepDive(BaseModel):
    evidence_strength: Literal["strong", "moderate", "weak"] = Field(
        description="How well the paper's evidence supports its central claim: strong (solid baselines, "
                    "ablations, scale), moderate (reasonable but with clear gaps), weak (little or no "
                    "real evidence, self-constructed or tiny evaluations, mostly argument).")
    tldr: str = Field(description="One sentence (max 30 words) a busy engineer can remember.")
    problem: str = Field(description="The problem, explained simply, and why it is hard. Max 70 words.")
    contribution: str = Field(description="What the authors changed or invented. Max 70 words.")
    method: str = Field(description="How the method works: the core mechanism only, concretely. Max 130 words.")
    evidence: str = Field(description="The 2-3 key results with numbers, attributed to the authors, and whether "
                                      "they genuinely support the central claim (baselines, ablations, scale, "
                                      "statistical care). Max 120 words.")
    limitations: list[str] = Field(description="The 2-4 most important limitations, one sentence each, labelled "
                                               "'(authors)' if stated in the paper or '(assessment)' if yours.")
    prior_work: str = Field(description="How it relates to the key earlier work the paper itself cites. Max 50 words.")
    why_it_matters: str = Field(description="Why it matters for the reader specifically; concrete, not generic. "
                                            "Max 60 words.")
    try_this: str = Field(description="One concrete thing the reader could try or read next. Max 40 words.")


class VerifierIssue(BaseModel):
    section: str = Field(description="Which summary field the statement is in, e.g. 'evidence'.")
    statement: str = Field(description="The problematic statement, copied verbatim from the SUMMARY "
                                       "(never from the paper).")
    verdict: Literal["unsupported", "overstated", "misattributed"]
    explanation: str = Field(description="What the paper actually says instead, quoting it where possible.")


class Verification(BaseModel):
    issues: list[VerifierIssue]


class Headline(BaseModel):
    title: str
    summary: str = Field(description="2-3 sentences.")
    arxiv_ids: list[str]


class Editorial(BaseModel):
    overview: str = Field(description="A ~400-word briefing covering the week, readable in a few minutes. "
                                      "Plain prose, may use short paragraphs.")
    headlines: list[Headline] = Field(description="The 3-5 most important developments this week.")
    themes: list[str] = Field(description="2-4 cross-cutting trends observed across the papers.")


# ---------------------------------------------------------------- client wrapper

class Claude:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = anthropic.Anthropic(max_retries=4)
        self._lock = Lock()
        self.usage: dict[str, list[int]] = {}

    def _record(self, model: str, usage) -> None:
        with self._lock:
            tot = self.usage.setdefault(model, [0, 0])
            tot[0] += usage.input_tokens + (usage.cache_read_input_tokens or 0) + (usage.cache_creation_input_tokens or 0)
            tot[1] += usage.output_tokens

    def cost_breakdown(self) -> dict[str, float]:
        out = {}
        for model, (inp, out_tok) in self.usage.items():
            pin, pout = PRICES.get(model, (5.0, 25.0))
            out[model] = round(inp / 1e6 * pin + out_tok / 1e6 * pout, 4)
        return out

    @property
    def cost_usd(self) -> float:
        cost = 0.0
        for model, (inp, out) in self.usage.items():
            pin, pout = PRICES.get(model, (5.0, 25.0))
            cost += inp / 1e6 * pin + out / 1e6 * pout
        return cost

    def call(self, model: str, system: str, prompt: str, schema: type[T], *,
             max_tokens: int = 16000, effort: str | None = None) -> T:
        kwargs: dict = {}
        if model in ADAPTIVE_MODELS:
            kwargs["thinking"] = {"type": "adaptive"}
            if effort:
                kwargs["output_config"] = {"effort": effort}
        if model in FALLBACK_MODELS:
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"

        response = self.client.beta.messages.parse(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
            **kwargs,
        )
        self._record(model, response.usage)
        if response.stop_reason == "refusal":
            raise RuntimeError(f"{model} declined the request")
        if response.stop_reason == "max_tokens":
            raise RuntimeError(f"{model} hit max_tokens={max_tokens}; output truncated")
        if response.parsed_output is None:
            raise RuntimeError(f"{model} returned no parseable output")
        return response.parsed_output


# ---------------------------------------------------------------- pipeline stages

TRIAGE_SYSTEM = """You screen new AI research abstracts for one reader. Score each paper honestly:
most papers are incremental, so most importance scores should be 3-6. Reserve 8-10 for work that
is likely to change practice (new capability, strong evidence at scale, a widely useful method or
benchmark, or a result that overturns a common belief). Dramatic framing is not evidence: score
down grand claims that the abstract does not back with experiments. Judge from the abstract only.

Reader profile:
{profile}

Interest areas (use these keys for `area`):
{areas}"""


def triage(claude: Claude, papers: list[Paper], workers: int = 4) -> None:
    """Score every paper's abstract in batches; mutates papers in place."""
    cfg = claude.cfg
    system = TRIAGE_SYSTEM.format(profile=cfg.reader.profile, areas=_area_list(cfg))
    size = cfg.pipeline.triage_batch_size
    batches = [papers[i:i + size] for i in range(0, len(papers), size)]
    by_id = {p.arxiv_id: p for p in papers}

    def run(batch: list[Paper]) -> None:
        listing = "\n\n".join(
            f"<paper id=\"{p.arxiv_id}\">\nTitle: {p.title}\nAbstract: {p.abstract}\n</paper>" for p in batch
        )
        prompt = f"Score each of these {len(batch)} papers. Return one item per paper id.\n\n{listing}"
        try:
            result = claude.call(cfg.models.triage, system, prompt, TriageBatch, max_tokens=8000)
        except Exception as e:
            log.error("triage batch failed (%d papers skipped): %s", len(batch), e)
            return
        for item in result.items:
            if p := by_id.get(item.arxiv_id):
                p.relevance, p.importance = float(item.relevance), float(item.importance)
                if item.area in cfg.areas:  # otherwise keep the keyword-based area
                    p.area = item.area
                p.extra["triaged"] = True

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(run, batches))
    missing = sum(1 for p in papers if not p.extra.get("triaged"))
    if missing:
        log.warning("%d papers received no triage score", missing)


def _area_list(cfg: Config) -> str:
    return "\n".join(f"- {key}: {a.label} (target share {a.share:.0%})" for key, a in cfg.areas.items())


RANK_SYSTEM = """You choose which new AI papers go into one reader's weekly research briefing.
You see the strongest candidates from a first-pass screen, side by side. Pick the best ones, best first.

Rank by: relevance to the reader, likely importance to the field, and how well the abstract backs its
claims with evidence. Put hype, grand claims without experiments, and thin incremental results lower.
Order the picks strictly by overall merit across all areas; do not group them by area.
Keep a rough balance across interest areas, following their target shares, but never include a weak
paper just to fill an area. Only use arXiv ids from the list.

Reader profile:
{profile}

Interest areas:
{areas}"""


def rank(claude: Claude, pool: list[Paper], n: int) -> list[Paper]:
    """Compare the pool side by side and return the best `n`, best first, with gists filled in."""
    cfg = claude.cfg
    listing = "\n\n".join(
        f"<paper id=\"{p.arxiv_id}\" area=\"{p.area}\" status=\"{p.venue_status}\">\n"
        f"Title: {p.title}\nAbstract: {p.abstract}\n</paper>" for p in pool
    )
    prompt = f"Choose the best {n} of these {len(pool)} candidates, best first.\n\n{listing}"
    system = RANK_SYSTEM.format(profile=cfg.reader.profile, areas=_area_list(cfg))
    result = claude.call(cfg.models.judge, system, prompt, Ranking, effort=cfg.models.judge_effort)
    by_id = {p.arxiv_id: p for p in pool}
    ranked: list[Paper] = []
    for pick in result.picks:
        p = by_id.get(pick.arxiv_id)
        if p and p not in ranked:
            p.triage_note = pick.gist
            ranked.append(p)
    return ranked[:n]


DEEP_SYSTEM = """You write the detailed paper explanations in a weekly AI research briefing for one reader.

Rules:
- Use only the paper text provided. Never add numbers, datasets, baselines or comparisons that are not in it.
- Attribute results: write "the authors report/claim ..." for anything not independently verified.
- Be a sharp, fair reviewer: say plainly when evidence is thin (small benchmarks, missing baselines,
  no ablations, single seed, cherry-picked settings, self-constructed benchmarks).
- Explain for a strong ML engineer who has not read the paper. Prefer concrete mechanisms over adjectives.
- Be concise: the whole deep dive should be roughly 500-700 words. Respect each field's word limit;
  pick the most important points rather than covering everything.

Reader profile:
{profile}"""


def deep_dive(claude: Claude, paper: Paper, text: str) -> DeepDive:
    cfg = claude.cfg
    prompt = (
        f"Title: {paper.title}\nAuthors: {', '.join(paper.authors[:12])}\n"
        f"Publication status (from arXiv metadata): {paper.venue_status}\n\n"
        f"<paper_text>\n{text}\n</paper_text>\n\nWrite the deep dive."
    )
    return claude.call(cfg.models.writer, DEEP_SYSTEM.format(profile=cfg.reader.profile), prompt, DeepDive,
                       effort=cfg.models.writer_effort)


VERIFY_SYSTEM = """You fact-check a SUMMARY of a research paper against the paper's text.
Your job is to judge whether the summary represents the paper faithfully, not whether the paper
itself is correct or convincing. Criticising the paper is out of scope.

Flag only statements in the summary that: the paper does not say (unsupported), state a result more
strongly than the paper does (overstated), or present an authors' claim as established fact
(misattributed). Opinions clearly framed as the summarizer's assessment (e.g. about limitations,
evidence quality or relevance) are fine, and so is scepticism about the paper.

Before flagging a statement, search the paper text for it. Do not flag it if the paper supports it,
including after unit conversion, rounding, or reasonable paraphrase. A small number of precise,
real problems is far more useful than many borderline ones. Quote each flagged statement verbatim
from the summary. Return an empty list if the summary is faithful."""


def verify(claude: Claude, text: str, dive: DeepDive) -> Verification:
    prompt = (
        f"<paper_text>\n{text}\n</paper_text>\n\n"
        f"<summary>\n{dive.model_dump_json(indent=2)}\n</summary>\n\nCheck the summary."
    )
    cfg = claude.cfg
    result = claude.call(cfg.models.judge, VERIFY_SYSTEM, prompt, Verification, effort=cfg.models.judge_effort)
    # Keep only flags that actually quote the summary; anything else is the verifier critiquing the paper.
    summary_text = _norm(" ".join(str(v) for v in dive.model_dump().values()))
    kept = [i for i in result.issues if _norm(i.statement).strip(" .…") in summary_text]
    if len(kept) < len(result.issues):
        log.info("dropped %d verifier flags that did not quote the summary", len(result.issues) - len(kept))
    return Verification(issues=kept)


def _norm(text: str) -> str:
    return " ".join(text.replace("“", '"').replace("”", '"').replace("’", "'").split()).lower()


EDITOR_SYSTEM = """You are the editor of a weekly AI research briefing for one reader. From the
paper summaries below, write the week's overview. Rank by genuine importance, not hype. Keep the
authors-claim framing: do not upgrade claims to facts. Only reference arXiv ids that appear below.

Reader profile:
{profile}"""


def editorial(claude: Claude, deep: list[tuple[Paper, DeepDive]], skim: list[Paper]) -> Editorial:
    cfg = claude.cfg
    parts = [
        f"<paper id=\"{p.arxiv_id}\" tier=\"deep\" status=\"{p.venue_status}\">\n{p.title}\n"
        f"TL;DR: {d.tldr}\nContribution: {d.contribution}\nEvidence: {d.evidence}\n</paper>"
        for p, d in deep
    ] + [
        f"<paper id=\"{p.arxiv_id}\" tier=\"skim\">\n{p.title}\n{p.triage_note}\n</paper>" for p in skim
    ]
    prompt = "\n\n".join(parts) + "\n\nWrite this week's editorial."
    return claude.call(cfg.models.writer, EDITOR_SYSTEM.format(profile=cfg.reader.profile), prompt, Editorial,
                       effort=cfg.models.writer_effort)
