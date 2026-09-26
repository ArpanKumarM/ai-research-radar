"""SQLite history: every paper seen, and which ones made it into a report."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .paper import Paper

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id     TEXT PRIMARY KEY,
    version      INTEGER NOT NULL,
    title        TEXT NOT NULL,
    abstract     TEXT NOT NULL,
    authors      TEXT NOT NULL,
    categories   TEXT NOT NULL,
    published    TEXT NOT NULL,
    comment      TEXT,
    journal_ref  TEXT,
    first_seen   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS report_items (
    report_date  TEXT NOT NULL,
    arxiv_id     TEXT NOT NULL,
    tier         TEXT NOT NULL,          -- 'deep' | 'skim'
    area         TEXT,
    relevance    REAL,
    importance   REAL,
    summary_json TEXT,
    PRIMARY KEY (report_date, arxiv_id)
);
CREATE TABLE IF NOT EXISTS runs (
    report_date  TEXT PRIMARY KEY,
    collected    INTEGER,
    prefiltered  INTEGER,
    shortlisted  INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd     REAL,
    finished_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class Store:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    def upsert_papers(self, papers: list[Paper]) -> None:
        self.db.executemany(
            """INSERT INTO papers (arxiv_id, version, title, abstract, authors, categories,
                                   published, comment, journal_ref)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(arxiv_id) DO UPDATE SET
                   version=excluded.version, title=excluded.title, abstract=excluded.abstract,
                   comment=excluded.comment, journal_ref=excluded.journal_ref
               WHERE excluded.version >= papers.version""",
            [
                (p.arxiv_id, p.version, p.title, p.abstract, json.dumps(p.authors),
                 json.dumps(p.categories), p.published.isoformat(), p.comment, p.journal_ref)
                for p in papers
            ],
        )
        self.db.commit()

    def reported_ids(self, before_date: str) -> set[str]:
        rows = self.db.execute(
            "SELECT DISTINCT arxiv_id FROM report_items WHERE report_date < ?", (before_date,)
        )
        return {r[0] for r in rows}

    def save_report(self, report_date: str, items: list[tuple[Paper, str, dict | None]]) -> None:
        self.db.execute("DELETE FROM report_items WHERE report_date = ?", (report_date,))
        self.db.executemany(
            "INSERT INTO report_items VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (report_date, p.arxiv_id, tier, p.area, p.relevance, p.importance,
                 json.dumps(summary) if summary else None)
                for p, tier, summary in items
            ],
        )
        self.db.commit()

    def save_run(self, report_date: str, stats: dict) -> None:
        self.db.execute(
            """INSERT OR REPLACE INTO runs (report_date, collected, prefiltered, shortlisted,
                                            input_tokens, output_tokens, cost_usd)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (report_date, stats.get("collected"), stats.get("screened"), stats.get("shortlisted"),
             stats.get("input_tokens"), stats.get("output_tokens"), stats.get("cost_usd")),
        )
        self.db.commit()
