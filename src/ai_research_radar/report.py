"""Render the weekly report to Markdown and HTML."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import markdown
from jinja2 import Environment, PackageLoader

from .config import Config


def _anchor(arxiv_id: str) -> str:
    return "p-" + re.sub(r"[^\w-]", "-", arxiv_id)


def _linklabel(url: str) -> str:
    u = urlparse(url)
    if u.netloc.endswith("github.com"):
        return "code"
    if u.netloc.endswith("huggingface.co"):
        return "Hugging Face"
    return u.netloc.removeprefix("www.")


def _env() -> Environment:
    env = Environment(loader=PackageLoader("ai_research_radar"), trim_blocks=False, lstrip_blocks=False)
    env.filters["anchor"] = _anchor
    env.filters["linklabel"] = _linklabel
    return env


def render(cfg: Config, date: str, context: dict) -> tuple[Path, Path]:
    env = _env()
    md = env.get_template("report.md.j2").render(
        date=date, areas=cfg.areas, reader=cfg.reader.name, **context
    )
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"

    body = markdown.markdown(md, extensions=["extra", "sane_lists"])
    # Style the verifier-flag callouts.
    body = body.replace("<blockquote>\n<p>⚠️", '<blockquote class="warn">\n<p>⚠️')
    html = env.get_template("report.html.j2").render(date=date, body=body)

    out = Path(cfg.output.reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path, html_path = out / f"{date}.md", out / f"{date}.html"
    md_path.write_text(md)
    html_path.write_text(html)
    return md_path, html_path
