"""AI Research Radar: a weekly, evidence-first AI research briefing."""

import argparse
import json
import logging
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="radar", description="Generate the weekly AI research report.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="No model calls: rank by keyword score and render abstracts only.")
    parser.add_argument("--date", help="Report date (YYYY-MM-DD); defaults to today in the configured timezone.")
    parser.add_argument("--days", type=int, help="Override arxiv.lookback_days.")
    parser.add_argument("--max-results", type=int, help="Override arxiv.max_results (useful for quick tests).")
    parser.add_argument("--deep", type=int, help="Override pipeline.deep_dives.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from dotenv import load_dotenv

    from .config import load_config

    load_dotenv()  # picks up ANTHROPIC_API_KEY from .env if present
    from .pipeline import run

    cfg = load_config(args.config)
    if args.days:
        cfg.arxiv.lookback_days = args.days
    if args.max_results:
        cfg.arxiv.max_results = args.max_results
    if args.deep is not None:
        cfg.pipeline.deep_dives = args.deep

    if not args.dry_run and not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        logging.warning("No ANTHROPIC_API_KEY set; relying on an `ant auth login` profile if one exists.")

    result = run(cfg, dry_run=args.dry_run, date=args.date)
    json.dump(result, sys.stdout, indent=2)
    print()
