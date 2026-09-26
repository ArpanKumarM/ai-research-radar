"""Typed loader for config.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Reader(BaseModel):
    name: str
    profile: str


class ArxivConfig(BaseModel):
    categories: list[str]
    lookback_days: int = 7
    max_results: int = 2000
    page_size: int = 200
    request_delay_s: float = 3.0


class Area(BaseModel):
    label: str
    share: float
    keywords: list[str]


class PipelineConfig(BaseModel):
    prefilter_keep: int = 0
    triage_batch_size: int = 40
    rank_pool: int = 60
    shortlist: int = 30
    deep_dives: int = 8
    skim: int = 15
    fulltext_max_chars: int = 120_000
    exclude_already_reported: bool = True


class ModelsConfig(BaseModel):
    triage: str = "claude-haiku-4-5"
    writer: str = "claude-opus-5"
    writer_effort: str = "high"
    judge: str = "claude-sonnet-5"
    judge_effort: str = "medium"


class OutputConfig(BaseModel):
    reports_dir: str = "reports"
    db_path: str = "data/radar.db"
    timezone: str = "America/Phoenix"


class Config(BaseModel):
    reader: Reader
    arxiv: ArxivConfig
    areas: dict[str, Area]
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_config(path: str | Path = "config.yaml") -> Config:
    with open(path) as f:
        return Config.model_validate(yaml.safe_load(f))
