"""Configuration for the standalone XBRL-conversion-tool.

Unlike the Self_Improving_Pipeline this project is adapted from, there is no
running Java backend here — this tool calls Gemini directly for
classification, extraction, and bbox verification. All paths are local to
this repo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _load_api_key() -> Optional[str]:
    """Read GEMINI_API_KEY/API_KEY from the environment, or a local .env file
    (shell-style `export KEY=VALUE` lines, same convention as SBI_Backend's
    .apikey.env, so a copied key file works unmodified)."""
    for key in ("GEMINI_API_KEY", "API_KEY"):
        if os.environ.get(key):
            return os.environ[key]

    env_path = _project_root() / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key in ("GEMINI_API_KEY", "API_KEY") and value:
            return value
    return None


@dataclass
class PipelineConfig:
    # Gemini models — set to the cheapest currently-active model
    # (gemini-2.5-flash-lite: $0.10/1M input, $0.40/1M output as of July 2026)
    # for all four roles while we're validating the pipeline mechanically
    # works. Verified 2026-07-23 to actually respond for the configured key
    # (several other listed model names 404 despite showing up in
    # client.models.list()). NOTE: bbox/critic/proposer reasoning quality may
    # suffer at this tier relative to a pro-tier model — revisit once
    # correctness, not cost, is the priority. gemini-2.5-flash-lite retires
    # 2026-10-16; gemini-3.1-flash-lite ($0.25/$1.50) is the next-cheapest
    # option after that.
    extraction_model: str = field(
        default_factory=lambda: os.environ.get("GEMINI_EXTRACTION_MODEL", "gemini-2.5-flash-lite")
    )
    bbox_model: str = field(
        default_factory=lambda: os.environ.get("GEMINI_BBOX_MODEL", "gemini-2.5-flash-lite")
    )
    proposer_model: str = field(
        default_factory=lambda: os.environ.get("GEMINI_PROPOSER_MODEL", "gemini-2.5-flash-lite")
    )
    ai_reviewer_model: str = field(
        default_factory=lambda: os.environ.get("GEMINI_CRITIC_MODEL", "gemini-2.5-flash-lite")
    )
    gemini_api_key: Optional[str] = field(default_factory=_load_api_key)

    # Reviewer policy thresholds (see reviewer.py:_classify) — same defaults
    # as Self_Improving_Pipeline; untuned for this domain yet.
    tau_high: float = 0.85
    tau_low: float = 0.5

    # Summarizer cross-document aggregation thresholds
    rate_threshold: float = 0.6
    min_files: int = 10

    # AI critic
    ai_reviewer_trust_threshold: float = 0.5
    ai_reviewer_max_concurrent: int = 4

    # Proposer bloat guard
    proposer_max_line_delta: int = 80
    proposer_max_total_lines: int = 250

    # Paths
    prompt_path: Path = field(
        default_factory=lambda: _project_root() / "prompts" / "construction_payment_prompt.txt"
    )
    taxonomy_path: Path = field(
        default_factory=lambda: _project_root() / "taxonomy" / "construction_payment_taxonomy.json"
    )
    runs_dir: Path = field(default_factory=lambda: _project_root() / "runs")
    review_log_path: Path = field(default_factory=lambda: _project_root() / "runs" / "review_log.jsonl")
    candidates_dir: Path = field(default_factory=lambda: _project_root() / "runs" / "candidates")

    def __post_init__(self) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
