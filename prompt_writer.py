"""Backup + apply helpers for the extraction prompt file.

Adapted from Self_Improving_Pipeline/prompt_writer.py. Simplified: the
original spliced one classifier's section within a large multi-classifier
Prompt_List.txt. This tool has exactly one prompt file, so applying a
candidate is a straight backup-then-overwrite — no splicing needed.

Nothing calls this automatically. Every application requires an explicit,
human-invoked call — see cli.py's `apply` subcommand. This mirrors the
source pipeline's deliberate, permanent human-approval gate: the Proposer
can draft, but never applies its own output.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def apply_prompt_update(prompt_path: Path, new_prompt_text: str) -> Path:
    """Back up the current prompt file, then overwrite it with new_prompt_text.

    Returns the backup path. Call only after a human has reviewed the
    candidate — see the `apply` subcommand in cli.py.
    """
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found at {prompt_path}")

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    backup_path = prompt_path.with_suffix(f".txt.backup_{ts}")
    shutil.copy(prompt_path, backup_path)
    logger.info("Backup created: %s", backup_path)

    prompt_path.write_text(new_prompt_text, encoding="utf-8")
    logger.info("Applied new prompt to %s", prompt_path)
    return backup_path
