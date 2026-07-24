"""Append-only JSONL writer/reader for cross-document aggregation.

Ported near-verbatim from Self_Improving_Pipeline/review_log.py — already
generic (one JSON line per document, judgments keyed by field path), so no
changes needed for this tool's schema.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from reviewer import ReviewOutput

logger = logging.getLogger(__name__)


def append(
    log_path: Path,
    pdf_path: Path,
    review: ReviewOutput,
    ai_reviews: Optional[List[Any]] = None,
    extraction_model: Optional[str] = None,
    bbox_model: Optional[str] = None,
) -> None:
    judgments = [asdict(j) for j in review.judgments]
    if ai_reviews:
        by_field: Dict[str, Any] = {}
        for r in ai_reviews:
            field_path = getattr(r, "field_path", None) if not isinstance(r, dict) else r.get("field_path")
            if not field_path:
                continue
            by_field[field_path] = r.to_dict() if hasattr(r, "to_dict") else dict(r)
        for j in judgments:
            ai = by_field.get(j.get("field"))
            if ai:
                j["ai_review"] = ai

    # extraction_model / bbox_model recorded per entry so past log lines can
    # always be traced back to which model version produced them -- added
    # after we started actively switching models mid-investigation.
    entry = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "pdf": str(pdf_path),
        "document_id": review.document_id,
        "extraction_model": extraction_model,
        "bbox_model": bbox_model,
        "judgments": judgments,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def read_all(log_path: Path) -> Iterator[Dict[str, Any]]:
    if not log_path.exists():
        return
    with open(log_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line %d in %s: %s", lineno, log_path, exc)


def read_all_list(log_path: Path) -> List[Dict[str, Any]]:
    return list(read_all(log_path))
