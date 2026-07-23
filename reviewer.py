"""Reviewer: pure policy that interprets the extractor's `confidence` block.

Ported near-verbatim from Self_Improving_Pipeline/reviewer.py — this module
was already schema-agnostic (it recursively walks whatever `confidence` tree
it's given), so it needs no changes to work on this tool's construction-
payment schema instead of utility bills. Only the module-level docstring and
the calibrated-confidence hook (not used by this tool) have been trimmed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import PipelineConfig

logger = logging.getLogger(__name__)


VERIFIED_CORRECT = "verified-correct"
LOW_CONFIDENCE = "low-confidence"
VERIFIED_SUSPICIOUS = "verified-suspicious"
UNVERIFIABLE_MISSING = "unverifiable-missing"

FLAG_WORTHY = {LOW_CONFIDENCE, VERIFIED_SUSPICIOUS, UNVERIFIABLE_MISSING}

_ALIGNMENT_EXACT = "EXACT"
_ALIGNMENT_NO_MATCH = "NO_MATCH"
_ALIGNMENT_MISSING = "MISSING"


@dataclass
class FieldJudgment:
    field: str
    extracted_value: Any
    confidence: Optional[float]
    alignment_status: str
    evidence_text: Optional[str]
    page: Optional[int]
    judgment: str
    is_flag_worthy: bool


@dataclass
class ReviewOutput:
    document_id: str
    judgments: List[FieldJudgment] = field(default_factory=list)

    def by_judgment(self, j: str) -> List[FieldJudgment]:
        return [x for x in self.judgments if x.judgment == j]

    def flag_worthy(self) -> List[FieldJudgment]:
        return [x for x in self.judgments if x.is_flag_worthy]


class Reviewer:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def review(self, extraction: Dict[str, Any], document_id: str) -> ReviewOutput:
        out = ReviewOutput(document_id=document_id)
        conf_block = extraction.get("confidence")
        if not isinstance(conf_block, dict):
            logger.warning("No 'confidence' block in extraction; reviewer has nothing to score.")
            return out
        self._walk(conf_block, extraction, prefix="", out=out.judgments)
        return out

    def _walk(self, conf_node: Any, extraction_node: Any, prefix: str, out: List[FieldJudgment]) -> None:
        if isinstance(conf_node, dict):
            if self._is_leaf(conf_node):
                out.append(self._make_judgment(prefix, conf_node, extraction_node))
                return
            for key, sub_conf in conf_node.items():
                sub_prefix = f"{prefix}.{key}" if prefix else key
                sub_extraction = extraction_node.get(key) if isinstance(extraction_node, dict) else None
                self._walk(sub_conf, sub_extraction, sub_prefix, out)
        elif isinstance(conf_node, list):
            for i, sub_conf in enumerate(conf_node):
                sub_prefix = f"{prefix}[{i}]"
                sub_extraction = (
                    extraction_node[i] if isinstance(extraction_node, list) and i < len(extraction_node) else None
                )
                self._walk(sub_conf, sub_extraction, sub_prefix, out)

    @staticmethod
    def _is_leaf(node: Dict[str, Any]) -> bool:
        return "alignment_status" in node

    def _make_judgment(self, field_path: str, leaf: Dict[str, Any], extracted_value: Any) -> FieldJudgment:
        alignment_status = str(leaf.get("alignment_status") or "")
        confidence_raw = leaf.get("confidence")
        try:
            confidence = float(confidence_raw) if confidence_raw is not None else None
        except (TypeError, ValueError):
            confidence = None

        judgment = self._classify(alignment_status, confidence)
        return FieldJudgment(
            field=field_path,
            extracted_value=extracted_value if not isinstance(extracted_value, (dict, list)) else None,
            confidence=confidence,
            alignment_status=alignment_status or "UNKNOWN",
            evidence_text=leaf.get("evidence_text"),
            page=leaf.get("page"),
            judgment=judgment,
            is_flag_worthy=judgment in FLAG_WORTHY,
        )

    def _classify(self, alignment_status: str, confidence: Optional[float]) -> str:
        if alignment_status == _ALIGNMENT_NO_MATCH:
            return VERIFIED_SUSPICIOUS
        if alignment_status == _ALIGNMENT_MISSING:
            return UNVERIFIABLE_MISSING
        if alignment_status == _ALIGNMENT_EXACT:
            if confidence is None:
                return LOW_CONFIDENCE
            if confidence < self.config.tau_low:
                return LOW_CONFIDENCE
            return VERIFIED_CORRECT
        logger.warning("Unknown alignment_status %r; treating as low-confidence", alignment_status)
        return LOW_CONFIDENCE
