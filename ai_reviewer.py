"""AI Reviewer: targeted Gemini critic on policy-flagged fields.

Adapted from Self_Improving_Pipeline/ai_reviewer.py. Two real changes from the
original, both driven by this tool's actual purpose:

1. The prompt is rewritten for construction-payment documents instead of
   utility bills.
2. ISSUE_CATEGORIES gains two new entries — `wrong_taxonomy_concept` and
   `wrong_hierarchy_position` — because tagging a value to the wrong XBRL
   concept (or the right concept at the wrong Presentation Linkbase depth) is
   a categorically different failure mode from "right neighborhood, wrong
   cell" (wrong_row/wrong_column), which is all the original categories
   covered. This is a new build item flagged during design, not carried over
   from the source pipeline.
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from config import PipelineConfig
from reviewer import FieldJudgment, LOW_CONFIDENCE, ReviewOutput, VERIFIED_SUSPICIOUS

logger = logging.getLogger(__name__)

CRITIC_TARGETS = {LOW_CONFIDENCE, VERIFIED_SUSPICIOUS}

ISSUE_CATEGORIES = (
    "wrong_row",
    "wrong_column",
    "wrong_taxonomy_concept",     # new: right value, tagged to the wrong XBRL concept
    "wrong_hierarchy_position",   # new: right concept, wrong Presentation Linkbase depth/parent
    "wrong_period",                # e.g. cumulative-to-date value used instead of this-period
    "hallucinated",
    "partial_match",
    "formatting_drift",
    "actually_correct",
    "unclear_from_evidence",
)

CRITIC_PROMPT = """You are an adversarial reviewer of a construction contractor payment-application extraction.

Your job is to second-opinion ONE extracted field. A cheap heuristic (comparing the model's quoted evidence_text against the PDF page) has already flagged this field as suspicious. That heuristic only checks that a quote can be located on the page; it does NOT check whether the value is the right value for the right taxonomy concept, correctly nested in the hierarchy. That's your job.

## Field under review
- field_path: `{field_path}`
- policy_judgment: `{policy_judgment}`   (low-confidence | verified-suspicious)
- extracted_value: {extracted_value}
- self_confidence: {self_confidence}
- alignment_status: {alignment_status}   (EXACT | NO_MATCH | MISSING)
- evidence_text: {evidence_text}
- page (1-indexed): {page}

## Extraction rules the extractor was following
```
{extraction_rules}
```

## How to think about this

Step 1 — Enumerate what would make this wrong. Before you decide, write out in `rationale` at least two ways the value could be incorrect, drawing on:
  - what the extraction rules say this field should be and where it belongs in the taxonomy hierarchy,
  - what the evidence_text actually contains,
  - what the PDF page shows (you have the PDF attached).

Step 2 — Pick an `issue_category` from this fixed list:
  - `wrong_row`               : value pulled from a sibling row/line item
  - `wrong_column`            : value pulled from a sibling column (e.g. cumulative-to-date instead of this-period)
  - `wrong_taxonomy_concept`  : the value itself is right, but it's tagged to the wrong concept (e.g. a subcontractor name extracted as prime_contractor)
  - `wrong_hierarchy_position`: the right concept, but nested under the wrong parent or at the wrong depth
  - `wrong_period`            : a cumulative/to-date figure used where a this-period figure belongs, or vice versa
  - `hallucinated`            : value does not appear on the page at all
  - `partial_match`           : value is a fragment / truncation / sign error
  - `formatting_drift`        : right value, wrong format (date format, missing normalization)
  - `actually_correct`        : value is correct; the policy flag was wrong
  - `unclear_from_evidence`   : cannot decide from the attached page + evidence

Step 3 — Score `trust`:
  - 0.0 = you are confident the extraction is wrong
  - 0.5 = unsure
  - 1.0 = you agree with the extractor

Step 4 — If you set `trust = 1.0` AND `policy_judgment` is `verified-suspicious`, you must explicitly justify in `rationale` why you override the policy flag. The default for verified-suspicious is to lean toward the policy heuristic being right.

## Output schema — STRICT JSON, no markdown fences

{{
  "field_path": "{field_path}",
  "trust": <float 0.0-1.0>,
  "issue_category": "<one of the categories above>",
  "rationale": "<one short paragraph; include the Step 1 enumeration>",
  "suggested_value": <string or null — only when trust < 0.5 and you are confident>,
  "evidence_quote": "<verbatim text from the page that supports your verdict, or null>"
}}

If you include `evidence_quote`, it MUST appear verbatim somewhere on the attached PDF page. Do not invent or paraphrase. If you cannot supply a verbatim quote, set it to null.
"""


@dataclass
class AIReview:
    field_path: str
    trust: Optional[float]
    issue_category: Optional[str]
    rationale: str
    suggested_value: Optional[str]
    evidence_quote: Optional[str]
    evidence_quote_verified: Optional[bool]
    model: str
    duration_seconds: float
    error: Optional[str] = None
    policy_judgment: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_path": self.field_path,
            "trust": self.trust,
            "issue_category": self.issue_category,
            "rationale": self.rationale,
            "suggested_value": self.suggested_value,
            "evidence_quote": self.evidence_quote,
            "evidence_quote_verified": self.evidence_quote_verified,
            "model": self.model,
            "duration_seconds": round(self.duration_seconds, 3),
            "error": self.error,
            "policy_judgment": self.policy_judgment,
        }


class AIReviewerError(RuntimeError):
    pass


class AIReviewer:
    def __init__(self, config: PipelineConfig, *, client: Any = None):
        self.config = config
        if client is not None:
            self._client = client
        else:
            if not GENAI_AVAILABLE:
                raise AIReviewerError("google-genai is required (pip install google-genai).")
            if not config.gemini_api_key:
                raise AIReviewerError("No Gemini API key configured (see config.py).")
            self._client = genai.Client(api_key=config.gemini_api_key)

    def flagged_targets(self, review: ReviewOutput) -> List[FieldJudgment]:
        return [j for j in review.judgments if j.judgment in CRITIC_TARGETS]

    def critique_document(self, pdf_path, extraction_rules: str, review: ReviewOutput, *, max_concurrent: Optional[int] = None) -> List[AIReview]:
        targets = self.flagged_targets(review)
        if not targets:
            return []
        try:
            pdf_bytes = pdf_path.read_bytes()
        except OSError as exc:
            return [self._error_review(j, f"pdf_not_readable: {exc}") for j in targets]

        concurrent = max(1, max_concurrent or self.config.ai_reviewer_max_concurrent)
        if concurrent == 1 or len(targets) == 1:
            return [self._critique_one_safe(pdf_bytes, extraction_rules, j) for j in targets]

        out: List[Optional[AIReview]] = [None] * len(targets)
        with ThreadPoolExecutor(max_workers=concurrent) as pool:
            futures = {pool.submit(self._critique_one_safe, pdf_bytes, extraction_rules, j): i for i, j in enumerate(targets)}
            for fut in as_completed(futures):
                out[futures[fut]] = fut.result()
        return out  # type: ignore[return-value]

    def _critique_one_safe(self, pdf_bytes: bytes, extraction_rules: str, judgment: FieldJudgment) -> AIReview:
        try:
            return self._critique_one_inner(pdf_bytes, extraction_rules, judgment)
        except Exception as exc:
            logger.exception("AI critic failed on field %s", judgment.field)
            return self._error_review(judgment, f"{type(exc).__name__}: {exc}")

    def _critique_one_inner(self, pdf_bytes: bytes, extraction_rules: str, judgment: FieldJudgment) -> AIReview:
        prompt_text = CRITIC_PROMPT.format(
            field_path=judgment.field,
            policy_judgment=judgment.judgment,
            extracted_value=json.dumps(judgment.extracted_value, ensure_ascii=False, default=str),
            self_confidence="null" if judgment.confidence is None else f"{judgment.confidence:.4f}",
            alignment_status=judgment.alignment_status,
            evidence_text=json.dumps(judgment.evidence_text, ensure_ascii=False, default=str),
            page="null" if judgment.page is None else judgment.page,
            extraction_rules=extraction_rules,
        )
        contents: Sequence[Any] = [types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"), prompt_text]
        gen_cfg = types.GenerateContentConfig(temperature=0.0, top_p=0.0, max_output_tokens=1024)

        start = time.time()
        response = self._client.models.generate_content(model=self.config.ai_reviewer_model, contents=contents, config=gen_cfg)
        duration = time.time() - start

        raw_text = getattr(response, "text", None) or ""
        if not raw_text.strip():
            return self._error_review(judgment, "empty_response", duration=duration)
        try:
            parsed = self._parse_json(raw_text)
        except json.JSONDecodeError as exc:
            return self._error_review(judgment, f"json_parse_error: {exc}", duration=duration)
        return self._build_review(judgment, parsed, duration)

    def _build_review(self, judgment: FieldJudgment, parsed: Dict[str, Any], duration: float) -> AIReview:
        trust = self._coerce_trust(parsed.get("trust"))
        issue_category = parsed.get("issue_category")
        if issue_category not in ISSUE_CATEGORIES:
            issue_category = None
        rationale = str(parsed.get("rationale") or "").strip()
        suggested_value = parsed.get("suggested_value")
        if isinstance(suggested_value, (dict, list)):
            suggested_value = json.dumps(suggested_value, ensure_ascii=False, default=str)
        elif suggested_value is not None:
            suggested_value = str(suggested_value)
        evidence_quote = parsed.get("evidence_quote")
        evidence_quote = str(evidence_quote) if evidence_quote is not None else None
        evidence_quote_verified = self._verify_evidence_quote(evidence_quote, judgment.evidence_text)

        return AIReview(
            field_path=judgment.field, trust=trust, issue_category=issue_category, rationale=rationale,
            suggested_value=suggested_value, evidence_quote=evidence_quote, evidence_quote_verified=evidence_quote_verified,
            model=self.config.ai_reviewer_model, duration_seconds=duration, policy_judgment=judgment.judgment,
        )

    @staticmethod
    def _coerce_trust(raw: Any) -> Optional[float]:
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(1.0, value))

    @staticmethod
    def _verify_evidence_quote(quote: Optional[str], evidence_text: Optional[str]) -> Optional[bool]:
        if not quote or not quote.strip() or not evidence_text:
            return None
        return quote.strip() in evidence_text

    def _error_review(self, judgment: FieldJudgment, error: str, *, duration: float = 0.0) -> AIReview:
        return AIReview(
            field_path=judgment.field, trust=None, issue_category=None, rationale="", suggested_value=None,
            evidence_quote=None, evidence_quote_verified=None, model=self.config.ai_reviewer_model,
            duration_seconds=duration, error=error, policy_judgment=judgment.judgment,
        )

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                raise
            obj = json.loads(m.group(0))
        if not isinstance(obj, dict):
            raise json.JSONDecodeError("Expected object", text, 0)
        return obj
