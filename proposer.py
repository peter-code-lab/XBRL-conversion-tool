"""Proposer: oracle-free Gemini call that drafts a candidate replacement extraction prompt.

Adapted from Self_Improving_Pipeline/proposer.py. Simplified relative to the
original: that version drafted a replacement for one classifier's *section*
within a larger multi-classifier Prompt_List.txt. This tool has exactly one
prompt file, so the Proposer drafts a replacement for the whole file.

Still oracle-free by design — see PROPOSAL_PROMPT below. It never sees ground
truth, only the Summarizer's aggregated flag-rate signals.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from config import PipelineConfig
from summarizer import FieldSignal

logger = logging.getLogger(__name__)


PROPOSAL_PROMPT = """You are an expert at surgically rewriting document-extraction rule blocks.

## Task
You are given:
  - The current extraction prompt (rules + JSON schema the extractor is using).
  - A cross-document flag report: fields where the extractor's self-confidence + bbox-alignment signal indicates a recurring problem across many documents.
  - For each flagged field: dominant judgment category, % of documents affected, and verbatim sample evidence_text and extracted-value strings.

You do NOT have ground truth. You have only what the model itself reported about its own output across many documents. Your job is to update the extraction prompt so the extractor stops producing the flagged failure pattern.

## Current Extraction Prompt
```
{current_prompt}
```

## Flagged Fields ({n_signals} total)
{flagged_fields_block}

## Judgment categories you'll see
  - `verified-suspicious`   : the model wrote a value the bbox pass could not relocate on the page. Likely hallucination, mislocation, or wrong-field pickup.
  - `low-confidence`        : the model wrote a value AND the bbox aligned, but the model's own self-confidence was below threshold.
  - `unverifiable-missing`  : the model returned no value for this field.
  - `ai-critic-flagged`     : a second-opinion LLM critic disagreed with the extracted value or its taxonomy tagging.

## CRITICAL RULES — READ BEFORE REWRITING

**JSON schema preservation:**
- DO NOT add, remove, or rename any fields in the JSON schema, and DO NOT change its nesting — the nesting mirrors this tool's taxonomy hierarchy and must stay intact.
- ONLY modify extraction instructions/rules BELOW the JSON schema.

**Surgical edits only — no bloat:**
- Touch ONLY the rules affecting the flagged fields. Leave everything else alone.
- Consolidate redundant rules rather than appending duplicates.
- The rewritten prompt MUST NOT grow by more than {max_line_delta} lines and MUST NOT exceed {max_total_lines} lines total. Generic rewrites will be rejected.

**WORKED EXAMPLES ARE MANDATORY for verified-suspicious fields:**
Quote the literal sample evidence_text and extracted-value strings from the flagged-fields block above. State the expected output value. Generic advice like "be more careful" is NOT acceptable — quote actual document strings from the samples.

**You only have what the model self-reported:**
You do NOT have ground truth. Phrase your rules in terms of "the document should show X next to label Y," not "the correct answer is Z." Where the samples are inconsistent or insufficient, prefer narrowing the extraction context over stating an absolute value.

## Response Format
Return ONLY valid JSON:
{{
  "reasoning": "Brief explanation of what you changed and why.",
  "improved_prompt": "The full rewritten extraction prompt, or null if no rewrite is justified given the signal strength."
}}

If the flagged fields look like noise or the rewrite would be bigger than the bloat limits, set `improved_prompt` to null and explain in `reasoning`.
"""


@dataclass
class ProposerOutput:
    candidate_prompt: Optional[str]
    reasoning: str
    rejected_for: Optional[str]
    duration_seconds: float
    token_usage: Dict[str, int] = field(default_factory=dict)
    signals_consumed: List[Dict[str, Any]] = field(default_factory=list)


class ProposerError(RuntimeError):
    pass


class Proposer:
    def __init__(self, config: PipelineConfig):
        self.config = config
        if not GENAI_AVAILABLE:
            raise ProposerError("google-genai is required (pip install google-genai).")
        if not config.gemini_api_key:
            raise ProposerError("No Gemini API key configured (see config.py).")
        self._client = genai.Client(api_key=config.gemini_api_key)

    def propose(self, signals: List[FieldSignal], current_prompt: str) -> ProposerOutput:
        if not signals:
            return ProposerOutput(candidate_prompt=None, reasoning="No signals provided; nothing to propose.", rejected_for="null", duration_seconds=0.0)

        flagged_block = self._build_flagged_fields_block(signals)
        prompt_text = PROPOSAL_PROMPT.format(
            current_prompt=current_prompt,
            n_signals=len(signals),
            flagged_fields_block=flagged_block,
            max_line_delta=self.config.proposer_max_line_delta,
            max_total_lines=self.config.proposer_max_total_lines,
        )
        gen_config = types.GenerateContentConfig(temperature=0.0, top_p=0.0, max_output_tokens=16384)

        start = time.time()
        response = self._client.models.generate_content(model=self.config.proposer_model, contents=[prompt_text], config=gen_config)
        duration = time.time() - start

        if not getattr(response, "text", None):
            raise ProposerError("Proposer got empty response from Gemini")

        parsed = self._parse_json(response.text)
        candidate = parsed.get("improved_prompt")
        reasoning = str(parsed.get("reasoning", ""))
        rejected_for = None

        if candidate is None or (isinstance(candidate, str) and not candidate.strip()):
            candidate = None
            rejected_for = "null"
        else:
            ok, why = self._bloat_guard(current_prompt, candidate)
            if not ok:
                logger.warning("Rejecting candidate: %s", why)
                rejected_for = f"bloat: {why}"
                candidate = None

        return ProposerOutput(
            candidate_prompt=candidate, reasoning=reasoning, rejected_for=rejected_for, duration_seconds=duration,
            token_usage=self._extract_tokens(response), signals_consumed=[self._signal_to_dict(s) for s in signals],
        )

    def write_candidate(self, output: ProposerOutput) -> Tuple[Optional[Path], Path]:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        stem = f"{ts}_candidate"
        meta_path = self.config.candidates_dir / f"{stem}.meta.json"
        candidate_path: Optional[Path] = None
        if output.candidate_prompt is not None:
            candidate_path = self.config.candidates_dir / f"{stem}.txt"
            candidate_path.write_text(output.candidate_prompt, encoding="utf-8")
        meta = {
            "candidate_path": str(candidate_path) if candidate_path else None,
            "rejected_for": output.rejected_for,
            "reasoning": output.reasoning,
            "duration_seconds": round(output.duration_seconds, 3),
            "token_usage": output.token_usage,
            "signals_consumed": output.signals_consumed,
        }
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return candidate_path, meta_path

    @staticmethod
    def _build_flagged_fields_block(signals: List[FieldSignal]) -> str:
        lines: List[str] = []
        for sig in signals:
            lines.append(f"### `{sig.field_path}` — {sig.judgment}")
            lines.append(f"  - Flag rate: {sig.n_flagged}/{sig.n_files_total} = {sig.flag_rate:.0%}")
            if sig.sample_evidence:
                lines.append("  - Sample evidence_text values (verbatim):")
                lines += [f"    - {s!r}" for s in sig.sample_evidence]
            if sig.sample_extracted_values:
                lines.append("  - Sample extracted values:")
                lines += [f"    - {s!r}" for s in sig.sample_extracted_values]
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _signal_to_dict(s: FieldSignal) -> Dict[str, Any]:
        return {
            "field_path": s.field_path, "judgment": s.judgment, "n_files_total": s.n_files_total,
            "n_flagged": s.n_flagged, "flag_rate": round(s.flag_rate, 4),
            "sample_evidence": s.sample_evidence, "sample_extracted_values": s.sample_extracted_values,
        }

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                raise
            return json.loads(m.group(0))

    @staticmethod
    def _extract_tokens(response) -> Dict[str, int]:
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            return {}
        return {
            "input_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
            "output_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
            "total_tokens": int(getattr(usage, "total_token_count", 0) or 0),
        }

    def _bloat_guard(self, original: str, candidate: str) -> Tuple[bool, str]:
        orig_lines = original.count("\n") + 1
        new_lines = candidate.count("\n") + 1
        delta = new_lines - orig_lines
        if delta > self.config.proposer_max_line_delta:
            return False, f"line_delta={delta} > {self.config.proposer_max_line_delta}"
        if new_lines > self.config.proposer_max_total_lines:
            return False, f"total_lines={new_lines} > {self.config.proposer_max_total_lines}"
        return True, ""
