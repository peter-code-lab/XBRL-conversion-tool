"""Summarizer: cross-document aggregation.

Adapted from Self_Improving_Pipeline/summarizer.py. Simplified relative to the
original: that version grouped signals by (classifier_code, field_path,
judgment) because it supported 30+ utility-bill classifiers. This tool has
exactly one document type, so signals are grouped by (field_path, judgment)
directly. Oracle-free by design — this never reads ground truth; it only
aggregates the extractor's own self-reported confidence/alignment judgments
across many documents.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import review_log
from config import PipelineConfig
from reviewer import FLAG_WORTHY

logger = logging.getLogger(__name__)


@dataclass
class FieldSignal:
    field_path: str
    judgment: str
    n_files_total: int
    n_flagged: int
    flag_rate: float
    sample_evidence: List[str] = field(default_factory=list)
    sample_extracted_values: List[str] = field(default_factory=list)
    source: str = "policy"  # "policy" | "ai_critic"
    sample_rationales: List[str] = field(default_factory=list)
    issue_category_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class SummaryReport:
    n_files_total: int
    signals: List[FieldSignal] = field(default_factory=list)
    ai_critic_signals: List[FieldSignal] = field(default_factory=list)


class Summarizer:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def aggregate(self) -> SummaryReport:
        entries = review_log.read_all_list(self.config.review_log_path)
        n_files_total = len(entries)

        flagged_docs: Dict[Tuple[str, str], set] = defaultdict(set)
        sample_evidence: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        sample_values: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        for entry in entries:
            doc = entry.get("pdf") or entry.get("document_id") or ""
            seen_in_doc = set()
            for j in entry.get("judgments", []):
                judgment = j.get("judgment")
                if judgment not in FLAG_WORTHY:
                    continue
                key = (j.get("field", "?"), judgment)
                if key in seen_in_doc:
                    continue
                seen_in_doc.add(key)
                flagged_docs[key].add(doc)
                if len(sample_evidence[key]) < 5 and j.get("evidence_text"):
                    sample_evidence[key].append(str(j["evidence_text"]))
                if len(sample_values[key]) < 5 and j.get("extracted_value") is not None:
                    sample_values[key].append(str(j["extracted_value"]))

        signals: List[FieldSignal] = []
        for (fld, judgment), docs in flagged_docs.items():
            if n_files_total < self.config.min_files:
                continue
            n_flagged = len(docs)
            rate = n_flagged / n_files_total
            if rate < self.config.rate_threshold:
                continue
            signals.append(FieldSignal(
                field_path=fld,
                judgment=judgment,
                n_files_total=n_files_total,
                n_flagged=n_flagged,
                flag_rate=rate,
                sample_evidence=sample_evidence[(fld, judgment)],
                sample_extracted_values=sample_values[(fld, judgment)],
            ))
        signals.sort(key=lambda s: (-s.flag_rate, s.field_path))

        ai_signals = self._aggregate_ai_critic(entries, n_files_total)

        return SummaryReport(n_files_total=n_files_total, signals=signals, ai_critic_signals=ai_signals)

    def _aggregate_ai_critic(self, entries: List[Dict], n_files_total: int) -> List[FieldSignal]:
        threshold = self.config.ai_reviewer_trust_threshold
        ai_flagged: Dict[str, set] = defaultdict(set)
        sample_rationales: Dict[str, List[str]] = defaultdict(list)
        sample_quotes: Dict[str, List[str]] = defaultdict(list)
        sample_values: Dict[str, List[str]] = defaultdict(list)
        issue_counts: Dict[str, Counter] = defaultdict(Counter)

        for entry in entries:
            doc = entry.get("pdf") or entry.get("document_id") or ""
            seen_in_doc = set()
            for j in entry.get("judgments", []):
                ai = j.get("ai_review") if isinstance(j, dict) else None
                if not isinstance(ai, dict) or ai.get("error"):
                    continue
                trust = ai.get("trust")
                try:
                    trust_f = float(trust) if trust is not None else None
                except (TypeError, ValueError):
                    trust_f = None
                if trust_f is None or trust_f >= threshold:
                    continue
                fld = j.get("field", "?")
                if fld in seen_in_doc:
                    continue
                seen_in_doc.add(fld)
                ai_flagged[fld].add(doc)
                if len(sample_rationales[fld]) < 3 and ai.get("rationale"):
                    sample_rationales[fld].append(str(ai["rationale"])[:240])
                if ai.get("evidence_quote") and len(sample_quotes[fld]) < 3:
                    sample_quotes[fld].append(str(ai["evidence_quote"]))
                if j.get("extracted_value") is not None and len(sample_values[fld]) < 3:
                    sample_values[fld].append(str(j["extracted_value"]))
                cat = ai.get("issue_category")
                if cat:
                    issue_counts[fld][str(cat)] += 1

        signals: List[FieldSignal] = []
        for fld, docs in ai_flagged.items():
            if n_files_total < self.config.min_files:
                continue
            n_flagged = len(docs)
            rate = n_flagged / n_files_total if n_files_total else 0.0
            if rate < self.config.rate_threshold:
                continue
            signals.append(FieldSignal(
                field_path=fld,
                judgment="ai-critic-flagged",
                n_files_total=n_files_total,
                n_flagged=n_flagged,
                flag_rate=rate,
                sample_evidence=sample_quotes[fld],
                sample_extracted_values=sample_values[fld],
                source="ai_critic",
                sample_rationales=sample_rationales[fld],
                issue_category_counts=dict(issue_counts[fld]),
            ))
        signals.sort(key=lambda s: (-s.flag_rate, s.field_path))
        return signals

    def write_suggested_updates(self, report: SummaryReport) -> Path:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out = self.config.runs_dir / f"suggested_updates_{ts}.md"
        lines: List[str] = [f"# Suggested Rule Updates ({ts})", "", f"Total documents seen: {report.n_files_total}", ""]

        if report.n_files_total < self.config.min_files:
            lines.append(f"Below min_files={self.config.min_files} — no signal can trigger yet.")
        elif not report.signals:
            lines.append(
                f"No signals crossed the trigger thresholds "
                f"(rate >= {self.config.rate_threshold:.0%}, n_files >= {self.config.min_files}). Nothing to suggest."
            )
        else:
            for sig in report.signals:
                lines.append(f"### Field: `{sig.field_path}`")
                lines.append(f"- Dominant judgment: **{sig.judgment}**")
                lines.append(f"- Flag rate: {sig.n_flagged}/{sig.n_files_total} = {sig.flag_rate:.0%}")
                if sig.sample_evidence:
                    lines.append("- Sample evidence_text values:")
                    lines += [f"  - {s!r}" for s in sig.sample_evidence]
                if sig.sample_extracted_values:
                    lines.append("- Sample extracted values:")
                    lines += [f"  - {s!r}" for s in sig.sample_extracted_values]
                lines.append("")

        if report.ai_critic_signals:
            lines += ["", "---", "", "# AI-critic-confirmed errors", ""]
            for sig in report.ai_critic_signals:
                lines.append(f"### Field: `{sig.field_path}`")
                lines.append(f"- Critic flag rate: {sig.n_flagged}/{sig.n_files_total} = {sig.flag_rate:.0%}")
                if sig.issue_category_counts:
                    cats = ", ".join(f"{k}×{v}" for k, v in sorted(sig.issue_category_counts.items(), key=lambda kv: -kv[1]))
                    lines.append(f"- Issue categories: {cats}")
                if sig.sample_rationales:
                    lines.append("- Sample critic rationales:")
                    lines += [f"  - {r!r}" for r in sig.sample_rationales]
                lines.append("")

        out.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Wrote suggested updates: %s", out)
        return out
