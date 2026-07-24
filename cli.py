"""CLI front door for the standalone XBRL conversion tool.

Subcommands:
  run <pdf> [--ai-review]           Extract + tag + review one PDF, log it, print the tagged output.
  run-batch <glob> [--ai-review]    Same, applied to many files.
  summarize                        Cross-document aggregation; writes runs/suggested_updates_<ts>.md.
  propose                          Aggregate + call the Proposer for a candidate prompt rewrite.
  apply --new-prompt <file>        Human-invoked only. Backs up and overwrites the live prompt file.
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import review_log
from ai_reviewer import AIReviewer
from config import PipelineConfig
from extractor import Extractor
from proposer import Proposer
from prompt_writer import apply_prompt_update
from reviewer import Reviewer
from summarizer import Summarizer
from taxonomy_tagger import TaxonomyTagger

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("cli")


def cmd_run(args: argparse.Namespace, config: PipelineConfig) -> None:
    pdf_path = Path(args.pdf)
    extractor = Extractor(config)
    reviewer = Reviewer(config)
    tagger = TaxonomyTagger(config.taxonomy_path)

    result = extractor.extract(pdf_path)
    review = reviewer.review(result.extraction, document_id=pdf_path.name)

    ai_reviews = None
    if args.ai_review:
        rules_text = config.prompt_path.read_text(encoding="utf-8")
        ai_reviewer = AIReviewer(config)
        ai_reviews = ai_reviewer.critique_document(pdf_path, rules_text, review)

    review_log.append(config.review_log_path, pdf_path, review, ai_reviews=ai_reviews)

    tagged = tagger.tag(result.extraction, source_document=str(pdf_path))
    tagged_dict = tagged.to_dict()
    print(json.dumps(tagged_dict, indent=2, ensure_ascii=False, default=str))

    # Persist every extraction's tagged output, timestamped so re-running the
    # same PDF accumulates history rather than overwriting the prior result —
    # previously this was only printed to stdout and lost once the terminal
    # scrolled past it.
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = config.extractions_dir / f"{ts}_{pdf_path.stem}.json"
    out_path.write_text(json.dumps(tagged_dict, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nExtraction saved: {out_path}", file=sys.stderr)

    flagged = review.flag_worthy()
    if flagged:
        print(f"\n{len(flagged)} flag-worthy field(s):", file=sys.stderr)
        for j in flagged:
            print(f"  - {j.field}: {j.judgment} (confidence={j.confidence})", file=sys.stderr)


def cmd_run_batch(args: argparse.Namespace, config: PipelineConfig) -> None:
    paths = sorted(Path(p) for p in globmod.glob(args.glob))
    if not paths:
        logger.warning("No files matched %r", args.glob)
        return
    for pdf_path in paths:
        try:
            ns = argparse.Namespace(pdf=str(pdf_path), ai_review=args.ai_review)
            cmd_run(ns, config)
        except Exception as exc:
            logger.error("Failed on %s: %s", pdf_path, exc)
            if not args.continue_on_error:
                raise


def cmd_summarize(args: argparse.Namespace, config: PipelineConfig) -> None:
    summarizer = Summarizer(config)
    report = summarizer.aggregate()
    out = summarizer.write_suggested_updates(report)
    print(f"Total documents: {report.n_files_total}")
    print(f"Signals: {len(report.signals)} policy, {len(report.ai_critic_signals)} AI-critic")
    print(f"Report: {out}")


def cmd_propose(args: argparse.Namespace, config: PipelineConfig) -> None:
    summarizer = Summarizer(config)
    report = summarizer.aggregate()
    all_signals = report.signals + report.ai_critic_signals
    if not all_signals:
        print("No signals — nothing to propose.")
        return
    proposer = Proposer(config)
    current_prompt = config.prompt_path.read_text(encoding="utf-8")
    output = proposer.propose(all_signals, current_prompt)
    candidate_path, meta_path = proposer.write_candidate(output)
    print(f"Reasoning: {output.reasoning}")
    if candidate_path:
        print(f"Candidate written: {candidate_path}")
    else:
        print(f"No candidate written (rejected_for={output.rejected_for})")
    print(f"Meta: {meta_path}")


def cmd_apply(args: argparse.Namespace, config: PipelineConfig) -> None:
    """Human-invoked only — nothing else in this tool calls this function."""
    new_prompt_path = Path(args.new_prompt)
    if not new_prompt_path.exists():
        raise FileNotFoundError(new_prompt_path)
    new_text = new_prompt_path.read_text(encoding="utf-8")
    backup = apply_prompt_update(config.prompt_path, new_text)
    print(f"Applied. Backup of previous prompt: {backup}")


def main() -> None:
    parser = argparse.ArgumentParser(description="XBRL conversion tool CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Extract + tag + review one PDF")
    p_run.add_argument("pdf")
    p_run.add_argument("--ai-review", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_batch = sub.add_parser("run-batch", help="Extract + tag + review many PDFs")
    p_batch.add_argument("glob")
    p_batch.add_argument("--ai-review", action="store_true")
    p_batch.add_argument("--continue-on-error", action="store_true")
    p_batch.set_defaults(func=cmd_run_batch)

    p_sum = sub.add_parser("summarize", help="Cross-document aggregation")
    p_sum.set_defaults(func=cmd_summarize)

    p_prop = sub.add_parser("propose", help="Aggregate + draft a candidate prompt rewrite")
    p_prop.set_defaults(func=cmd_propose)

    p_apply = sub.add_parser("apply", help="Human-invoked: apply a reviewed candidate prompt")
    p_apply.add_argument("--new-prompt", required=True)
    p_apply.set_defaults(func=cmd_apply)

    args = parser.parse_args()
    config = PipelineConfig()
    args.func(args, config)


if __name__ == "__main__":
    main()
