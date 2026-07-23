# XBRL Conversion Tool

Extracts structured data from construction contractor payment-application PDFs (Caltrans, Iowa DOT, TxDOT SiteManager, and similar state DOT reporting formats) and tags the extracted values against a bespoke XBRL-style taxonomy — concepts nested per a Presentation Linkbase hierarchy. Output today is a JSON structure that mirrors that hierarchy (facts tagged to taxonomy concepts, each carrying its parent concept), not yet a fully XBRL-conformant XML instance document — see "Scope" below.

This is a **standalone** project. It does not depend on any other repo at runtime — it calls Gemini directly for classification, extraction, and confidence/bbox verification, and reimplements (adapted, not imported) the self-correcting review loop pattern from a sibling project (`Self_Improving_Pipeline`, built for an unrelated utility-bill extraction system). Nothing here talks to that system's backend.

## Architecture

```
PDF → Extractor (classify-free, direct Gemini call + confidence + bbox)
    → Taxonomy Tagger (maps extracted values → taxonomy concepts + hierarchy)
    → Reviewer (policy walker: 5 judgment categories, no LLM call)
    → AI Critic (targeted second-opinion LLM call, only on flagged fields)
    → Review Log (JSONL, one line per document)
    → Summarizer (cross-document aggregation — oracle-free, no ground truth)
    → Proposer (drafts a candidate rewrite of the extraction prompt)
    → Human Approval (nothing auto-applies — see prompt_writer.py)
```

## Setup

```
pip install -r requirements.txt
```

Set a Gemini API key, either as an environment variable or in a local `.env` file (not committed — see `.gitignore`):

```
export GEMINI_API_KEY=...
```

## Running it

```
# Single PDF — extracts, tags, reviews, logs, prints the tagged JSON
python cli.py run path/to/estimate.pdf

# Same, plus a targeted AI second-opinion on any flagged fields
python cli.py run path/to/estimate.pdf --ai-review

# Batch
python cli.py run-batch "path/to/*.pdf" --continue-on-error

# Cross-document aggregation (needs >= min_files documents logged first — see config.py)
python cli.py summarize

# Draft a candidate rewrite of the extraction prompt from aggregated signal
python cli.py propose

# Human-reviewed apply — nothing else in this tool calls this
python cli.py apply --new-prompt path/to/reviewed_candidate.txt
```

## Scope of this stage

- **Taxonomy**: bespoke and minimal — scoped to 4 fields (Prime Contractor, Application for Payment Number, Application for Payment Date, Work Completed This Application), not derived from a published standard. The AIA G702/G703 construction-billing forms are an unconfirmed lead worth checking before expanding this.
- **Output format**: a JSON intermediate mirroring the taxonomy's concept + Presentation Linkbase structure, not a full XBRL XML instance document (schema + linkbases + instance with contexts/units/entities). That's a deliberate scoping decision for this stage, not an oversight.
- **Self-correction loop**: fully implemented and generic, but inert until real volume accumulates — `min_files` (default 10) gates all signal generation, and there's no ground-truth validation step in this tool yet (the source pipeline's `validate.py` was never ported, since it depended on a Postgres store specific to the other system).

## What's adapted from `Self_Improving_Pipeline`, and what's new

| Module | Status |
|---|---|
| `reviewer.py`, `review_log.py`, `prompt_writer.py` | Ported near-verbatim — already schema-agnostic |
| `summarizer.py`, `proposer.py` | Adapted — simplified from multi-classifier to this tool's single prompt file |
| `ai_reviewer.py` | Adapted — rewritten prompt + two new issue categories (`wrong_taxonomy_concept`, `wrong_hierarchy_position`) for tagging-specific errors the source pipeline never had to detect |
| `extractor.py` | Adapted from two source files (`Utility_Bill_Extractor_Local.py` + `auto_bbox.py`) into one standalone module with no Java backend dependency |
| `taxonomy_tagger.py` | New — nothing in the source pipeline does this |
| `taxonomy/construction_payment_taxonomy.json` | New |
