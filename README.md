# XBRL Conversion Tool

Extracts structured data from construction contractor payment-application PDFs (Caltrans, Iowa DOT, TxDOT SiteManager, and similar state DOT reporting formats), tags the extracted values against a bespoke XBRL-style taxonomy, and emits an XBRL instance document (`xbrli:xbrl` XML) per PDF — alongside a JSON intermediate that mirrors the same concept + Presentation Linkbase hierarchy. See "Scope" below for what "XBRL" does and doesn't mean here.

This is a **standalone** project. It does not depend on any other repo at runtime — it calls Gemini directly for extraction and bbox verification, and reimplements (adapted, not imported) the self-correcting review loop pattern from a sibling project (`Self_Improving_Pipeline`, built for an unrelated utility-bill extraction system). Nothing here talks to that system's backend.

## Architecture

Both entry points — the CLI and the Flask web UI — run the same seven stages. They orchestrate them **independently**, so a pipeline change has to be made in `cli.py` *and* `app.py`.

```
PDF
 1. Extractor          extractor.py        one Gemini call: all fields + a parallel
                                           self-reported confidence tree
 2. Bbox verifier      extractor.py        one Gemini call per page: render at 150 DPI,
                                           ask where the evidence_text appears
                                           → EXACT / NO_MATCH / MISSING
 3. Reviewer           reviewer.py         no LLM. confidence × alignment → one of
                                           verified-correct / low-confidence /
                                           verified-suspicious / unverifiable-missing
 4. AI critic          ai_reviewer.py      flagged fields only, one call each, in
                                           parallel. Adversarial second opinion:
                                           trust score + issue category
 5. Review log         review_log.py       one JSONL line per document run
 6. Taxonomy Tagger    taxonomy_tagger.py  values → TaggedFact list, each carrying its
                                           concept and parent concept
 7. XBRL Serializer    xbrl_serializer.py  TaggedDocument → xbrli:xbrl instance XML
```

Outputs per run: `runs/extractions/<ts>_<name>.json`, `runs/xbrl/<ts>_<name>.xml`, and the unparsed model text in `runs/raw_responses/` (kept because cheap-model malformations are the main source of null extractions).

Cross-document self-correction loop, run manually after volume accumulates:

```
review_log.jsonl → Summarizer   summarizer.py     aggregate flag rates per field
                 → Proposer     proposer.py       draft a candidate prompt rewrite
                 → HUMAN REVIEW                   nothing auto-applies
                 → apply        prompt_writer.py  backs up, then overwrites the prompt
```

### The taxonomy is the single source of truth

`taxonomy/construction_payment_taxonomy.json` defines three things everything else derives from:

- `concepts` — the fields plus abstract group headers, each with a `parent`
- `presentation_linkbase` — the display hierarchy
- `extraction_field_map` — dotted JSON path (`agreement_parties.prime_contractor`) → concept ID (`PrimeContractor`). **This is the join.** A field the extractor returns that is not in this map is silently dropped by the tagger.

`taxonomy/construction_payment_taxonomy.xsd` and `_pre.xml` are **generated derivatives**, not hand-edited:

```
python taxonomy/generate_taxonomy_artifacts.py
```

Run manually — never during extraction. Edit the JSON, re-run this, and commit all three together, or the `cpat_*` element IDs and linkbase locators drift out of sync with the instance documents that reference them.

## Setup

```
pip install -r requirements.txt
```

Set a Gemini API key, either as an environment variable or in a local `.env` file (not committed — see `.gitignore`):

```
export GEMINI_API_KEY=...
```

Input PDFs under `samples/` are gitignored — they are third-party agency documents, not project source.

## Running it

### Web UI

```
python app.py                     # http://127.0.0.1:5000
```

A drop zone that runs the full pipeline on one PDF and shows the per-field judgment table, the AI critic's verdict on anything flagged, the generated XBRL XML with a download link, and a self-correction loop status widget. The critic always runs on flagged fields here — unlike the CLI, there is no flag to remember.

Note: macOS binds port 5000 to AirPlay Receiver by default. To use another port:

```
python -c "from app import app; app.run(port=5001)"
```

### CLI

```
# Single PDF — extract, tag, review, log, write JSON + XBRL XML
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
- **XBRL output**: a real `xbrli:xbrl` instance document with a `schemaRef` to the generated schema, one shared context and unit, and `contextRef`/`unitRef`/`decimals` on each fact. It has **not** been validated with Arelle or any other XBRL processor, the entity identifier scheme is a placeholder rather than a real EIN/CIK, and there is no support for multiple contexts, dimensions, or footnotes. Well-formed and correctly shaped, not certified conformant.
- **Self-correction loop**: fully implemented and generic, but has never fired. See below.

## Known limitations in the quality signal

Worth understanding before trusting the flags or tuning the loop. Measured over the current review log (58 distinct documents):

- **The confidence axis is inert.** Of 432 logged confidence values, 425 are ≥ 0.9 (312 of them exactly 1.0), despite the prompt explicitly asking the model not to cluster at the extremes. Nothing has ever fallen below `tau_low = 0.5`, so the `low-confidence` judgment has never been produced. In practice the bbox alignment status is the *only* axis doing work.
- **The one live signal is mostly false positives.** Of the `NO_MATCH` flags the AI critic examined, it returned `actually_correct` on 32 of 35 and a genuine error on 3. Treat `verified-suspicious` as "worth a look", not "wrong".
- **Infrastructure failure is indistinguishable from a finding.** When the bbox retry exhausts (3 attempts), that page's fields are marked `NO_MATCH`, which the reviewer reads as `verified-suspicious`. An API timeout and a real extraction error produce identical data.
- **`rate_threshold = 0.6` is unreachable.** The highest observed per-field flag rate is 21%. No signal has ever crossed the gate, so `runs/candidates/` is empty and `apply` has never run. Lowering this threshold before improving flag precision would feed mostly-false-positive signal into a Gemini call that rewrites the extraction prompt.
- **Document identity is inconsistent across entry points.** `app.py` logs a bare uploaded filename while `cli.py` logs a full path, so the same PDF processed both ways counts as two documents in the summarizer's denominator.

## What's adapted from `Self_Improving_Pipeline`, and what's new

| Module | Status |
|---|---|
| `reviewer.py`, `review_log.py`, `prompt_writer.py` | Ported near-verbatim — already schema-agnostic |
| `summarizer.py`, `proposer.py` | Adapted — simplified from multi-classifier to this tool's single prompt file |
| `ai_reviewer.py` | Adapted — rewritten prompt + two new issue categories (`wrong_taxonomy_concept`, `wrong_hierarchy_position`) for tagging-specific errors the source pipeline never had to detect |
| `extractor.py` | Adapted from two source files (`Utility_Bill_Extractor_Local.py` + `auto_bbox.py`) into one standalone module with no Java backend dependency |
| `taxonomy_tagger.py` | New — nothing in the source pipeline does this |
| `xbrl_serializer.py` | New |
| `taxonomy/generate_taxonomy_artifacts.py` | New |
| `taxonomy/construction_payment_taxonomy.json` | New |
| `app.py` | New — Flask diagnostic UI over the same pipeline |
