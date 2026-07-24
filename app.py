"""Minimal local web UI: drag-and-drop a PDF, see full pipeline diagnostics --
not just the extracted values, but confidence, bbox-alignment match, the
policy reviewer's judgment, the AI critic's opinion (when it ran), and where
the self-correction loop currently stands.

Thin wrapper only -- reuses extractor.py / reviewer.py / taxonomy_tagger.py /
review_log.py / summarizer.py / ai_reviewer.py exactly as the CLI does. No
extraction or review logic lives here.

Run: .venv/bin/python app.py
Then open http://127.0.0.1:5000
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

import review_log
from ai_reviewer import AIReviewer
from config import PipelineConfig
from extractor import Extractor, ExtractorError
from reviewer import Reviewer
from summarizer import Summarizer
from taxonomy_tagger import TaxonomyTagger

app = Flask(__name__)
config = PipelineConfig()

PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>XBRL Conversion Tool</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 20px; color: #1a1a1a; }
  h1 { font-size: 1.4rem; }
  h2 { font-size: 1.05rem; margin-top: 32px; border-bottom: 1px solid #ddd; padding-bottom: 6px; }
  #drop {
    border: 2px dashed #999; border-radius: 10px; padding: 60px 20px;
    text-align: center; color: #666; cursor: pointer; transition: all 0.15s;
  }
  #drop.hover { border-color: #2b6cb0; background: #f0f6ff; color: #2b6cb0; }
  #status { margin-top: 16px; font-size: 0.95rem; }
  #status.error { color: #c0392b; }
  #status.ok { color: #2f855a; }
  input[type=file] { display: none; }

  .legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 0.8rem; color: #555; margin-top: 6px; }
  .legend span { display: flex; align-items: center; gap: 5px; }
  .legend .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }

  table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 0.88rem; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #eee; vertical-align: top; }
  th { color: #666; font-weight: 600; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.03em; cursor: help; }
  th small { display: block; text-transform: none; font-weight: 400; font-size: 0.68rem; color: #999; letter-spacing: 0; }

  .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.78rem; font-weight: 600; }
  .badge.verified-correct { background: #e6f4ea; color: #1e7e34; }
  .badge.low-confidence   { background: #fff4e5; color: #b7791f; }
  .badge.verified-suspicious, .badge.unverifiable-missing { background: #fdecea; color: #c0392b; }

  /* Confidence: deliberately "softer" styling -- outlined, not solid -- to
     visually signal this is a self-report, not a verified fact. */
  .conf-bar-wrap { display: flex; align-items: center; gap: 6px; }
  .conf-bar { width: 50px; height: 6px; background: #f2f2f2; border: 1px solid #ddd; border-radius: 3px; overflow: hidden; }
  .conf-bar-fill { height: 100%; opacity: 0.55; }
  .conf-good { background: #2f855a; } .conf-mid { background: #d69e2e; } .conf-bad { background: #c0392b; }
  .conf-label { color: #888; font-size: 0.8rem; }

  /* Page Match ("alignment_status"): solid, iconic -- this is the
     independently-verified signal, styled with more visual authority. */
  .verify-icon { font-weight: 700; font-size: 0.95rem; }
  .verify-EXACT { color: #1e7e34; }
  .verify-NO_MATCH { color: #c0392b; }
  .verify-MISSING { color: #888; }

  .critic-box { background: #fdf6ec; border: 1px solid #f0dfb8; border-radius: 6px; padding: 10px 12px; margin-top: 10px; font-size: 0.85rem; }
  .critic-box b { color: #975a16; }
  .critic-empty { color: #888; font-size: 0.88rem; font-style: italic; margin-top: 8px; }

  #loopStatus { background: #f7f9fc; border: 1px solid #dde3ec; border-radius: 8px; padding: 14px 16px; margin-top: 8px; }
  .progress-track { background: #e2e8f0; border-radius: 6px; height: 10px; overflow: hidden; margin: 6px 0; }
  .progress-fill { background: #2b6cb0; height: 100%; }
  .signal-yes { color: #c05621; font-weight: 600; }
  .signal-no { color: #2f855a; }
  .loop-row { margin-top: 6px; }

  details { margin-top: 20px; }
  summary { cursor: pointer; color: #666; font-size: 0.85rem; }
  pre {
    background: #1e1e1e; color: #d4d4d4; padding: 16px; border-radius: 8px;
    overflow-x: auto; font-size: 0.8rem; margin-top: 10px; white-space: pre-wrap; word-break: break-word;
  }
</style>
</head>
<body>
  <h1>XBRL Conversion Tool</h1>
  <p>Drop a construction-payment PDF below, or click to choose one.</p>
  <div id="drop">Drop PDF here</div>
  <input type="file" id="fileInput" accept="application/pdf">
  <div id="status"></div>

  <div id="results" style="display:none">
    <h2>Extracted fields</h2>
    <div class="legend">
      <span><span class="dot" style="background:#1e7e34"></span> verified-correct — page-matched AND self-confidence above threshold</span>
      <span><span class="dot" style="background:#b7791f"></span> low-confidence — page-matched, but the model itself wasn't sure</span>
      <span><span class="dot" style="background:#c0392b"></span> verified-suspicious — could NOT be located on the page (possible wrong value)</span>
      <span><span class="dot" style="background:#888"></span> unverifiable-missing — model returned no value at all</span>
    </div>
    <table id="fieldsTable"><thead>
      <tr>
        <th>Field</th>
        <th>Value</th>
        <th title="The model's OWN self-reported certainty, produced during extraction. Not independently checked -- treat as a soft signal only.">Confidence <small>(self-reported)</small></th>
        <th title="A SEPARATE Gemini call, given only the rendered page image, tried to actually locate this text. EXACT = found it. NO_MATCH = could not.">Verified on Page <small>(independent check)</small></th>
        <th title="The combined verdict: Page-verification result + confidence threshold.">Judgment</th>
      </tr>
    </thead><tbody></tbody></table>

    <h2>AI Critic</h2>
    <div id="criticSection"></div>

    <h2>Self-correction loop status</h2>
    <div id="loopStatus"></div>

    <details>
      <summary>Raw JSON</summary>
      <pre id="output"></pre>
    </details>
  </div>

<script>
const drop = document.getElementById('drop');
const fileInput = document.getElementById('fileInput');
const statusEl = document.getElementById('status');
const results = document.getElementById('results');
const output = document.getElementById('output');
const fieldsBody = document.querySelector('#fieldsTable tbody');
const criticSection = document.getElementById('criticSection');
const loopStatus = document.getElementById('loopStatus');

drop.addEventListener('click', () => fileInput.click());
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('hover'); });
drop.addEventListener('dragleave', () => drop.classList.remove('hover'));
drop.addEventListener('drop', e => {
  e.preventDefault();
  drop.classList.remove('hover');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', e => {
  if (e.target.files.length) handleFile(e.target.files[0]);
});

function confClass(c) {
  if (c === null || c === undefined) return 'conf-bad';
  if (c >= 0.8) return 'conf-good';
  if (c >= 0.5) return 'conf-mid';
  return 'conf-bad';
}

const VERIFY_ICON = { EXACT: '\\u2713', NO_MATCH: '\\u2717', MISSING: '\\u2013' };

function renderFields(judgments) {
  fieldsBody.innerHTML = '';
  for (const j of judgments) {
    const tr = document.createElement('tr');
    const conf = j.confidence;
    const confPct = conf === null || conf === undefined ? '?' : Math.round(conf * 100) + '%';
    const icon = VERIFY_ICON[j.alignment_status] || '?';
    tr.innerHTML = `
      <td>${j.field}</td>
      <td>${j.extracted_value === null ? '<i>N/A</i>' : String(j.extracted_value)}</td>
      <td><div class="conf-bar-wrap"><div class="conf-bar"><div class="conf-bar-fill ${confClass(conf)}" style="width:${conf ? conf*100 : 0}%"></div></div><span class="conf-label">${confPct}</span></div></td>
      <td class="verify-${j.alignment_status}"><span class="verify-icon">${icon}</span> ${j.alignment_status}</td>
      <td><span class="badge ${j.judgment}">${j.judgment}</span></td>
    `;
    fieldsBody.appendChild(tr);
  }
}

function renderCritic(judgments, criticTriggered) {
  criticSection.innerHTML = '';
  const withCritic = judgments.filter(j => j.ai_review);
  if (!withCritic.length) {
    const msg = criticTriggered
      ? 'Critic was triggered but returned no usable result (check raw JSON for an error).'
      : 'Not triggered this run -- no fields were flagged, so there was nothing to second-opinion. (This is the normal, expected state most of the time.)';
    criticSection.innerHTML = `<div class="critic-empty">${msg}</div>`;
    return;
  }
  for (const j of withCritic) {
    const box = document.createElement('div');
    box.className = 'critic-box';
    const ai = j.ai_review;
    box.innerHTML = `<b>${j.field}</b> — trust: ${ai.trust ?? '?'}, issue: ${ai.issue_category ?? '?'}<br>${ai.rationale ?? ''}`;
    criticSection.appendChild(box);
  }
}

function renderLoopStatus(loop) {
  const pct = Math.min(100, Math.round(100 * loop.documents_logged / loop.min_files_required));
  const signalLine = loop.signals_found > 0
    ? `<span class="signal-yes">${loop.signals_found} signal(s) found</span> — a field is being flagged consistently enough to be a systemic pattern.`
    : `<span class="signal-no">No systemic signal yet</span> — either not enough documents logged, or nothing is failing consistently.`;
  const proposerLine = loop.candidates_drafted > 0
    ? `Proposer has drafted <b>${loop.candidates_drafted}</b> candidate rewrite(s) so far.`
    : `Proposer has never been invoked -- it only runs once a signal crosses threshold, which hasn't happened yet.`;
  loopStatus.innerHTML = `
    <div>Documents logged so far: <b>${loop.documents_logged}</b> / ${loop.min_files_required} needed before any signal can trigger</div>
    <div class="progress-track"><div class="progress-fill" style="width:${pct}%"></div></div>
    <div class="loop-row">${signalLine}</div>
    <div class="loop-row" style="color:#666">${proposerLine}</div>
  `;
}

async function handleFile(file) {
  results.style.display = 'none';
  statusEl.className = '';
  statusEl.textContent = `Extracting ${file.name} ... (this calls Gemini live, may take 10-30s)`;

  const form = new FormData();
  form.append('file', file);

  try {
    const resp = await fetch('/extract', { method: 'POST', body: form });
    const data = await resp.json();
    if (!resp.ok) {
      statusEl.className = 'error';
      statusEl.textContent = 'Error: ' + (data.error || resp.statusText);
      return;
    }
    const flaggedCount = data.judgments.filter(j => j.is_flag_worthy).length;
    let msg = `Done — ${data.judgments.length} field(s) reviewed.`;
    if (flaggedCount) msg += ` <b style="color:#c05621">${flaggedCount} flagged.</b>`;
    statusEl.className = 'ok';
    statusEl.innerHTML = msg;

    renderFields(data.judgments);
    renderCritic(data.judgments, data.critic_triggered);
    renderLoopStatus(data.loop_status);
    output.textContent = JSON.stringify(data, null, 2);
    results.style.display = 'block';
  } catch (err) {
    statusEl.className = 'error';
    statusEl.textContent = 'Request failed: ' + err;
  }
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/extract", methods=["POST"])
def extract():
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "No file uploaded"}), 400
    if not uploaded.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / uploaded.filename
        uploaded.save(pdf_path)

        try:
            extractor = Extractor(config)
            result = extractor.extract(pdf_path)
        except ExtractorError as exc:
            return jsonify({"error": str(exc)}), 502
        except Exception as exc:  # noqa: BLE001 -- surface any live-call failure to the UI
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 502

        reviewer = Reviewer(config)
        review = reviewer.review(result.extraction, document_id=uploaded.filename)

        # Auto-run the AI critic on whatever got flagged -- the CLI gates this
        # behind an explicit --ai-review flag; for a visual diagnostic tool,
        # always showing the critic's opinion on flagged fields is more useful
        # than requiring the user to remember a flag.
        ai_reviews = None
        flagged = review.flag_worthy()
        critic_triggered = bool(flagged)
        if flagged:
            rules_text = config.prompt_path.read_text(encoding="utf-8")
            ai_reviewer = AIReviewer(config)
            ai_reviews = ai_reviewer.critique_document(pdf_path, rules_text, review)

        review_log.append(
            config.review_log_path, Path(uploaded.filename), review, ai_reviews=ai_reviews,
            extraction_model=config.extraction_model, bbox_model=config.bbox_model,
        )

        tagger = TaxonomyTagger(config.taxonomy_path)
        tagged = tagger.tag(
            result.extraction, source_document=uploaded.filename,
            extraction_model=config.extraction_model, bbox_model=config.bbox_model,
        )
        tagged_dict = tagged.to_dict()

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = config.extractions_dir / f"{ts}_{pdf_path.stem}.json"
        out_path.write_text(json.dumps(tagged_dict, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

        ai_review_by_field = {}
        if ai_reviews:
            for r in ai_reviews:
                ai_review_by_field[r.field_path] = r.to_dict()

        judgments = []
        for j in review.judgments:
            entry = {
                "field": j.field,
                "extracted_value": j.extracted_value,
                "confidence": j.confidence,
                "alignment_status": j.alignment_status,
                "judgment": j.judgment,
                "is_flag_worthy": j.is_flag_worthy,
            }
            if j.field in ai_review_by_field:
                entry["ai_review"] = ai_review_by_field[j.field]
            judgments.append(entry)

        # Self-correction loop status: how close are we to the point where
        # the Summarizer can even compute a signal, has it found one, and has
        # the Proposer (gated behind a real signal) ever actually run?
        summarizer = Summarizer(config)
        report = summarizer.aggregate()
        candidates_drafted = len(list(config.candidates_dir.glob("*.meta.json")))
        loop_status = {
            "documents_logged": report.n_files_total,
            "min_files_required": config.min_files,
            "signals_found": len(report.signals) + len(report.ai_critic_signals),
            "candidates_drafted": candidates_drafted,
        }

    return jsonify({
        "tagged": tagged_dict,
        "judgments": judgments,
        "critic_triggered": critic_triggered,
        "loop_status": loop_status,
        "saved_to": str(out_path),
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)
