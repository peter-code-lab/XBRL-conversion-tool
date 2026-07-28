"""Standalone extractor: classify-free, direct Gemini extraction + confidence +
bbox verification for construction contractor payment-application PDFs.

Adapted from SBI_Backend's Utility_Bill_Extractor_Local.py (extraction call +
CONFIDENCE_PROMPT_SUFFIX pattern) and auto_bbox.py (page rendering + targeted
bbox verification), generalized for this tool's simpler, array-free schema and
with no dependency on a running Java backend — this module talks to Gemini
directly.

Unlike the system it's adapted from, this tool has exactly one document type,
so there is no two-stage classifier here. Every PDF submitted to this tool is
assumed to be a construction contractor payment application; the extraction
prompt itself is written to be robust across differing state DOT formats
(Caltrans, Iowa DOT, TxDOT SiteManager, ...) rather than routing each format
to a different schema.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from config import PipelineConfig

logger = logging.getLogger(__name__)


# Appended to the extraction prompt so the model self-reports, per leaf field,
# a confidence score + verbatim evidence quote + page number. Schema-agnostic
# by construction — "mirror the structure of the value JSON" works for any
# nested schema, not just this one. Ported from
# SBI_Backend/.../Utility_Bill_Extractor_Local.py:CONFIDENCE_PROMPT_SUFFIX,
# with the usage_info-array-specific language removed since this schema has
# no arrays.
CONFIDENCE_PROMPT_SUFFIX = """

After the value JSON, append a top-level "confidence" object that mirrors the structure of the value JSON exactly (same keys, same nesting). For every leaf field, emit an object:
{"confidence": <0.0-1.0 float>, "evidence_text": "<verbatim short substring quoted from the document>", "page": <1-indexed page number or null if unknown>}

Rules for the confidence block:
- "confidence" is your own self-assessment. Use the full 0.0-1.0 range, not just 0 or 1. The meaning depends on whether the value is concrete or "N/A":

  When the value is CONCRETE (not N/A), confidence = how sure you are the value is correct:
    * 0.95-1.00 — Verbatim from a clearly labeled field on the document. No ambiguity.
    * 0.80-0.94 — On the document but the label is partial, abbreviated, or in a non-standard position; you had to interpret slightly which printed value matches the schema field.
    * 0.50-0.79 — Inferred or computed from related fields, or picking the matching value among several candidates. A different reasonable reader could pick a different value.
    * 0.20-0.49 — A contextual guess. Something on the document suggests it but no field is explicitly labeled with it.
    * 0.00-0.19 — Wholly invented — you produced something but it's nearly random.

  When the value is "N/A", confidence = how sure you are the field is genuinely absent:
    * 0.95-1.00 — Field is clearly not applicable to this document, or you scanned every page and confirmed it isn't there.
    * 0.80-0.94 — Document template doesn't appear to include this field; you scanned the relevant pages and saw nothing matching.
    * 0.50-0.79 — Field MIGHT be on the document but you couldn't find it; could go either way.
    * 0.20-0.49 — You suspect the field IS on the document but you couldn't extract it (illegible, ambiguous, multiple candidates with no clear winner).
    * 0.00-0.19 — Total guess that N/A is correct; no real assessment.

  Avoid clustering at 0.0 and 1.0 — pick the band that best describes WHY.

- "evidence_text" must be a short verbatim quote (<= 80 chars) copied character-for-character from the document, including any nearby label that disambiguates the value. For "N/A" values leave evidence_text as "" (there is no value to quote).
- "page": for concrete values, the 1-indexed page number where the value appears. For "N/A" values, set page to null (or the page you most carefully scanned for the field, if any).
- Do NOT add commentary, explanations, or extra keys. Only the schema described above.
"""


_BBOX_PROMPT = """You are given one page image of a construction contractor payment-application document plus a list of items to locate.

For each item, find the smallest rectangular region on the page that contains the field's value. The `evidence_text` is a HINT — it tells you what the field looks like on the document, but the exact substring may not appear as one continuous visual phrase (label and value are often in different positions). In that case, box the value portion — that's what the user wants to see highlighted.

Items to box:
__ITEMS__

Return JSON with this exact schema:
{"boxes": [{"label": "...", "evidence_text": "...", "box_2d": [ymin, xmin, ymax, xmax]}]}

box_2d format:
- Coordinates normalized to 0-1000 (top-left origin), [ymin, xmin, ymax, xmax].

Rules:
- One entry per input item, in the same order. Use the exact same `label` and `evidence_text` strings (do not modify either — return them verbatim).
- ALWAYS try to find a box. Only set `box_2d` to null if the value genuinely is not on this page at all.
- If unsure between two possible locations, pick the more prominent / larger occurrence.
- Return only the JSON object. No commentary, no markdown fences.
"""

_RENDER_DPI = 150


class ExtractorError(RuntimeError):
    pass


@dataclass
class ExtractionResult:
    pdf_path: Path
    extraction: Dict[str, Any]  # value JSON + top-level "confidence" block, post-bbox
    duration_seconds: float


class Extractor:
    def __init__(self, config: PipelineConfig, *, client: Any = None):
        self.config = config
        if client is not None:
            self._client = client
        else:
            if not GENAI_AVAILABLE:
                raise ExtractorError("google-genai is required (pip install google-genai).")
            if not config.gemini_api_key:
                raise ExtractorError(
                    "No Gemini API key found. Set GEMINI_API_KEY/API_KEY in the "
                    "environment, or create a .env file (see config.py)."
                )
            self._client = genai.Client(api_key=config.gemini_api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, pdf_path: Path) -> ExtractionResult:
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)

        # One id shared by every raw response saved for this document's run
        # (extraction + every per-page bbox call), so they can all be found
        # and correlated together afterward.
        run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{pdf_path.stem}"

        start = time.time()
        remote_file = self._upload(pdf_path)
        prompt = self.config.prompt_path.read_text(encoding="utf-8") + CONFIDENCE_PROMPT_SUFFIX

        response = self._client.models.generate_content(
            model=self.config.extraction_model,
            contents=[remote_file, prompt],
            config=types.GenerateContentConfig(temperature=0.0, top_p=0.0),
        )
        raw = (response.text or "").strip()
        self._save_raw_response(run_id, "extraction", raw)
        if not raw:
            raise ExtractorError(f"Empty response from Gemini for {pdf_path}")

        extraction = self._parse_json(raw)
        extraction = self._compute_bboxes(pdf_path, extraction, run_id)

        duration = time.time() - start
        return ExtractionResult(pdf_path=pdf_path, extraction=extraction, duration_seconds=duration)

    def _save_raw_response(self, run_id: str, kind: str, raw_text: str) -> None:
        """Save the exact, pre-parsed Gemini response text -- so a future
        anomaly (a null value despite correct evidence, an unexpected
        response shape, etc.) can be diagnosed by reading exactly what the
        model said, instead of trying to reproduce it after the fact (which
        doesn't reliably work, since a fresh call can come back differently)."""
        try:
            out_path = self.config.raw_responses_dir / f"{run_id}_{kind}.txt"
            out_path.write_text(raw_text, encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not save raw response for %s/%s: %s", run_id, kind, exc)

    # ------------------------------------------------------------------
    # Gemini upload
    # ------------------------------------------------------------------

    def _upload(self, pdf_path: Path):
        file_obj = self._client.files.upload(file=str(pdf_path))
        for _ in range(12):
            info = self._client.files.get(name=file_obj.name)
            if info.state in ("ACTIVE", 2):
                return info
            time.sleep(5)
        raise ExtractorError(f"File {pdf_path.name} not ACTIVE after waiting.")

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """Parse Gemini's response into a single dict with value fields plus
        a top-level "confidence" key.

        The prompt asks for one JSON object shaped that way, but cheaper
        models are inconsistent in practice — three distinct malformed shapes
        observed across live test runs on gemini-2.5-flash-lite /
        gemini-3.5-flash-lite:

          1. Two fenced blocks: the value object, then a separate
             `{"confidence": {...}}` object.
          2. Two fenced blocks: the value object, then a SECOND full copy of
             the same schema shape with each leaf's plain value replaced by
             a `{confidence, evidence_text, page}` dict — no "confidence"
             wrapper key at all.
          3. ONE object, but the real value fields are nested one level too
             deep under an extra, unrequested `"values"` key alongside
             `"confidence"` -- e.g. `{"values": {"agreement_parties": ...},
             "confidence": {...}}` instead of `agreement_parties` sitting
             directly at the top level. Diagnosed via a saved raw response
             where this caused every value to silently resolve to null
             (Reviewer's tree-walk expects value keys at the top level to
             match the confidence tree's structure) despite the confidence
             block itself being completely correct.

        A naive `dict.update()` merge across all top-level objects handles
        case 1 correctly but silently destroys the real values in case 2
        (the second object's top-level keys collide with and overwrite the
        first object's). So: parse every top-level JSON value in the
        response, classify each one as a "confidence tree" (every leaf is a
        confidence-shaped dict, no plain scalar values anywhere) or a "value
        object" (has at least one real scalar leaf), and combine accordingly
        instead of blindly overwriting. Then unwrap a stray "values" wrapper
        (case 3) if the merge produced one instead of real top-level keys.
        """
        cleaned = re.sub(r"```(?:json)?", "", text).strip()

        decoder = json.JSONDecoder()
        parsed_objects: List[Dict[str, Any]] = []
        idx = 0
        n = len(cleaned)
        while idx < n:
            while idx < n and cleaned[idx] in " \t\r\n":
                idx += 1
            if idx >= n:
                break
            try:
                obj, end = decoder.raw_decode(cleaned, idx)
            except json.JSONDecodeError:
                break
            if isinstance(obj, dict):
                parsed_objects.append(obj)
            idx = end

        if not parsed_objects:
            raise ExtractorError(f"Could not parse any JSON object from Gemini response: {text[:300]!r}")

        merged: Dict[str, Any] = {}
        confidence_tree: Optional[Dict[str, Any]] = None
        for obj in parsed_objects:
            if Extractor._is_pure_confidence_tree(obj):
                # Unwrap a `{"confidence": {...}}` wrapper if that's all this
                # object is; otherwise the object itself IS the tree (case 2).
                confidence_tree = obj["confidence"] if list(obj.keys()) == ["confidence"] else obj
            else:
                merged.update(obj)

        # Case 3: unwrap a stray "values" key. "values" is never a real
        # taxonomy field name in this tool's schema, so if the model added
        # one as an extra wrapper around the actual fields, lift its
        # contents up to the top level instead of leaving them nested one
        # level too deep (where the Reviewer's tree-walk would never find
        # them, since it expects value keys to mirror the confidence tree's
        # top-level structure).
        if isinstance(merged.get("values"), dict):
            wrapped = merged.pop("values")
            for key, value in wrapped.items():
                merged.setdefault(key, value)

        if confidence_tree is not None:
            merged["confidence"] = confidence_tree
        return merged

    @staticmethod
    def _is_pure_confidence_tree(node: Any) -> bool:
        """True if every leaf in this JSON value is a confidence-shaped dict
        (has a scalar "confidence" key) — i.e. there are NO plain scalar
        values anywhere, so this can't be (part of) the real value object."""
        if isinstance(node, dict):
            if "confidence" in node and not isinstance(node.get("confidence"), (dict, list)):
                return True
            if not node:
                return False
            return all(Extractor._is_pure_confidence_tree(v) for v in node.values())
        if isinstance(node, list):
            if not node:
                return False
            return all(Extractor._is_pure_confidence_tree(v) for v in node)
        return False

    # ------------------------------------------------------------------
    # Bbox verification — adapted from SBI_Backend/.../auto_bbox.py, generalized
    # for a schema with no arrays (no usage_info-style special-casing needed).
    # ------------------------------------------------------------------

    def _compute_bboxes(self, pdf_path: Path, response: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        if not PYMUPDF_AVAILABLE:
            logger.warning("PyMuPDF not installed; skipping bbox verification, alignment_status will be absent")
            return response
        conf = response.get("confidence")
        if not isinstance(conf, dict):
            logger.warning("No 'confidence' block in extraction; skipping bbox verification")
            return response

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            logger.warning("Failed to open PDF %s for bbox pass: %s", pdf_path, exc)
            return response

        page_groups: Dict[int, List[Tuple[str, str, Dict[str, Any]]]] = defaultdict(list)
        for label, evidence, page_num, leaf in self._walk_confidence(conf):
            if not evidence or not str(evidence).strip():
                leaf["alignment_status"] = "MISSING"
                continue
            if not isinstance(page_num, int) or page_num < 1 or page_num > len(doc):
                leaf["alignment_status"] = "NO_MATCH"
                continue
            page_groups[page_num].append((label, str(evidence), leaf))

        try:
            for page_num in sorted(page_groups.keys()):
                items = page_groups[page_num]
                page = doc[page_num - 1]
                try:
                    img_bytes = self._render_page(page)
                    boxes = self._call_bbox_gemini_with_retry(img_bytes, items, page_num, pdf_path, run_id)
                except Exception as exc:
                    # Only reached after retries are exhausted. Marking NO_MATCH
                    # here is a fallback, not a genuine "checked and it's not
                    # there" result -- log the full exception so a persistent
                    # failure is actually diagnosable instead of silently
                    # looking identical to a real suspicious-field finding.
                    logger.warning(
                        "Bbox verification failed for page %d of %s after retries: %s",
                        page_num, pdf_path, exc, exc_info=True,
                    )
                    for _, _, leaf in items:
                        leaf["alignment_status"] = "NO_MATCH"
                    continue

                by_label = {label: leaf for label, _, leaf in items}
                matched = set()
                for box_entry in boxes:
                    label = box_entry.get("label")
                    leaf = by_label.get(label)
                    if leaf is None:
                        continue
                    matched.add(label)
                    box_2d = box_entry.get("box_2d")
                    if not isinstance(box_2d, list) or len(box_2d) != 4:
                        leaf["alignment_status"] = "NO_MATCH"
                        continue
                    leaf["alignment_status"] = "EXACT"
                for label, _, leaf in items:
                    if label not in matched:
                        leaf["alignment_status"] = "NO_MATCH"
        finally:
            doc.close()

        return response

    @staticmethod
    def _walk_confidence(conf_node: Any, prefix: str = ""):
        """Generic recursive walk — yields (label, evidence_text, page, leaf_dict)
        for every leaf that has a scalar 'confidence' key. No array special-casing
        since this tool's schema has none."""
        if not isinstance(conf_node, dict):
            return
        if "confidence" in conf_node and not isinstance(conf_node.get("confidence"), (dict, list)):
            yield prefix, conf_node.get("evidence_text") or "", conf_node.get("page"), conf_node
            return
        for key, sub in conf_node.items():
            sub_prefix = f"{prefix}.{key}" if prefix else key
            yield from Extractor._walk_confidence(sub, sub_prefix)

    def _render_page(self, page: "fitz.Page") -> bytes:
        mat = fitz.Matrix(_RENDER_DPI / 72, _RENDER_DPI / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")

    def _call_bbox_gemini_with_retry(
        self,
        img_bytes: bytes,
        items: List[Tuple[str, str, Dict[str, Any]]],
        page_num: int,
        pdf_path: Path,
        run_id: str,
        max_attempts: int = 3,
    ) -> List[Dict[str, Any]]:
        """Retry the bbox call on transient failures (network hiccups, rate
        limits, occasional malformed JSON) before giving up. Added after
        observing the same document produce all-EXACT on some runs and
        all-NO_MATCH on others -- an all-or-nothing failure pattern across
        every field on a page is a sign the whole call errored out, not that
        the vision model individually struggled to find each piece of text.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                return self._call_bbox_gemini(img_bytes, items, run_id, page_num, attempt)
            except Exception as exc:  # noqa: BLE001 -- retry on anything, let the caller's handler log the final one
                last_exc = exc
                if attempt < max_attempts:
                    logger.info(
                        "Bbox call attempt %d/%d failed for page %d of %s: %s -- retrying",
                        attempt, max_attempts, page_num, pdf_path, exc,
                    )
                    time.sleep(1.5 * attempt)
        raise last_exc  # type: ignore[misc]

    def _call_bbox_gemini(
        self,
        img_bytes: bytes,
        items: List[Tuple[str, str, Dict[str, Any]]],
        run_id: Optional[str] = None,
        page_num: Optional[int] = None,
        attempt: int = 1,
    ) -> List[Dict[str, Any]]:
        items_payload = [{"label": label, "evidence_text": ev} for label, ev, _ in items]
        prompt = _BBOX_PROMPT.replace("__ITEMS__", json.dumps(items_payload, ensure_ascii=False, indent=2))

        resp = self._client.models.generate_content(
            model=self.config.bbox_model,
            contents=[types.Part.from_bytes(data=img_bytes, mime_type="image/png"), prompt],
            config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"),
        )
        raw = resp.text or ""
        if run_id is not None:
            self._save_raw_response(run_id, f"bbox_page{page_num}_attempt{attempt}", raw)
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
        boxes = parsed.get("boxes")
        if not isinstance(boxes, list):
            raise ExtractorError(f"Gemini bbox response missing 'boxes' array: {parsed}")
        return boxes
