"""Taxonomy Tagger: maps extracted values onto the taxonomy's concepts and
Presentation Linkbase hierarchy.

This is the genuinely new component — nothing in the pipeline this tool is
adapted from does this; the source system only ever produced flat extracted
JSON. Output here is a JSON structure that mirrors XBRL's shape (facts tagged
to taxonomy concepts, each carrying its position in the Presentation
Linkbase hierarchy) rather than a fully XBRL-conformant XML instance
document — that's the deliberately scoped-down "JSON intermediate" decision
for this stage of the project, not a full XBRL serializer.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TaggedFact:
    concept: str
    label: str
    parent_concept: Optional[str]
    parent_label: Optional[str]
    value: Any
    confidence: Optional[float] = None
    evidence_text: Optional[str] = None
    page: Optional[int] = None
    alignment_status: Optional[str] = None


@dataclass
class TaggedDocument:
    taxonomy_name: str
    taxonomy_version: str
    source_document: str
    facts: List[TaggedFact] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "taxonomy_name": self.taxonomy_name,
            "taxonomy_version": self.taxonomy_version,
            "source_document": self.source_document,
            "facts": [asdict(f) for f in self.facts],
        }


class TaxonomyTagger:
    def __init__(self, taxonomy_path: Path):
        self.taxonomy = json.loads(Path(taxonomy_path).read_text(encoding="utf-8"))
        self.concepts: Dict[str, Any] = self.taxonomy.get("concepts", {})
        self.field_map: Dict[str, str] = self.taxonomy.get("extraction_field_map", {})

    def tag(self, extraction: Dict[str, Any], source_document: str) -> TaggedDocument:
        """Walk the extractor's value JSON (not the parallel `confidence` block)
        and, for every leaf whose field path is registered in the taxonomy's
        extraction_field_map, emit a TaggedFact carrying its concept + parent
        concept per the Presentation Linkbase.
        """
        confidence_block = extraction.get("confidence") if isinstance(extraction.get("confidence"), dict) else {}
        value_block = {k: v for k, v in extraction.items() if k != "confidence"}

        facts: List[TaggedFact] = []
        self._walk(value_block, confidence_block, prefix="", facts=facts)

        return TaggedDocument(
            taxonomy_name=self.taxonomy.get("taxonomy_name", "unknown"),
            taxonomy_version=self.taxonomy.get("version", "0.0.0"),
            source_document=source_document,
            facts=facts,
        )

    def _walk(self, value_node: Any, conf_node: Any, prefix: str, facts: List[TaggedFact]) -> None:
        if isinstance(value_node, dict):
            for key, sub_value in value_node.items():
                sub_prefix = f"{prefix}.{key}" if prefix else key
                sub_conf = conf_node.get(key) if isinstance(conf_node, dict) else None
                self._walk(sub_value, sub_conf, sub_prefix, facts)
            return

        # Leaf value — only emit a fact if this field path is registered in the taxonomy.
        concept_id = self.field_map.get(prefix)
        if concept_id is None:
            return

        concept = self.concepts.get(concept_id, {})
        parent_id = concept.get("parent")
        parent = self.concepts.get(parent_id, {}) if parent_id else {}

        leaf_conf = conf_node if isinstance(conf_node, dict) else {}
        facts.append(TaggedFact(
            concept=concept_id,
            label=concept.get("label", concept_id),
            parent_concept=parent_id,
            parent_label=parent.get("label") if parent_id else None,
            value=value_node,
            confidence=leaf_conf.get("confidence"),
            evidence_text=leaf_conf.get("evidence_text"),
            page=leaf_conf.get("page"),
            alignment_status=leaf_conf.get("alignment_status"),
        ))

    def write(self, tagged: TaggedDocument, out_path: Path) -> Path:
        out_path.write_text(json.dumps(tagged.to_dict(), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return out_path
