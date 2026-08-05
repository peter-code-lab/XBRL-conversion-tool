"""Generates the taxonomy schema (.xsd) and Presentation Linkbase (_pre.xml)
from construction_payment_taxonomy.json.

Run manually whenever the taxonomy JSON changes:
    python taxonomy/generate_taxonomy_artifacts.py

These artifacts are a pure function of the taxonomy definition, not of any
extracted document, so they aren't regenerated per-extraction the way
instance documents are (see xbrl_serializer.py).
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict

TAXONOMY_DIR = Path(__file__).resolve().parent
TAXONOMY_PATH = TAXONOMY_DIR / "construction_payment_taxonomy.json"
SCHEMA_FILENAME = "construction_payment_taxonomy.xsd"
PRE_FILENAME = "construction_payment_taxonomy_pre.xml"
SCHEMA_PATH = TAXONOMY_DIR / SCHEMA_FILENAME
PRE_PATH = TAXONOMY_DIR / PRE_FILENAME

CPAT_NS = "http://example.org/xbrl/construction-payment-application"
XBRLI_INSTANCE_XSD = "http://www.xbrl.org/2003/xbrl-instance-2003-12-31.xsd"

NS = {
    "xsd": "http://www.w3.org/2001/XMLSchema",
    "link": "http://www.xbrl.org/2003/linkbase",
    "xlink": "http://www.w3.org/1999/xlink",
    "xbrli": "http://www.xbrl.org/2003/instance",
    "cpat": CPAT_NS,
}
for _prefix, _uri in NS.items():
    ET.register_namespace(_prefix, _uri)

_ITEM_TYPE_BY_CONCEPT_TYPE = {
    "string": "xbrli:stringItemType",
    "date": "xbrli:dateItemType",
    "monetary": "xbrli:monetaryItemType",
    "group": "xbrli:stringItemType",
}


def _qname(prefix: str, tag: str) -> str:
    return f"{{{NS[prefix]}}}{tag}"


def build_schema(taxonomy: Dict[str, Any]) -> ET.Element:
    schema = ET.Element(_qname("xsd", "schema"))
    schema.set("targetNamespace", CPAT_NS)
    schema.set("elementFormDefault", "qualified")

    imp = ET.SubElement(schema, _qname("xsd", "import"))
    imp.set("namespace", NS["xbrli"])
    imp.set("schemaLocation", XBRLI_INSTANCE_XSD)

    annotation = ET.SubElement(schema, _qname("xsd", "annotation"))
    appinfo = ET.SubElement(annotation, _qname("xsd", "appinfo"))
    linkbase_ref = ET.SubElement(appinfo, _qname("link", "linkbaseRef"))
    linkbase_ref.set(_qname("xlink", "type"), "simple")
    linkbase_ref.set(_qname("xlink", "href"), PRE_FILENAME)
    linkbase_ref.set(_qname("xlink", "role"), "http://www.xbrl.org/2003/role/presentationLinkbaseRef")
    linkbase_ref.set(_qname("xlink", "arcrole"), "http://www.w3.org/1999/xlink/properties/linkbase")

    for concept_id, concept in taxonomy.get("concepts", {}).items():
        concept_type = concept.get("type")
        elem = ET.SubElement(schema, _qname("xsd", "element"))
        elem.set("name", concept_id)
        elem.set("id", f"cpat_{concept_id}")
        elem.set("type", _ITEM_TYPE_BY_CONCEPT_TYPE.get(concept_type, "xbrli:stringItemType"))
        elem.set("substitutionGroup", "xbrli:item")
        elem.set(_qname("xbrli", "periodType"), "instant")
        elem.set("nillable", "true")
        if concept_type == "group":
            elem.set("abstract", "true")

    return schema


def build_presentation_linkbase(taxonomy: Dict[str, Any]) -> ET.Element:
    rows = taxonomy.get("presentation_linkbase", [])

    linkbase = ET.Element(_qname("link", "linkbase"))

    # No roleRef: http://www.xbrl.org/2003/role/link is a *standard* XBRL link
    # role, which per spec (3.5.2.4.5) doesn't require a roleRef declaration
    # -- only custom/extension roles do.
    presentation_link = ET.SubElement(linkbase, _qname("link", "presentationLink"))
    presentation_link.set(_qname("xlink", "type"), "extended")
    presentation_link.set(_qname("xlink", "role"), "http://www.xbrl.org/2003/role/link")

    concept_ids = []
    seen = set()
    for row in rows:
        for concept_id in (row.get("parent"), row.get("concept")):
            if concept_id and concept_id not in seen:
                seen.add(concept_id)
                concept_ids.append(concept_id)

    for concept_id in concept_ids:
        loc = ET.SubElement(presentation_link, _qname("link", "loc"))
        loc.set(_qname("xlink", "type"), "locator")
        loc.set(_qname("xlink", "href"), f"{SCHEMA_FILENAME}#cpat_{concept_id}")
        loc.set(_qname("xlink", "label"), concept_id)

    for row in rows:
        parent_id = row.get("parent")
        if parent_id is None:
            continue
        arc = ET.SubElement(presentation_link, _qname("link", "presentationArc"))
        arc.set(_qname("xlink", "type"), "arc")
        arc.set(_qname("xlink", "arcrole"), "http://www.xbrl.org/2003/arcrole/parent-child")
        arc.set(_qname("xlink", "from"), parent_id)
        arc.set(_qname("xlink", "to"), row["concept"])
        arc.set("order", str(row.get("order", 1)))

    return linkbase


def main() -> None:
    taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))

    schema = build_schema(taxonomy)
    ET.ElementTree(schema).write(SCHEMA_PATH, xml_declaration=True, encoding="UTF-8")
    print(f"Wrote {SCHEMA_PATH}")

    presentation_linkbase = build_presentation_linkbase(taxonomy)
    ET.ElementTree(presentation_linkbase).write(PRE_PATH, xml_declaration=True, encoding="UTF-8")
    print(f"Wrote {PRE_PATH}")


if __name__ == "__main__":
    main()
