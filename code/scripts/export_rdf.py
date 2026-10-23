"""Export the knowledge base as an RDF graph for the Open Knowledge Network.

The knowledge base is an Obsidian vault: each page is a document, the
``semantic_relations`` in its frontmatter are typed edges, and every
``[[wiki link]]`` in its body is an untyped edge. This module turns that
structure into RDF using vocabularies other graphs already use, so that the
graph joins with them instead of standing apart:

* SKOS for concepts, labels, aliases, domains and concept-to-concept relations,
* Dublin Core Terms for titles, descriptions, dates, containment and links,
* schema.org for pages, keywords, authors and publication years,
* PROV-O for where the graph came from and what generated it,
* DCAT and VoID for the dataset description.

Topic-bearing documents describe separate ``/concept/`` resources through
``schema:about`` and ``dcterms:subject``. Reviewed conceptual relations such as
``implements`` and ``foundation_for`` have explicit Concept scope. Documentary
and ambiguous assertions use document-scoped predicates without SKOS entailment.
Explicit citations use BIBO. Tags point to concepts; reviewed SKOS
mappings carry statement evidence, while unreviewed Wikidata identifiers are
discovery links only.

Entity IRIs live under ``https://purl.org/okn/frink/kg/cognitive-kb/``, the
prefix the FRINK registry reserves for the ``cognitive-kb`` graph. They are
slash IRIs throughout.

The export is deterministic: the same vault always produces the same bytes. No
timestamp is written; the dataset's ``dcterms:modified`` is the latest page
``updated`` date. Link resolution is the one ``verify_links`` uses, so the graph
and the link checker agree on what a link points to.

Usage::

    python -m scripts.export_rdf                 # N-Quads + Turtle into output/okn/
    python -m scripts.export_rdf --write-docs    # refresh docs/okn/ sample + statistics
    python -m scripts.export_rdf --check         # fail if docs/okn/ is stale
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

try:
    from scripts.kb_index import _split
    from scripts.rdf_alignment import RELATIONS, checked_iri, reviewed_mappings
    from scripts.rdf_keywords import KEYWORD_TERMS as KEYWORD_TERMS
    from scripts.rdf_keywords import keyword_iri
    from scripts.rdf_quality import GraphIndex, semantic_quality
    from scripts.rdf_relations import CONCEPT_RELATIONS, DOCUMENT_RELATIONS, TOPIC_PAGE_TYPES
    from scripts.verify_links import (
        MATRIX_LITERAL_PATTERN,
        WIKI_LINK_PATTERN,
        LinkIndex,
        _alias_key,
        _extract_wiki_target,
        _frontmatter_aliases,
        _iter_files,
        _normalise_key,
        _resolve_target_path,
        _strip_markdown_code,
    )
except ImportError:  # run as a file: the script directory is on sys.path
    from kb_index import _split  # type: ignore[no-redef]
    from rdf_alignment import RELATIONS, checked_iri, reviewed_mappings  # type: ignore[no-redef]
    from rdf_keywords import KEYWORD_TERMS as KEYWORD_TERMS  # type: ignore[no-redef]
    from rdf_keywords import keyword_iri  # type: ignore[no-redef]
    from rdf_quality import GraphIndex, semantic_quality  # type: ignore[no-redef]
    from rdf_relations import (  # type: ignore[no-redef]
        CONCEPT_RELATIONS,
        DOCUMENT_RELATIONS,
        TOPIC_PAGE_TYPES,
    )
    from verify_links import (  # type: ignore[no-redef]
        MATRIX_LITERAL_PATTERN,
        WIKI_LINK_PATTERN,
        LinkIndex,
        _alias_key,
        _extract_wiki_target,
        _frontmatter_aliases,
        _iter_files,
        _normalise_key,
        _resolve_target_path,
        _strip_markdown_code,
    )

KB_NAME = "knowledge_base"
BASE = "https://purl.org/okn/frink/kg/cognitive-kb/"
# The FRINK federation names each graph https://purl.org/okn/frink/kg/<shortname>.
GRAPH = "https://purl.org/okn/frink/kg/cognitive-kb"
DATASET = BASE + "dataset"
SCHEME = BASE + "scheme"
ACTIVITY = BASE + "activity/export"
VOCAB = BASE + "vocab/"
PAGE = BASE + "page/"
CONCEPT = BASE + "concept/"
DOMAIN = BASE + "domain/"
RELATION = BASE + "relation/"
DOCUMENT_RELATION = RELATION + "document/"

SOURCE_REPOSITORY = "https://github.com/ActiveInferenceInstitute/cognitive"
EXPORTER_URL = SOURCE_REPOSITORY + "/blob/main/code/scripts/export_rdf.py"
ORGANISATION = "https://activeinference.institute/"
LICENSE = "https://creativecommons.org/licenses/by/4.0/"

DOCS_DIR = Path("docs/okn")
SAMPLE_NAME = "cognitive-kb.sample.ttl"
STATISTICS_NAME = "statistics.md"
ALIGNMENT_NAME = "alignment-review.md"
RUNTIME_SOURCE_NAMES = (
    "__init__.py",
    "export_rdf.py",
    "rdf_alignment.py",
    "rdf_quality.py",
    "rdf_keywords.py",
    "rdf_relations.py",
    "kb_index.py",
    "verify_links.py",
    "markdown_fences.py",
)
SAMPLE_PAGES = (
    "mathematics/markov_blanket",
    "citations/smith_2022",
    "citations/README",
    "biology/cortisol",
    "biology/serotonin",
    "biology/gene_expression",
    "mathematics/probability_distributions",
    "mathematics/entropy",
    "cognitive/emotion_regulation",
    "cognitive/active_inference_agent",
)

PREFIXES = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "dcterms": "http://purl.org/dc/terms/",
    "schema": "https://schema.org/",
    "prov": "http://www.w3.org/ns/prov#",
    "dcat": "http://www.w3.org/ns/dcat#",
    "void": "http://rdfs.org/ns/void#",
    "bibo": "http://purl.org/ontology/bibo/",
    "sio": "http://semanticscience.org/resource/",
    "ckb": VOCAB,
    "rel": RELATION,
    "page": PAGE,
    "concept": CONCEPT,
    "domain": DOMAIN,
    "kg": BASE,
}

RDF_TYPE = PREFIXES["rdf"] + "type"
XSD_DATE = PREFIXES["xsd"] + "date"
XSD_GYEAR = PREFIXES["xsd"] + "gYear"

# Candidate predicate spellings; _route_relation additionally checks role/scope.
STANDARD_RELATIONS = {
    "relates": PREFIXES["skos"] + "related",
    "related": PREFIXES["skos"] + "related",
    "relates_to": PREFIXES["skos"] + "related",
    "related_to": PREFIXES["skos"] + "related",
    "part_of": PREFIXES["dcterms"] + "isPartOf",
    "component_of": PREFIXES["dcterms"] + "isPartOf",
    "has_parent": PREFIXES["dcterms"] + "isPartOf",
    "organizes": PREFIXES["dcterms"] + "hasPart",
    "cites": PREFIXES["bibo"] + "cites",
    "cited_by": PREFIXES["bibo"] + "citedBy",
}
WIKILINK_PREDICATE = PREFIXES["dcterms"] + "references"
REDIRECT_PREDICATE = PREFIXES["dcterms"] + "isReplacedBy"
WIKIDATA_ENTITY = "http://www.wikidata.org/entity/"
WIKIDATA_ID = re.compile(r"Q[1-9][0-9]*")
NAVIGATION_STEMS = {"readme", "agents", "catalog", "index"}
# Links are resolved only against what the repository ships. A checkout also holds
# local tool folders (editor, index and agent directories); letting a link land in
# one would make the graph differ between machines.
SHIPPED_DIRECTORIES = {KB_NAME, "docs", "code"}


# --------------------------------------------------------------------------- terms
def iri(value: str) -> str:
    """Return ``value`` as an N-Triples IRI term."""
    return f"<{value}>"


_ESCAPES = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\r": "\\r", "\t": "\\t"}


def literal(value: str, *, lang: str | None = None, datatype: str | None = None) -> str:
    """Return ``value`` as an N-Triples literal term."""
    out = []
    for char in value:
        if char in _ESCAPES:
            out.append(_ESCAPES[char])
        elif ord(char) < 0x20 or ord(char) == 0x7F:
            out.append(f"\\u{ord(char):04X}")
        else:
            out.append(char)
    text = '"' + "".join(out) + '"'
    if lang:
        return f"{text}@{lang}"
    if datatype:
        return f"{text}^^<{datatype}>"
    return text


def date_text(value: Any) -> str:
    """Return a YAML date or string as ``YYYY-MM-DD`` text when it is a date."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value).strip()


def typed_date(value: Any) -> str:
    """Return a date as an ``xsd:date`` literal, or a plain literal if it is not one."""
    text = date_text(value)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return literal(text, datatype=XSD_DATE)
    return literal(text)


def page_iri(node_id: str) -> str:
    """IRI of the page whose vault-relative path (no ``.md``) is ``node_id``."""
    return PAGE + quote(node_id, safe="/-_.~")


def concept_iri(node_id: str) -> str:
    """Stable topic IRI, distinct from the source document's existing page IRI."""
    return CONCEPT + quote(node_id, safe="/-_.~")


def relation_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_") or "relates"


def relation_predicate(name: str) -> str:
    text = name.strip()
    if text.startswith(("http://", "https://")):
        return checked_iri(text)
    prefix, sep, local = text.partition(":")
    if sep and prefix in {"skos", "dcterms", "schema", "bibo", "sio", "prov"} and local:
        return checked_iri(PREFIXES[prefix] + local)
    slug = relation_slug(name)
    return STANDARD_RELATIONS.get(slug, RELATION + slug)


Triple = tuple[str, str, str]


@dataclass
class Graph:
    """A set of triples held as N-Triples terms, kept in insertion-independent order."""

    triples: set[Triple] = field(default_factory=set)

    def add(self, subject: str, predicate: str, obj: str) -> None:
        self.triples.add((subject, predicate, obj))

    def sorted(self) -> list[Triple]:
        return sorted(self.triples)


# ----------------------------------------------------------------------- parsing
def _flatten(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return [leaf for item in value for leaf in _flatten(item)]
    return [value]


def _link_target(raw: Any) -> str:
    """Return the page name in a wiki-link scalar, with display text and anchor removed."""
    if not isinstance(raw, str):
        return ""
    text = raw.strip()
    match = WIKI_LINK_PATTERN.fullmatch(text)
    return _extract_wiki_target(match.group(1) if match else text)


def semantic_relations(value: Any) -> list[tuple[str, str]]:
    """Return ``(relation, target)`` pairs from a ``semantic_relations`` value.

    Unquoted ``[[a]]`` entries are parsed by YAML as nested lists, so values are
    flattened recursively before the wiki-link syntax is removed.
    """
    pairs: list[tuple[str, str]] = []
    if not isinstance(value, list):
        return pairs
    for entry in value:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("type") or "relates")
        for key, relation in (("links", name), ("parent", "has_parent")):
            for leaf in _flatten(entry.get(key)):
                target = _link_target(leaf)
                if target:
                    pairs.append((relation, target))
    return pairs


def body_links(body: str) -> list[str]:
    """Return wiki-link targets in a page body, ignoring code and matrix literals."""
    targets = []
    for inner in WIKI_LINK_PATTERN.findall(_strip_markdown_code(body)):
        if MATRIX_LITERAL_PATTERN.match(inner):
            continue
        target = _extract_wiki_target(inner)
        if target:
            targets.append(target)
    return targets


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    return [str(item).strip() for item in _flatten(value) if str(item).strip()]


@dataclass
class Report:
    """Counts describing what an export contained and what it left out."""

    pages: int = 0
    concepts: int = 0
    document_relation_assertions: int = 0
    document_relation_routes: list[dict[str, str]] = field(default_factory=list)
    navigation_pages: int = 0
    redirects: int = 0
    aligned_pages: int = 0
    discovery_pages: int = 0
    keyword_concepts: int = 0
    keyword_links: int = 0
    mapping_relations: Counter = field(default_factory=Counter)
    typed_edges: int = 0
    link_edges: int = 0
    unresolved_links: int = 0
    links_outside_graph: int = 0
    self_loops_dropped: int = 0
    triples: int = 0
    modified: str = ""
    page_types: Counter = field(default_factory=Counter)
    relations: Counter = field(default_factory=Counter)
    domains: Counter = field(default_factory=Counter)


def _is_navigation(path: Path) -> bool:
    return path.stem.lower() in NAVIGATION_STEMS


def _has_topic(path: Path, frontmatter: dict[str, Any]) -> bool:
    """Only declared topic types (or untyped explicitly mapped topics) mint topics."""
    page_type = str(frontmatter.get("type") or "").strip()
    return (
        not _is_navigation(path)
        and "citations" not in path.parts
        and (
            page_type in TOPIC_PAGE_TYPES
            or (not page_type and bool(frontmatter.get("external_mappings")))
        )
    )


def _title(frontmatter: dict[str, Any], node_id: str) -> str:
    raw = str(frontmatter.get("title") or "").strip()
    match = WIKI_LINK_PATTERN.fullmatch(raw)
    if match:
        inner = match.group(1)
        raw = inner.split("|", 1)[1].strip() if "|" in inner else _extract_wiki_target(inner)
    return raw or node_id.rsplit("/", 1)[-1].replace("_", " ")


def _describe_page(
    graph: Graph, node_id: str, path: Path, frontmatter: dict[str, Any], report: Report
) -> None:
    subject = iri(page_iri(node_id))
    title = _title(frontmatter, node_id)
    page_type = str(frontmatter.get("type") or "").strip()
    navigation = _is_navigation(path) or page_type in {"index", "catalog", "agents"}
    topic = iri(concept_iri(node_id)) if _has_topic(path, frontmatter) else None

    graph.add(subject, iri(RDF_TYPE), iri(PREFIXES["schema"] + "CreativeWork"))
    if navigation:
        graph.add(subject, iri(RDF_TYPE), iri(PREFIXES["schema"] + "CollectionPage"))
        report.navigation_pages += 1
    if topic:
        graph.add(topic, iri(RDF_TYPE), iri(PREFIXES["skos"] + "Concept"))
        graph.add(topic, iri(PREFIXES["skos"] + "inScheme"), iri(SCHEME))
        graph.add(topic, iri(PREFIXES["skos"] + "prefLabel"), literal(title, lang="en"))
        graph.add(subject, iri(PREFIXES["schema"] + "about"), topic)
        graph.add(subject, iri(PREFIXES["dcterms"] + "subject"), topic)
        report.concepts += 1
    graph.add(subject, iri(PREFIXES["dcterms"] + "title"), literal(title, lang="en"))

    description = str(frontmatter.get("description") or "").strip()
    if description:
        graph.add(
            subject, iri(PREFIXES["dcterms"] + "description"), literal(description, lang="en")
        )
    for key, predicate in (("definition", "definition"), ("scope_note", "scopeNote")):
        if frontmatter.get(key):
            graph.add(
                topic or subject,
                iri(
                    PREFIXES["skos"] + predicate
                    if topic
                    else VOCAB + "source" + key.title().replace("_", "")
                ),
                literal(str(frontmatter[key]), lang="en"),
            )
    for alias in _strings(frontmatter.get("aliases")):
        graph.add(subject, iri(PREFIXES["schema"] + "alternateName"), literal(alias, lang="en"))
        if topic and alias != title:
            graph.add(topic, iri(PREFIXES["skos"] + "altLabel"), literal(alias, lang="en"))
    if frontmatter.get("created") is not None:
        graph.add(subject, iri(PREFIXES["dcterms"] + "created"), typed_date(frontmatter["created"]))
    updated = frontmatter.get("updated") or frontmatter.get("modified")
    if updated is not None:
        graph.add(subject, iri(PREFIXES["dcterms"] + "modified"), typed_date(updated))
    for raw_tag in _flatten(frontmatter.get("tags", [])):
        if raw_tag is None or not str(raw_tag).strip():
            continue
        # WHY: reviewed keyword reuse is exact; whitespace is not an alias.
        tag = str(raw_tag)
        if external_keyword := keyword_iri(tag):
            graph.add(subject, iri(PREFIXES["schema"] + "keywords"), iri(external_keyword))
            continue
        keyword = iri(BASE + "keyword/" + quote(tag, safe="-._~"))
        graph.add(subject, iri(PREFIXES["schema"] + "keywords"), keyword)
        graph.add(keyword, iri(RDF_TYPE), iri(PREFIXES["skos"] + "Concept"))
        graph.add(keyword, iri(RDF_TYPE), iri(PREFIXES["schema"] + "DefinedTerm"))
        graph.add(
            keyword, iri(PREFIXES["skos"] + "prefLabel"), literal(tag.replace("_", " "), lang="en")
        )
        graph.add(keyword, iri(PREFIXES["skos"] + "notation"), literal(tag))
        graph.add(keyword, iri(PREFIXES["skos"] + "inScheme"), iri(SCHEME))
    for key in ("status", "complexity", "processing_priority"):
        if frontmatter.get(key) is not None:
            name = "processingPriority" if key == "processing_priority" else key
            graph.add(subject, iri(VOCAB + name), literal(str(frontmatter[key])))
    if page_type:
        graph.add(subject, iri(VOCAB + "pageType"), literal(page_type))
        report.page_types[page_type] += 1
    source = f"{SOURCE_REPOSITORY}/blob/main/{KB_NAME}/{node_id}.md"
    graph.add(subject, iri(PREFIXES["dcterms"] + "source"), iri(quote(source, safe=":/-_.~")))
    if frontmatter.get("license"):
        graph.add(subject, iri(PREFIXES["dcterms"] + "license"), iri(str(frontmatter["license"])))

    parts = node_id.split("/")
    if len(parts) > 1:
        domain = DOMAIN + quote(parts[0], safe="-_.~")
        if topic:
            graph.add(iri(domain), iri(PREFIXES["skos"] + "member"), topic)
        report.domains[parts[0]] += 1

    for author in _strings(frontmatter.get("authors")):
        graph.add(subject, iri(PREFIXES["schema"] + "author"), literal(author))
    if frontmatter.get("year") is not None:
        graph.add(
            subject,
            iri(PREFIXES["schema"] + "datePublished"),
            literal(str(frontmatter["year"]), datatype=XSD_GYEAR),
        )
    if frontmatter.get("journal"):
        graph.add(subject, iri(VOCAB + "journal"), literal(str(frontmatter["journal"])))
    if frontmatter.get("wikidata") is not None:
        qid = str(frontmatter["wikidata"]).strip()
        if not WIKIDATA_ID.fullmatch(qid):
            raise ValueError(f"{node_id}: wikidata must look like Q123, got {qid!r}")
        graph.add(subject, iri(PREFIXES["rdfs"] + "seeAlso"), iri(WIKIDATA_ENTITY + qid))
        report.discovery_pages += 1
    mappings = reviewed_mappings(frontmatter.get("external_mappings"))
    if mappings and not topic:
        raise ValueError(f"non-concept page {node_id!r} cannot carry SKOS mappings")
    if mappings:
        report.aligned_pages += 1
    for mapping in mappings:
        predicate, target = iri(PREFIXES["skos"] + mapping.relation), iri(mapping.target)
        assert topic is not None  # mappings on a document-only role were rejected above
        graph.add(topic, predicate, target)
        key = hashlib.sha256(
            f"{topic[1:-1]}\n{predicate[1:-1]}\n{target[1:-1]}".encode()
        ).hexdigest()
        statement = iri(BASE + "mapping/" + key)
        for p, o in (
            (RDF_TYPE, iri(PREFIXES["rdf"] + "Statement")),
            (PREFIXES["rdf"] + "subject", topic),
            (PREFIXES["rdf"] + "predicate", predicate),
            (PREFIXES["rdf"] + "object", target),
            (PREFIXES["dcterms"] + "source", iri(mapping.evidence)),
            (PREFIXES["dcterms"] + "description", literal(mapping.rationale, lang="en")),
            (PREFIXES["dcterms"] + "date", literal(mapping.reviewed_on, datatype=XSD_DATE)),
        ):
            graph.add(statement, iri(p), o)
        report.mapping_relations[mapping.relation] += 1
    if frontmatter.get("doi"):
        doi = quote(str(frontmatter["doi"]).strip(), safe="/:;()-._~")
        graph.add(subject, iri(PREFIXES["schema"] + "sameAs"), iri(f"https://doi.org/{doi}"))


def _is_shipped(relative: Path) -> bool:
    return len(relative.parts) == 1 or relative.parts[0] in SHIPPED_DIRECTORIES


def _shipped_index(root: Path) -> LinkIndex:
    """The link index restricted to files the repository ships, in a fixed order.

    ``verify_links`` keeps the first page it meets for a repeated name or alias,
    which depends on the order the filesystem lists directories. The graph must
    not, so repeated names resolve to the alphabetically first path.
    """
    files = [
        path.relative_to(root) for path in _iter_files(root) if _is_shipped(path.relative_to(root))
    ]
    by_rel = {
        _normalise_key(path.with_suffix("").as_posix()): path
        for path in sorted(files)
        if path.suffix.casefold() == ".md"
    }
    by_stem: dict[str, list[Path]] = {}
    by_alias: dict[str, Path] = {}
    for path in sorted(by_rel.values(), key=lambda item: item.as_posix().lower()):
        _require_regular_source(root / path, root)
        by_stem.setdefault(path.stem.lower(), []).append(path)
        for alias in _frontmatter_aliases(root / path):
            by_alias.setdefault(_alias_key(alias), path)
    return LinkIndex(
        markdown_by_stem=by_stem,
        markdown_by_rel_no_suffix=by_rel,
        all_files_by_rel={_normalise_key(path.as_posix()) for path in files},
        markdown_by_alias=by_alias,
    )


def _resolve(root: Path, source: Path, target: str, index: LinkIndex) -> str | None:
    """Return the node id a link points to, ``""`` if it leaves the vault, ``None`` if broken."""
    resolved = _resolve_target_path(root, source, target, index)
    if resolved is None:
        return None
    resolved = Path(os.path.normpath(resolved))  # may carry ".." segments
    try:
        if not _is_shipped(resolved.relative_to(root)):
            return None
    except ValueError:
        return None
    if resolved.suffix.lower() != ".md":
        return None
    try:
        relative = resolved.relative_to(root / KB_NAME)
    except ValueError:
        return ""
    return relative.with_suffix("").as_posix()


def _add_edge(
    graph: Graph,
    report: Report,
    source_id: str,
    target_id: str,
    predicate: str,
    typed: bool,
    *,
    conceptual: bool = False,
) -> None:
    if source_id == target_id:
        report.self_loops_dropped += 1
        return
    resource = concept_iri if conceptual else page_iri
    graph.add(iri(resource(source_id)), iri(predicate), iri(resource(target_id)))
    if typed:
        report.typed_edges += 1
    else:
        report.link_edges += 1


def _route_relation(name: str, both_topics: bool) -> tuple[str, bool, str]:
    """Return predicate, conceptual scope and reason without inventing topic roles."""
    slug = relation_slug(name)
    predicate = relation_predicate(name)
    if (
        slug in {"organizes", "has_parent", "cites", "cited_by"}
        or predicate
        in {
            PREFIXES["bibo"] + "cites",
            PREFIXES["bibo"] + "citedBy",
            PREFIXES["dcterms"] + "references",
            PREFIXES["dcterms"] + "hasPart",
            PREFIXES["dcterms"] + "isPartOf",
        }
        and slug not in {"part_of", "component_of"}
    ):
        return predicate, False, "document relation"
    if slug in {"part_of", "component_of"} and not both_topics:
        return PREFIXES["dcterms"] + "isPartOf", False, "document containment"
    if slug in DOCUMENT_RELATIONS:
        return DOCUMENT_RELATION + slug, False, DOCUMENT_RELATIONS[slug]
    if both_topics:
        if slug in CONCEPT_RELATIONS:
            return RELATION + slug, True, CONCEPT_RELATIONS[slug]
        if predicate in {
            PREFIXES["skos"] + r
            for r in (
                "related",
                "broader",
                "narrower",
                "broaderTransitive",
                "narrowerTransitive",
            )
        }:
            return predicate, True, "standard SKOS concept relation"
    # WHY: a typed source assertion to a guide, citation or navigation page is
    # evidence of the written cross-reference, not evidence of another concept.
    return (
        DOCUMENT_RELATION + slug,
        False,
        (
            f"Source document lists target document under semantic_relations type '{name}'. "
            "Conceptual interpretation is unresolved; no relation between topics is asserted."
        ),
    )


def _describe_vocabulary(graph: Graph, relations: dict[str, tuple[str, bool, str]]) -> None:
    """Declare the graph's own terms so a consumer can interpret them."""
    skos_related = iri(PREFIXES["skos"] + "semanticRelation")
    label, comment = iri(PREFIXES["rdfs"] + "label"), iri(PREFIXES["rdfs"] + "comment")
    sub_property = iri(PREFIXES["rdfs"] + "subPropertyOf")
    prop = iri(PREFIXES["rdf"] + "Property")
    for predicate, (name, conceptual, meaning) in sorted(relations.items()):
        term = iri(predicate)
        graph.add(term, iri(RDF_TYPE), prop)
        if conceptual:
            graph.add(term, sub_property, skos_related)
        scope = PREFIXES["skos"] + "Concept" if conceptual else PREFIXES["schema"] + "CreativeWork"
        graph.add(term, iri(PREFIXES["rdfs"] + "domain"), iri(scope))
        graph.add(term, iri(PREFIXES["rdfs"] + "range"), iri(scope))
        graph.add(term, iri(VOCAB + "sourceRelation"), literal(name))
        graph.add(term, label, literal(relation_slug(name).replace("_", " "), lang="en"))
        graph.add(
            term,
            comment,
            literal(
                meaning
                + " Direction: declaring source to named target. "
                + ("Scope: topic concepts." if conceptual else "Scope: document records."),
                lang="en",
            ),
        )
    for name, text in (
        ("pageType", "The page's `type` frontmatter value, for example mathematical_concept."),
        ("status", "The page's editorial status, for example stable or verified."),
        ("complexity", "The page's declared difficulty level."),
        ("processingPriority", "The page's source-declared editorial processing priority."),
        ("journal", "Journal named by a citation page."),
        (
            "sourceDefinition",
            "Definition frontmatter retained on a document without a declared topic.",
        ),
        (
            "sourceScopeNote",
            "Scope note frontmatter retained on a document without a declared topic.",
        ),
        (
            "sourceRelation",
            "Exact source semantic_relations name whose routing defines this predicate.",
        ),
    ):
        term = iri(VOCAB + name)
        graph.add(term, iri(RDF_TYPE), prop)
        graph.add(term, label, literal(name, lang="en"))
        graph.add(term, comment, literal(text, lang="en"))


def _describe_dataset(graph: Graph, modified: str) -> None:
    dataset = iri(DATASET)
    dcterms, prov = PREFIXES["dcterms"], PREFIXES["prov"]
    graph.add(dataset, iri(RDF_TYPE), iri(PREFIXES["dcat"] + "Dataset"))
    graph.add(dataset, iri(RDF_TYPE), iri(PREFIXES["void"] + "Dataset"))
    graph.add(
        dataset,
        iri(dcterms + "title"),
        literal("Cognitive Active Inference knowledge base", lang="en"),
    )
    graph.add(
        dataset,
        iri(dcterms + "description"),
        literal(
            "Concept pages on Active Inference and the sciences around it, linked by typed "
            "and untyped relations, with citation records and navigation pages.",
            lang="en",
        ),
    )
    graph.add(dataset, iri(dcterms + "license"), iri(LICENSE))
    graph.add(dataset, iri(dcterms + "creator"), iri(ORGANISATION))
    graph.add(dataset, iri(dcterms + "publisher"), iri(ORGANISATION))
    graph.add(dataset, iri(dcterms + "source"), iri(SOURCE_REPOSITORY))
    graph.add(dataset, iri(PREFIXES["dcat"] + "landingPage"), iri(SOURCE_REPOSITORY))
    for target in ("wikidata", "identifier-mappings", "ubergraph"):
        graph.add(
            dataset,
            iri(PREFIXES["rdfs"] + "seeAlso"),
            iri("https://registry.okn.us/registry/kgs/" + target + "/"),
        )
    graph.add(dataset, iri(PREFIXES["void"] + "vocabulary"), iri(PREFIXES["skos"]))
    graph.add(dataset, iri(PREFIXES["void"] + "vocabulary"), iri(dcterms))
    graph.add(dataset, iri(PREFIXES["void"] + "vocabulary"), iri(PREFIXES["schema"]))
    graph.add(dataset, iri(PREFIXES["void"] + "vocabulary"), iri(prov))
    graph.add(dataset, iri(PREFIXES["void"] + "vocabulary"), iri(PREFIXES["bibo"]))
    if modified:
        graph.add(dataset, iri(dcterms + "modified"), typed_date(modified))
    graph.add(dataset, iri(prov + "wasDerivedFrom"), iri(SOURCE_REPOSITORY))
    graph.add(dataset, iri(prov + "wasGeneratedBy"), iri(ACTIVITY))

    activity = iri(ACTIVITY)
    graph.add(activity, iri(RDF_TYPE), iri(prov + "Activity"))
    graph.add(activity, iri(prov + "used"), iri(SOURCE_REPOSITORY))
    graph.add(activity, iri(prov + "wasAssociatedWith"), iri(EXPORTER_URL))
    graph.add(iri(EXPORTER_URL), iri(RDF_TYPE), iri(prov + "SoftwareAgent"))
    graph.add(iri(EXPORTER_URL), iri(RDF_TYPE), iri(PREFIXES["schema"] + "SoftwareSourceCode"))

    scheme = iri(SCHEME)
    graph.add(scheme, iri(RDF_TYPE), iri(PREFIXES["skos"] + "ConceptScheme"))
    graph.add(
        scheme, iri(dcterms + "title"), literal("Cognitive Active Inference concepts", lang="en")
    )
    graph.add(scheme, iri(dcterms + "license"), iri(LICENSE))
    graph.add(scheme, iri(PREFIXES["void"] + "inDataset"), dataset)


def build_graph(repo_root: Path) -> tuple[Graph, Report]:
    """Build the knowledge-base graph from the vault under ``repo_root``."""
    root = Path(repo_root).resolve()
    kb_root = root / KB_NAME
    if not kb_root.is_dir():
        raise FileNotFoundError(f"knowledge base not found: {kb_root}")
    index = _shipped_index(root)
    graph, report = Graph(), Report()
    used_relations: dict[str, tuple[str, bool, str]] = {}
    updated_dates: list[str] = []
    domain_names: set[str] = set()

    records: dict[str, tuple[Path, dict[str, Any], str]] = {}
    for path in _markdown_paths(kb_root):
        _require_regular_source(path, root)
        node_id = path.relative_to(kb_root).with_suffix("").as_posix()
        frontmatter, body = _split(path.read_text(encoding="utf-8"))
        records[node_id] = path, frontmatter, body
    topics = {node_id for node_id, (path, fm, _) in records.items() if _has_topic(path, fm)}
    for node_id, (path, frontmatter, body) in records.items():
        report.pages += 1
        _describe_page(graph, node_id, path, frontmatter, report)
        for key in ("updated", "modified"):
            if frontmatter.get(key) is not None:
                updated_dates.append(date_text(frontmatter[key]))
        if "/" in node_id:
            domain_names.add(node_id.split("/", 1)[0])

        redirect_target = ""
        if str(frontmatter.get("type")) in {"redirect", "citation_redirect"}:
            report.redirects += 1
            redirect_target = _link_target(frontmatter.get("redirect"))

        links: list[tuple[str, str, str]] = [
            ("typed", relation, target)
            for relation, target in semantic_relations(frontmatter.get("semantic_relations"))
        ]
        links += [("link", "", target) for target in body_links(body)]
        if redirect_target:
            links.append(("redirect", "", redirect_target))
        for kind, relation, target in links:
            resolved = _resolve(root, path, target, index)
            if resolved is None:
                report.unresolved_links += 1
                continue
            if resolved == "":
                report.links_outside_graph += 1
                continue
            conceptual = False
            if kind == "redirect":
                predicate = REDIRECT_PREDICATE
            elif kind == "typed":
                predicate, conceptual, reason = _route_relation(
                    relation, node_id in topics and resolved in topics
                )
                if predicate.startswith(RELATION):
                    used_relations[predicate] = (relation, conceptual, reason)
                    graph.add(iri(predicate), iri(VOCAB + "sourceRelation"), literal(relation))
                if predicate.startswith(DOCUMENT_RELATION) and node_id != resolved:
                    report.document_relation_assertions += 1
                    report.document_relation_routes.append(
                        {
                            "source": node_id,
                            "relation": relation,
                            "target": resolved,
                            "predicate": predicate,
                            "reason": reason,
                        }
                    )
                report.relations[relation_slug(relation)] += 1
            else:
                predicate = WIKILINK_PREDICATE
            _add_edge(
                graph, report, node_id, resolved, predicate, kind != "link", conceptual=conceptual
            )

    for name in sorted(domain_names):
        domain = iri(DOMAIN + quote(name, safe="-_.~"))
        graph.add(domain, iri(RDF_TYPE), iri(PREFIXES["skos"] + "Collection"))
        graph.add(
            domain,
            iri(PREFIXES["skos"] + "prefLabel"),
            literal(name.replace("_", " ").title(), lang="en"),
        )
        graph.add(domain, iri(PREFIXES["void"] + "inDataset"), iri(DATASET))

    report.modified = max(updated_dates) if updated_dates else ""
    _describe_vocabulary(graph, used_relations)
    _describe_dataset(graph, report.modified)
    keywords = {o for _, p, o in graph.triples if p == iri(PREFIXES["schema"] + "keywords")}
    report.keyword_concepts = sum(o.startswith("<" + BASE + "keyword/") for o in keywords)
    report.keyword_links = sum(
        p == iri(PREFIXES["schema"] + "keywords") for _, p, _ in graph.triples
    )
    report.triples = len(graph.triples)
    return graph, report


# --------------------------------------------------------------------- serializers
def to_nquads(graph: Graph) -> str:
    """Serialise ``graph`` as N-Quads in the graph named ``GRAPH``."""
    named = iri(GRAPH)
    return "".join(f"{s} {p} {o} {named} .\n" for s, p, o in graph.sorted())


_LOCAL = re.compile(r"[A-Za-z0-9_](?:[A-Za-z0-9_\-/]|\.(?=[A-Za-z0-9_\-/]))*")


def _compact(term: str) -> str:
    """Shorten an N-Triples IRI term to a Turtle prefixed name when that is exact."""
    if not term.startswith("<"):
        return term
    value = term[1:-1]
    best = ""
    for prefix, namespace in sorted(PREFIXES.items(), key=lambda item: -len(item[1])):
        if value.startswith(namespace):
            local = value[len(namespace) :]
            if local and _LOCAL.fullmatch(local):
                escaped = local.replace("/", "\\/")  # Turtle escapes a slash in a local name
                best = f"{prefix}:{escaped}"
                break
    return best or term


def _compact_literal(term: str) -> str:
    if term.endswith(">") and "^^<" in term:
        text, datatype = term.rsplit("^^", 1)
        return f"{text}^^{_compact(datatype)}"
    return term


def to_turtle(graph: Graph, subjects: set[str] | None = None) -> str:
    """Serialise ``graph`` as Turtle, optionally keeping only some subjects."""
    lines = [f"@prefix {prefix}: <{namespace}> ." for prefix, namespace in PREFIXES.items()]
    lines.append("")
    triples = [t for t in graph.sorted() if subjects is None or t[0] in subjects]
    by_subject: dict[str, list[tuple[str, str]]] = {}
    for subject, predicate, obj in triples:
        by_subject.setdefault(subject, []).append((predicate, obj))
    for subject in sorted(by_subject):
        by_predicate: dict[str, list[str]] = {}
        for predicate, obj in by_subject[subject]:
            by_predicate.setdefault(predicate, []).append(obj)
        lines.append(_compact(subject))
        predicates = sorted(by_predicate, key=lambda p: (p != iri(RDF_TYPE), p))
        for position, predicate in enumerate(predicates):
            name = "a" if predicate == iri(RDF_TYPE) else _compact(predicate)
            objects = ", ".join(_compact_literal(_compact(o)) for o in by_predicate[predicate])
            end = " ." if position == len(predicates) - 1 else " ;"
            lines.append(f"    {name} {objects}{end}")
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------- docs outputs
def sample_turtle(graph: Graph) -> str:
    """The committed sample: the dataset description plus a few pages in full."""
    pages = {iri(page_iri(page)) for page in SAMPLE_PAGES}
    pages |= {
        o for s, p, o in graph.triples if s in pages and p == iri(PREFIXES["schema"] + "about")
    }
    pages |= {
        s for s, p, o in graph.triples if p == iri(PREFIXES["rdf"] + "subject") and o in pages
    }
    pages |= {
        o for s, p, o in graph.triples if s in pages and p == iri(PREFIXES["schema"] + "keywords")
    }
    fixed = {iri(DATASET), iri(SCHEME), iri(ACTIVITY), iri(EXPORTER_URL)}
    fixed |= {iri(DOMAIN + page.split("/", 1)[0]) for page in SAMPLE_PAGES if "/" in page}
    member = iri(PREFIXES["skos"] + "member")
    terms = {p for s, p, _ in graph.triples if s in pages and p.startswith(f"<{VOCAB}")}
    terms |= {p for s, p, _ in graph.triples if s in pages and p.startswith(f"<{RELATION}")}

    def keep(triple: Triple) -> bool:
        subject, predicate, obj = triple
        if subject in pages or subject in terms:
            return True
        # A domain lists every member; the sample keeps only the sampled ones.
        return subject in fixed and (predicate != member or obj in pages)

    narrowed = Graph({t for t in graph.triples if keep(t)})
    header = (
        "# Sample of the cognitive-kb graph: the dataset description, selected pages described\n"
        "# in full, and the terms those pages use. The complete graph is produced by\n"
        "# `python -m scripts.export_rdf`. Regenerate this file with `--write-docs`.\n\n"
    )
    return header + to_turtle(narrowed)


def statistics_markdown(report: Report) -> str:
    """The generated statistics page; every figure in the OKN docs comes from here."""
    lines = [
        "---",
        "title: cognitive-kb graph statistics",
        "description: Counts from the RDF export of the knowledge base; generated, do not edit.",
        "type: reference",
        "status: stable",
        "---",
        "",
        "# cognitive-kb graph statistics",
        "",
        "Generated by `python -m scripts.export_rdf --write-docs`. Edit the exporter or the",
        "pages, then regenerate; never edit this file.",
        "",
        "| Quantity | Count |",
        "|---|---|",
        f"| Pages (all described as `schema:CreativeWork`) | {report.pages} |",
        f"| Separate page-topic concepts (`skos:Concept`) | {report.concepts} |",
        f"| Document-scoped source relation assertions | {report.document_relation_assertions} |",
        f"| Additional local keyword concepts | {report.keyword_concepts} |",
        f"| Concept-valued keyword links | {report.keyword_links} |",
        f"| Navigation pages (`schema:CollectionPage`) | {report.navigation_pages} |",
        f"| Redirect pages | {report.redirects} |",
        f"| Documents whose topics have reviewed external mappings | {report.aligned_pages} |",
        f"| Wikidata discovery links (`rdfs:seeAlso`, not mappings) | {report.discovery_pages} |",
        f"| Reviewed exact matches | {report.mapping_relations['exactMatch']} |",
        f"| Reviewed close matches | {report.mapping_relations['closeMatch']} |",
        f"| Reviewed associative matches | {report.mapping_relations['relatedMatch']} |",
        f"| Typed edges from `semantic_relations` | {report.typed_edges} |",
        f"| Untyped edges from body links (`dcterms:references`) | {report.link_edges} |",
        f"| Triples | {report.triples} |",
        f"| Links to concepts that have no page yet (not exported) | {report.unresolved_links} |",
        f"| Links to files outside the knowledge base (not exported) | "
        f"{report.links_outside_graph} |",
        f"| Self-links dropped | {report.self_loops_dropped} |",
        f"| Latest page update | {report.modified} |",
        "",
        "## Pages by domain",
        "",
        "| Domain | Pages |",
        "|---|---|",
    ]
    lines += [f"| `{name}` | {count} |" for name, count in sorted(report.domains.items())]
    lines += ["", "## Pages by `type`", "", "| Type | Pages |", "|---|---|"]
    lines += [
        f"| `{name}` | {count} |"
        for name, count in sorted(report.page_types.items(), key=lambda i: (-i[1], i[0]))
    ]
    lines += ["", "## Typed edges by relation", "", "| Relation | Edges |", "|---|---|"]
    lines += [
        f"| `{name}` | {count} |"
        for name, count in sorted(report.relations.items(), key=lambda i: (-i[1], i[0]))
    ]
    lines += [
        "",
        "## Document-scoped source assertions",
        "",
        "These preserve source labels and directions without asserting relations between topics.",
        "",
        "| Source document | Source relation | Target document | Routing reason |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| `{r['source']}` | `{r['relation']}` | `{r['target']}` | {r['reason']} |"
        for r in sorted(
            report.document_relation_routes, key=lambda r: (r["source"], r["relation"], r["target"])
        )
    ]
    return "\n".join(lines) + "\n"


def alignment_markdown(graph: Graph) -> str:
    """Generate mapping coverage and an incoming-link review queue from RDF."""
    title = iri(PREFIXES["dcterms"] + "title")
    labels = {s: o for s, p, o in graph.triples if p == title and s.startswith("<" + PAGE)}
    topic_pages = {o: s for s, p, o in graph.triples if p == iri(PREFIXES["schema"] + "about")}
    predicates = {
        iri(PREFIXES["skos"] + r)
        for r in ("exactMatch", "closeMatch", "broadMatch", "narrowMatch", "relatedMatch")
    }
    mappings: dict[str, list[tuple[str, str]]] = {}
    incoming: Counter = Counter()
    for subject, predicate, target in graph.triples:
        source_page = topic_pages.get(subject, subject)
        target_page = topic_pages.get(target, target)
        if source_page in labels and target_page in labels and source_page != target_page:
            incoming[target_page] += 1
        if predicate in predicates:
            mappings.setdefault(topic_pages.get(subject, subject), []).append((predicate, target))
    lines = [
        "---",
        "title: Mapping review and priority queue",
        "type: reference",
        "status: stable",
        "---",
        "",
        "# Mapping review and priority queue",
        "",
        "Generated from the current export; regenerate with `--write-docs`.",
        "Incoming links count exported document references and topic relations projected "
        "to their source documents. Mapping dispositions describe the separate topic concepts.",
        "Only explicit source-reviewed records become SKOS mappings. "
        "Wikidata discovery links are not equivalence evidence.",
        "",
        "## Most frequently referenced pages",
        "",
        "| Page | Incoming links | Mapping disposition |",
        "|---|---|---|",
    ]
    for subject, count in sorted(incoming.items(), key=lambda item: (-item[1], item[0]))[:25]:
        node = subject[len(PAGE) + 1 : -1]
        disposition = "; ".join(
            p.rsplit("#", 1)[-1][:-1] + " " + o for p, o in sorted(mappings.get(subject, []))
        )
        lines.append(f"| `{node}` | {count} | {disposition or 'No reviewed equivalent asserted'} |")
    lines += [
        "",
        "## Reviewed correspondences",
        "",
        "| Document path | Separate topic IRI | Predicate | Target |",
        "|---|---|---|---|",
    ]
    for subject in sorted(mappings):
        for predicate, target in sorted(mappings[subject]):
            lines.append(
                f"| `{subject[len(PAGE) + 1 : -1]}` | "
                f"{iri(CONCEPT + subject[len(PAGE) + 1 : -1])} | "
                f"`{predicate.rsplit('#', 1)[-1][:-1]}` | {target} |"
            )
    return "\n".join(lines) + "\n"


def expected_docs(root: Path) -> dict[Path, str]:
    """Map each generated file under ``docs/okn`` to the content it should have."""
    graph, report = build_graph(root)
    return {
        root / DOCS_DIR / SAMPLE_NAME: sample_turtle(graph),
        root / DOCS_DIR / STATISTICS_NAME: statistics_markdown(report),
        root / DOCS_DIR / ALIGNMENT_NAME: alignment_markdown(graph),
    }


def stale_docs(root: Path) -> list[str]:
    """Return the generated docs that are missing or out of date.

    A tree with no knowledge base has nothing to export, so nothing is stale.
    """
    if not (root / KB_NAME).is_dir():
        return []
    stale = []
    for path, content in expected_docs(root).items():
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            stale.append(path.relative_to(root).as_posix())
    return sorted(stale)


def submission_errors(graph: Graph) -> list[str]:
    """Check machine-observable OKN criteria; acceptance remains a curator decision."""
    return submission_diagnostics(graph)["errors"]


def submission_diagnostics(graph: Graph) -> dict[str, Any]:
    """Combine submission metadata checks and semantic quality using one index."""
    index = GraphIndex.build(graph.triples)
    quality = semantic_quality(graph, index=index)
    errors: list[str] = []
    license_predicate = iri(PREFIXES["dcterms"] + "license")
    allowed = {
        iri("https://creativecommons.org/licenses/by/4.0/"),
        iri("https://creativecommons.org/publicdomain/zero/1.0/"),
    }
    for subject in (DATASET, SCHEME):
        licenses = index.objects(iri(subject), PREFIXES["dcterms"] + "license")
        if len(licenses) != 1 or not licenses <= allowed:
            errors.append(f"{subject}: requires one CC-BY 4.0 or CC0 1.0 license")
    for subject, obj in index.by_predicate.get(license_predicate, set()):
        if obj not in allowed:
            errors.append(f"{subject}: incompatible license {obj}")
    pages = {s for s, o in index.pairs(RDF_TYPE) if o == iri(PREFIXES["schema"] + "CreativeWork")}
    if not pages:
        errors.append("graph contains no pages")
    dangling = {
        o
        for pairs in index.by_predicate.values()
        for _, o in pairs
        if o.startswith(iri(PAGE)[:-1]) and o not in pages
    }
    if dangling:
        errors.append(f"dangling page objects: {len(dangling)}")
    topics = {o for _, o in index.pairs(PREFIXES["schema"] + "about")}
    if not any(s in topics for r in RELATIONS for s, _ in index.pairs(PREFIXES["skos"] + r)):
        errors.append("no external concept mappings; document an interoperability review")
    errors.extend(
        f"{finding['code']}: {finding['message']} {' '.join(finding['terms'])}"
        for finding in quality["errors"]
    )
    provenance = (iri(DATASET), iri(PREFIXES["prov"] + "wasDerivedFrom"), iri(SOURCE_REPOSITORY))
    if provenance not in graph.triples:
        errors.append("dataset source provenance is missing")
    return {"machine_checks_passed": not errors, "errors": sorted(set(errors)), "quality": quality}


def _markdown_paths(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.suffix.casefold() == ".md")


def _require_regular_source(path: Path, root: Path) -> None:
    if (
        not path.is_file()
        or path.is_symlink()
        or any(parent.is_symlink() for parent in path.parents if parent.is_relative_to(root))
    ):
        raise ValueError(f"export source must be a regular file without symlinks: {path}")


def _source_hashes(root: Path) -> dict[str, str]:
    paths = {path for path in root.iterdir() if path.suffix.casefold() == ".md"}
    for directory in SHIPPED_DIRECTORIES:
        paths.update(_markdown_paths(root / directory))
    paths.update(
        path
        for name in RUNTIME_SOURCE_NAMES
        if (path := Path(__file__).resolve().with_name(name)).is_relative_to(root)
    )
    for name in ("pyproject.toml", "uv.lock", ".python-version"):
        path = root / name
        if path.exists() or path.is_symlink():
            paths.add(path)
    hashes = {}
    for path in sorted(paths):
        _require_regular_source(path, root)
        hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _require_stable_sources(root: Path, expected: dict[str, str]) -> None:
    if _source_hashes(root) != expected:
        raise ValueError("source inputs changed during export; retry after source writers finish")


def _resolution_files(root: Path) -> list[str]:
    names: set[str] = set()

    def fail(error: OSError) -> None:
        raise error

    for path in root.iterdir():
        if path.name in SHIPPED_DIRECTORIES:
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f"resolution input must be a real directory: {path}")
            for parent, directories, files in os.walk(path, followlinks=False, onerror=fail):
                for name in directories:
                    if (Path(parent) / name).is_symlink():
                        raise ValueError(f"resolution directory cannot be a symlink: {name}")
                for name in files:
                    source = Path(parent) / name
                    _require_regular_source(source, root)
                    names.add(source.relative_to(root).as_posix())
        elif path.name in {"pyproject.toml", "uv.lock", ".python-version"} or not path.is_dir():
            _require_regular_source(path, root)
            names.add(path.name)
    return sorted(names)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", nargs="?", default=".", help="repository root")
    parser.add_argument(
        "--output", default="output/okn", help="directory for cognitive-kb.nq and .ttl"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="exit 1 if docs/okn is stale")
    mode.add_argument(
        "--check-submission", action="store_true", help="exit 1 if machine OKN criteria fail"
    )
    mode.add_argument(
        "--write-docs", action="store_true", help="refresh docs/okn sample and statistics"
    )
    mode.add_argument(
        "--quality-report",
        action="store_true",
        help="print deterministic semantic JSON; exit 1 on fatal findings",
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()

    if args.check:
        stale = stale_docs(root)
        for item in stale:
            print(item)
        print(
            "RDF docs are current"
            if not stale
            else f"{len(stale)} RDF doc(s) stale (run --write-docs)"
        )
        return 1 if stale else 0
    if args.write_docs:
        for path, content in expected_docs(root).items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            print(f"wrote {path.relative_to(root)}")
        return 0

    source_hashes = {} if args.quality_report or args.check_submission else _source_hashes(root)
    resolution_files = (
        [] if args.quality_report or args.check_submission else _resolution_files(root)
    )
    graph, report = build_graph(root)
    if args.quality_report:
        quality = semantic_quality(graph)
        print(json.dumps(quality, indent=2, sort_keys=True))
        return 0 if quality["machine_checks_passed"] else 1
    if args.check_submission:
        diagnostics = submission_diagnostics(graph)
        result = {
            **diagnostics,
            "aligned_pages": report.aligned_pages,
            "reviewed_mapping_relations": dict(sorted(report.mapping_relations.items())),
            "discovery_pages": report.discovery_pages,
            "unresolved_links": report.unresolved_links,
        }
        print(json.dumps(result, indent=2))
        return 0 if diagnostics["machine_checks_passed"] else 1
    _require_stable_sources(root, source_hashes)
    if _resolution_files(root) != resolution_files:
        raise ValueError("resolution file-name inventory changed during export; retry when stable")
    out = (root / args.output) if not Path(args.output).is_absolute() else Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    artifacts = {"cognitive-kb.nq": to_nquads(graph), "cognitive-kb.ttl": to_turtle(graph)}
    for name, content in artifacts.items():
        (out / name).write_text(content, encoding="utf-8")
    manifest = {
        "graph": GRAPH,
        "license": LICENSE,
        "artifacts": {
            name: hashlib.sha256(content.encode()).hexdigest()
            for name, content in artifacts.items()
        },
        "sources": source_hashes,
        "resolution_files": resolution_files,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "pages": report.pages,
        "concepts": report.concepts,
        "typed_edges": report.typed_edges,
        "link_edges": report.link_edges,
        "triples": report.triples,
        "unresolved_links": report.unresolved_links,
        "output": str(out),
    }
    print(
        json.dumps(summary, indent=2)
        if args.json
        else "\n".join(f"{k}: {v}" for k, v in summary.items())
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
