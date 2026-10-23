"""Deterministic, offline semantic diagnostics for the cognitive RDF profile.

This checks the exporter's N-Triples terms, not arbitrary RDF syntax or OWL.
SKOS entailments are limited to the documented standard relation properties.
``exactMatch`` closure never merges resource identities or propagates hierarchy.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

try:
    from scripts.rdf_alignment import RELATIONS, checked_iri
    from scripts.rdf_relations import CONCEPT_RELATIONS, DOCUMENT_PAGE_TYPES
except ImportError:
    from rdf_alignment import RELATIONS, checked_iri  # type: ignore[no-redef]
    from rdf_relations import CONCEPT_RELATIONS, DOCUMENT_PAGE_TYPES  # type: ignore[no-redef]

BASE = "https://purl.org/okn/frink/kg/cognitive-kb/"
SKOS = "http://www.w3.org/2004/02/skos/core#"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
DCT = "http://purl.org/dc/terms/"
SCHEMA = "https://schema.org/"
BIBO = "http://purl.org/ontology/bibo/"
XSD = "http://www.w3.org/2001/XMLSchema#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
Triple = tuple[str, str, str]


def _iri(value: str) -> str:
    return f"<{value}>"


class TripleGraph(Protocol):
    """The stdlib exporter graph interface, without importing the exporter."""

    triples: set[Triple]


@dataclass
class GraphIndex:
    """Index the input once; per-mapping evidence queries use keyed lookups."""

    by_subject: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    by_predicate: dict[str, set[tuple[str, str]]] = field(default_factory=dict)
    triples_indexed: int = 0

    @classmethod
    def build(cls, triples: Iterable[Triple]) -> GraphIndex:
        index = cls()
        for subject, predicate, obj in triples:
            index.by_subject.setdefault(subject, {}).setdefault(predicate, set()).add(obj)
            index.by_predicate.setdefault(predicate, set()).add((subject, obj))
            index.triples_indexed += 1
        return index

    def objects(self, subject: str, predicate: str) -> set[str]:
        """Return the indexed objects of one subject and full predicate IRI."""
        return self.by_subject.get(subject, {}).get(_iri(predicate), set())

    def pairs(self, predicate: str) -> set[tuple[str, str]]:
        """Return the indexed subject/object pairs of a full predicate IRI."""
        return self.by_predicate.get(_iri(predicate), set())


_LITERAL = re.compile(
    r'"((?:[^"\\\n\r]|\\(?:[tbnrf"\x27\\]|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}))*)"'
    r"(?:@([a-zA-Z]+(?:-[a-zA-Z0-9]+)*)|\^\^<([^<>]+)>)?"
)
_ESCAPES = {"t": "\t", "b": "\b", "n": "\n", "r": "\r", "f": "\f", '"': '"', "'": "'", "\\": "\\"}


def _literal(term: str) -> tuple[str, str, str] | None:
    match = _LITERAL.fullmatch(term)
    if not match:
        return None

    def unescape(escape: re.Match[str]) -> str:
        value = escape.group()[1:]
        return chr(int(value[1:], 16)) if value[0] in "uU" else _ESCAPES[value]

    try:
        text = re.sub(r"\\(?:u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|.)", unescape, match[1])
    except (ValueError, OverflowError):
        return None
    if any(0xD800 <= ord(char) <= 0xDFFF for char in text):
        return None
    return text, (match[2] or "").lower(), match[3] or ""


def _http_term(term: str) -> bool:
    if not term.startswith("<") or not term.endswith(">"):
        return False
    try:
        checked_iri(term[1:-1])
    except ValueError:
        return False
    return True


def _date_term(term: str) -> bool:
    parsed = _literal(term)
    if parsed is None or parsed[1] or parsed[2] != XSD + "date":
        return False
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", parsed[0]):
        return False
    try:
        date.fromisoformat(parsed[0])
    except ValueError:
        return False
    return True


def _reachable(adjacency: dict[str, set[str]], source: str, target: str) -> bool:
    pending = list(adjacency.get(source, ()))
    seen: set[str] = set()
    while pending:
        node = pending.pop()
        if node == target:
            return True
        if node not in seen:
            seen.add(node)
            pending.extend(adjacency.get(node, ()))
    return False


def _components(adjacency: dict[str, set[str]]) -> list[list[str]]:
    """Find directed strongly connected components without recursion depth limits."""
    nodes = set(adjacency) | {n for values in adjacency.values() for n in values}
    reverse: dict[str, set[str]] = defaultdict(set)
    for source, targets in adjacency.items():
        for target in targets:
            reverse[target].add(source)
    seen: set[str] = set()
    order: list[str] = []
    for node in sorted(nodes):
        if node in seen:
            continue
        pending = [(node, False)]
        while pending:
            current, finished = pending.pop()
            if finished:
                order.append(current)
            elif current not in seen:
                seen.add(current)
                pending.append((current, True))
                pending.extend((n, False) for n in sorted(adjacency.get(current, ()), reverse=True))
    seen.clear()
    components = []
    for node in reversed(order):
        if node in seen:
            continue
        component = []
        pending_nodes = [node]
        seen.add(node)
        while pending_nodes:
            current = pending_nodes.pop()
            component.append(current)
            for target in reverse.get(current, ()):
                if target not in seen:
                    seen.add(target)
                    pending_nodes.append(target)
        components.append(sorted(component))
    return sorted(components)


def _entity_model_findings(
    idx: GraphIndex,
    concepts: set[str],
    pages: set[str],
    emit: Callable[[str, str, Iterable[str], str], None],
) -> None:
    """Reject role contamination even when a corrupt graph omits its type triples."""
    resources = set(idx.by_subject) | {
        o for pairs in idx.by_predicate.values() for _, o in pairs if o.startswith("<")
    }
    page_iris = {s for s in resources if s.startswith("<" + BASE + "page/")}
    topics = {s for s in resources if s.startswith("<" + BASE + "concept/")}
    for subject in sorted((concepts & pages) | (concepts & page_iris) | (topics & pages)):
        emit(
            "error",
            "document_concept_overlap",
            [subject],
            "Document and topic concept identities must be distinct.",
        )
    for page in sorted(page_iris - pages):
        emit("error", "page_declaration", [page], "Local page requires CreativeWork declaration.")
    for topic in sorted(topics):
        page = _iri(BASE + "page/" + topic[len("<" + BASE + "concept/") : -1])
        if (
            topic not in concepts
            or page not in pages
            or any(idx.objects(page, p) != {topic} for p in (SCHEMA + "about", DCT + "subject"))
            or _iri(BASE + "scheme") not in idx.objects(topic, SKOS + "inScheme")
        ):
            emit(
                "error",
                "topic_document_link",
                [page, topic],
                "Local topic requires Concept, scheme and matching document about/subject links.",
            )
    for predicate in (SCHEMA + "about", DCT + "subject"):
        for page, topic in sorted(idx.pairs(predicate)):
            expected = _iri(BASE + "concept/" + page[len("<" + BASE + "page/") : -1])
            page_types = {
                parsed[0]
                for t in idx.objects(page, BASE + "vocab/pageType")
                if (parsed := _literal(t))
            }
            node_id = page[len("<" + BASE + "page/") : -1]
            if (
                page not in pages
                or page not in page_iris
                or topic not in concepts
                or topic != expected
                or page_types & DOCUMENT_PAGE_TYPES
                or node_id.rsplit("/", 1)[-1].lower() in {"readme", "agents", "index", "catalog"}
                or node_id.startswith("citations/")
                or _iri(SCHEMA + "CollectionPage") in idx.objects(page, RDF + "type")
            ):
                emit(
                    "error",
                    "document_topic_endpoint",
                    [page, _iri(predicate), topic],
                    "Document-topic links require the matching declared topic "
                    "of a topic-bearing page.",
                )
    document_metadata = {
        DCT + p for p in ("title", "description", "created", "modified", "source", "license")
    } | {SCHEMA + p for p in ("author", "datePublished", "sameAs", "keywords", "alternateName")}
    document_metadata |= {
        BASE + "vocab/" + p
        for p in (
            "status",
            "pageType",
            "complexity",
            "processingPriority",
            "journal",
            "sourceDefinition",
            "sourceScopeNote",
        )
    }
    for predicate in sorted(document_metadata):
        for subject, obj in sorted(idx.pairs(predicate)):
            if subject in concepts or subject in topics:
                emit(
                    "error",
                    "concept_document_metadata",
                    [subject, _iri(predicate), obj],
                    "Source-document metadata belongs on the document record.",
                )
    for predicate in (
        "prefLabel",
        "altLabel",
        "hiddenLabel",
        "definition",
        "scopeNote",
        "inScheme",
    ):
        for subject, obj in sorted(idx.pairs(SKOS + predicate)):
            if subject in pages or subject in page_iris:
                emit(
                    "error",
                    "document_skos_metadata",
                    [subject, _iri(SKOS + predicate), obj],
                    "SKOS concept metadata must not be attached to a document.",
                )
    for collection, member in sorted(idx.pairs(SKOS + "member")):
        if collection.startswith("<" + BASE + "domain/") and (
            member not in concepts or member not in topics
        ):
            emit(
                "error",
                "domain_member_endpoint",
                [collection, member],
                "Domain collections contain topic concepts, not documents.",
            )
    document_predicates = {DCT + p for p in ("references", "isReplacedBy", "isPartOf", "hasPart")}
    document_predicates |= {BIBO + "cites", BIBO + "citedBy"}
    for predicate in sorted(document_predicates):
        for subject, obj in sorted(idx.pairs(predicate)):
            if any(
                t in concepts or t in topics or (t.startswith("<" + BASE) and t not in pages)
                for t in (subject, obj)
            ):
                emit(
                    "error",
                    "document_relation_endpoint",
                    [subject, _iri(predicate), obj],
                    "Document references, citations, redirects and containment require documents.",
                )
    semantic = {
        _iri(SKOS + r)
        for r in RELATIONS
        | {
            "semanticRelation",
            "related",
            "broader",
            "narrower",
            "broaderTransitive",
            "narrowerTransitive",
        }
    }
    # Follow declared sub-properties as well: removing an explicit Concept type
    # must not let an indirect SKOS domain/range implication pass unnoticed.
    changed = True
    while changed:
        parents = {s for s, o in idx.pairs(RDFS + "subPropertyOf") if o in semantic}
        changed = not parents <= semantic
        semantic |= parents
    for predicate in sorted(semantic):
        for subject, obj in sorted(idx.by_predicate.get(predicate, set())):
            if any(
                t in pages or t in page_iris or (t.startswith("<" + BASE) and t not in concepts)
                for t in (subject, obj)
            ):
                emit(
                    "error",
                    "concept_relation_endpoint",
                    [subject, predicate, obj],
                    "Conceptual relations and mapping endpoints require concepts, never documents.",
                )
    local_properties = {p for p in idx.by_predicate if p.startswith("<" + BASE + "relation/")}
    for predicate in sorted(local_properties):
        document = predicate.startswith("<" + BASE + "relation/document/")
        name = predicate[len("<" + BASE + "relation/") : -1]
        scope = _iri(SCHEMA + "CreativeWork" if document else SKOS + "Concept")
        meanings = [_literal(t) for t in idx.objects(predicate, RDFS + "comment")]
        source_names = [_literal(t) for t in idx.objects(predicate, BASE + "vocab/sourceRelation")]
        if (
            not document
            and name not in CONCEPT_RELATIONS
            or idx.objects(predicate, RDFS + "domain") != {scope}
            or idx.objects(predicate, RDFS + "range") != {scope}
            or not any(v and v[0].strip() and v[1:] == ("en", "") for v in meanings)
            or not any(v and v[0].strip() and not v[2] for v in source_names)
            or (predicate in semantic) == document
            or _iri(RDF + "Property") not in idx.objects(predicate, RDF + "type")
        ):
            emit(
                "error",
                "relation_scope_declaration",
                [predicate],
                "Local relation requires documented meaning, source name "
                "and correct document/concept scope.",
            )
        expected_resources = pages if document else concepts
        for subject, obj in sorted(idx.by_predicate[predicate]):
            if subject not in expected_resources or obj not in expected_resources:
                emit(
                    "error",
                    "relation_scope_endpoint",
                    [subject, predicate, obj],
                    "Local relation endpoints must match its declared scope.",
                )


def semantic_quality(graph: TripleGraph, *, index: GraphIndex | None = None) -> dict[str, Any]:
    """Return versioned findings and coverage; only ``errors`` block submission.

    Missing editorial fields and SKOS-permitted cycles are warnings. Mapping
    evidence is a stricter local submission profile, not a SKOS integrity axiom.
    Every finding includes N-Triples terms to locate its source assertion.
    """
    idx = index if index is not None else GraphIndex.build(graph.triples)
    findings: list[dict[str, Any]] = []

    def emit(severity: str, code: str, terms: Iterable[str], message: str) -> None:
        findings.append(
            {"severity": severity, "code": code, "terms": list(terms), "message": message}
        )

    types = idx.pairs(RDF + "type")
    concepts = {s for s, o in types if o == _iri(SKOS + "Concept")}
    pages = {s for s, o in types if o == _iri(SCHEMA + "CreativeWork")}
    _entity_model_findings(idx, concepts, pages, emit)
    mappings = {(s, _iri(SKOS + r), o) for r in RELATIONS for s, o in idx.pairs(SKOS + r)}
    semantic_relations = RELATIONS | {
        "semanticRelation",
        "broader",
        "narrower",
        "broaderTransitive",
        "narrowerTransitive",
        "related",
    }
    for relation in sorted(semantic_relations):
        for subject, target in sorted(idx.pairs(SKOS + relation)):
            if not _http_term(subject) or not _http_term(target):
                emit(
                    "error",
                    "semantic_relation_endpoint",
                    [subject, _iri(SKOS + relation), target],
                    "SKOS object properties require resource IRIs in this profile.",
                )

    iris = set(idx.by_subject) | set(idx.by_predicate)
    iris |= {o for pairs in idx.by_predicate.values() for _, o in pairs if o.startswith("<")}
    for term in sorted(iris):
        if not _http_term(term):
            emit(
                "error",
                "invalid_http_iri",
                [term],
                "Profile requires a valid absolute HTTP(S) IRI.",
            )

    label_predicates = [SKOS + p for p in ("prefLabel", "altLabel", "hiddenLabel")]
    label_subjects = {s for p in label_predicates for s, _ in idx.pairs(p)}
    for subject in sorted(label_subjects):
        labels: dict[str, set[tuple[str, str, str]]] = {}
        preferred: dict[str, set[str]] = defaultdict(set)
        for predicate in label_predicates:
            values: set[tuple[str, str, str]] = set()
            for term in sorted(idx.objects(subject, predicate)):
                parsed = _literal(term)
                if parsed is None or parsed[2] not in {"", XSD + "string"}:
                    emit(
                        "error",
                        "invalid_label",
                        [subject, _iri(predicate), term],
                        "SKOS labels must be text literals.",
                    )
                    continue
                text, lang, _ = parsed
                values.add((text, lang, ""))
                if not text.strip() or not lang:
                    emit(
                        "warning",
                        "label_language_or_text",
                        [subject, term],
                        "Review empty or language-neutral label.",
                    )
                if predicate == SKOS + "prefLabel":
                    preferred[lang].add(text)
            labels[predicate] = values
        for lang, preferred_texts in sorted(preferred.items()):
            if len(preferred_texts) > 1:
                emit(
                    "error",
                    "preferred_label_cardinality",
                    [subject, lang, *sorted(preferred_texts)],
                    "SKOS S14: multiple preferred labels in one language.",
                )
        for i, first in enumerate(label_predicates):
            for second in label_predicates[i + 1 :]:
                for text, lang, _ in sorted(labels[first] & labels[second]):
                    emit(
                        "error",
                        "label_disjointness",
                        [subject, _iri(first), _iri(second), text, lang],
                        "SKOS S13: label properties are pairwise disjoint.",
                    )

    for predicate, pairs in sorted(idx.by_predicate.items()):
        for subject, term in sorted(pairs):
            parsed = _literal(term)
            if parsed and parsed[2]:
                if not _http_term(_iri(parsed[2])):
                    emit(
                        "error",
                        "invalid_http_iri",
                        [_iri(parsed[2])],
                        "Profile requires a valid datatype IRI.",
                    )
                if parsed[2] == XSD + "date" and not _date_term(term):
                    emit(
                        "error",
                        "invalid_date",
                        [subject, predicate, term],
                        "Profile requires a real YYYY-MM-DD xsd:date.",
                    )
                if parsed[2] == XSD + "gYear" and not re.fullmatch(r"[0-9]{4}", parsed[0]):
                    emit(
                        "error",
                        "invalid_year",
                        [subject, predicate, term],
                        "Profile requires a four-digit xsd:gYear.",
                    )
            if predicate in {_iri(DCT + "created"), _iri(DCT + "modified")} and not _date_term(
                term
            ):
                emit(
                    "warning",
                    "untyped_page_date",
                    [subject, predicate, term],
                    "Page date is missing a valid typed calendar date.",
                )

    for subject, target in sorted(idx.pairs(SCHEMA + "keywords")):
        if (
            not _http_term(target)
            or target in pages
            or target.startswith("<" + BASE + "page/")
            or (target.startswith("<" + BASE) and target not in concepts)
        ):
            emit(
                "error",
                "keyword_iri",
                [subject, target],
                "schema:keywords must be concept IRIs in this profile.",
            )
    keywords = {s for s in idx.by_subject if s.startswith("<" + BASE + "keyword/")}
    keywords |= {
        o for _, o in idx.pairs(SCHEMA + "keywords") if o.startswith("<" + BASE + "keyword/")
    }
    for keyword in sorted(keywords):
        if (
            keyword not in concepts
            or (keyword, _iri(SCHEMA + "DefinedTerm")) not in types
            or not idx.objects(keyword, SKOS + "prefLabel")
            or not any(
                (v := _literal(t)) and v[0].strip() for t in idx.objects(keyword, SKOS + "notation")
            )
            or _iri(BASE + "scheme") not in idx.objects(keyword, SKOS + "inScheme")
        ):
            emit(
                "error",
                "local_keyword_description",
                [keyword],
                "Local keyword requires Concept, DefinedTerm, label, notation and local scheme.",
            )

    statements = {s for s, o in types if o == _iri(RDF + "Statement")}
    statements |= {s for p in ("subject", "predicate", "object") for s, _ in idx.pairs(RDF + p)}
    statements |= {s for s in idx.by_subject if s.startswith("<" + BASE + "mapping/")}
    evidenced: set[Triple] = set()
    for statement in sorted(statements):
        fields = [idx.objects(statement, RDF + p) for p in ("subject", "predicate", "object")]
        if (statement, _iri(RDF + "Statement")) not in types or any(len(v) != 1 for v in fields):
            emit(
                "error",
                "mapping_statement_shape",
                [statement],
                "Review statement requires type and exactly one subject, predicate and object.",
            )
            continue
        triple = tuple(next(iter(v)) for v in fields)
        if triple not in mappings:
            emit(
                "error",
                "orphan_mapping_statement",
                [statement, *triple],
                "Review statement must describe an asserted SKOS mapping.",
            )
            continue
        sources = idx.objects(statement, DCT + "source")
        descriptions = idx.objects(statement, DCT + "description")
        dates = idx.objects(statement, DCT + "date")
        description = _literal(next(iter(descriptions))) if len(descriptions) == 1 else None
        if (
            len(sources) != 1
            or not all(_http_term(t) for t in sources)
            or not description
            or not description[0].strip()
            or description[1:] != ("en", "")
            or len(dates) != 1
            or not all(_date_term(t) for t in dates)
        ):
            emit(
                "error",
                "mapping_statement_evidence",
                [statement],
                "Requires one HTTP(S) source, nonempty English rationale and valid review date.",
            )
            continue
        evidenced.add((triple[0], triple[1], triple[2]))
    for triple in sorted(mappings):
        if (
            triple not in evidenced
            or not _http_term(triple[0])
            or not _http_term(triple[2])
            or (triple[0].startswith("<" + BASE) and triple[0] not in concepts)
        ):
            emit(
                "error",
                "mapping_review_evidence",
                triple,
                "Mapping lacks review evidence or a local concept source declaration.",
            )

    # Exact match is symmetric/transitive, but never an identity substitution.
    exact: dict[str, set[str]] = defaultdict(set)
    for source, target in idx.pairs(SKOS + "exactMatch"):
        exact[source].add(target)
        exact[target].add(source)
    exact_groups = _components(exact)
    exact_id = {node: i for i, group in enumerate(exact_groups) for node in group}
    for relation in ("broadMatch", "narrowMatch", "relatedMatch"):
        for source, target in sorted(idx.pairs(SKOS + relation)):
            if source in exact_id and exact_id.get(target) == exact_id[source]:
                emit(
                    "error",
                    "mapping_disjointness",
                    [source, _iri(SKOS + relation), target],
                    "SKOS S46 conflicts with symmetric/transitive exactMatch closure.",
                )

    hierarchy: dict[str, set[str]] = defaultdict(set)
    for relation in ("broader", "broaderTransitive", "broadMatch"):
        for source, target in idx.pairs(SKOS + relation):
            hierarchy[source].add(target)
    for relation in ("narrower", "narrowerTransitive", "narrowMatch"):
        for source, target in idx.pairs(SKOS + relation):
            hierarchy[target].add(source)
    cycles = [g for g in _components(hierarchy) if len(g) > 1 or g[0] in hierarchy.get(g[0], ())]
    for group in cycles:
        emit(
            "warning",
            "hierarchy_cycle",
            group,
            "SKOS permits hierarchy cycles; review whether this is intended.",
        )
    for source, target in sorted(idx.pairs(SKOS + "related") | idx.pairs(SKOS + "relatedMatch")):
        if _reachable(hierarchy, source, target) or _reachable(hierarchy, target, source):
            emit(
                "error",
                "related_hierarchy_disjointness",
                [source, target],
                "SKOS S27: related conflicts with transitive hierarchy in either direction.",
            )

    citation_pairs: set[tuple[str, str]] = set(idx.pairs(BIBO + "cites"))
    citation_pairs |= {(target, source) for source, target in idx.pairs(BIBO + "citedBy")}
    for citing, cited in sorted(citation_pairs):
        if any(
            not _http_term(t) or t in concepts or (t.startswith("<" + BASE) and t not in pages)
            for t in (citing, cited)
        ):
            emit(
                "error",
                "citation_endpoint",
                [citing, cited],
                "Citation direction is citing to cited; endpoints require document IRIs "
                "and declared local pages.",
            )
    for name in ("cites", "cited_by"):
        if idx.pairs(BASE + "relation/" + name):
            emit(
                "error",
                "citation_predicate",
                [_iri(BASE + "relation/" + name)],
                "Use the standard BIBO citation predicate with its documented direction.",
            )

    local_concepts = {s for s in concepts if s.startswith("<" + BASE)}
    page_concepts = {s for s in local_concepts if s.startswith("<" + BASE + "concept/")}
    mapped_concepts = {s for s, _, _ in evidenced}
    mapped_pages = {s for s, o in idx.pairs(SCHEMA + "about") if o in mapped_concepts}
    citation_records = {s for s, o in idx.pairs(BASE + "vocab/pageType") if o == '"citation"'}
    coverage = {
        "pages": len(pages),
        "local_concepts": len(local_concepts),
        "page_concepts": len(page_concepts),
        "page_concepts_with_definition": sum(
            bool(idx.objects(s, SKOS + "definition")) for s in page_concepts
        ),
        "page_concepts_with_scope_note": sum(
            bool(idx.objects(s, SKOS + "scopeNote")) for s in page_concepts
        ),
        "page_concepts_with_description": sum(
            bool(idx.objects(s, DCT + "description"))
            for s, o in idx.pairs(SCHEMA + "about")
            if o in page_concepts
        ),
        "document_topic_links": len(idx.pairs(SCHEMA + "about")),
        "document_relation_assertions": sum(
            len(pairs)
            for p, pairs in idx.by_predicate.items()
            if p.startswith("<" + BASE + "relation/document/")
        ),
        "concepts_with_preferred_label": sum(
            bool(idx.objects(s, SKOS + "prefLabel")) for s in local_concepts
        ),
        "concepts_with_definition": sum(
            bool(idx.objects(s, SKOS + "definition")) for s in local_concepts
        ),
        "concepts_with_description": sum(
            bool(idx.objects(s, DCT + "description")) for s in local_concepts
        ),
        "pages_with_reviewed_mapping": len(mapped_pages),
        "mapping_assertions": len(mappings),
        "evidenced_mapping_assertions": len(evidenced),
        "mapping_statements": len(statements),
        "local_keywords": len(keywords),
        "hierarchy_cycle_components": len(cycles),
        "citation_pairs": len(citation_pairs),
        "citation_record_pages": len(citation_records),
        "citation_records_with_identifier": sum(
            bool(idx.objects(s, SCHEMA + "sameAs")) for s in citation_records
        ),
        "citing_local_pages": len({s for s, _ in citation_pairs if s in pages}),
        "cited_local_pages": len({o for _, o in citation_pairs if o in pages}),
        "mapping_relations": {r: len(idx.pairs(SKOS + r)) for r in sorted(RELATIONS)},
    }
    for subject in sorted(local_concepts):
        if not idx.objects(subject, SKOS + "prefLabel"):
            emit(
                "warning",
                "concept_preferred_label_missing",
                [subject],
                "Local concept has no preferred label.",
            )
    counts = Counter(f["code"] for f in findings)
    findings.sort(key=lambda f: (f["severity"], f["code"], f["terms"], f["message"]))
    return {
        "schema_version": 2,
        "profile": "cognitive-kb-semantic-2",
        "machine_checks_passed": not any(f["severity"] == "error" for f in findings),
        "errors": [f for f in findings if f["severity"] == "error"],
        "warnings": [f for f in findings if f["severity"] == "warning"],
        "finding_counts": dict(sorted(counts.items())),
        "coverage": coverage,
        "work": {
            "triples_indexed": idx.triples_indexed,
            "mapping_statement_lookups": len(statements),
        },
    }
