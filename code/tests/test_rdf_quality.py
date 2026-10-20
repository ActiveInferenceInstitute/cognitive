import json
from pathlib import Path

import pytest
from scripts import export_rdf as rdf
from scripts.rdf_quality import GraphIndex, semantic_quality

SKOS = rdf.PREFIXES["skos"]
RDF = rdf.PREFIXES["rdf"]
DCT = rdf.PREFIXES["dcterms"]
SCHEMA = rdf.PREFIXES["schema"]
BIBO = rdf.PREFIXES["bibo"]
A, B, C = (rdf.iri(rdf.page_iri(n)) for n in ("a", "b", "c"))


def concept_graph():
    graph = rdf.Graph()
    for node in (A, B, C):
        graph.add(node, rdf.iri(rdf.RDF_TYPE), rdf.iri(SKOS + "Concept"))
        graph.add(node, rdf.iri(rdf.RDF_TYPE), rdf.iri(SCHEMA + "CreativeWork"))
        graph.add(node, rdf.iri(SKOS + "prefLabel"), rdf.literal(node, lang="en"))
    return graph


def add_review(graph, source=A, relation="exactMatch", target=B, key="a"):
    predicate = rdf.iri(SKOS + relation)
    statement = rdf.iri(rdf.BASE + "mapping/" + key)
    graph.add(source, predicate, target)
    for p, obj in (
        (RDF + "type", rdf.iri(RDF + "Statement")),
        (RDF + "subject", source),
        (RDF + "predicate", predicate),
        (RDF + "object", target),
        (DCT + "source", rdf.iri("https://example.org/source")),
        (DCT + "description", rdf.literal("Reviewed source comparison.", lang="en")),
        (DCT + "date", rdf.literal("2024-02-29", datatype=rdf.XSD_DATE)),
    ):
        graph.add(statement, rdf.iri(p), obj)
    return statement


def error_codes(graph):
    return {finding["code"] for finding in semantic_quality(graph)["errors"]}


def test_clean_real_graph_terms_and_coverage_are_deterministic():
    graph = concept_graph()
    add_review(graph)
    first = semantic_quality(graph)
    second = semantic_quality(rdf.Graph(set(reversed(sorted(graph.triples)))))
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["errors"] == first["warnings"] == []
    assert first["coverage"]["mapping_assertions"] == 1
    assert first["coverage"]["evidenced_mapping_assertions"] == 1
    assert first["coverage"]["concepts_with_definition"] == 0
    assert first["coverage"]["concepts_with_preferred_label"] == 3


@pytest.mark.parametrize(
    "first,second",
    [("prefLabel", "altLabel"), ("prefLabel", "hiddenLabel"), ("altLabel", "hiddenLabel")],
)
def test_label_disjointness_compares_rdf_values_and_case_insensitive_language(first, second):
    graph = concept_graph()
    graph.add(A, rdf.iri(SKOS + first), '"Caf\\u00E9"@en')
    graph.add(A, rdf.iri(SKOS + second), '"Café"@EN')
    assert "label_disjointness" in error_codes(graph)


def test_label_cardinality_allows_languages_and_identical_escaped_values():
    graph = concept_graph()
    graph.triples = {t for t in graph.triples if t[:2] != (A, rdf.iri(SKOS + "prefLabel"))}
    for value in ('"Café"@en', '"Caf\\u00E9"@EN', '"Café"@fr'):
        graph.add(A, rdf.iri(SKOS + "prefLabel"), value)
    assert error_codes(graph) == set()
    graph.add(A, rdf.iri(SKOS + "prefLabel"), '"Coffee"@EN')
    assert "preferred_label_cardinality" in error_codes(graph)


@pytest.mark.parametrize(
    "term",
    [
        A,
        '"12"^^<http://www.w3.org/2001/XMLSchema#integer>',
        '"bad\\UFFFFFFFF"@en',
        '"bad\\uD800"@en',
    ],
)
def test_label_values_must_be_well_formed_text(term):
    graph = concept_graph()
    graph.add(A, rdf.iri(SKOS + "altLabel"), term)
    assert "invalid_label" in error_codes(graph)


def test_missing_labels_and_language_and_untyped_dates_are_nonfatal():
    graph = concept_graph()
    graph.triples = {t for t in graph.triples if t[:2] != (A, rdf.iri(SKOS + "prefLabel"))}
    graph.add(B, rdf.iri(SKOS + "altLabel"), rdf.literal("Alias"))
    graph.add(C, rdf.iri(DCT + "modified"), rdf.literal("spring 2024"))
    report = semantic_quality(graph)
    assert report["machine_checks_passed"]
    assert {w["code"] for w in report["warnings"]} == {
        "concept_preferred_label_missing",
        "label_language_or_text",
        "untyped_page_date",
    }


@pytest.mark.parametrize(
    "term",
    [
        "https://example.org:bad/a",
        "https://example.org/%ZZ",
        "https://example.org:99999/a",
        "https://user:pass@example.org/a",
    ],
)
@pytest.mark.parametrize("position", [0, 1, 2])
def test_all_iri_positions_reject_invalid_http_terms(term, position):
    graph = concept_graph()
    triple = [A, rdf.iri(DCT + "references"), B]
    triple[position] = rdf.iri(term)
    graph.add(*triple)
    assert "invalid_http_iri" in error_codes(graph)


@pytest.mark.parametrize("text", ["2026-02-30", "2023-02-29", "20261020", "2024-01-01Z"])
def test_profile_dates_reject_impossible_or_nonprofile_typed_values(text):
    graph = concept_graph()
    graph.add(A, rdf.iri(DCT + "created"), rdf.literal(text, datatype=rdf.XSD_DATE))
    assert "invalid_date" in error_codes(graph)


def test_dates_accept_leap_day_and_year_requires_local_four_digit_format():
    graph = concept_graph()
    graph.add(A, rdf.iri(DCT + "created"), rdf.literal("2024-02-29", datatype=rdf.XSD_DATE))
    graph.add(A, rdf.iri(SCHEMA + "datePublished"), rdf.literal("2024", datatype=rdf.XSD_GYEAR))
    assert error_codes(graph) == set()
    graph.add(A, rdf.iri(SCHEMA + "datePublished"), rdf.literal("twenty", datatype=rdf.XSD_GYEAR))
    assert "invalid_year" in error_codes(graph)


@pytest.mark.parametrize("missing", ["Concept", "DefinedTerm", "prefLabel", "notation", "inScheme"])
def test_local_keyword_requires_complete_declaration(missing):
    graph = concept_graph()
    keyword = rdf.iri(rdf.BASE + "keyword/quality")
    graph.add(A, rdf.iri(SCHEMA + "keywords"), keyword)
    for p, value in (
        (rdf.RDF_TYPE, rdf.iri(SKOS + "Concept")),
        (rdf.RDF_TYPE, rdf.iri(SCHEMA + "DefinedTerm")),
        (SKOS + "prefLabel", rdf.literal("quality", lang="en")),
        (SKOS + "notation", rdf.literal("quality")),
        (SKOS + "inScheme", rdf.iri(rdf.SCHEME)),
    ):
        graph.add(keyword, rdf.iri(p), value)
    assert error_codes(graph) == set()
    graph.triples = {
        t
        for t in graph.triples
        if not (
            t[0] == keyword
            and (
                t[2] == rdf.iri(SKOS + missing)
                or t[2] == rdf.iri(SCHEMA + missing)
                or t[1] == rdf.iri(SKOS + missing)
            )
        )
    }
    assert "local_keyword_description" in error_codes(graph), missing


@pytest.mark.parametrize(
    "field", ["subject", "predicate", "object", "source", "description", "date"]
)
@pytest.mark.parametrize("corruption", ["missing", "multiple"])
def test_every_statement_field_is_checked_even_with_another_valid_review(field, corruption):
    graph = concept_graph()
    bad = add_review(graph, key="bad")
    add_review(graph, key="good")
    namespace = RDF if field in {"type", "subject", "predicate", "object"} else DCT
    predicate = rdf.iri(namespace + field)
    if corruption == "missing":
        graph.triples = {t for t in graph.triples if t[:2] != (bad, predicate)}
    else:
        graph.add(bad, predicate, rdf.literal("extra"))
    assert error_codes(graph) & {"mapping_statement_shape", "mapping_statement_evidence"}


def test_orphan_review_and_detached_mapping_metadata_are_fatal():
    graph = concept_graph()
    add_review(graph)
    graph.triples.remove((A, rdf.iri(SKOS + "exactMatch"), B))
    assert "orphan_mapping_statement" in error_codes(graph)
    graph = concept_graph()
    graph.add(rdf.iri(rdf.BASE + "mapping/detached"), rdf.iri(DCT + "source"), B)
    assert "mapping_statement_shape" in error_codes(graph)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source", rdf.iri("https://example.org/%QQ")),
        ("source", rdf.literal("https://example.org/source")),
        ("description", rdf.literal("   ", lang="en")),
        ("description", rdf.literal("Review", lang="fr")),
        ("description", rdf.iri("https://example.org/rationale")),
        ("date", rdf.literal("2023-02-29", datatype=rdf.XSD_DATE)),
        ("date", rdf.literal("2024-02-29")),
    ],
)
def test_single_but_invalid_statement_metadata_cannot_cover_a_mapping(field, value):
    graph = concept_graph()
    statement = add_review(graph)
    predicate = rdf.iri(DCT + field)
    graph.triples = {t for t in graph.triples if t[:2] != (statement, predicate)}
    graph.add(statement, predicate, value)
    assert {"mapping_statement_evidence", "mapping_review_evidence"} <= error_codes(graph)


def test_mapping_requires_local_concept_source_even_when_its_page_types_are_absent():
    graph = concept_graph()
    add_review(graph, rdf.iri(rdf.BASE + "dataset"))
    assert "mapping_review_evidence" in error_codes(graph)


def test_literal_semantic_endpoints_fail_and_unrelated_local_relations_have_no_hierarchy_axioms():
    graph = concept_graph()
    graph.add(A, rdf.iri(SKOS + "broader"), rdf.literal("a class"))
    assert "semantic_relation_endpoint" in error_codes(graph)
    graph = concept_graph()
    graph.add(A, rdf.iri(rdf.RELATION + "implements"), B)
    graph.add(A, rdf.iri(SKOS + "related"), B)
    assert error_codes(graph) == set()


@pytest.mark.parametrize(
    "relation,reverse", [("broadMatch", False), ("narrowMatch", True), ("relatedMatch", True)]
)
def test_mapping_conflicts_include_inverse_predicates_and_exact_match_closure(relation, reverse):
    graph = concept_graph()
    add_review(graph, A, "exactMatch", B, "ab")
    add_review(graph, C, "exactMatch", B, "cb")
    assert error_codes(graph) == set()
    add_review(graph, C if reverse else A, relation, A if reverse else C, "conflict")
    assert "mapping_disjointness" in error_codes(graph)


def test_exact_and_close_are_compatible_and_close_match_is_not_transitive():
    graph = concept_graph()
    add_review(graph, A, "exactMatch", B, "ab")
    add_review(graph, B, "closeMatch", C, "bc")
    add_review(graph, A, "closeMatch", B, "close-ab")
    add_review(graph, A, "broadMatch", C, "ac")
    assert error_codes(graph) == set()


@pytest.mark.parametrize(
    "relation,reverse",
    [
        ("broader", False),
        ("narrower", True),
        ("broaderTransitive", False),
        ("narrowerTransitive", True),
        ("broadMatch", False),
        ("narrowMatch", True),
    ],
)
def test_related_conflicts_with_direct_and_transitive_inverse_hierarchy(relation, reverse):
    graph = concept_graph()
    if relation.endswith("Match"):
        add_review(graph, B if reverse else A, relation, A if reverse else B)
    else:
        graph.add(B if reverse else A, rdf.iri(SKOS + relation), A if reverse else B)
    graph.add(B, rdf.iri(SKOS + "broader"), C)
    graph.add(C, rdf.iri(SKOS + "related"), A)
    assert "related_hierarchy_disjointness" in error_codes(graph)


def test_hierarchy_cycle_is_warning_and_exact_match_does_not_substitute_identity():
    graph = concept_graph()
    graph.add(A, rdf.iri(SKOS + "broader"), B)
    graph.add(B, rdf.iri(SKOS + "broader"), A)
    report = semantic_quality(graph)
    assert report["machine_checks_passed"]
    assert report["warnings"][0]["code"] == "hierarchy_cycle"
    add_review(graph, A, "exactMatch", C)
    graph.add(C, rdf.iri(SKOS + "related"), B)
    assert error_codes(graph) == set()


def test_citation_directions_normalize_to_one_citing_to_cited_pair():
    graph = concept_graph()
    graph.add(A, rdf.iri(BIBO + "citedBy"), B)
    graph.add(B, rdf.iri(BIBO + "cites"), A)
    report = semantic_quality(graph)
    assert report["errors"] == []
    assert report["coverage"]["citation_pairs"] == 1
    graph.add(A, rdf.iri(BIBO + "cites"), rdf.literal("not a document IRI"))
    assert "citation_endpoint" in error_codes(graph)


@pytest.mark.parametrize("relation", ["cites", "citedBy"])
def test_citation_endpoints_allow_external_documents_and_reject_absent_local_pages(relation):
    graph = concept_graph()
    graph.add(A, rdf.iri(BIBO + relation), rdf.iri("https://doi.org/10.1000/example"))
    assert error_codes(graph) == set()
    graph.add(A, rdf.iri(BIBO + relation), rdf.iri(rdf.page_iri("absent")))
    assert "citation_endpoint" in error_codes(graph)


def test_large_deterministic_mapping_graph_uses_one_index_and_all_reviews():
    graph = concept_graph()
    for i in range(1000):
        add_review(graph, A, "closeMatch", rdf.iri(f"https://example.org/concept/{i}"), str(i))
    for i in range(68000):
        graph.add(
            rdf.iri(f"https://example.org/document/{i}"),
            rdf.iri(DCT + "title"),
            rdf.literal(str(i)),
        )
    index = GraphIndex.build(graph.triples)
    assert index.triples_indexed == len(graph.triples) > 75000
    assert len(index.pairs(RDF + "subject")) == 1000
    report = semantic_quality(graph, index=index)
    assert report["errors"] == []
    assert report["work"] == {
        "triples_indexed": len(graph.triples),
        "mapping_statement_lookups": 1000,
    }
    assert report["coverage"]["evidenced_mapping_assertions"] == 1000
    graph.triples.remove(
        (
            rdf.iri(rdf.BASE + "mapping/999"),
            rdf.iri(DCT + "date"),
            rdf.literal("2024-02-29", datatype=rdf.XSD_DATE),
        )
    )
    failed = semantic_quality(graph)
    assert failed["coverage"]["evidenced_mapping_assertions"] == 999
    assert len(failed["errors"]) == 2


def test_real_vault_quality_and_submission_share_semantic_findings():
    graph, _ = rdf.build_graph(Path(__file__).resolve().parents[2])
    report = rdf.submission_diagnostics(graph)
    assert report["errors"] == []
    assert report["quality"]["coverage"]["pages"] > 1000
    graph.add(A, rdf.iri(SKOS + "prefLabel"), rdf.literal("First", lang="en"))
    graph.add(A, rdf.iri(SKOS + "prefLabel"), rdf.literal("Second", lang="en"))
    assert any("preferred_label_cardinality" in error for error in rdf.submission_errors(graph))


def test_quality_cli_warnings_are_nonfatal_errors_fail_and_no_export_written(tmp_path, capsys):
    kb = tmp_path / "knowledge_base"
    kb.mkdir()
    page = kb / "concept.md"
    page.write_text('---\ntitle: Concept\ncreated: "spring 2026"\n---\n', encoding="utf-8")
    assert rdf.main([str(tmp_path), "--quality-report"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["warnings"][0]["code"] == "untyped_page_date"
    assert not (tmp_path / "output").exists()
    page.write_text('---\ntitle: Concept\ncreated: "2026-02-30"\n---\n', encoding="utf-8")
    assert rdf.main([str(tmp_path), "--quality-report"]) == 1
    assert "invalid_date" in {f["code"] for f in json.loads(capsys.readouterr().out)["errors"]}
