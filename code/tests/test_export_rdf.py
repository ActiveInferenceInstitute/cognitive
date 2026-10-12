import json
import re
from datetime import date, datetime
from pathlib import Path

import pytest
import yaml
from scripts import export_rdf as rdf

REPO_ROOT = Path(__file__).resolve().parents[2]
SKOS = rdf.PREFIXES["skos"]
DCT = rdf.PREFIXES["dcterms"]

# One N-Quads statement: subject IRI, predicate IRI, an IRI or a literal, graph IRI.
_NQUAD = re.compile(
    r'^<[^<>\s"]+> <[^<>\s"]+> '
    r'(<[^<>\s"]+>|"(?:[^"\\\n]|\\["\\nrt]|\\u[0-9A-F]{4})*"(?:@[a-z]+|\^\^<[^<>\s"]+>)?) '
    r"<[^<>\s\"]+> \.$"
)


def _page(path: Path, front: dict | None, body: str = "\nBody.\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"---\n{yaml.safe_dump(front, sort_keys=False)}---\n" if front is not None else ""
    path.write_text(header + body, encoding="utf-8")


def _vault(tmp_path: Path) -> Path:
    kb = tmp_path / "knowledge_base"
    _page(
        kb / "math" / "entropy.md",
        {
            "title": "Entropy",
            "type": "mathematical_concept",
            "status": "stable",
            "complexity": "basic",
            "created": date(2024, 3, 15),
            "updated": date(2026, 1, 2),
            "description": 'Uncertainty of a "distribution".',
            "aliases": ["Shannon entropy"],
            "tags": ["information", "probability"],
            "semantic_relations": [
                {"type": "relates", "links": ["[[kl_divergence]]"]},
                {"type": "Foundation For", "links": [[["[[bayes|Bayes' rule]]"]]]},
                {"type": "part_of", "links": ["[[math/README]]"], "parent": "[[math/README]]"},
                {"type": "implements", "links": ["[[entropy]]", "[[no_such_concept]]"]},
            ],
        },
        "\nSee [[kl_divergence#Definition|KL]] and [[bayes]], not `[[in_code]]` nor [[1, 0]].\n"
        "Outside: [[docs/guide]]. Spaced: [[Spaced Page]].\n",
    )
    _page(kb / "math" / "kl_divergence.md", {"title": "KL divergence", "type": "concept"})
    _page(kb / "math" / "bayes.md", {"title": "[[bayes|Bayes rule]]", "type": "concept"})
    _page(kb / "math" / "README.md", {"title": "Math", "type": "index"})
    _page(kb / "math" / "Spaced Page.md", {"type": "concept"})
    _page(kb / "bare.md", None, "\nNo frontmatter at all.\n")
    _page(
        kb / "citations" / "smith_2022.md",
        {
            "title": "A tutorial",
            "type": "citation",
            "authors": ["Ryan Smith", "Karl J. Friston"],
            "year": 2022,
            "journal": "J Math Psychol",
            "doi": "10.1016/j.jmp.2021.102632",
            "created": "not a date",
        },
    )
    _page(
        kb / "old" / "moved.md",
        {"title": "Moved", "type": "redirect", "redirect": "[[math/entropy]]"},
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# Guide\n", encoding="utf-8")
    return tmp_path


def _has(graph: rdf.Graph, subject: str, predicate: str, obj: str | None = None) -> bool:
    return any(
        s == rdf.iri(subject) and p == rdf.iri(predicate) and (obj is None or o == obj)
        for s, p, o in graph.triples
    )


def test_literal_escapes_quotes_newlines_and_control_characters():
    assert rdf.literal('say "hi"\n\tback\\slash\x01') == '"say \\"hi\\"\\n\\tback\\\\slash\\u0001"'
    assert rdf.literal("café", lang="en") == '"café"@en'
    assert rdf.literal("2022", datatype=rdf.XSD_GYEAR).endswith("#gYear>")


def test_dates_are_typed_only_when_they_are_dates():
    assert (
        rdf.typed_date(date(2026, 1, 2)) == '"2026-01-02"^^<http://www.w3.org/2001/XMLSchema#date>'
    )
    assert rdf.date_text(datetime(2026, 1, 2, 9, 30)) == "2026-01-02"
    assert rdf.typed_date("2026-01-02").endswith("#date>")
    assert rdf.typed_date("spring 2026") == '"spring 2026"'


def test_relation_names_map_to_standard_terms_or_graph_terms():
    assert rdf.relation_predicate("relates") == SKOS + "related"
    assert rdf.relation_predicate("Related To") == SKOS + "related"
    assert rdf.relation_predicate("part_of") == DCT + "isPartOf"
    assert rdf.relation_predicate("organizes") == DCT + "hasPart"
    assert rdf.relation_predicate("Foundation For") == rdf.RELATION + "foundation_for"
    assert rdf.relation_slug("???") == "relates"


def test_page_iri_quotes_spaces_and_keeps_directories():
    assert rdf.page_iri("math/Spaced Page") == rdf.PAGE + "math/Spaced%20Page"


def test_semantic_relations_flatten_nested_lists_and_strip_wiki_syntax():
    value = [
        {"type": "extends", "links": [[["[[a|shown]]"]], "[[b#frag]]", "c", 7, ""]},
        {"links": ["[[d]]"], "parent": "[[p]]"},
        "not a mapping",
    ]
    assert rdf.semantic_relations(value) == [
        ("extends", "a"),
        ("extends", "b"),
        ("extends", "c"),
        ("relates", "d"),
        ("has_parent", "p"),
    ]
    assert rdf.semantic_relations(None) == []


def test_body_links_skip_code_matrices_and_empty_targets():
    body = "[[a|x]] `[[inline]]` [[0, 1.5]] [[#only_anchor]]\n```\n[[fenced]]\n```\n[[b#h]]\n"
    assert rdf.body_links(body) == ["a", "b"]


def test_graph_describes_pages_with_standard_vocabularies(tmp_path):
    graph, report = rdf.build_graph(_vault(tmp_path))
    entropy = rdf.page_iri("math/entropy")
    assert _has(graph, entropy, rdf.RDF_TYPE, rdf.iri(SKOS + "Concept"))
    assert _has(graph, entropy, SKOS + "prefLabel", '"Entropy"@en')
    assert _has(graph, entropy, SKOS + "altLabel", '"Shannon entropy"@en')
    assert _has(graph, entropy, DCT + "description", '"Uncertainty of a \\"distribution\\"."@en')
    assert _has(graph, entropy, DCT + "created", rdf.typed_date("2024-03-15"))
    assert _has(graph, entropy, DCT + "modified", rdf.typed_date("2026-01-02"))
    assert _has(graph, entropy, rdf.PREFIXES["schema"] + "keywords", '"probability"')
    assert _has(graph, entropy, rdf.VOCAB + "pageType", '"mathematical_concept"')
    assert _has(graph, entropy, rdf.VOCAB + "complexity", '"basic"')
    assert _has(
        graph,
        entropy,
        DCT + "source",
        rdf.iri(f"{rdf.SOURCE_REPOSITORY}/blob/main/knowledge_base/math/entropy.md"),
    )
    assert _has(graph, entropy, SKOS + "inScheme", rdf.iri(rdf.SCHEME))
    assert _has(graph, rdf.DOMAIN + "math", SKOS + "member", rdf.iri(entropy))
    assert report.modified == "2026-01-02"


def test_typed_and_untyped_edges_use_the_declared_predicates(tmp_path):
    graph, report = rdf.build_graph(_vault(tmp_path))
    entropy = rdf.page_iri("math/entropy")
    kl, bayes, readme = (
        rdf.iri(rdf.page_iri(f"math/{n}")) for n in ("kl_divergence", "bayes", "README")
    )
    assert _has(graph, entropy, SKOS + "related", kl)
    assert _has(graph, entropy, rdf.RELATION + "foundation_for", bayes)
    assert _has(graph, entropy, DCT + "isPartOf", readme)
    assert _has(graph, entropy, DCT + "references", kl)
    assert _has(graph, entropy, DCT + "references", bayes)
    spaced = rdf.iri(rdf.page_iri("math/Spaced Page"))
    assert _has(graph, entropy, DCT + "references", spaced)
    # A self-reference is dropped, a missing concept and a file outside the vault are counted.
    assert not _has(graph, entropy, rdf.RELATION + "implements", rdf.iri(entropy))
    assert report.self_loops_dropped == 1
    assert report.unresolved_links == 1
    assert report.links_outside_graph == 1
    assert report.relations["foundation_for"] == 1


def test_graph_terms_are_declared_as_sub_properties_of_skos_semantic_relation(tmp_path):
    graph, _ = rdf.build_graph(_vault(tmp_path))
    term = rdf.RELATION + "foundation_for"
    assert _has(
        graph, term, rdf.PREFIXES["rdfs"] + "subPropertyOf", rdf.iri(SKOS + "semanticRelation")
    )
    assert _has(graph, term, rdf.PREFIXES["rdfs"] + "label", '"foundation for"@en')
    assert _has(graph, rdf.VOCAB + "pageType", rdf.PREFIXES["rdfs"] + "label", '"pageType"@en')


def test_navigation_citation_and_redirect_pages_are_modelled_distinctly(tmp_path):
    graph, report = rdf.build_graph(_vault(tmp_path))
    readme = rdf.page_iri("math/README")
    assert _has(graph, readme, rdf.RDF_TYPE, rdf.iri(rdf.PREFIXES["schema"] + "CollectionPage"))
    assert not _has(graph, readme, rdf.RDF_TYPE, rdf.iri(SKOS + "Concept"))

    citation = rdf.page_iri("citations/smith_2022")
    assert _has(graph, citation, rdf.PREFIXES["schema"] + "author", '"Ryan Smith"')
    assert _has(
        graph, citation, rdf.PREFIXES["schema"] + "datePublished", '"2022"^^<' + rdf.XSD_GYEAR + ">"
    )
    assert _has(
        graph,
        citation,
        rdf.PREFIXES["schema"] + "sameAs",
        rdf.iri("https://doi.org/10.1016/j.jmp.2021.102632"),
    )
    assert _has(
        graph,
        citation,
        rdf.VOCAB + "journal",
        '"J Math Psychol"',
    )
    assert _has(graph, citation, DCT + "created", '"not a date"')

    moved = rdf.page_iri("old/moved")
    assert _has(graph, moved, rdf.REDIRECT_PREDICATE, rdf.iri(rdf.page_iri("math/entropy")))
    assert not _has(graph, moved, rdf.RDF_TYPE, rdf.iri(SKOS + "Concept"))
    assert report.redirects == 1
    assert report.navigation_pages == 1
    assert report.concepts == 6
    assert report.pages == 8


def test_pages_without_frontmatter_or_title_still_get_a_label(tmp_path):
    graph, _ = rdf.build_graph(_vault(tmp_path))
    assert _has(graph, rdf.page_iri("bare"), DCT + "title", '"bare"@en')
    assert _has(graph, rdf.page_iri("math/Spaced Page"), DCT + "title", '"Spaced Page"@en')
    assert _has(graph, rdf.page_iri("math/bayes"), DCT + "title", '"Bayes rule"@en')


def test_dataset_description_carries_license_provenance_and_modified_date(tmp_path):
    graph, _ = rdf.build_graph(_vault(tmp_path))
    assert _has(graph, rdf.DATASET, DCT + "license", rdf.iri(rdf.LICENSE))
    assert _has(
        graph, rdf.DATASET, rdf.PREFIXES["prov"] + "wasDerivedFrom", rdf.iri(rdf.SOURCE_REPOSITORY)
    )
    assert _has(graph, rdf.DATASET, rdf.PREFIXES["prov"] + "wasGeneratedBy", rdf.iri(rdf.ACTIVITY))
    assert _has(graph, rdf.DATASET, DCT + "modified", rdf.typed_date("2026-01-02"))
    assert _has(graph, rdf.SCHEME, rdf.RDF_TYPE, rdf.iri(SKOS + "ConceptScheme"))


def test_links_into_local_tool_folders_stay_unresolved(tmp_path):
    root = _vault(tmp_path)
    _page(root / "knowledge_base" / "ghost.md", {"title": "Ghost"}, "\n[[tool_card]] [[entropy]]\n")
    _page(root / "graft" / "tool_card.md", {"title": "Card"})
    _page(root / "graft" / "entropy.md", {"title": "Shadow of entropy"})
    graph, report = rdf.build_graph(root)
    # The card is not shipped, so the link stays broken; the shadow does not win the stem.
    assert report.unresolved_links == 2 and report.links_outside_graph == 1
    assert _has(
        graph, rdf.page_iri("ghost"), DCT + "references", rdf.iri(rdf.page_iri("math/entropy"))
    )


def test_missing_knowledge_base_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="knowledge base not found"):
        rdf.build_graph(tmp_path)


def test_nquads_are_well_formed_and_name_the_frink_graph(tmp_path):
    graph, report = rdf.build_graph(_vault(tmp_path))
    text = rdf.to_nquads(graph)
    lines = text.splitlines()
    assert len(lines) == report.triples == len(graph.triples)
    assert all(_NQUAD.match(line) for line in lines), [x for x in lines if not _NQUAD.match(x)][:3]
    assert {line.rsplit(" ", 2)[1] for line in lines} == {
        "<https://purl.org/okn/frink/kg/cognitive-kb>"
    }
    assert lines == sorted(lines, key=lambda line: line.rsplit(" <", 1)[0])


def test_turtle_uses_prefixes_and_escapes_slashes_in_local_names(tmp_path):
    graph, _ = rdf.build_graph(_vault(tmp_path))
    text = rdf.to_turtle(graph)
    assert text.startswith("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .")
    assert "page:math\\/entropy" in text
    assert "skos:related" in text and "a skos:Concept" in text
    assert '"2024-03-15"^^xsd:date' in text
    # A page whose name needs percent-encoding keeps its full IRI.
    assert "<https://purl.org/okn/frink/kg/cognitive-kb/page/math/Spaced%20Page>" in text
    only = rdf.to_turtle(graph, {rdf.iri(rdf.DATASET)})
    assert "kg:dataset" in only and "page:math" not in only


def test_export_is_deterministic(tmp_path):
    root = _vault(tmp_path)
    first = rdf.to_nquads(rdf.build_graph(root)[0])
    second = rdf.to_nquads(rdf.build_graph(root)[0])
    assert first == second


def test_statistics_come_from_the_report(tmp_path):
    _, report = rdf.build_graph(_vault(tmp_path))
    text = rdf.statistics_markdown(report)
    assert f"| Pages (all described as `schema:CreativeWork`) | {report.pages} |" in text
    assert f"| Triples | {report.triples} |" in text
    assert "| `math` | 5 |" in text and "| `foundation_for` | 1 |" in text
    assert text.startswith("---\ntitle: cognitive-kb graph statistics")


def test_docs_are_written_checked_and_detected_when_stale(tmp_path, capsys):
    root = _vault(tmp_path)
    assert rdf.main([str(root), "--write-docs"]) == 0
    assert rdf.stale_docs(root) == []
    assert rdf.main([str(root), "--check"]) == 0
    assert "current" in capsys.readouterr().out
    (root / rdf.DOCS_DIR / rdf.STATISTICS_NAME).write_text("edited\n", encoding="utf-8")
    assert rdf.stale_docs(root) == ["docs/okn/statistics.md"]
    assert rdf.main([str(root), "--check"]) == 1
    assert "docs/okn/statistics.md" in capsys.readouterr().out


def test_a_tree_without_a_knowledge_base_has_no_stale_docs(tmp_path):
    assert rdf.stale_docs(tmp_path) == []


def test_command_line_writes_both_serialisations(tmp_path, capsys):
    root = _vault(tmp_path)
    out = tmp_path / "out"
    assert rdf.main([str(root), "--output", str(out), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["pages"] == 8 and summary["unresolved_links"] == 1
    nq = (out / "cognitive-kb.nq").read_text(encoding="utf-8")
    ttl = (out / "cognitive-kb.ttl").read_text(encoding="utf-8")
    assert len(nq.splitlines()) == summary["triples"]
    assert "@prefix skos:" in ttl
    assert rdf.main([str(root), "--output", str(out)]) == 0
    assert "triples:" in capsys.readouterr().out


def test_committed_docs_match_the_knowledge_base():
    assert rdf.stale_docs(REPO_ROOT) == []


def test_real_graph_has_no_dangling_page_objects():
    graph, report = rdf.build_graph(REPO_ROOT)
    pages = {s for s, p, _ in graph.triples if p == rdf.iri(DCT + "title")}
    page_prefix = f"<{rdf.PAGE}"
    dangling = {o for _, _, o in graph.triples if o.startswith(page_prefix) and o not in pages}
    assert dangling == set()
    assert report.unresolved_links > 0 and report.typed_edges > 1000
