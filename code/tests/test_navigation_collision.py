"""Extensionless document links must not be redirected to directory indexes."""

from pathlib import Path

from scripts import export_rdf as rdf


def test_markdown_page_precedes_same_named_directory_index(tmp_path: Path) -> None:
    kb = tmp_path / "knowledge_base"
    (kb / "AGENTS").mkdir(parents=True)
    (kb / "AGENTS.md").write_text("---\ntitle: Agents\ntype: agents\n---\n# Agents\n")
    (kb / "AGENTS/README.md").write_text("---\ntitle: Directory\ntype: index\n---\n# Directory\n")
    (kb / "index.md").write_text("---\ntitle: Index\ntype: index\n---\n[[AGENTS]]\n")
    graph, _ = rdf.build_graph(tmp_path)
    source = rdf.iri(rdf.page_iri("index"))
    predicate = rdf.iri(rdf.PREFIXES["dcterms"] + "references")
    assert (source, predicate, rdf.iri(rdf.page_iri("AGENTS"))) in graph.triples
    assert (source, predicate, rdf.iri(rdf.page_iri("AGENTS/README"))) not in graph.triples


def test_directory_index_is_used_when_there_is_no_same_named_page(tmp_path: Path) -> None:
    kb = tmp_path / "knowledge_base"
    (kb / "section").mkdir(parents=True)
    (kb / "section/README.md").write_text("---\ntitle: Section\ntype: index\n---\n# Section\n")
    (kb / "index.md").write_text("---\ntitle: Index\ntype: index\n---\n[[section]]\n")
    graph, _ = rdf.build_graph(tmp_path)
    assert (
        rdf.iri(rdf.page_iri("index")),
        rdf.iri(rdf.PREFIXES["dcterms"] + "references"),
        rdf.iri(rdf.page_iri("section/README")),
    ) in graph.triples


def test_exact_directory_spelling_precedes_a_case_folded_page(tmp_path: Path) -> None:
    kb = tmp_path / "knowledge_base"
    (kb / "agents").mkdir(parents=True)
    (kb / "AGENTS.md").write_text("---\ntitle: Instructions\ntype: agents\n---\n# Instructions\n")
    (kb / "agents/README.md").write_text("---\ntitle: Agents\ntype: index\n---\n# Agents\n")
    (kb / "index.md").write_text(
        "---\ntitle: Index\ntype: index\n---\n[[AGENTS]] [[agents]] [[aGeNtS.md]]\n"
    )
    graph, _ = rdf.build_graph(tmp_path)
    source = rdf.iri(rdf.page_iri("index"))
    predicate = rdf.iri(rdf.PREFIXES["dcterms"] + "references")
    assert (source, predicate, rdf.iri(rdf.page_iri("AGENTS"))) in graph.triples
    assert (source, predicate, rdf.iri(rdf.page_iri("agents/README"))) in graph.triples
    assert (source, predicate, rdf.iri(rdf.page_iri("agents"))) not in graph.triples
