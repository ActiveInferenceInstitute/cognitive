"""The knowledge-base report: graph construction, measurements, figures, command line."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import networkx as nx
import pytest
import yaml
from scripts import export_rdf as rdf
from scripts import kb_report as report_module

REPO_ROOT = Path(__file__).resolve().parents[2]


def page(path: Path, front: dict | None, body: str = "\nBody.\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"---\n{yaml.safe_dump(front, sort_keys=False)}---\n" if front is not None else ""
    path.write_text(header + body, encoding="utf-8")


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    kb = tmp_path / "knowledge_base"
    page(
        kb / "math" / "entropy.md",
        {
            "title": "Entropy",
            "type": "concept",
            "created": date(2026, 1, 1),
            "semantic_relations": [
                {"type": "implements", "links": ["[[kl]]"]},
                {"type": "relates", "links": ["[[free_energy]]"]},
            ],
        },
        "\nSee [[kl]] and [[bayes]].\n",
    )
    page(kb / "math" / "kl.md", {"title": "KL", "type": "concept"}, "\n[[entropy]]\n")
    page(kb / "math" / "bayes.md", {"title": "Bayes", "type": "concept"})
    page(kb / "math" / "README.md", {"title": "Math", "type": "index"}, "\n[[entropy]]\n")
    page(
        kb / "bio" / "free_energy.md", {"title": "Free energy", "type": "concept"}, "\n[[bayes]]\n"
    )
    page(kb / "bio" / "lonely.md", {"title": "Lonely", "type": "concept"})
    page(kb / "top.md", {"title": "Top", "type": "concept"})
    return tmp_path


def test_the_graph_has_pages_domains_kinds_and_unique_directed_links(vault: Path) -> None:
    graph = report_module.knowledge_graph(vault)
    assert set(graph.nodes) == {
        "math/entropy",
        "math/kl",
        "math/bayes",
        "math/README",
        "bio/free_energy",
        "bio/lonely",
        "top",
    }
    assert graph.nodes["math/entropy"] == {"title": "Entropy", "domain": "math", "kind": "concept"}
    assert graph.nodes["math/README"]["kind"] == "navigation"
    assert graph.nodes["top"]["domain"] == ""
    # entropy -> kl is both a typed relation and a body link: one edge with two names
    assert graph.edges["math/entropy", "math/kl"]["relations"] == ["implements", "references"]
    assert graph.edges["math/entropy", "bio/free_energy"]["relations"] == ["related"]
    assert graph.has_edge("math/kl", "math/entropy")
    assert not graph.has_edge("math/bayes", "math/entropy")


def test_measurements_match_a_hand_count(vault: Path) -> None:
    graph = report_module.knowledge_graph(vault)
    result = report_module.compute_report(graph)
    assert result["pages"] == 7 and result["links"] == 6
    assert result["isolated_pages"] == ["bio/lonely", "top"]
    assert result["weak_components"] == 3 and result["largest_component"] == 5
    assert result["max_in_degree"] == 2  # bayes is linked from entropy and free_energy
    assert result["reciprocity"] == pytest.approx(2 / 6)  # entropy <-> kl
    assert result["relations"] == {"references": 5, "implements": 1, "related": 1}
    assert result["kinds"] == {"concept": 6, "navigation": 1}
    by_domain = {row["domain"]: row for row in result["domains"]}
    assert by_domain["math"]["pages"] == 4 and by_domain["math"]["outgoing_links"] == 5
    assert by_domain["math"]["share_within_domain"] == pytest.approx(4 / 5)
    assert by_domain["(top level)"]["outgoing_links"] == 0
    assert result["domain_links"]["math"]["bio"] == 1
    assert result["top_in_degree"][0] == {
        "page": "math/bayes",
        "title": "Bayes",
        "domain": "math",
        "degree": 2,
    }
    assert result["in_degree_histogram"] == {0: 3, 1: 2, 2: 2}


def test_markdown_summary_names_the_domains_hubs_and_relations(vault: Path) -> None:
    result = report_module.compute_report(report_module.knowledge_graph(vault))
    text = report_module.report_markdown(result)
    assert "7 pages and 6 links" in text and "2 page(s) have no links" in text
    assert "| math | 4 | 5 | 80% |" in text
    assert "| Bayes | math | 2 |" in text and "| references | 5 |" in text


def test_an_empty_graph_is_an_error() -> None:
    with pytest.raises(ValueError, match="no pages"):
        report_module.compute_report(nx.DiGraph())


def test_command_line_writes_numbers_and_four_figures(vault: Path, tmp_path: Path) -> None:
    out = tmp_path / "report"
    assert report_module.main([str(vault), "--output", str(out), "--format", "svg"]) == 0
    assert json.loads((out / "report.json").read_text())["pages"] == 7
    assert (out / "report.md").is_file()
    for name in ("in_degree_distribution", "domain_sizes", "domain_links", "relations"):
        assert (
            (out / f"{name}.svg").read_text(encoding="utf-8").lstrip().startswith(("<?xml", "<svg"))
        )


def test_no_figures_writes_only_the_numbers_and_relative_output_is_under_the_root(
    vault: Path,
) -> None:
    assert report_module.main([str(vault), "--no-figures", "--output", "results"]) == 0
    assert sorted(path.name for path in (vault / "results").iterdir()) == [
        "report.json",
        "report.md",
    ]


def test_the_real_knowledge_base_is_one_connected_graph_with_its_exported_pages() -> None:
    graph = report_module.knowledge_graph(REPO_ROOT)
    _, export = rdf.build_graph(REPO_ROOT)
    assert graph.number_of_nodes() == export.pages
    result = report_module.compute_report(graph)
    assert result["links"] <= export.typed_edges + export.link_edges
    assert result["largest_component"] > 0.99 * result["pages"]
    assert {"cognitive", "mathematics"} <= {row["domain"] for row in result["domains"]}
