"""Measure and draw the knowledge base as a graph.

The RDF export (``export_rdf``) turns every page and link into triples. This
script reads those triples back into a directed graph and reports how the
vault is connected: how big each domain is, how much of its linking stays
inside the domain, which pages are hubs, how the degree distribution falls
off, and which relation names carry the typed edges. Building the graph from
the exported triples rather than from the Markdown keeps the report honest
about what the published graph contains.

Usage::

    python -m scripts.kb_report                      # JSON, Markdown, figures: output/kb_report
    python -m scripts.kb_report --no-figures         # numbers only
    python -m scripts.kb_report --output /tmp/report --format svg

The report is derived data and is not committed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import networkx as nx

try:
    from scripts import export_rdf as rdf
except ImportError:  # run as a file: the script directory is on sys.path
    import export_rdf as rdf  # type: ignore[no-redef]

REPORT_SCHEMA_VERSION = 1
TOP = 15

_SHORT = {
    rdf.PREFIXES["skos"] + "related": "related",
    rdf.PREFIXES["dcterms"] + "isPartOf": "isPartOf",
    rdf.PREFIXES["dcterms"] + "hasPart": "hasPart",
    rdf.PREFIXES["dcterms"] + "references": "references",
    rdf.PREFIXES["dcterms"] + "isReplacedBy": "isReplacedBy",
}


def _bare(term: str) -> str:
    return term[1:-1] if term.startswith("<") else term


def _page_id(iri_text: str) -> str | None:
    return unquote(iri_text[len(rdf.PAGE) :]) if iri_text.startswith(rdf.PAGE) else None


def _literal(term: str) -> str:
    return term[1 : term.rindex('"')].replace('\\"', '"').replace("\\\\", "\\")


def knowledge_graph(repo_root: Path) -> nx.DiGraph:
    """The page graph of ``repo_root``: nodes are pages, edges are links.

    Node attributes: ``title``, ``domain`` (top-level folder, ``""`` for a
    page at the top) and ``kind`` (``concept``, ``navigation`` or ``other``).
    An edge carries ``relations``, the set of relation names that link the pair
    in that direction (``references`` for a body link, ``related`` for
    ``skos:related``, the local name for a graph-defined relation).
    """
    triples, _ = rdf.build_graph(repo_root)
    graph = nx.DiGraph()
    titles: dict[str, str] = {}
    kinds: dict[str, str] = {}
    domains: dict[str, str] = {}
    edges: dict[tuple[str, str], set[str]] = defaultdict(set)
    type_predicate = rdf.RDF_TYPE
    title_predicate = rdf.PREFIXES["dcterms"] + "title"
    member = rdf.PREFIXES["skos"] + "member"
    for subject, predicate, obj in triples.triples:
        predicate_iri = _bare(predicate)
        source = _page_id(_bare(subject))
        if predicate_iri == member and _bare(subject).startswith(rdf.DOMAIN):
            target = _page_id(_bare(obj))
            if target is not None:
                domains[target] = unquote(_bare(subject)[len(rdf.DOMAIN) :])
            continue
        if source is None:
            continue
        if predicate_iri == title_predicate:
            titles[source] = _literal(obj)
        elif predicate_iri == type_predicate:
            if _bare(obj) == rdf.PREFIXES["schema"] + "CollectionPage":
                kinds[source] = "navigation"
            elif _bare(obj) == rdf.PREFIXES["skos"] + "Concept":
                kinds.setdefault(source, "concept")
        else:
            target = _page_id(_bare(obj))
            if target is None:
                continue
            if predicate_iri in _SHORT:
                name = _SHORT[predicate_iri]
            elif predicate_iri.startswith(rdf.RELATION):
                name = predicate_iri[len(rdf.RELATION) :]
            else:
                continue
            edges[(source, target)].add(name)
    for page in titles:
        graph.add_node(
            page,
            title=titles[page],
            domain=domains.get(page, ""),
            kind=kinds.get(page, "other"),
        )
    for (source, target), names in edges.items():
        graph.add_edge(source, target, relations=sorted(names))
    return graph


def _degree_table(graph: nx.DiGraph, degrees: Any) -> list[dict[str, Any]]:
    ranked = sorted(degrees, key=lambda item: (-item[1], item[0]))[:TOP]
    return [
        {
            "page": page,
            "title": graph.nodes[page]["title"],
            "domain": graph.nodes[page]["domain"],
            "degree": degree,
        }
        for page, degree in ranked
    ]


def compute_report(graph: nx.DiGraph) -> dict[str, Any]:
    """Structural measurements of the page graph, JSON-serializable."""
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    if nodes == 0:
        raise ValueError("the knowledge graph has no pages")
    weak = sorted((len(c) for c in nx.weakly_connected_components(graph)), reverse=True)
    undirected = graph.to_undirected()
    domain_pages: Counter[str] = Counter(d["domain"] for _, d in graph.nodes(data=True))
    within: Counter[str] = Counter()
    leaving: Counter[str] = Counter()
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for source, target in graph.edges:
        a, b = graph.nodes[source]["domain"], graph.nodes[target]["domain"]
        matrix[a][b] += 1
        (within if a == b else leaving)[a] += 1
    relations: Counter[str] = Counter()
    for _, _, data in graph.edges(data=True):
        relations.update(data["relations"])
    in_degrees = [degree for _, degree in graph.in_degree()]
    domains = []
    for name in sorted(domain_pages, key=lambda key: (-domain_pages[key], key)):
        outgoing = within[name] + leaving[name]
        domains.append(
            {
                "domain": name or "(top level)",
                "pages": domain_pages[name],
                "outgoing_links": outgoing,
                "share_within_domain": (within[name] / outgoing) if outgoing else 0.0,
            }
        )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "pages": nodes,
        "links": edges,
        "kinds": dict(Counter(d["kind"] for _, d in graph.nodes(data=True))),
        "density": nx.density(graph),
        "reciprocity": nx.reciprocity(graph) if edges else 0.0,
        "mean_out_degree": edges / nodes,
        "max_in_degree": max(in_degrees),
        "median_in_degree": float(sorted(in_degrees)[len(in_degrees) // 2]),
        "isolated_pages": sorted(nx.isolates(graph)),
        "weak_components": len(weak),
        "largest_component": weak[0],
        "transitivity": nx.transitivity(undirected),
        "domains": domains,
        "domain_links": {a: dict(row) for a, row in matrix.items()},
        "relations": dict(relations.most_common()),
        "top_in_degree": _degree_table(graph, graph.in_degree()),
        "top_out_degree": _degree_table(graph, graph.out_degree()),
        "in_degree_histogram": dict(sorted(Counter(in_degrees).items())),
    }


def report_markdown(report: dict[str, Any]) -> str:
    """A human-readable summary of ``report``."""
    lines = [
        "# Knowledge base connectivity",
        "",
        f"{report['pages']} pages and {report['links']} links "
        f"(density {report['density']:.4f}, reciprocity {report['reciprocity']:.3f}, "
        f"transitivity {report['transitivity']:.3f}).",
        f"{report['weak_components']} weakly connected component(s); the largest holds "
        f"{report['largest_component']} pages. {len(report['isolated_pages'])} page(s) have no "
        "links in or out.",
        "",
        "## Domains",
        "",
        "| Domain | Pages | Outgoing links | Share staying in domain |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {row['domain']} | {row['pages']} | {row['outgoing_links']} | "
        f"{row['share_within_domain']:.0%} |"
        for row in report["domains"]
    ]
    lines += [
        "",
        "## Most linked-to pages",
        "",
        "| Page | Domain | Incoming links |",
        "|---|---|---|",
    ]
    lines += [
        f"| {row['title']} | {row['domain'] or '(top level)'} | {row['degree']} |"
        for row in report["top_in_degree"]
    ]
    lines += ["", "## Relations", "", "| Relation | Edges |", "|---|---|"]
    lines += [f"| {name} | {count} |" for name, count in report["relations"].items()]
    return "\n".join(lines) + "\n"


def draw_figures(
    graph: nx.DiGraph, report: dict[str, Any], output: Path, *, image_format: str, dpi: int
) -> list[Path]:
    """Write four figures into ``output`` and return their paths."""
    import matplotlib.pyplot as plt
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def save(fig: Any, name: str) -> None:
        path = output / f"{name}.{image_format}"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        written.append(path)

    histogram = report["in_degree_histogram"]
    degrees = np.array([int(k) for k in histogram if int(k) > 0], dtype=float)
    counts = np.array([histogram[k] for k in histogram if int(k) > 0], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
    if degrees.size:
        survival = counts[::-1].cumsum()[::-1] / counts.sum()
        ax.loglog(degrees, survival, "o", color="#2563eb", markersize=4)
    ax.set(
        title="Incoming links per page",
        xlabel="incoming links k",
        ylabel="share of pages with at least k",
    )
    save(fig, "in_degree_distribution")

    rows = report["domains"]
    labels = [row["domain"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.5), constrained_layout=True)
    positions = np.arange(len(rows))
    ax.barh(positions, [row["pages"] for row in rows], color="#2563eb")
    for position, row in zip(positions, rows, strict=True):
        ax.text(
            row["pages"],
            position,
            f"  {row['share_within_domain']:.0%} of links stay inside",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set(title="Pages per domain", xlabel="pages")
    ax.set_xlim(0, max(row["pages"] for row in rows) * 1.45)
    save(fig, "domain_sizes")

    named = [name for name in labels if name != "(top level)"]
    matrix = np.array(
        [[report["domain_links"].get(a, {}).get(b, 0) for b in named] for a in named], dtype=float
    )
    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    image = ax.imshow(np.log1p(matrix), cmap="viridis")
    ax.set_xticks(range(len(named)))
    ax.set_xticklabels(named, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(named)))
    ax.set_yticklabels(named, fontsize=8)
    ax.set(title="Links between domains (from row to column)", xlabel="target", ylabel="source")
    fig.colorbar(image, ax=ax, label="log(1 + links)")
    save(fig, "domain_links")

    relations = list(report["relations"].items())[:20]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(relations) + 1.4), constrained_layout=True)
    ax.barh([name for name, _ in relations], [count for _, count in relations], color="#7c3aed")
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set(title="Edges by relation", xlabel="edges (log scale)")
    save(fig, "relations")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure and draw the knowledge-base graph.")
    parser.add_argument("root", nargs="?", default=".", help="repository root")
    parser.add_argument("--output", default="output/kb_report", help="directory for the results")
    parser.add_argument("--format", default="png", choices=["png", "svg", "pdf"])
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--no-figures", action="store_true", help="write only the numbers")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    output = Path(args.output)
    output = output if output.is_absolute() else root / output

    graph = knowledge_graph(root)
    report = compute_report(graph)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(report_markdown(report), encoding="utf-8")
    written = [output / "report.json", output / "report.md"]
    if not args.no_figures:
        import matplotlib

        matplotlib.use("Agg")
        written += draw_figures(graph, report, output, image_format=args.format, dpi=args.dpi)
    print(
        f"{report['pages']} pages, {report['links']} links, "
        f"{report['weak_components']} component(s)"
    )
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
