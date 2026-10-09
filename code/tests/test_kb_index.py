from pathlib import Path

import yaml
from scripts import kb_index as kb


def _page(path: Path, front: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\n{front}\n---\n{body}", encoding="utf-8")


def _tree(tmp_path: Path) -> Path:
    base = tmp_path / "knowledge_base"
    _page(
        base / "math" / "entropy.md",
        "title: Entropy",
        "\n# Entropy\n\n## Overview\n\nEntropy measures the uncertainty of a distribution. More.\n",
    )
    _page(
        base / "math" / "deep" / "kl.md",
        "title: KL Divergence",
        "\nKL divergence compares two beliefs.\n",
    )
    _page(base / "math" / "README.md", "title: Math", "\nNavigation page.\n")
    _page(
        base / "bio" / "ants.md", "title: Ants", "\nAnts forage by stigmergy, leaving pheromone.\n"
    )
    _page(base / "glossary.md", "title: Glossary", "\nDefinitions of the terms used throughout.\n")
    return tmp_path


def test_describe_prefers_declared_description_then_overview():
    assert kb.describe({"description": "Declared text here that is long enough."}, "x") == (
        "Declared text here that is long enough."
    )
    body = "Intro line that is ignored here.\n\n## Overview\n\nThe real summary sentence. Second.\n"
    assert kb.describe({}, body) == "The real summary sentence."


def test_describe_rejects_lead_ins_math_and_boilerplate():
    assert kb.describe({}, "The update is given by the following equation:\n") == ""
    assert kb.describe({}, r"Posterior \(q(s)\) is computed from the prior." + "\n") == ""
    boilerplate = (
        "This knowledge base entry is a foundation. Contributions welcome — see the guide.\n"
    )
    assert "foundation" not in kb.describe({}, boilerplate)
    assert kb.describe({}, "```python\nx = 1\n```\n\n- bullet only\n") == ""


def test_describe_strips_markdown_and_truncates():
    text = "A [[concept|linked]] term with **bold** and [site](http://x.y) markup."
    assert kb.describe({}, text + "\n") == "A linked term with bold and site markup."
    long = " ".join(["word"] * 80)
    out = kb.describe({}, long + ".\n")
    assert len(out) <= 180 and out.endswith("…")


def test_page_title_uses_frontmatter_then_heading_then_stem():
    assert kb.page_title({"title": "A | [B]"}, "", "s") == "A B"
    assert kb.page_title({}, "# From Heading\n", "s") == "From Heading"
    assert kb.page_title({}, "no heading", "free_energy") == "Free Energy"


def test_catalogs_cover_every_page_and_skip_navigation_files(tmp_path):
    root = _tree(tmp_path)
    catalogs = kb.expected_catalogs(root)

    math = catalogs[root / "knowledge_base" / "math" / "catalog.md"]
    assert "[[knowledge_base/math/entropy|Entropy]] — Entropy measures" in math
    assert "[[knowledge_base/math/deep/kl|KL Divergence]]" in math
    assert not [
        line for line in math.splitlines() if line.startswith("- [[knowledge_base/math/README")
    ]
    assert "## `deep/`" in math
    assert math.index("## Top level") < math.index("## `deep/`")

    top = catalogs[root / "knowledge_base" / "catalog.md"]
    assert "[[knowledge_base/math/catalog|math]] — 2 pages" in top
    assert "[[knowledge_base/bio/catalog|bio]] — 1 pages" in top
    assert "[[knowledge_base/glossary|Glossary]]" in top


def test_empty_files_are_not_catalogued(tmp_path):
    root = _tree(tmp_path)
    (root / "knowledge_base" / "math" / "empty.md").write_text("", encoding="utf-8")

    catalog = kb.expected_catalogs(root)[root / "knowledge_base" / "math" / "catalog.md"]

    assert "empty" not in catalog
    assert (
        "catalog|math]] — 2 pages"
        in kb.expected_catalogs(root)[root / "knowledge_base" / "catalog.md"]
    )


def test_generated_frontmatter_is_valid_yaml(tmp_path):
    for content in kb.expected_catalogs(_tree(tmp_path)).values():
        front = yaml.safe_load(content.split("---")[1])
        assert front["type"] == "catalog"
        assert front["title"].startswith("Catalog")


def test_check_write_cycle_detects_staleness_and_new_pages(tmp_path, capsys):
    root = _tree(tmp_path)

    assert kb.main([str(root), "--check"]) == 1
    assert "knowledge_base/math/catalog.md" in capsys.readouterr().out

    assert kb.main([str(root), "--write"]) == 0
    assert kb.stale_catalogs(root) == []
    assert kb.main([str(root), "--check"]) == 0

    _page(
        root / "knowledge_base" / "math" / "fisher.md", "title: Fisher", "\nFisher information.\n"
    )
    assert kb.stale_catalogs(root) == [
        "knowledge_base/catalog.md",
        "knowledge_base/math/catalog.md",
    ]

    (root / "knowledge_base" / "bio" / "catalog.md").unlink()
    assert "knowledge_base/bio/catalog.md" in kb.stale_catalogs(root)
