from pathlib import Path

from scripts.check_markdown_links import main, verify_markdown_link_report


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_broken_relative_file_link_is_reported(tmp_path):
    write_markdown(tmp_path / "index.md", "See [missing](nope.md).")

    report = verify_markdown_link_report(tmp_path)

    assert len(report.broken_links) == 1
    assert report.broken_links[0]["source"] == "index.md"
    assert report.broken_links[0]["target"] == "nope.md"


def test_existing_relative_and_root_relative_links_resolve(tmp_path):
    write_markdown(tmp_path / "docs" / "a.md", "Link [b](../guide.md).")
    write_markdown(tmp_path / "guide.md", "# Guide")
    write_markdown(tmp_path / "docs" / "b.md", "Link [root](guide.md).")

    report = verify_markdown_link_report(tmp_path)

    assert report.broken_links == []
    assert report.checked_links == 2


def test_image_links_are_checked(tmp_path):
    write_markdown(tmp_path / "doc.md", "![alt](picture.png)")
    (tmp_path / "picture.png").write_bytes(b"x")
    write_markdown(tmp_path / "other.md", "![alt](missing.png)")

    report = verify_markdown_link_report(tmp_path)

    assert [b["target"] for b in report.broken_links] == ["missing.png"]


def test_external_urls_are_skipped(tmp_path):
    write_markdown(
        tmp_path / "doc.md",
        "[site](https://example.com) [mail](mailto:a@example.com)",
    )

    report = verify_markdown_link_report(tmp_path)

    assert report.checked_links == 0
    assert report.broken_links == []


def test_same_file_anchor_mismatch_is_a_warning(tmp_path):
    write_markdown(tmp_path / "doc.md", "# Hello\n\n[bad](#nope) [good](#hello)")

    report = verify_markdown_link_report(tmp_path)

    assert report.broken_links == []
    assert [w["link"] for w in report.anchor_warnings] == ["#nope"]


def test_cross_file_anchor_is_checked_against_heading_slug(tmp_path):
    write_markdown(tmp_path / "a.md", "[ok](b.md#my-section) [bad](b.md#missing)")
    write_markdown(tmp_path / "b.md", "## My Section")

    report = verify_markdown_link_report(tmp_path)

    assert report.broken_links == []
    assert [w["link"] for w in report.anchor_warnings] == ["[bad](b.md#missing)"]


def test_code_fences_and_latex_are_not_scanned(tmp_path):
    write_markdown(
        tmp_path / "doc.md",
        "```\n[not-a-link](fake.md)\n```\n\nInline $[x](y)$ math.\n",
    )

    report = verify_markdown_link_report(tmp_path)

    assert report.checked_links == 0
    assert report.broken_links == []


def test_vendored_rxinfer_docs_subtree_is_skipped(tmp_path):
    vendored = tmp_path / "docs" / "implementation" / "rxinfer" / "docs"
    write_markdown(vendored / "index.md", "[ref](@ref some-page) [id](@id anchor)")
    write_markdown(tmp_path / "docs" / "README.md", "# Docs")

    report = verify_markdown_link_report(tmp_path)

    assert report.indexed_markdown_files == 1
    assert report.checked_links == 0
    assert report.broken_links == []


def test_manuscript_figure_references_are_allowed(tmp_path):
    write_markdown(
        tmp_path / "docs" / "manuscript" / "03_results.md",
        "![Figure 1](figures/belief_updates.png)",
    )

    report = verify_markdown_link_report(tmp_path)

    assert report.checked_links == 1
    assert report.broken_links == []


def test_cli_json_exit_code_reflects_broken_links(tmp_path, capsys):
    write_markdown(tmp_path / "doc.md", "[broken](gone.md)")

    exit_code = main([str(tmp_path), "--json"])

    assert exit_code == 1
    assert "broken_links" in capsys.readouterr().out
