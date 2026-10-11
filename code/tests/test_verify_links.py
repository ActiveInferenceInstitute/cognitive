from pathlib import Path

from scripts.verify_links import main, verify_link_report


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_extensionless_wiki_links_can_be_intentional_concepts(tmp_path):
    write_markdown(
        tmp_path / "notes.md",
        "Related: [[systems_engineering]], [[Active Inference Education]].",
    )

    report = verify_link_report(tmp_path)

    assert report.broken_links == []
    assert report.skipped_concept_links == 2


def test_existing_wiki_targets_resolve_by_path_directory_and_alias(tmp_path):
    write_markdown(tmp_path / "docs" / "README.md", "# Docs")
    write_markdown(tmp_path / "knowledge_base" / "concept.md", "# Concept")
    write_markdown(tmp_path / "folder" / "index.md", "# Folder")
    write_markdown(
        tmp_path / "index.md",
        "\n".join(
            [
                "[[docs/README|Documentation]]",
                "[[knowledge_base/concept#Heading]]",
                "[[folder]]",
            ]
        ),
    )

    report = verify_link_report(tmp_path)

    assert report.broken_links == []
    assert report.resolved_links == 3


def test_missing_explicit_markdown_file_is_broken(tmp_path):
    write_markdown(tmp_path / "index.md", "See [[missing.md]].")

    broken = verify_link_report(tmp_path).broken_links

    assert broken == [
        {
            "source": "index.md",
            "link": "missing.md",
            "target": "missing.md",
        }
    ]


def test_missing_explicit_non_markdown_file_is_broken(tmp_path):
    write_markdown(tmp_path / "simulation.md", "# Simulation")
    write_markdown(tmp_path / "index.md", "See [[simulation.py]].")

    broken = verify_link_report(tmp_path).broken_links

    assert broken == [
        {
            "source": "index.md",
            "link": "simulation.py",
            "target": "simulation.py",
        }
    ]


def test_generated_output_directories_are_not_scanned(tmp_path):
    write_markdown(
        tmp_path / "code" / "Thing" / "outputs" / "request.md",
        "Generated data link: [[Saffir.json]].",
    )
    write_markdown(tmp_path / "index.md", "See [[source_concept]].")

    report = verify_link_report(tmp_path)

    assert report.broken_links == []
    assert report.skipped_concept_links == 1


def test_strict_wiki_mode_reports_unresolved_concepts(tmp_path):
    write_markdown(tmp_path / "index.md", "See [[systems_engineering]].")

    report = verify_link_report(tmp_path, strict_wiki_links=True)

    assert report.broken_links == [
        {
            "source": "index.md",
            "link": "systems_engineering",
            "target": "systems_engineering",
        }
    ]
    assert report.skipped_concept_links == 0


def test_cli_defaults_to_concept_aware_validation(tmp_path, capsys):
    write_markdown(tmp_path / "index.md", "See [[systems_engineering]].")

    exit_code = main([str(tmp_path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Skipped 1 unresolved concept/wiki links" in output


def test_valid_anchor_fragment_does_not_warn(tmp_path):
    write_markdown(tmp_path / "concept.md", "# Concept\n\n## Sub Section\n")
    write_markdown(
        tmp_path / "index.md",
        "See [[concept#Sub-Section|the subsection]].",
    )

    report = verify_link_report(tmp_path)

    assert report.broken_links == []
    assert report.anchor_warnings == []


def test_missing_anchor_fragment_warns(tmp_path):
    write_markdown(tmp_path / "concept.md", "# Concept\n")
    write_markdown(
        tmp_path / "index.md",
        "See [[concept#Missing-Anchor|missing]].",
    )

    report = verify_link_report(tmp_path)

    assert report.broken_links == []
    assert len(report.anchor_warnings) == 1
    assert report.anchor_warnings[0]["fragment"] == "Missing-Anchor"
    assert report.anchor_warnings[0]["source"] == "index.md"


def test_stem_ambiguous_concept_link_is_not_broken(tmp_path):
    write_markdown(tmp_path / "a" / "README.md", "# A")
    write_markdown(tmp_path / "b" / "README.md", "# B")
    write_markdown(tmp_path / "index.md", "See [[README]].")

    report = verify_link_report(tmp_path)

    assert report.broken_links == []


def test_cli_text_mode_reports_broken_links_and_returns_one(tmp_path, capsys):
    write_markdown(tmp_path / "index.md", "See [[missing-file.md|the manual]].")
    write_markdown(tmp_path / "other.md", "Also [[gone.md]].")

    exit_code = main([str(tmp_path)])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "Index built with" in output
    assert "Checked" in output
    assert "Found 2 broken file links:" in output
    assert "missing-file.md" in output
    assert "gone.md" in output


def test_cli_text_mode_reports_anchor_warnings(tmp_path, capsys):
    write_markdown(tmp_path / "concept.md", "# Concept\n")
    write_markdown(tmp_path / "index.md", "See [[concept#No-Such-Heading]].")

    exit_code = main([str(tmp_path)])

    output = capsys.readouterr().out
    # A missing anchor is a warning, not a broken link.
    assert exit_code == 0
    assert "Found 1 anchor warnings:" in output
    assert "No-Such-Heading" in output


def test_cli_json_mode_outputs_the_full_report(tmp_path, capsys):
    import json

    write_markdown(tmp_path / "index.md", "See [[missing.md]].")

    exit_code = main([str(tmp_path), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["broken_links"] == [
        {"source": "index.md", "link": "missing.md", "target": "missing.md"}
    ]
    assert payload["checked_links"] == 1
    assert payload["anchor_warnings"] == []


def test_cli_json_mode_exit_zero_without_broken_links(tmp_path, capsys):
    import json

    write_markdown(tmp_path / "index.md", "See [[systems_engineering]].")

    exit_code = main([str(tmp_path), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["broken_links"] == []
    assert payload["skipped_concept_links"] == 1


def test_heading_slugs_return_empty_on_unreadable_file(tmp_path):
    from scripts.verify_links import _heading_slugs

    # A directory has no readable text content; the OSError guard returns no slugs.
    assert _heading_slugs(tmp_path / "does-not-exist.md") == set()


def test_target_exists_mirror_of_resolve(tmp_path):
    from scripts.verify_links import _build_index, _resolve_target_path, _target_exists

    write_markdown(tmp_path / "concept.md", "# Concept\n")
    index = _build_index(tmp_path)
    source = tmp_path / "index.md"
    assert _target_exists(tmp_path, source, "concept", index) is True
    assert _target_exists(tmp_path, source, "missing.md", index) is False
    assert _resolve_target_path(tmp_path, source, "concept", index) == tmp_path / "concept.md"
    assert _resolve_target_path(tmp_path, source, "missing.md", index) is None


def test_a_frontmatter_alias_resolves_a_concept_link(tmp_path):
    """Obsidian opens the page that lists a name under ``aliases:``; so must the checker."""
    write_markdown(
        tmp_path / "plasticity_page.md",
        "---\ntitle: Plasticity\naliases:\n  - plasticity\n  - Synaptic Change\n---\n\n# B\n",
    )
    write_markdown(
        tmp_path / "index.md", "See [[plasticity]], [[synaptic_change]] and [[unrelated]]."
    )

    report = verify_link_report(tmp_path)

    assert report.resolved_links == 2
    assert report.skipped_concept_links == 1
    assert report.broken_links == []


def test_alias_anchors_are_checked_against_the_aliased_page(tmp_path):
    write_markdown(tmp_path / "page.md", "---\naliases: [short]\n---\n\n# Real heading\n")
    write_markdown(tmp_path / "index.md", "[[short#real-heading]] [[short#missing]]")

    report = verify_link_report(tmp_path)

    assert [w["fragment"] for w in report.anchor_warnings] == ["missing"]


def test_malformed_or_odd_aliases_are_ignored(tmp_path):
    write_markdown(tmp_path / "bad.md", "---\naliases: [unclosed\n---\n\n# Bad\n")
    write_markdown(tmp_path / "odd.md", "---\naliases: 7\n---\n\n# Odd\n")
    write_markdown(tmp_path / "text.md", "---\naliases: single\n---\n\n# Text\n")
    write_markdown(tmp_path / "index.md", "[[single]] [[nothing]]")

    report = verify_link_report(tmp_path)

    assert report.resolved_links == 1  # the string form is accepted, nothing else
    assert report.skipped_concept_links == 1


def test_alias_names_ignore_case_hyphens_and_underscores(tmp_path):
    write_markdown(
        tmp_path / "geometry.md",
        "---\naliases:\n- riemannian-geometry\n- Free Energy\n---\n\n# G\n",
    )
    write_markdown(tmp_path / "index.md", "[[riemannian_geometry]] [[free_energy]] [[Free-Energy]]")

    report = verify_link_report(tmp_path)

    assert report.resolved_links == 3
    assert report.skipped_concept_links == 0
