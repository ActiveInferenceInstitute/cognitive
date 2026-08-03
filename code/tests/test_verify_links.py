from pathlib import Path

from scripts.verify_links import main, verify_link_report, verify_links


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

    broken = verify_links(tmp_path)

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

    broken = verify_links(tmp_path)

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
