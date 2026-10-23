"""Real-vault oracles for the read-only semantic review view."""

import hashlib
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest
import yaml
from scripts import export_rdf as rdf
from scripts.semantic_review import (
    build_review_queue,
    expected_outputs,
    render_markdown,
    write_outputs,
)


def page(root: Path, name: str, **metadata: object) -> Path:
    path = root / "knowledge_base" / (name + ".md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\n" + yaml.safe_dump(metadata, sort_keys=False) + "---\nBody.\n")
    return path


def relation(label: str, *targets: str) -> dict:
    return {"type": label, "links": list(targets)}


def vault(root: Path) -> Path:
    page(
        root,
        "a",
        type="concept",
        semantic_relations=[
            relation(
                "implements", "[[guide alias#Details|Alias]]", "[[guide alias#Details|Alias]]"
            ),
            relation("Implements", "[[guide]]"),
            relation("Novel Policy", "[[b]]"),
            relation("documents", "[[guide]]"),
        ],
    )
    page(root, "b", type="concept")
    page(
        root,
        "guide",
        type="guide",
        title="Guide | title",
        aliases=["guide alias"],
        updated=date(2026, 10, 23),
        processing_priority=4,
        semantic_relations=[relation("extends", "[[a]]"), relation("implements", "[[citation]]")],
    )
    page(
        root,
        "citation",
        type="citation",
        doi="10.1234/example",
        semantic_relations=[relation("implements", "[[a]]")],
    )
    return root


def fingerprints(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def cli(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scripts.semantic_review", "--root", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_queue_covers_routes_roles_duplicates_aliases_and_overlapping_union(tmp_path):
    root = vault(tmp_path)
    before = fingerprints(root)
    graph, report = rdf.build_graph(root)
    triple_bytes = rdf.to_nquads(graph)
    queue = build_review_queue(root)
    assert fingerprints(root) == before
    assert rdf.to_nquads(rdf.build_graph(root)[0]) == triple_bytes
    rows = queue["assertions"]
    assert sum(r["occurrence_count"] for r in rows) == len(report.document_relation_routes) == 8
    assert len(rows) == 7
    assert len({r["effective_assertion_id"] for r in rows}) == 6
    ambiguous = {r["assertion_id"] for r in rows if r["disposition"] == "unresolved_interpretation"}
    ordinary = {r["assertion_id"] for r in rows if r["disposition"] == "ordinary_document_relation"}
    assert len(ambiguous) == 6 and len(ordinary) == 1
    groups = queue["endpoint_groups"]
    assert [(g["endpoint"], g["incident_assertion_count"]) for g in groups] == [
        ("guide", 4),
        ("citation", 2),
    ]
    grouped = set().union(*(set(g["assertion_ids"]) for g in groups))
    unassigned = set(queue["relation_policy_review"])
    assert len(grouped) == 5 and len(unassigned) == 1
    assert grouped.isdisjoint(unassigned)
    assert grouped | unassigned == ambiguous
    assert not ordinary & grouped
    assert queue["counts"]["ambiguous_distinct_union"] == 6
    assert queue["counts"]["grouped_distinct_union"] == 5
    assert queue["counts"]["group_memberships"] == 6
    assert [g["cumulative_distinct_union"] for g in groups] == [4, 5]
    assert queue["counts"]["source_declaration_occurrences"] == 8
    assert queue["counts"]["unique_effective_assertions"] == 6
    guide = queue["endpoints"]["guide"]
    assert guide["has_topic"] is False and guide["source_types"] == ["guide"]
    assert "document" in guide["role_explanation"].lower()
    assert guide["source_metadata"]["processing_priority"] == 4
    assert guide["source_metadata"]["updated"] == {"yaml_type": "date", "value": "2026-10-23"}
    assert queue["endpoints"]["a"]["has_topic"] is True
    assert "publication" in queue["endpoints"]["citation"]["role_explanation"].lower()
    repeated = next(r for r in rows if r["source"] == "a" and r["relation"] == "implements")
    assert repeated["target"] == "guide" and repeated["occurrence_count"] == 2
    assert len({d["occurrence_id"] for d in repeated["declarations"]}) == 2
    for declaration in repeated["declarations"]:
        assert declaration["raw_target"] == "[[guide alias#Details|Alias]]"
        start, end = declaration["line"], declaration["end_line"]
        source = (root / "knowledge_base/a.md").read_text().splitlines()
        assert "guide alias#Details|Alias" in "\n".join(source[start - 1 : end])
        assert declaration["source_url"].endswith(
            f"#L{start}" + (f"-L{end}" if end != start else "")
        )
    actual_routes = {
        (r["source"], r["relation"], r["target"], r["predicate"], r["reason"])
        for r in report.document_relation_routes
    }
    assert {
        (r["source"], r["relation"], r["target"], r["predicate"], r["reason"]) for r in rows
    } == actual_routes
    policy = next(r for r in rows if r["assertion_id"] in unassigned)
    assert policy["relation"] == "Novel Policy"
    assert queue["endpoints"][policy["source"]]["has_topic"]
    assert queue["endpoints"][policy["target"]]["has_topic"]
    markdown = render_markdown(queue)
    assert "Ordinary document relations" in markdown and "Relation-policy review" in markdown
    assert "[[guide alias#Details|Alias]]" in markdown
    assert "10.1234/example" in markdown
    assert "Guide \\| title" in markdown
    assert "/Volumes/" not in markdown and str(tmp_path) not in json.dumps(queue)


def test_deterministic_ids_lexical_rows_ties_and_no_topic_eligibility_copy(tmp_path):
    page(
        tmp_path,
        "z",
        type="concept",
        semantic_relations=[relation("implements", "[[docs/README]]")],
    )
    page(tmp_path, "a", type="concept", semantic_relations=[relation("extends", "[[guide]]")])
    page(tmp_path, "docs/README", type="concept")  # navigation overrides incidental source type
    page(tmp_path, "guide", type="implementation")
    first = build_review_queue(tmp_path)
    second = build_review_queue(tmp_path)
    assert expected_outputs(first) == expected_outputs(second)
    assert [g["endpoint"] for g in first["endpoint_groups"]] == ["docs/README", "guide"]
    assert [r["source"] for r in first["assertions"]] == ["a", "z"]
    assert first["endpoints"]["docs/README"]["has_topic"] is False
    assert "navigation" in first["endpoints"]["docs/README"]["role_explanation"].lower()
    ids = {r["assertion_id"] for r in first["assertions"]}
    page(tmp_path, "unrelated", type="concept")
    assert {r["assertion_id"] for r in build_review_queue(tmp_path)["assertions"]} == ids


def test_yaml_alias_occurrences_keep_scalar_origin_without_guessing_alias_use_line(tmp_path):
    page(tmp_path, "g", type="guide")
    source = tmp_path / "knowledge_base/a.md"
    source.write_text(
        "---\ntype: concept\nsemantic_relations:\n  - type: implements\n"
        '    links: [&target "[[g]]", *target]\n---\n'
    )
    row = build_review_queue(tmp_path)["assertions"][0]
    assert row["occurrence_count"] == 2
    assert [d["line"] for d in row["declarations"]] == [5, 5]
    assert all(d["location_kind"] == "yaml_scalar_origin" for d in row["declarations"])
    assert len({d["occurrence_id"] for d in row["declarations"]}) == 2


def test_cli_reporting_is_read_only_and_checks_reject_missing_and_stale(tmp_path):
    root = vault(tmp_path / "vault")
    before = fingerprints(root)
    result = cli(root, "--json")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["schema_version"] == 1
    assert fingerprints(root) == before
    output = root / "output/review"
    missing = cli(root, "--output", str(output), "--check", "--json")
    assert missing.returncode == 1 and not output.exists()
    assert len(json.loads(missing.stdout)["stale"]) == 2
    write = cli(root, "--output", str(output), "--write", "--json")
    assert write.returncode == 0, write.stderr
    sentinel = output / "other.txt"
    sentinel.write_text("preserve")
    assert cli(root, "--output", str(output), "--check").returncode == 0
    queue_file = output / "semantic-review.json"
    stale_queue = json.loads(queue_file.read_text())
    stale_queue["counts"]["unique_source_assertions"] = -1
    queue_file.write_text(json.dumps(stale_queue))
    stale_bytes = fingerprints(root)
    stale = cli(root, "--output", str(output), "--check", "--json")
    assert stale.returncode == 1 and json.loads(stale.stdout)["stale"] == ["semantic-review.json"]
    assert fingerprints(root) == stale_bytes
    assert cli(root, "--output", str(output), "--write").returncode == 0
    assert sentinel.read_text() == "preserve"
    assert cli(root, "--output", str(output)).returncode == 2


@pytest.mark.parametrize(
    "unsafe", ["symlink_directory", "symlink_file", "collision", "escape", "source", "hardlink"]
)
def test_output_preflight_preserves_every_file_on_rejected_destination(tmp_path, unsafe):
    root = vault(tmp_path / "vault")
    queue = build_review_queue(root)
    output = root / "output/review"
    output.mkdir(parents=True)
    target = output / "semantic-review.json"
    if unsafe == "symlink_directory":
        real = tmp_path / "real"
        real.mkdir()
        (root / "linked").symlink_to(real, target_is_directory=True)
        output = root / "linked/review"
    elif unsafe == "symlink_file":
        target.symlink_to(root / "knowledge_base/a.md")
    elif unsafe == "collision":
        target.mkdir()
    elif unsafe == "escape":
        output = root / "output/../../../escaped"
    elif unsafe == "source":
        output = root / "knowledge_base"
    elif unsafe == "hardlink":
        os.link(root / "knowledge_base/a.md", target)
    before = fingerprints(root)
    with pytest.raises(ValueError):
        write_outputs(root, output, queue)
    assert fingerprints(root) == before
    assert not (output / "semantic-review.md").exists()


def test_unowned_destination_collision_and_symlink_source_are_rejected(tmp_path):
    root = vault(tmp_path / "vault")
    queue = build_review_queue(root)
    output = root / "output/review"
    output.mkdir(parents=True)
    occupied = output / "semantic-review.md"
    occupied.write_text("Unrelated existing document\n")
    with pytest.raises(ValueError, match="collision"):
        write_outputs(root, output, queue)
    assert occupied.read_text() == "Unrelated existing document\n"
    assert not (output / "semantic-review.json").exists()
    (root / "knowledge_base/link.md").symlink_to(root / "knowledge_base/a.md")
    with pytest.raises(ValueError, match="symlink|regular"):
        build_review_queue(root)


def test_empty_vault_and_missing_knowledge_base(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_review_queue(tmp_path)
    (tmp_path / "knowledge_base").mkdir()
    queue = build_review_queue(tmp_path)
    assert queue["assertions"] == queue["endpoint_groups"] == queue["relation_policy_review"] == []
    assert queue["counts"]["ambiguous_distinct_union"] == 0


def test_rendering_roundtrip_and_different_checkout_paths_are_byte_identical(tmp_path):
    first = build_review_queue(vault(tmp_path / "one"))
    other = build_review_queue(vault(tmp_path / "two"))
    outputs = expected_outputs(first)
    assert outputs == expected_outputs(other)
    assert outputs == expected_outputs(json.loads(outputs["semantic-review.json"]))


def test_repeated_entries_keep_actual_leaf_indices_and_unique_occurrences(tmp_path):
    page(tmp_path, "g", type="guide")
    (tmp_path / "knowledge_base/a.md").write_text(
        "---\ntype: concept\nsemantic_relations:\n  - type: implements\n"
        '    links: [null, "[[g]]", "[[g]]"]\n  - type: implements\n'
        '    links: ["[[g]]"]\n---\n'
    )
    queue = build_review_queue(tmp_path)
    assert queue["counts"]["source_declaration_occurrences"] == 3
    assert queue["counts"]["unique_source_assertions"] == 1
    assert queue["counts"]["unique_effective_assertions"] == 1
    declarations = queue["assertions"][0]["declarations"]
    assert [(d["entry_index"], d["leaf_index"]) for d in declarations] == [(0, 1), (0, 2), (1, 0)]
    assert len({d["occurrence_id"] for d in declarations}) == 3


def test_rendered_source_examples_do_not_become_new_wiki_references(tmp_path):
    from scripts.verify_links import verify_link_report

    page(tmp_path, "g", type="guide")
    page(tmp_path, "a", type="concept", semantic_relations=[relation("implements", "[[./g]]")])
    queue = build_review_queue(tmp_path)
    write_outputs(tmp_path, Path("docs/okn"), queue)
    report = verify_link_report(tmp_path, strict_wiki_links=True)
    assert report.broken_links == []
    assert report.checked_links == 1  # Only the original source declaration is an active wiki link.
