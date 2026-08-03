from scripts.validate_docs import _validate_frontmatter


def test_validate_frontmatter_accepts_valid_metadata(tmp_path):
    content = (
        "---\n"
        "title: Concept\n"
        "type: concept\n"
        "id: concept_001\n"
        "created: 2024-02-05\n"
        "---\n"
        "\n"
        "# Concept\n"
    )

    assert _validate_frontmatter(tmp_path / "concept.md", content) == []


def test_validate_frontmatter_rejects_template_placeholder_values(tmp_path):
    content = "---\ntype: matrix_spec\nid: unique_identifier\ncreated: timestamp\n---\n"

    errors = _validate_frontmatter(tmp_path / "spec.md", content)

    assert len(errors) == 1
    assert "unique_identifier" in errors[0]


def test_validate_frontmatter_rejects_malformed_yaml(tmp_path):
    content = "---\ntitle: [unclosed\n---\n"

    errors = _validate_frontmatter(tmp_path / "bad.md", content)

    assert len(errors) == 1
    assert "invalid YAML" in errors[0]
