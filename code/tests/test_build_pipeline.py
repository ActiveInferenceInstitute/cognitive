"""End-to-end coverage for the manuscript builder and the documentation gates.

Both are executable parts of the repository that a reader is invited to run.
Before this module existed neither was exercised by the suite, so a break in
the builder or in the validator surfaced only when someone ran a release.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
from scripts import build_manuscript
from scripts.validate_docs import load_term_policy, validate

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_builder_produces_every_declared_figure_and_variable(tmp_path: Path) -> None:
    """The pipeline runs without a document toolchain and declares what it made."""
    output = build_manuscript.build(
        REPO_ROOT / "docs" / "manuscript" / "config.yaml",
        tmp_path / "manuscript",
        render_documents=False,
    )
    registry = json.loads((output / "figure_registry.json").read_text(encoding="utf-8"))
    variables = json.loads((output / "manuscript_variables.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "build_manifest.json").read_text(encoding="utf-8"))

    assert registry, "the builder must declare at least one figure"
    for entry in registry:
        assert (output / entry["file"]).is_file(), f"declared but missing: {entry['file']}"
        assert (output / entry["file"]).stat().st_size > 0
    assert variables["FIGURE_COUNT"] == str(len(registry))
    assert manifest["rendered_documents"] is False
    assert manifest["rendered_pdf"] is False
    assert manifest["combined_sha256"]
    assert (output / "combined.md").is_file()
    assert (output / "references.bib").is_file()


def test_builder_is_deterministic_for_a_fixed_seed(tmp_path: Path) -> None:
    """Two builds of the same configuration hydrate byte-identical source."""
    first = build_manuscript.build(
        REPO_ROOT / "docs" / "manuscript" / "config.yaml",
        tmp_path / "first",
        render_documents=False,
    )
    second = build_manuscript.build(
        REPO_ROOT / "docs" / "manuscript" / "config.yaml",
        tmp_path / "second",
        render_documents=False,
    )
    first_manifest = json.loads((first / "build_manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "build_manifest.json").read_text(encoding="utf-8"))
    assert first_manifest["combined_sha256"] == second_manifest["combined_sha256"]
    assert first_manifest["variables"] == second_manifest["variables"]


def test_builder_rebuilds_over_its_own_output_only(tmp_path: Path) -> None:
    """A previous build is replaced; an unrelated directory is refused."""
    output = tmp_path / "manuscript"
    build_manuscript.build(
        REPO_ROOT / "docs" / "manuscript" / "config.yaml", output, render_documents=False
    )
    assert build_manuscript._is_build_output(output)
    build_manuscript.build(
        REPO_ROOT / "docs" / "manuscript" / "config.yaml", output, render_documents=False
    )

    occupied = tmp_path / "not-a-build"
    occupied.mkdir()
    (occupied / "important.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(ValueError, match="Refusing to delete non-build directory"):
        build_manuscript.build(
            REPO_ROOT / "docs" / "manuscript" / "config.yaml", occupied, render_documents=False
        )
    assert (occupied / "important.txt").read_text(encoding="utf-8") == "keep me"

    empty = tmp_path / "empty"
    empty.mkdir()
    build_manuscript._prepare_output(empty)
    assert empty.is_dir() and not any(empty.iterdir())


def test_builder_cli_accepts_no_render(tmp_path: Path) -> None:
    """The documented flag combination runs and reports success."""
    exit_code = build_manuscript.main(
        [
            "--config",
            str(REPO_ROOT / "docs" / "manuscript" / "config.yaml"),
            "--output",
            str(tmp_path / "cli"),
            "--no-render",
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "cli" / "build_manifest.json").is_file()


def test_manuscript_figures_reflect_the_corrected_expected_free_energy() -> None:
    """The manuscript's experiment uses the package objective, not a copy of it."""
    config = yaml.safe_load(
        (REPO_ROOT / "docs" / "manuscript" / "config.yaml").read_text(encoding="utf-8")
    )
    model, results = build_manuscript._model_and_results(config)
    energies = np.asarray(results["efe"], dtype=float)
    beliefs = np.asarray(results["efe_beliefs"], dtype=float)
    for action in range(model.num_actions):
        total, risk, ambiguity, _epistemic = model.expected_free_energy(beliefs, action)
        assert energies[action, 0] == pytest.approx(total, abs=1e-12)
        assert energies[action, 0] == pytest.approx(risk + ambiguity, abs=1e-12)
        assert energies[action, 1] == pytest.approx(risk, abs=1e-12)
        assert energies[action, 2] == pytest.approx(ambiguity, abs=1e-12)


def test_manuscript_experiment_distinguishes_its_actions() -> None:
    """The reported decomposition must actually separate the available actions.

    Evaluated at the uniform prior, both transition matrices in this experiment
    predict the same state distribution, so every action carries an identical
    decomposition and the figure conveys nothing. The builder evaluates at the
    posterior instead; this pins that the resulting figure is informative.
    """
    config = yaml.safe_load(
        (REPO_ROOT / "docs" / "manuscript" / "config.yaml").read_text(encoding="utf-8")
    )
    model, results = build_manuscript._model_and_results(config)
    energies = np.asarray(results["efe"], dtype=float)
    assert energies.shape[0] >= 2
    totals = energies[:, 0]
    assert totals.max() - totals.min() > 1e-3, "the actions must be distinguishable"
    beliefs = np.asarray(results["efe_beliefs"], dtype=float)
    assert not np.allclose(beliefs, model.D), "the posterior must differ from the prior"


def _minimal_repo(root: Path) -> None:
    """Create a tiny tracked repository the validator can run against."""
    import subprocess

    (root / "docs" / "policy").mkdir(parents=True)
    shutil.copy(
        REPO_ROOT / "docs" / "policy" / "documentation_terms.yaml",
        root / "docs" / "policy" / "documentation_terms.yaml",
    )
    (root / "README.md").write_text("# Sample\n\nA sentence.\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)


def test_validator_reads_its_policy_from_the_declared_file(tmp_path: Path) -> None:
    """The term list comes from the YAML policy, not from the validator's source."""
    terms, exempt = load_term_policy(REPO_ROOT)
    assert terms, "the policy must declare at least one term"
    assert "docs/policy/documentation_terms.yaml" in exempt
    assert "code/scripts/validate_docs.py" in exempt

    with pytest.raises(FileNotFoundError):
        load_term_policy(tmp_path)

    broken = tmp_path / "broken"
    (broken / "docs" / "policy").mkdir(parents=True)
    policy = broken / "docs" / "policy" / "documentation_terms.yaml"
    policy.write_text("terms: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty list of terms"):
        load_term_policy(broken)
    policy.write_text("terms: [sample]\nexempt_paths: 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exempt_paths"):
        load_term_policy(broken)
    policy.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a mapping"):
        load_term_policy(broken)


def test_validator_flags_a_declared_term_and_honours_exemptions(tmp_path: Path) -> None:
    """A banned word fails the gate, and only exempt files may contain it."""
    import subprocess

    root = tmp_path / "repo"
    root.mkdir()
    _minimal_repo(root)
    clean = validate(root)
    assert clean["forbidden_terms"] == []
    assert clean["term_policy"]["terms"] > 0

    terms, _exempt = load_term_policy(REPO_ROOT)
    offending = terms[0]
    (root / "page.md").write_text(f"# Page\n\nThis section is a {offending}.\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    flagged = validate(root)
    assert flagged["ok"] is False
    assert {entry["term"] for entry in flagged["forbidden_terms"]} == {offending}
    assert {entry["path"] for entry in flagged["forbidden_terms"]} == {"page.md"}


def test_validator_rejects_broken_frontmatter_and_python_examples(tmp_path: Path) -> None:
    """Both documentation checks fire on real malformed input."""
    import subprocess

    root = tmp_path / "repo"
    root.mkdir()
    _minimal_repo(root)
    (root / "bad_frontmatter.md").write_text(
        "---\ntitle: [unclosed\n---\n\nBody.\n", encoding="utf-8"
    )
    (root / "bad_example.md").write_text(
        "# Example\n\n```python\ndef broken(:\n```\n", encoding="utf-8"
    )
    (root / "template_value.md").write_text(
        "---\nid: unique_identifier\n---\n\nBody.\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    report = validate(root)
    joined = " ".join(report["documentation_errors"])
    assert "bad_frontmatter.md" in joined
    assert "bad_example.md" in joined
    assert "template_value.md" in joined
    assert report["ok"] is False


def test_repository_passes_its_own_documentation_gate() -> None:
    """The gate is green on the repository it ships with."""
    report = validate(REPO_ROOT)
    assert report["forbidden_terms"] == [], report["forbidden_terms"]
    assert report["documentation_errors"] == [], report["documentation_errors"]
    assert report["manuscript"]["ok"] is True
    assert report["ok"] is True


def _has_document_toolchain() -> bool:
    return all(shutil.which(tool) for tool in ("pandoc", "pandoc-crossref"))


@pytest.mark.skipif(
    not _has_document_toolchain(),
    reason="requires pandoc and pandoc-crossref on PATH",
)
def test_builder_renders_html_when_the_toolchain_is_present(tmp_path: Path) -> None:
    """The Pandoc path produces a standalone document with resolved references."""
    output = build_manuscript.build(
        REPO_ROOT / "docs" / "manuscript" / "config.yaml",
        tmp_path / "rendered",
        render_pdf=False,
    )
    html = (output / "manuscript.html").read_text(encoding="utf-8")
    assert "<html" in html
    # Unresolved cross-references and unhydrated variables must not survive.
    assert "{{" not in html
    assert "[@fig:" not in html
    assert "[@sec:" not in html
    manifest = json.loads((output / "build_manifest.json").read_text(encoding="utf-8"))
    assert manifest["rendered_documents"] is True
    assert manifest["rendered_pdf"] is False
