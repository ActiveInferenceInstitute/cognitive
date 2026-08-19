"""The mutation harness must apply cleanly, restore the tree, and be able to fail.

A mutation gate that cannot report a survivor is worse than no gate: it prints a
perfect score whatever the suite does. These tests prove the harness detects a
survivor, restores every file it edits, and refuses to run against a source
whose anchor has moved.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from scripts.mutation_check import MUTATIONS, Mutation, check

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_every_declared_mutation_anchor_still_exists_exactly_once() -> None:
    """A refactor that moves an anchor must fail loudly, not silently pass.

    ``check`` raises on a missing anchor, but only for the mutations it runs.
    This asserts the whole declared set stays applicable, so a stale entry is
    caught even when someone runs a filtered subset.
    """
    assert MUTATIONS, "the harness must declare at least one mutation"
    for mutation in MUTATIONS:
        source = mutation.target(REPO_ROOT).read_text(encoding="utf-8")
        occurrences = source.count(mutation.original)
        assert occurrences == 1, (
            f"{mutation.name}: anchor appears {occurrences} times in {mutation.path}"
        )
        assert mutation.original != mutation.mutated
        assert mutation.rationale.strip(), f"{mutation.name} has no rationale"
        for test_path in mutation.tests:
            assert (REPO_ROOT / test_path).is_file(), f"{mutation.name} names a missing test file"


def test_mutation_names_are_unique() -> None:
    names = [mutation.name for mutation in MUTATIONS]
    assert len(names) == len(set(names))


def _digest(paths: set[Path]) -> dict[Path, str]:
    return {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)
    }


def test_harness_detects_a_survivor_and_restores_the_source(tmp_path: Path) -> None:
    """A mutation no test can see is reported, and the file is put back.

    The mutation used here replaces a docstring, which changes the source
    without changing behaviour, so nothing can kill it. If the harness reported
    it as killed, every other result would be meaningless.
    """
    import scripts.mutation_check as harness

    target = REPO_ROOT / "code/tools/src/utils/matrix_utils.py"
    original_bytes = target.read_bytes()
    survivor = Mutation(
        name="probe_docstring_only",
        path="code/tools/src/utils/matrix_utils.py",
        original='"""Validated probability and expected-free-energy utilities."""',
        mutated='"""Utilities."""',
        rationale="A comment-only edit cannot change behaviour, so nothing can kill it.",
        tests=("code/tests/test_version_consistency.py",),
    )
    declared = harness.MUTATIONS
    harness.MUTATIONS = (survivor,)
    try:
        report = check(REPO_ROOT)
    finally:
        harness.MUTATIONS = declared

    assert report["ok"] is False
    assert report["killed"] == 0
    assert [entry["name"] for entry in report["survivors"]] == ["probe_docstring_only"]
    assert target.read_bytes() == original_bytes, "the harness must restore what it edits"


def test_running_the_declared_set_leaves_the_tree_untouched() -> None:
    """Every file the harness touches is byte-identical afterwards."""
    touched = {mutation.target(REPO_ROOT) for mutation in MUTATIONS}
    before = _digest(touched)
    check(REPO_ROOT, name_filter="policy_limit_boundary")
    assert _digest(touched) == before


def test_a_moved_anchor_is_refused_rather_than_skipped(tmp_path: Path) -> None:
    """An inapplicable mutation raises instead of counting as a pass."""
    import scripts.mutation_check as harness

    declared = harness.MUTATIONS
    harness.MUTATIONS = (
        Mutation(
            name="probe_absent_anchor",
            path="code/tools/src/utils/matrix_utils.py",
            original="this text does not appear anywhere in the module",
            mutated="nor does this",
            rationale="probe",
            tests=("code/tests/test_version_consistency.py",),
        ),
    )
    try:
        with pytest.raises(ValueError, match="does not apply"):
            check(REPO_ROOT)
    finally:
        harness.MUTATIONS = declared


def test_an_unmatched_filter_is_an_error() -> None:
    with pytest.raises(ValueError, match="No mutation matches filter"):
        check(REPO_ROOT, name_filter="no-such-mutation")


def test_declared_mutations_all_die(request: pytest.FixtureRequest) -> None:
    """The full gate, run only when explicitly requested.

    Each mutation spawns a pytest subprocess, so the whole set costs far more
    than the rest of the suite. Run it with ``--mutation`` locally, or through
    ``python -m scripts.mutation_check`` in CI.
    """
    if not request.config.getoption("--mutation"):
        pytest.skip("pass --mutation to run the full mutation gate")
    report = check(REPO_ROOT)
    assert report["ok"], report["survivors"]


def test_mutation_cli_reports_both_outcomes(capsys: pytest.CaptureFixture[str]) -> None:
    """Text and JSON output, and the exit code, follow the survivor list."""
    import scripts.mutation_check as harness

    assert harness.main([str(REPO_ROOT), "--filter", "policy_limit_boundary"]) == 0
    text = capsys.readouterr().out
    assert "killed" in text
    assert "1/1 mutations killed." in text

    assert harness.main([str(REPO_ROOT), "--filter", "policy_limit_boundary", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["mutations"] == 1
    assert report["survivors"] == []

    declared = harness.MUTATIONS
    harness.MUTATIONS = (
        Mutation(
            name="probe_cli_survivor",
            path="code/tools/src/utils/matrix_utils.py",
            original='"""Validated probability and expected-free-energy utilities."""',
            mutated='"""Utilities."""',
            rationale="A comment-only edit cannot change behaviour.",
            tests=("code/tests/test_version_consistency.py",),
        ),
    )
    try:
        assert harness.main([str(REPO_ROOT)]) == 1
    finally:
        harness.MUTATIONS = declared
    survivor_output = capsys.readouterr().out
    assert "SURVIVED" in survivor_output
    assert "Survivor: probe_cli_survivor" in survivor_output
