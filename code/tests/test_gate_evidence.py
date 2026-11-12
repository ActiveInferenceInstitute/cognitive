"""Reject stale measurement claims and unresolved publication output."""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import chdir
from pathlib import Path

import pytest
from scripts import gate, gate_evidence


def _rendered(output: Path, html: str = "<html>resolved</html>", pdf: bool = False) -> None:
    output.mkdir(exist_ok=True)
    (output / "build_manifest.json").write_text(
        json.dumps(
            {
                "rendered_documents": True,
                "rendered_pdf": pdf,
            }
        )
    )
    (output / "manuscript.html").write_text(html)
    if pdf:
        (output / "manuscript.pdf").write_bytes(b"%PDF-1.7\n")


@pytest.mark.parametrize(
    "marker", ["{{SEED}}", "[@fig:missing]", "[@sec:missing]", "[@eq:missing]", "[@tbl:missing]"]
)
def test_unresolved_render_is_not_a_pass(tmp_path: Path, marker: str) -> None:
    _rendered(tmp_path, f"<html>{marker}</html>")
    with pytest.raises(ValueError, match="Unresolved manuscript marker"):
        gate.verify_render(tmp_path, False)


def test_skipped_render_and_invalid_pdf_are_not_passes(tmp_path: Path) -> None:
    _rendered(tmp_path, pdf=True)
    gate.verify_render(tmp_path, True)
    (tmp_path / "manuscript.pdf").write_bytes(b"not a PDF")
    with pytest.raises(ValueError, match="did not produce a PDF"):
        gate.verify_render(tmp_path, True)
    with pytest.raises(ValueError, match="flags"):
        gate.verify_render(tmp_path, False)
    (tmp_path / "build_manifest.json").write_text(
        json.dumps(
            {
                "rendered_documents": False,
                "rendered_pdf": False,
            }
        )
    )
    with pytest.raises(ValueError, match="flags"):
        gate.verify_render(tmp_path, False)


def test_evidence_check_rejects_changed_claims_and_source(tmp_path: Path) -> None:
    with chdir(tmp_path):
        for name in gate_evidence.SOURCES:
            target = Path(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("source bytes\n")
        gate_evidence.DATA.parent.mkdir(parents=True, exist_ok=True)
        before = Path("before.json")
        after = Path("after.json")
        record = {
            "test-full": {
                "seconds": 4.25,
                "exit_code": 0,
                "commands": [["python", "-m", "pytest"]],
            }
        }
        before.write_text(json.dumps(record))
        after.write_text(json.dumps(record))
        Path("before-test-full.log").write_text("12 passed\nTotal coverage: 91.25%\n")
        refresh = [
            "--refresh",
            "--before",
            str(before),
            "--after",
            str(after),
            "--baseline-revision",
            "a" * 40,
        ]
        assert gate_evidence.main(refresh) == 0
        assert gate_evidence.main(["--check"]) == 0
        data = json.loads(gate_evidence.DATA.read_text())
        assert data["before"]["test-full"]["coverage_percent"] == 91.25
        data["after"]["test-full"]["seconds"] = 0.01
        gate_evidence.DATA.write_text(json.dumps(data))
        assert gate_evidence.main(["--check"]) == 1
        assert gate_evidence.main(refresh) == 0
        Path("run.sh").write_text("changed source bytes\n")
        assert gate_evidence.main(["--check"]) == 1


def test_missing_required_parser_is_a_recorded_failure(tmp_path: Path) -> None:
    record = tmp_path / "result.json"
    code = gate.main(
        [
            "gnn",
            "--runtime",
            str(tmp_path / "missing-python"),
            "--record",
            str(record),
        ]
    )
    assert code == 2
    assert json.loads(record.read_text())["gnn"]["exit_code"] == 2


def test_wrong_tag_is_refused_before_dependency_installation(tmp_path: Path) -> None:
    remote = tmp_path / "upstream.git"
    subprocess.run(
        ["git", "init", "--bare", str(remote)],
        check=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
    )
    with chdir(tmp_path):
        Path("specs").mkdir()
        Path("specs/gnn-pin.json").write_text(
            json.dumps(
                {
                    "repository": str(remote),
                    "tag": "v99",
                    "commit": "a" * 40,
                }
            )
        )
        runtime = tmp_path / "runtime/bin/python"
        with pytest.raises(ValueError, match="pin says"):
            gate.provision(runtime)
        assert not runtime.parent.parent.exists()


@pytest.mark.parametrize("prefix", [sys.prefix, sys.base_prefix])
def test_provision_refuses_the_running_interpreter_prefix(prefix: str) -> None:
    with pytest.raises(ValueError, match="running Python environment"):
        gate.provision(Path(prefix) / "bin/python")


def test_provision_preserves_existing_non_venv_directory(tmp_path: Path) -> None:
    original = tmp_path / "keep.txt"
    original.write_bytes(b"unrelated original bytes")
    with pytest.raises(ValueError, match="existing non-venv"):
        gate.provision(tmp_path / "bin/python")
    assert original.read_bytes() == b"unrelated original bytes"
