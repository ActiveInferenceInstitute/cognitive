"""Every example script runs from a clean checkout and writes what its docstring promises."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "examples"

EXPECTED = {
    "01_run_episode.py": ["trajectory.json", "episode.png", "efe_step0.png"],
    "02_compare_aggregation.py": ["summary.csv", "aggregation.png"],
    "03_sensor_noise_sweep.py": ["summary.csv", "sensor_noise.png"],
    "04_model_mismatch.py": ["correct_model.png", "wrong_model.png"],
    "05_knowledge_base_report.py": ["report.json", "report.md", "domain_sizes.png"],
    "06_export_rdf.py": ["cognitive-kb.nq", "cognitive-kb.ttl"],
}


# Examples that read the whole knowledge base are pointed at a three-page vault,
# and the sweeps at two seeds, so the suite stays fast; the defaults are what a
# reader runs.
FAST = {
    "02_compare_aggregation.py": ["--seeds", "2"],
    "03_sensor_noise_sweep.py": ["--seeds", "2"],
}
VAULT_PAGES = {
    "a/one.md": "---\ntitle: One\ntype: concept\n---\n[[two]]\n",
    "a/two.md": "---\ntitle: Two\ntype: concept\n---\n[[one]]\n",
    "b/three.md": "---\ntitle: Three\ntype: concept\n---\n[[one]]\n",
}


def small_vault(root: Path) -> Path:
    for relative, text in VAULT_PAGES.items():
        target = root / "knowledge_base" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return root


def scripts() -> list[Path]:
    return sorted(EXAMPLES.glob("[0-9][0-9]_*.py"))


def test_every_example_script_has_an_expectation_and_vice_versa() -> None:
    assert {path.name for path in scripts()} == set(EXPECTED)


@pytest.mark.parametrize("script", scripts(), ids=lambda path: path.name)
def test_example_is_thin_documented_and_parses(script: Path) -> None:
    tree = ast.parse(script.read_text(encoding="utf-8"))
    assert ast.get_docstring(tree), f"{script.name} needs a module docstring"
    assert not [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))], (
        f"{script.name} should orchestrate library calls, not define functions or classes"
    )
    assert len(script.read_text(encoding="utf-8").splitlines()) <= 50


@pytest.mark.parametrize("script", scripts(), ids=lambda path: path.name)
def test_example_runs_and_writes_its_outputs(script: Path, tmp_path: Path) -> None:
    environment = {**os.environ, "MPLBACKEND": "Agg"}
    extra = list(FAST.get(script.name, []))
    if script.name in ("05_knowledge_base_report.py", "06_export_rdf.py"):
        extra += ["--root", str(small_vault(tmp_path / "vault"))]
    result = subprocess.run(
        [sys.executable, str(script), "--output", str(tmp_path / "out"), *extra],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=environment,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    for name in EXPECTED[script.name]:
        written = tmp_path / "out" / name
        assert written.is_file() and written.stat().st_size > 0, (
            f"{script.name} did not write {name}"
        )
