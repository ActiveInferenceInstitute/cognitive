"""Gallery captions and hash receipts are checked against their recorded data."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from pathlib import Path

import pytest
from PIL import Image
from scripts.regenerate_gallery import FIGURES, gallery_markdown, main, regenerate_gallery


def test_gallery_parameters_match_the_generator_defaults() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "docs/visualization/gallery.md").read_text()
    parameters = json.loads(re.search(r"```json\n(.*?)\n```", text, re.S).group(1))
    signature = inspect.signature(regenerate_gallery)
    for name in ("steps", "seed", "seeds"):
        assert parameters[name] == signature.parameters[name].default
    assert parameters["temperatures"] == [0.1, 0.3, 1.0, 3.0]


def test_gallery_generator_writes_captions_figures_and_verifiable_hashes(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    for name, content in {"one": "[[two]]", "two": "[[one]]"}.items():
        path = root / "knowledge_base/cognitive" / f"{name}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"---\ntitle: {name}\ntype: concept\n---\n{content}\n")
    output = tmp_path / "generated"
    receipt = regenerate_gallery(root, output, steps=3, seed=2, seeds=1)
    assert "code/tools/src/utils/matrix_utils.py" in receipt["sources"]
    assert "code/Things/Continuous_Generic/__init__.py" in receipt["sources"]
    assert receipt["runtime"]["python"] and receipt["runtime"]["numpy"]
    for name in FIGURES:
        with Image.open(output / "figures" / name) as image:
            image.verify()
    for path, digest in receipt["outputs"].items():
        assert hashlib.sha256((output / path).read_bytes()).hexdigest() == digest
    report = json.loads((output / "report.json").read_text())
    page = (output / "gallery.md").read_text()
    assert page == gallery_markdown(receipt["parameters"], report)
    data = json.loads(re.search(r"```json\n(.*?)\n```", page, re.S).group(1))
    assert data["steps"] == 3 and data["seed"] == 2 and data["seeds"] == 1
    assert not (root / "docs").exists()
    target = output / "figures" / "episode.png"
    target.write_bytes(b"corrupted")
    assert (
        hashlib.sha256(target.read_bytes()).hexdigest() != receipt["outputs"]["figures/episode.png"]
    )
    with pytest.raises(ValueError):
        regenerate_gallery(root, output, seeds=0)


def test_gallery_cli_publishes_only_with_the_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "vault"
    for name, link in (("one", "two"), ("two", "one")):
        path = root / "knowledge_base/cognitive" / f"{name}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"---\ntitle: {name}\ntype: concept\n---\n[[{link}]]\n")
    output = tmp_path / "generated"
    assert (
        main(
            [
                "--root",
                str(root),
                "--output",
                str(output),
                "--publish",
                "--steps",
                "3",
                "--seeds",
                "1",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["figures"] == len(FIGURES)
    assert (root / "docs/visualization/gallery.md").read_bytes() == (
        output / "gallery.md"
    ).read_bytes()
    assert sorted(p.name for p in (root / "docs/visualization/figures").iterdir()) == sorted(
        FIGURES
    )
