"""Consumer tests for the generated flat documentation index."""

from pathlib import Path

import pytest
from scripts import docs_index


def _page(root: Path, relative: str, text: str = "# A documentation page\n") -> Path:
    path = root / "docs" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_check_write_cycle_detects_added_and_removed_pages(tmp_path: Path) -> None:
    _page(tmp_path, "README.md")
    root = str(tmp_path)
    index = tmp_path / "docs" / "documentation-index.md"
    assert docs_index.main(["--check", root]) == 1
    assert not index.exists()
    assert docs_index.main(["--write", root]) == 0
    assert docs_index.main(["--check", root]) == 0
    assert "- [README.md](README.md)" in index.read_text(encoding="utf-8")

    page = _page(tmp_path, "guides/new-page.md")
    before = index.read_bytes()
    assert docs_index.main(["--check", root]) == 1
    assert index.read_bytes() == before
    assert docs_index.main(["--write", root]) == 0
    assert docs_index.main(["--check", root]) == 0
    assert "- [guides/new-page.md](guides/new-page.md)" in index.read_text(encoding="utf-8")

    page.unlink()
    assert docs_index.main(["--check", root]) == 1
    assert docs_index.main(["--write", root]) == 0
    assert docs_index.main(["--check", root]) == 0
    assert "guides/new-page.md" not in index.read_text(encoding="utf-8")


def test_index_is_recursive_flat_sorted_and_excludes_instructions(tmp_path: Path) -> None:
    included = [
        "tools/z.md",
        "README.md",
        "manuscript/sections/a.md",
        "development/gate-measurements.md",
        "guides/a [b] #c.md",
    ]
    for name in included:
        _page(tmp_path, name)
    for name in ["AGENTS.md", "tools/AGENTS.md", ".cache/page.md", "tools/.cache.md"]:
        _page(tmp_path, name)
    (tmp_path / "docs" / "directory.md").mkdir()
    _page(tmp_path, "tools/data.txt")
    assert docs_index.main(["--write", str(tmp_path)]) == 0
    index = tmp_path / "docs" / "documentation-index.md"
    text = index.read_text(encoding="utf-8")
    entries = [line for line in text.splitlines() if line.startswith("- [")]
    assert entries == [
        "- [README.md](README.md)",
        "- [development/gate-measurements.md](development/gate-measurements.md)",
        "- [guides/a \\[b\\] #c.md](guides/a%20%5Bb%5D%20%23c.md)",
        "- [manuscript/sections/a.md](manuscript/sections/a.md)",
        "- [tools/z.md](tools/z.md)",
    ]
    assert "AGENTS.md" not in text
    assert ".cache" not in text
    assert "directory.md" not in text
    assert not any("documentation-index.md" in line for line in entries)
    before = index.read_bytes()
    assert docs_index.main(["--write", str(tmp_path)]) == 0
    assert index.read_bytes() == before


@pytest.mark.parametrize("mode", [[], ["--check"]])
def test_default_and_explicit_check_do_not_rewrite_files(
    tmp_path: Path, mode: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    source = _page(tmp_path, "README.md")
    index = _page(tmp_path, "documentation-index.md", "outdated index\n")
    outside = tmp_path / "outside.md"
    outside.write_text("outside documentation tree\n", encoding="utf-8")
    before = {path: path.read_bytes() for path in [source, index, outside]}
    assert docs_index.main([*mode, str(tmp_path)]) == 1
    output = capsys.readouterr()
    assert "docs/documentation-index.md" in output.out
    assert "stale" in output.err
    assert {path: path.read_bytes() for path in before} == before
    assert docs_index.main(["--write", str(tmp_path)]) == 0
    assert docs_index.main([*mode, str(tmp_path)]) == 0
    assert source.read_bytes() == before[source]
    assert outside.read_bytes() == before[outside]


def test_missing_docs_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        docs_index.main(["--write", str(tmp_path)])
    assert error.value.code == 2
    assert not (tmp_path / "docs").exists()


def test_symlink_pages_are_excluded_and_index_cannot_redirect_writes(tmp_path: Path) -> None:
    _page(tmp_path, "README.md")
    external = tmp_path / "outside.md"
    external.write_text("outside content\n", encoding="utf-8")
    docs = tmp_path / "docs"
    (docs / "external.md").symlink_to(external)
    assert docs_index.main(["--write", str(tmp_path)]) == 0
    index = docs / "documentation-index.md"
    assert "external.md" not in index.read_text(encoding="utf-8")
    index.unlink()
    index.symlink_to(external)
    with pytest.raises(SystemExit) as error:
        docs_index.main(["--write", str(tmp_path)])
    assert error.value.code == 2
    assert external.read_text(encoding="utf-8") == "outside content\n"
