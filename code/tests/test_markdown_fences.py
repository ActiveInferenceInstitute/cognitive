"""Fence tracking and silent-failure paths in the documentation scripts.

The link scripts collected heading anchors after removing code with the pattern
```` ```[\\s\\S]*?``` ````. That pattern ends a block at the first three backticks,
so a four-backtick fence that shows a three-backtick example closed early and
the text after the example was read as prose, and it ignored a fence that was
never closed, so a ``# comment`` after it was read as a heading.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from scripts import check_markdown_links, kb_index, normalize_frontmatter, verify_links
from scripts.build_manuscript import _latex_preamble
from scripts.markdown_fences import iter_fenced_blocks, strip_fenced_blocks
from scripts.validate_docs import unterminated_fences, validate

NESTED = """# Real

````markdown
```bash
# inner comment
```
# outer comment
````

after the fence
"""


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# --- the tracker -------------------------------------------------------------


def test_a_longer_fence_contains_a_shorter_one() -> None:
    blocks = list(iter_fenced_blocks(NESTED))
    assert len(blocks) == 1
    assert blocks[0].info == "markdown"
    assert blocks[0].terminated
    assert "# outer comment" in blocks[0].body
    assert strip_fenced_blocks(NESTED).split() == ["#", "Real", "after", "the", "fence"]


def test_a_closing_fence_needs_the_same_character_and_enough_length() -> None:
    text = "```python\n~~~\n``\n```not a close\nx = 1\n````\nafter\n"
    (block,) = iter_fenced_blocks(text)
    assert block.terminated
    assert block.end_line == 6
    assert block.body.splitlines()[-1] == "x = 1"
    tilde = "~~~\n```\n# still code\n~~~\n# heading\n"
    (only,) = iter_fenced_blocks(tilde)
    assert only.end_line == 4


def test_an_unterminated_fence_runs_to_the_end_of_the_file() -> None:
    text = "# Title\n\n```bash\n# comment\nmore\n"
    (block,) = iter_fenced_blocks(text)
    assert not block.terminated
    assert (block.start_line, block.end_line) == (3, 5)
    assert block.body == "# comment\nmore"
    assert strip_fenced_blocks(text) == "# Title\n\n"


def test_inline_triple_backticks_do_not_open_a_block() -> None:
    assert list(iter_fenced_blocks("Use ```code``` inline.\n# Heading\n")) == []
    assert len(list(iter_fenced_blocks("``` text\nbody\n"))) == 1
    assert list(iter_fenced_blocks("``` a`b\ntext\n")) == []


def test_indented_fences_are_found_and_dedented() -> None:
    text = "- item\n\n  ```python\n  x = 1\n    y = 2\n  ```\n"
    (block,) = iter_fenced_blocks(text)
    assert block.indent == 2
    assert block.body == "x = 1\n  y = 2"


def test_crlf_line_endings_are_handled() -> None:
    (block,) = iter_fenced_blocks("```py\r\nx = 1\r\n```\r\ntext\r\n")
    assert block.terminated
    assert block.body == "x = 1"


def test_replacement_marks_each_removed_block() -> None:
    text = "a\n```\nb\n```\nc\n"
    assert strip_fenced_blocks(text, replacement="\n") == "a\n\nc\n"


def test_a_fence_opened_on_a_list_item_line_is_found() -> None:
    """``- ```` opened a block the tracker could not see, then its closer opened a new one."""
    text = "- ```python\n  [[inside]]\n  ```\n[[after]]\n"
    blocks = list(iter_fenced_blocks(text))
    assert [(b.start_line, b.end_line, b.terminated, b.info) for b in blocks] == [
        (1, 3, True, "python")
    ]
    assert blocks[0].body == "[[inside]]"
    assert strip_fenced_blocks(text) == "[[after]]\n"
    numbered = "1. ```\n   code\n   ```\nprose\n"
    assert strip_fenced_blocks(numbered) == "prose\n"


def test_a_fence_inside_a_blockquote_or_callout_is_found() -> None:
    text = "> [!note]\n> ```python\n> x = 1\n> ```\nafter [[y]]\n"
    blocks = list(iter_fenced_blocks(text))
    assert [(b.start_line, b.end_line, b.terminated) for b in blocks] == [(2, 4, True)]
    assert blocks[0].body == "x = 1"  # quote markers are removed, so it can be compiled
    assert strip_fenced_blocks(text) == "> [!note]\nafter [[y]]\n"


def test_only_line_feeds_and_carriage_returns_end_a_line() -> None:
    """``str.splitlines`` also splits at U+2028 and U+0085, which faked a fence opener."""
    for separator in ("\u2028", "\u0085", "\u2029", "\x0b", "\x0c"):
        text = f"foo{separator}```\n[[x]]\n"
        assert list(iter_fenced_blocks(text)) == []
        assert strip_fenced_blocks(text) == text


def test_an_anchor_into_a_file_that_is_not_text_is_not_a_crash(tmp_path: Path) -> None:
    """The anchor helper caught ``OSError`` only, so ``[x](data.bin#a)`` raised."""
    (tmp_path / "data.bin").write_bytes(b"\xff\xfe\x00\x80binary")
    _write(tmp_path / "a.md", "[x](data.bin#foo)\n\n[[data.bin#foo]]\n")
    standard = check_markdown_links.verify_markdown_link_report(tmp_path)
    wiki = verify_links.verify_link_report(tmp_path)
    assert [w["target"] for w in standard.anchor_warnings] == ["data.bin#foo"]
    assert [w["fragment"] for w in wiki.anchor_warnings] == ["foo"]


# --- the two link checkers ---------------------------------------------------


def test_comments_in_nested_and_unterminated_fences_are_not_anchors(tmp_path: Path) -> None:
    _write(tmp_path / "nested.md", NESTED)
    _write(tmp_path / "open.md", "# Real\n\n```bash\n# unclosed comment\n")
    _write(
        tmp_path / "index.md",
        "[[nested#outer-comment]] [[nested#inner-comment]] [[open#unclosed-comment]] "
        "[[nested#real]] [[open#real]]\n\n"
        "[a](nested.md#outer-comment) [b](nested.md#inner-comment) "
        "[c](open.md#unclosed-comment) [d](nested.md#real) [e](open.md#real)\n",
    )
    wiki = verify_links.verify_link_report(tmp_path)
    assert sorted(w["fragment"] for w in wiki.anchor_warnings) == [
        "inner-comment",
        "outer-comment",
        "unclosed-comment",
    ]
    standard = check_markdown_links.verify_markdown_link_report(tmp_path)
    assert sorted(w["target"] for w in standard.anchor_warnings) == [
        "nested.md#inner-comment",
        "nested.md#outer-comment",
        "open.md#unclosed-comment",
    ]


def test_links_inside_a_nested_fence_are_not_checked(tmp_path: Path) -> None:
    """With two inner blocks the old pattern paired the wrong fences.

    It then scanned the first inner block as prose and reported its example link
    as broken.
    """
    _write(
        tmp_path / "page.md",
        "````markdown\n```text\n[example](missing.md) [[fake.md]]\n```\n"
        "```text\nmore\n```\n````\n\n[real](missing_real.md)\n",
    )
    standard = check_markdown_links.verify_markdown_link_report(tmp_path)
    assert [b["target"] for b in standard.broken_links] == ["missing_real.md"]
    assert verify_links.verify_link_report(tmp_path).broken_links == []


# --- kb_index and validate_docs use the same tracker -------------------------


def test_catalog_description_skips_a_nested_fence() -> None:
    body = (
        "# Page\n\n````markdown\n```text\nThis sentence sits inside the inner fence, at length.\n"
        "```\n````\n\nThe prose that really opens this page is long enough.\n"
    )
    assert kb_index.describe({}, body) == "The prose that really opens this page is long enough."


def test_python_examples_inside_a_longer_fence_are_not_compiled(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "docs" / "policy").mkdir(parents=True)
    (root / "docs" / "policy" / "documentation_terms.yaml").write_text(
        "terms: [zzzunlikelyterm]\n", encoding="utf-8"
    )
    _write(
        root / "page.md",
        "# Example\n\n````markdown\n```python\ndef broken(:\n```\n````\n\n"
        "```python\ndef also_broken(:\n```\n",
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    errors = [e for e in validate(root)["documentation_errors"] if "page.md" in e]
    assert len(errors) == 1  # only the top-level fence; the nested one is example text


def test_unterminated_fences_are_reported(tmp_path: Path) -> None:
    page = tmp_path / "page.md"
    messages = unterminated_fences(page, "ok\n```python\nx = 1\n```\n\n```text\nnever closed\n")
    assert len(messages) == 1
    assert "line 6" in messages[0]
    assert unterminated_fences(page, "```\nx\n```\n") == []


def test_validator_fails_on_an_unterminated_fence_without_compiling_it(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "docs" / "policy").mkdir(parents=True)
    (root / "docs" / "policy" / "documentation_terms.yaml").write_text(
        "terms: [zzzunlikelyterm]\n", encoding="utf-8"
    )
    _write(root / "open.md", "# Open\n\n```python\ndef broken(:\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    report = validate(root)
    assert any("open.md" in item for item in report["unterminated_fences"])
    messages = [e for e in report["documentation_errors"] if "open.md" in e]
    assert len(messages) == 1 and "never closed" in messages[0]  # reported, and not compiled
    assert report["ok"] is False


# --- silent failure paths ----------------------------------------------------


def test_validator_does_not_skip_a_file_it_cannot_decode(tmp_path: Path) -> None:
    """``except UnicodeDecodeError: continue`` exempted the file from every check."""
    root = tmp_path / "repo"
    (root / "docs" / "policy").mkdir(parents=True)
    (root / "docs" / "policy" / "documentation_terms.yaml").write_text(
        "terms: [zzzunlikelyterm]\n", encoding="utf-8"
    )
    (root / "latin1.md").write_bytes("# Caf\xe9\n\nzzzunlikelyterm\n".encode("latin-1"))
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    report = validate(root)
    assert report["ok"] is False
    assert any("latin1.md" in e and "UTF-8" in e for e in report["documentation_errors"])


def _broken_link_file(root: Path) -> None:
    (root / "good.md").write_text("# Good\n", encoding="utf-8")
    (root / "gone.md").symlink_to(root / "does_not_exist.md")  # os.walk lists it; reading fails


def test_wiki_link_checker_reports_files_it_could_not_read(tmp_path: Path, capsys) -> None:
    _broken_link_file(tmp_path)
    report = verify_links.verify_link_report(tmp_path)
    assert [item["source"] for item in report.unreadable_files] == ["gone.md"]
    assert verify_links.main([str(tmp_path)]) == 1
    assert "gone.md" in capsys.readouterr().out
    assert verify_links.main([str(tmp_path), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["unreadable_files"][0]["source"] == "gone.md"


def test_markdown_link_checker_reports_files_it_could_not_read(tmp_path: Path, capsys) -> None:
    _broken_link_file(tmp_path)
    report = check_markdown_links.verify_markdown_link_report(tmp_path)
    assert [item["source"] for item in report.unreadable_files] == ["gone.md"]
    assert check_markdown_links.main([str(tmp_path)]) == 1
    assert "gone.md" in capsys.readouterr().out
    assert check_markdown_links.main([str(tmp_path), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["unreadable_files"][0]["source"] == "gone.md"


def test_undecodable_markdown_is_reported_not_a_crash(tmp_path: Path) -> None:
    (tmp_path / "bad.md").write_bytes(b"\xff\xfe\x00 not utf-8 [x](y.md)")
    assert [i["source"] for i in verify_links.verify_link_report(tmp_path).unreadable_files] == [
        "bad.md"
    ]
    report = check_markdown_links.verify_markdown_link_report(tmp_path)
    assert [i["source"] for i in report.unreadable_files] == ["bad.md"]


def test_clean_trees_still_pass_the_link_checkers(tmp_path: Path) -> None:
    _write(tmp_path / "a.md", "# A\n\n[b](b.md) [[b]]\n")
    _write(tmp_path / "b.md", "# B\n")
    assert verify_links.main([str(tmp_path)]) == 0
    assert check_markdown_links.main([str(tmp_path)]) == 0


def test_validator_reports_unreadable_files_from_the_link_scans(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "docs" / "policy").mkdir(parents=True)
    (root / "docs" / "policy" / "documentation_terms.yaml").write_text(
        "terms: [zzzunlikelyterm]\n", encoding="utf-8"
    )
    _broken_link_file(root)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    errors = validate(root)["documentation_errors"]
    assert any("gone.md" in e and "wiki links not checked" in e for e in errors)
    assert any("gone.md" in e and "Markdown links not checked" in e for e in errors)


@pytest.mark.parametrize("links", [0, "", False])
def test_falsy_non_list_relation_links_are_reported(links: object) -> None:
    """``entry.get('links') or []`` turned these into an empty, valid list."""
    front = {"semantic_relations": [{"type": "related", "links": links}]}
    problems = normalize_frontmatter.unquoted_links(front)
    assert problems == ["links of 'related' is not a list"]


def test_missing_or_null_relation_links_still_mean_no_links() -> None:
    for entry in ({"type": "related"}, {"type": "related", "links": None}):
        assert normalize_frontmatter.unquoted_links({"semantic_relations": [entry]}) == []
    assert (
        normalize_frontmatter.unquoted_links(
            {"semantic_relations": [{"type": "related", "links": []}]}
        )
        == []
    )


def test_normalizer_reports_a_file_it_cannot_decode(tmp_path: Path, capsys) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "bad.md").write_bytes(b"---\ntitle: caf\xe9\n---\n")
    (tmp_path / "ok.md").write_text("# fine\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    assert normalize_frontmatter.main([str(tmp_path), "--check"]) == 1
    assert "bad.md" in capsys.readouterr().err
    assert normalize_frontmatter.main([str(tmp_path), "--write"]) == 1


def test_a_preamble_without_a_latex_fence_is_an_error() -> None:
    """An empty ``preamble.tex`` was written and the PDF lost its macros silently."""
    with pytest.raises(ValueError, match="no closed latex fence"):
        _latex_preamble("# Preamble\n\nNo fence here.\n")
    with pytest.raises(ValueError, match="no closed latex fence"):
        _latex_preamble("```latex\n\\usepackage{amsmath}\n")  # never closed
    source = "```latex\n\\usepackage{amsmath}\n```\n\ntext\n\n```latex\n\\newcommand{\\x}{1}\n```\n"
    assert _latex_preamble(source) == "\\usepackage{amsmath}\n\n\\newcommand{\\x}{1}\n"


def test_the_shipped_preamble_has_a_latex_fence() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "docs" / "manuscript" / "preamble.md").read_text(encoding="utf-8")
    assert _latex_preamble(source).strip()


def test_a_crlf_preamble_uses_one_line_ending() -> None:
    source = "x\r\n```latex\r\n\\a\r\n\\b\r\n```\r\n\r\n```latex\r\n\\c\r\n```\r\n"
    assert _latex_preamble(source) == "\\a\n\\b\n\n\\c\n"
