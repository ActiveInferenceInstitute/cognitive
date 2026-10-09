import subprocess

import yaml
from scripts import normalize_frontmatter as nf
from scripts.validate_docs import _validate_frontmatter

DAMAGED = """---
title: Prediction Error
tags:
- cognition
semantic_relations:
- type: implements
  links:
  - - predictive_processing
- type: related
  parent: knowledge_base/cognitive/prediction_error
  links:
  - - - free_energy_minimization
  - - - active_inference
- type: documents
  links: [[[a]], [[knowledge_base/mathematics/b|B]]]
- type: empty
  links:
---

# Body

Text with [[a_link]] and a code fence.
"""


def _front(text: str) -> dict:
    match = nf._FRONTMATTER.match(text)
    assert match is not None
    return yaml.safe_load(match.group(1))


def test_flatten_handles_arbitrary_nesting():
    assert nf._flatten([["a"], [["b", "c"]], "d", None]) == ["a", "b", "c", "d"]
    assert nf._flatten(None) == []


def test_wiki_link_does_not_double_wrap():
    assert nf.wiki_link("x") == "[[x]]"
    assert nf.wiki_link("[[x]]") == "[[x]]"


def test_normalize_text_quotes_every_link_and_preserves_everything_else():
    result = nf.normalize_text(DAMAGED)

    relations = _front(result)["semantic_relations"]
    assert relations[0]["links"] == ["[[predictive_processing]]"]
    assert relations[1]["links"] == ["[[free_energy_minimization]]", "[[active_inference]]"]
    assert relations[1]["parent"] == "knowledge_base/cognitive/prediction_error"
    assert relations[2]["links"] == ["[[a]]", "[[knowledge_base/mathematics/b|B]]"]
    assert relations[3]["links"] == []
    assert _front(result)["tags"] == ["cognition"]
    assert result.endswith(DAMAGED[DAMAGED.index("# Body") :])


def test_normalize_text_is_idempotent():
    once = nf.normalize_text(DAMAGED)
    assert nf.normalize_text(once) == once


def test_normalize_text_leaves_other_documents_alone():
    plain = "# No frontmatter\n"
    no_key = "---\ntitle: T\n---\nbody\n"
    broken = "---\ntitle: [unclosed\n---\nbody\n"
    assert nf.normalize_text(plain) == plain
    assert nf.normalize_text(no_key) == no_key
    assert nf.normalize_text(broken) == broken


def test_empty_relations_render_as_an_empty_list():
    text = "---\ntitle: T\nsemantic_relations:\n- bogus\ntags: [x]\n---\nbody\n"
    result = nf.normalize_text(text)
    assert _front(result)["semantic_relations"] == []
    assert _front(result)["tags"] == ["x"]


def test_unquoted_links_reports_only_unquoted_entries():
    assert nf.unquoted_links({}) == []
    assert nf.unquoted_links({"semantic_relations": [{"type": "t", "links": ["[[x]]"]}]}) == []
    assert nf.unquoted_links({"semantic_relations": [{"type": "t", "links": [["x"]]}]})
    assert nf.unquoted_links({"semantic_relations": [{"type": "t", "links": "[[x]]"}]})
    assert nf.unquoted_links({"semantic_relations": "nope"})
    assert nf.unquoted_links({"semantic_relations": ["bare"]})


def test_validator_rejects_damaged_frontmatter_and_accepts_normalized(tmp_path):
    path = tmp_path / "page.md"
    assert _validate_frontmatter(path, DAMAGED)
    assert _validate_frontmatter(path, nf.normalize_text(DAMAGED)) == []


def _git_repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    page = tmp_path / "page.md"
    page.write_text(DAMAGED, encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "page.md"], check=True)
    return page


def test_main_check_fails_then_write_repairs_then_check_passes(tmp_path, capsys):
    page = _git_repo(tmp_path)

    assert nf.main([str(tmp_path), "--check"]) == 1
    assert "page.md" in capsys.readouterr().out
    assert page.read_text(encoding="utf-8") == DAMAGED

    assert nf.main([str(tmp_path), "--write"]) == 0
    assert page.read_text(encoding="utf-8") == nf.normalize_text(DAMAGED)

    assert nf.main([str(tmp_path), "--check"]) == 0
