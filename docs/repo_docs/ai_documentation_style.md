---
title: Documentation style
type: documentation
status: stable
created: 2024-01-01
updated: 2026-10-10
tags:
  - documentation
  - style
  - standards
semantic_relations:
  - type: documents
    links:
      - "[[docs/repo_docs/documentation_standards|Documentation standards]]"
      - "[[docs/repo_docs/naming_conventions|Naming conventions]]"
      - "[[docs/repo_docs/content_management|Content management]]"
      - "[[docs/repo_docs/validation|Validation]]"
---

# Documentation style

Style rules for documentation in this repository, with attention to content
written or extended with AI assistance.

## Style rules

- **Accurate above all.** Describe only behavior that exists. A page must not
  introduce classes, commands, packages, citations, or statistics that are not
  in the repository. Run `python code/scripts/validate_docs.py --json` before
  merging.
- **Frontmatter first.** Every page starts with YAML frontmatter (`title`,
  `type`, `status`, dates, optional `tags`, and `semantic_relations` whose
  links are quoted wiki-link strings). The frontmatter must be the first thing
  in the file, above the H1.
- **Wiki links.** Use `[[wiki links]]` for internal cross-references. A target
  with a path or a file suffix must resolve; a bare concept name may stay
  unresolved (see [[docs/repo_docs/obsidian_linking|Obsidian linking]]).
- **Code examples.** `validate_docs.py` only checks that `python` fences
  compile, so the author is responsible for the rest: a fence that claims to
  use `cognitive` or `Things` must run against the installed package. Label
  pseudocode as such and keep it out of `python` fences.
- **Structure.** One H1 per page; short paragraphs; blank lines around
  headings, lists, and fenced blocks; tables for reference data.
- **No duplication.** Link to the authoritative page instead of repeating it.
- **Plain wording.** Avoid filler adjectives and promotional phrasing. State
  what a thing does and what it requires.

## Checking a page

```bash
python code/scripts/validate_docs.py --json
python code/scripts/verify_links.py . --json
python code/scripts/check_markdown_links.py . --json
python code/scripts/normalize_frontmatter.py --check .
```

`validate_docs.py` aggregates the rest: frontmatter validity, Python fence
syntax, public package exports, manuscript references, forbidden terms, wiki
links, standard Markdown links, and knowledge-base catalog freshness. See
[[docs/repo_docs/ai_validation_framework|the validation framework]] for what
each check covers.

## Related documentation

- [[docs/repo_docs/documentation_standards|Documentation standards]]
- [[docs/repo_docs/naming_conventions|Naming conventions]]
- [[docs/repo_docs/ai_semantic_processing|Semantic structure]]
- [[docs/repo_docs/validation|Validation]]
- [[docs/repo_docs/content_management|Content management]]
