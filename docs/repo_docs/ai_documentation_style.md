---
title: AI Documentation Style
type: documentation
status: stable
created: 2024-01-01
updated: 2026-08-02
tags:
  - documentation
  - style
  - standards
  - ai_generated
semantic_relations:
  - type: documents
    links:
      - '[[documentation_standards|Documentation standards]]'
      - '[[naming_conventions|Naming conventions]]'
      - '[[content_management|Content management]]'
      - '[[validation|Validation]]'
---

# AI Documentation Style

This page records the style rules for documentation in this repository,
with attention to content written or extended with AI assistance.

## Style rules

- **Accurate above all**: describe only behavior that exists in the
  repository. AI-generated pages must not introduce classes, commands,
  packages, citations, or statistics that do not exist. Run
  `python code/scripts/validate_docs.py --json` before merging.
- **Frontmatter first**: every Markdown page starts with YAML frontmatter
  (`title`, `type`, `status`; `created`/`updated` dates; `tags`;
  `semantic_relations` with links that resolve). Do not bury the frontmatter
  below a heading.
- **Wiki links**: use `[[wiki links]]` for internal cross-references. File-like
  targets must resolve; extensionless concept links are allowed by convention.
- **Code examples**: Python fences must compile and use symbols from the
  installed `cognitive` or `Things` packages. Illustrative pseudocode must be
  explicitly labelled and kept out of `python` fences. Prefer YAML
  frontmatter examples for metadata conventions.
- **Structure**: one H1 per page; short paragraphs; blank lines around
  headings, lists, and fenced blocks; tables for reference data.
- **No duplication**: do not repeat whole sections within a page; link to the
  authoritative page instead.

## Validating generated content

```bash
python code/scripts/validate_docs.py --json
python code/scripts/verify_links.py . --json
python code/scripts/check_markdown_links.py . --json
```

The documentation validator checks frontmatter validity, Python fenced
blocks (syntax), public package exports, manuscript references, forbidden
terms, wiki links, and standard Markdown links.

## Related documentation

- [[documentation_standards]] — repository-wide standards.
- [[naming_conventions]] — file and folder naming.
- [[ai_semantic_processing]] — machine-readable structure.
- [[validation]] — validation workflow.
- [[content_management]] — maintenance practices.
