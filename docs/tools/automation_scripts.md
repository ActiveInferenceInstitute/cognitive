---
title: Automation scripts
type: tool
status: stable
created: 2026-08-02
tags:
  - automation
  - scripts
  - tools
semantic_relations:
  - type: documents
    links:
      - '[[dev_scripts|Development scripts]]'
      - '[[../repo_docs/README|Repository documentation]]'
---

# Automation scripts

The repository's automation is the executable scripts it ships — there are
no external automation services.

## Repository scripts (`code/scripts/`)

- `validate_docs.py` — documentation gate (frontmatter, Python blocks,
  exports, links, manuscript).
- `verify_links.py` — explicit wiki-link audit.
- `check_markdown_links.py` — standard Markdown link and anchor audit.
- `build_manuscript.py` — manuscript build pipeline.

## Documentation helpers (`docs/repo_docs/repo_scripts/`)

- `list_file_directory.py` — repository-wide file inventory.
- `fix_links.py` — broken/ambiguous link analysis and repair.
- `markdown_autofix.py` and `markdown_autofix_low_risk.py` — formatting
  fixes.
- `link_fix_prompt.md` — guidance for link repairs.

Run any script with `python <path> --help` for its options. The CI workflow
(`.github/workflows/quality.yml`) automates the gates on every push.
