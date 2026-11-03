---
title: Repository helper scripts
type: documentation
status: stable
updated: 2026-11-03
---

# `docs/repo_docs/repo_scripts/`

Standalone maintenance helpers, not part of the installed `scripts` package:

| Script | Purpose |
|---|---|
| `fix_links.py` | Report and repair wiki links from a file listing. |
| `list_file_directory.py` | Write a directory listing with MIME types. |
| `markdown_autofix.py` | Fix low-risk Markdown formatting across a tree. |
| `markdown_autofix_low_risk.py` | Apply the narrower normalization pass. |
| `link_fix_prompt.md` | The prompt describing the link-repair procedure. |

They are run directly with `python <script>`; `ruff` excludes this directory
(`extend-exclude` in `pyproject.toml`) and the `maintenance` extra supplies
`tqdm`. The enforced link checks are the `scripts` package:
[`../../../code/scripts/README.md`](../../../code/scripts/README.md).
