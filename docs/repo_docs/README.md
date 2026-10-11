---
title: Repository documentation index
type: documentation
status: stable
updated: 2026-10-10
semantic_relations:
  - type: organizes
    links:
      - "[[docs/repo_docs/AGENTS|Repository documentation instructions]]"
      - "[[docs/repo_docs/getting_started|Getting started]]"
      - "[[docs/repo_docs/documentation_standards|Documentation standards]]"
      - "[[docs/repo_docs/folder_structure|Folder structure]]"
      - "[[docs/repo_docs/obsidian_linking|Obsidian linking]]"
---

# Repository documentation index

This directory records how the repository documents itself: the conventions
for pages, links, and frontmatter, and the tools that enforce them. The
enforcing tools are the scripts in `code/scripts/`; these pages describe what
those scripts really check.

## Standards

- [[docs/repo_docs/documentation_standards|Documentation standards]]: page types, formatting, diagrams.
- [[docs/repo_docs/ai_documentation_style|Documentation style]]: accuracy rules for human and AI-assisted writing.
- [[docs/repo_docs/naming_conventions|Naming conventions]]: file, folder, and code names.
- [[docs/repo_docs/folder_structure|Folder structure]], [[docs/repo_docs/ai_folder_structure|AI-oriented folder notes]], and [[docs/repo_docs/ai_file_organization|file organization]]: where things live.
- [[docs/repo_docs/content_management|Content management]]: the maintenance workflow.
- [[docs/repo_docs/package_documentation|Package documentation]]: docstrings, API pages, and examples.

## Linking and metadata

- [[docs/repo_docs/obsidian_linking|Obsidian linking]] and [[docs/repo_docs/obsidian_usage|Obsidian usage]]: wiki-link syntax and the vault.
- [[docs/repo_docs/linking_patterns|Linking patterns]], [[docs/repo_docs/linking_analysis|linking analysis]],
  [[docs/repo_docs/linking_completeness|linking completeness]], and
  [[docs/repo_docs/linking_validation|linking validation]]: how links are written, measured, and checked.
- [[docs/repo_docs/ai_semantic_processing|Semantic structure]]: frontmatter relations and generated catalogs.

## Development and validation

- [[docs/repo_docs/unit_testing|Unit testing]] and [[docs/repo_docs/testing_guidelines|testing guidelines]]: the pytest suite.
- [[docs/repo_docs/validation|Validation]] and [[docs/repo_docs/ai_validation_framework|validation framework]]: the documentation and runtime gates.
- [[docs/repo_docs/model_implementation|Model implementation]], [[docs/repo_docs/api_development|API development]],
  [[docs/repo_docs/simulation|simulation]], [[docs/repo_docs/machine_learning|machine learning]], and
  [[docs/repo_docs/research|research]]: pointers into the package and the knowledge base.
- [[docs/repo_docs/getting_started|Getting started]]: the fresh-checkout guide (keep it in sync with the root `README.md`).

## Maintenance scripts

`repo_scripts/` holds four helper scripts: `list_file_directory.py` and
`fix_links.py` (both write reports to `repo_scripts/output/`; both import
`tqdm`, declared in the `maintenance` extra), and `markdown_autofix.py` and
`markdown_autofix_low_risk.py` (whitespace and blank-line normalizers with
`--path` and `--dry-run`). They are excluded from ruff. The enforceable checks
are the validators in `code/scripts/`.
