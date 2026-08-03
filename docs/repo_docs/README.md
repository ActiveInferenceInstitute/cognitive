---
title: Repository documentation index
type: documentation
status: stable
semantic_relations:
  - type: organizes
    links:
      - '[[AGENTS|Repository documentation instructions]]'
      - '[[getting_started|Getting started]]'
      - '[[documentation_standards|Documentation standards]]'
      - '[[ai_documentation_style|Documentation style]]'
      - '[[folder_structure|Folder structure]]'
      - '[[obsidian_linking|Obsidian linking]]'
---

# Repository documentation index

This directory records how the repository documents itself: standards,
maintenance practices, linking rules, and the getting-started guide.

## Standards

- `documentation_standards.md` — quality and consistency guidelines.
- `ai_documentation_style.md` — writing style for documentation.
- `naming_conventions.md` — file and folder naming.
- `folder_structure.md` and `ai_folder_structure.md` — repository layout.
- `content_management.md` — documentation maintenance workflow.

## Linking

- `obsidian_linking.md` and `obsidian_usage.md` — wiki-link conventions.
- `linking_patterns.md`, `linking_analysis.md`, `linking_completeness.md`,
  `linking_validation.md` — link quality checks.

## Development and validation

- `unit_testing.md`, `testing_guidelines.md`, `validation.md` — test and
  validation guidance.
- `model_implementation.md`, `api_development.md`, `package_documentation.md`
  — implementation documentation.
- `getting_started.md` — the fresh-checkout guide (keep in sync with the root
  `README.md`).

## Maintenance scripts

`repo_scripts/` contains helper scripts for link fixing and directory
inventories (`list_file_directory.py`, `fix_links.py`,
`markdown_autofix.py`). The enforceable checks are the repository validators
in `code/scripts/`.
