---
title: Folders beyond the documentation tree
type: guide
status: stable
created: 2024-02-06
updated: 2026-10-10
tags:
  - organization
  - structure
semantic_relations:
  - type: relates
    links:
      - "[[docs/repo_docs/ai_file_organization|File organization]]"
      - "[[docs/repo_docs/folder_structure|Folder structure]]"
---

# Folders beyond the documentation tree

[[docs/repo_docs/folder_structure|Folder structure]] covers `code/`, `docs/`,
and `knowledge_base/`. These are the other top-level entries and what each is
for. The repository has no embedding store, research-artifact directory, or
interactive-learning directory; the only machine-generated documents are the
knowledge-base catalogs.

| Entry | Purpose | Tracked |
|---|---|---|
| `.obsidian/` | Obsidian vault settings (core plugins, graph view settings, one community plugin entry, `3d-graph`) | yes |
| `.github/workflows/quality.yml` | The CI workflow: tests, static checks, documentation gates, manuscript build, mutation gate, packaging | yes |
| `.aii/config.yaml` | InstituteOS sidecar metadata (repository name, category, ecosystem relations) | yes |
| `.cursorrules` | Editor rules for AI assistants, mirroring the folder-level documentation requirements | yes |
| `output/` | Generated run output | no (git-ignored) |
| `graft/` | Local code-graph cache, regenerable | no (git-ignored) |
| `.venv/`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, `.coverage` | Local environment and tool caches | no |
| `cognitive_active_inference.egg-info/` | Editable-install metadata | no |

The link checkers (`verify_links.py`, `check_markdown_links.py`) skip `.git`,
virtual environments, tool caches, and any directory named `output` or
`outputs`, so generated files there are never validated as documentation.

## Related documentation

- [[docs/repo_docs/ai_file_organization|File organization]]
- [[docs/repo_docs/folder_structure|Folder structure]]
- [[docs/repo_docs/ai_semantic_processing|Semantic structure]]
