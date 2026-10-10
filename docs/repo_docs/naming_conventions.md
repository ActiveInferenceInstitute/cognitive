---
title: Naming conventions
type: guide
status: stable
created: 2024-02-06
updated: 2026-10-10
tags:
  - conventions
  - naming
  - standards
semantic_relations:
  - type: implements
    links:
      - "[[docs/repo_docs/documentation_standards|Documentation standards]]"
  - type: relates
    links:
      - "[[docs/repo_docs/folder_structure|Folder structure]]"
      - "[[docs/repo_docs/ai_file_organization|File organization]]"
---

# Naming conventions

Conventions as practised in the tree. The only automated check on names is link
resolution: a wiki link to a misnamed page fails or is silently treated as a
concept (see [[docs/repo_docs/obsidian_linking|Obsidian linking]]).

## Documentation files

- Markdown pages are lowercase `snake_case`, such as `active_inference.md` or
  `free_energy_principle.md`. Use singular nouns for concept pages.
- `README.md` and `AGENTS.md` are the capitalized exceptions, one pair per
  folder. `CHANGELOG.md`, `CLAUDE.md`, `LICENSE`, and `REVIEW_LOG_YYYY-MM-DD.md`
  sit at the root.
- A handful of older pages keep capitals or spaces (for example
  `knowledge_base/systems/Social-Ecological Systems.md`, `SWR_Swarm.md`, and the
  `Simple_POMDP_README.md` pages under `code/Things/`). Do not add new ones; if
  you rename one, fix every link to it, since the repository is case-sensitive on
  Linux.
- Generated files are named by their generator: `catalog.md` (from `kb_index.py`).
  Do not hand-create a file with that name.
- Template files are `<node_type>_template.md` when `cognitive-create-node` is to
  render them (`agent_template.md`, `belief_template.md`, `goal_template.md`,
  `action_template.md`, `observation_template.md`, `environment_template.md`,
  and others in `docs/templates/`).

## Folders

- Documentation and knowledge-base folders are lowercase `snake_case`
  (`free_energy_principle`, `learning_paths`); `BioFirm` is the one capitalized
  knowledge-base domain.
- Agent folders under `code/Things/` use capitalized words joined by
  underscores (`Simple_POMDP`, `Continuous_Generic`) because the folder is the
  importable package name.
- Group by subject, not by file type or date.

## Python

- Modules and functions: `snake_case`. Classes: `PascalCase`
  (`ActiveInferenceModel`, `DiscreteGenerativeModel`). Constants: `UPPER_CASE`.
- Do not prefix names with non-semantic adjectives such as `enhanced_`, `new_`,
  or `improved_`. Words that carry meaning (`continuous_time`, `hierarchical_`)
  are fine.
- Test files are `test_<subject>.py`; test functions are `test_<behaviour>`.
- Console commands are `cognitive-<verb>` and map to `module:main` in
  `[project.scripts]`.
- Matrices follow the package convention: `A` likelihood, `B` transition, `C`
  log-preferences, `D` state prior, `E` action prior.

## Frontmatter and links

- Frontmatter keys are lowercase (`title`, `type`, `status`, `created`,
  `updated`, `tags`, `semantic_relations`).
- Wiki-link targets use the path from the repository root without the suffix,
  for example `[[knowledge_base/cognitive/active_inference|Active Inference]]`.
  A bare name such as `[[active_inference]]` resolves by file stem anywhere in
  the vault, but is ambiguous when two pages share a stem (`README` and
  `AGENTS` always do), so use the path form for those.
- Heading anchors in `[[page#heading]]` links are matched against the slugified
  heading text (lowercase, punctuation removed, spaces to hyphens).

## Related documentation

- [[docs/repo_docs/documentation_standards|Documentation standards]]
- [[docs/repo_docs/folder_structure|Folder structure]]
- [[docs/repo_docs/ai_file_organization|File organization]]
