---
title: Documentation standards
type: guide
status: stable
created: 2024-02-12
updated: 2026-10-10
tags:
  - documentation
  - standards
  - guidelines
semantic_relations:
  - type: implements
    links:
      - "[[docs/repo_docs/ai_documentation_style|Documentation style]]"
  - type: relates
    links:
      - "[[docs/repo_docs/package_documentation|Package documentation]]"
      - "[[docs/implementation/implementation_guides|Implementation guides]]"
---

# Documentation standards

The repository is a Python package and an Obsidian vault of linked Markdown.
These standards cover the Markdown. The gates that enforce them are
`code/scripts/validate_docs.py` and its companions; see
[[docs/repo_docs/validation|Validation]].

## Where documentation lives

| Location | Content |
|---|---|
| `knowledge_base/` | Theory: concepts, mathematics, biology, philosophy, ontology, citations. Each domain has a generated `catalog.md`. |
| `docs/api/` | API reference and examples for the `cognitive` package. |
| `docs/guides/`, `docs/examples/`, `docs/implementation/` | How-to material, worked examples, and implementation notes. |
| `docs/research/` | Research notes and architectures. |
| `docs/templates/` | Node and document templates (see [[docs/templates/template_guide|Template guide]]). |
| `docs/tools/` | Notes on the repository's tooling. |
| `docs/manuscript/` | The executable manuscript, built by `cognitive-build-manuscript`. |
| `docs/policy/` | Machine-readable documentation policy (forbidden terms). |
| `docs/repo_docs/` | These repository-wide standards. |
| `code/**/README.md`, `AGENTS.md` | Per-folder overview and technical notes. |

Most folders carry a `README.md` (overview and navigation) and an `AGENTS.md`
(technical notes for agents working in it); some leaf folders, such as
`code/tools/src/models` and `docs/research/*`, have only one or neither. The
folder map is in
[[docs/repo_docs/folder_structure|Folder structure]] and
[`project_structure.md`](../../project_structure.md).

## Page format

Every page starts with YAML frontmatter, then one H1.

```yaml
---
title: Example page
type: concept
status: stable
created: 2024-02-12
updated: 2026-10-10
tags:
  - example
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/cognitive/active_inference|Active Inference]]"
---
```

- `title`, `type`, and `status` are expected on every page. The `type` values in
  use include `concept`, `mathematical_concept`, `guide`, `agents`,
  `documentation`, `reference`, `index`, `citation`, `learning_path`, and
  `template`.
- `semantic_relations` links must be quoted strings. An unquoted `[[link]]`
  parses as a nested YAML list, so Obsidian no longer sees it as a link.
  `normalize_frontmatter.py --write` repairs drift and `validate_docs.py`
  rejects it.
- Template values such as `unique_identifier`, `timestamp`, or `lorem_ipsum`
  left in frontmatter fail validation.
- The frontmatter block must be the first thing in the file. A page that opens
  with an H1 is not recognized as having frontmatter.

## Writing

- Describe what exists. Do not document tools, classes, flags, or directories
  that are not in the repository.
- Prefer short paragraphs, descriptive headings, and tables for reference data.
- State the assumptions behind a number.
- Cite sources as links into `knowledge_base/citations/` or as BibTeX keys in
  `docs/manuscript/references.bib`.
- Terms listed in `docs/policy/documentation_terms.yaml` may not appear in any
  tracked text file.

## Links

Wiki links (`[[target]]`, `[[target|text]]`, `[[target#heading]]`) are the
internal link syntax; standard Markdown links (`[text](path.md)`) are also
checked. Resolution rules and the checkers are described in
[[docs/repo_docs/obsidian_linking|Obsidian linking]] and
[[docs/repo_docs/linking_validation|Link validation]].

## Code blocks

- Label every fence with a language (`python`, `bash`, `yaml`, `text`).
- `validate_docs.py` syntax-checks `python` and `py` fences. A fence that
  claims to use `cognitive` or `Things` must also run; run the examples in
  [[docs/api/api_reference|the API reference]] against the tree whenever you
  change them.
- Show shell commands exactly as they run from the repository root.

## Diagrams

Mermaid fences are used throughout the vault (about 150 pages) and render in
Obsidian and on GitHub. Keep diagrams small enough to read, label edges that
carry meaning, and put the explanation in text as well, since the diagram source
is not validated. PlantUML is not used here.

## Review checklist

1. Frontmatter parses, is first in the file, and has quoted relation links.
2. Every named path, command, flag, and class exists.
3. `python` fences compile, and repository examples run.
4. Links resolve; no forbidden terms.
5. For `knowledge_base/` pages, `python code/scripts/kb_index.py --write`
   regenerates the catalogs (never edit a `catalog.md` by hand).

## Related documentation

- [[docs/repo_docs/ai_documentation_style|Documentation style]]
- [[docs/repo_docs/naming_conventions|Naming conventions]]
- [[docs/repo_docs/content_management|Content management]]
- [[docs/repo_docs/package_documentation|Package documentation]]
