---
title: Document Linking Template
type: template
status: stable
created: 2024-02-06
updated: 2026-10-10
tags:
  - template
  - linking
  - documentation
semantic_relations:
  - type: implements
    links:
      - "[[docs/repo_docs/linking_completeness|Linking completeness]]"
  - type: extends
    links:
      - "[[docs/repo_docs/ai_documentation_style|AI documentation style]]"
---

# Document Linking Template

A template for the links a page should carry. Copy the sections you need and
replace every example target with a page that exists. The conventions are
described in [[docs/repo_docs/linking_patterns|Obsidian linking patterns]] and
[[docs/repo_docs/linking_completeness|Linking completeness]].

## Frontmatter links

`semantic_relations` links are quoted strings. An unquoted `[[...]]` parses as
a nested YAML list and is no longer a link.

```yaml
---
title: Document Title
type: concept
status: draft
created: YYYY-MM-DD
updated: YYYY-MM-DD
semantic_relations:
  - type: implements
    links:
      - "[[path/to/base_concept|Base concept]]"
  - type: extends
    links:
      - "[[path/to/parent_concept|Parent concept]]"
  - type: relates
    links:
      - "[[path/to/related_page_1]]"
      - "[[path/to/related_page_2]]"
---
```

Relation types in use across the knowledge base include `relates`,
`implements`, `extends`, `foundation`, `organizes`, `documents` and `uses`.

## Link sections in the body

### Concept pages

```markdown
## Concept links

- Parent concept: [[parent_concept]]
- Implementation: [[implementation]]
- Validation: [[validation_spec]]
- Examples: [[usage_examples]]
```

### Implementation pages

```markdown
## Implementation links

- Interface: [[api_spec]]
- Tests: [[test_suite]]
- Examples: [[implementation_examples]]
```

### Hierarchy

```markdown
## Hierarchy

- Parent: [[parent_concept]]
- Children:
  - [[child_concept_1]]
  - [[child_concept_2]]
- Siblings:
  - [[sibling_1]]
  - [[sibling_2]]
```

### Documentation and research

```markdown
## Documentation

- Guide: [[user_guide]]
- API: [[api_reference]]
- Examples: [[example_collection]]

## Research

- Papers: [[research_papers]]
- Experiments: [[experiment_results]]
- Analysis: [[data_analysis]]
```

## Link syntax

```markdown
[[filename]]
[[filename|Display text]]
[[filename#Section name]]
[[folder/filename]]
```

## Checking links

- `cognitive-verify-links` scans the repository's Markdown files. By default an
  unresolved extensionless target counts as a concept link;
  `cognitive-verify-links --strict-wiki-links` requires every wiki link to
  resolve to a file.
- `cognitive-normalize-frontmatter --check` lists pages whose
  `semantic_relations` links are not quoted; `--write` fixes them.
- `cognitive-validate-docs` runs the full documentation gate.

## Best practices

- Group related links under a heading and keep the structure consistent.
- Prefer path-qualified targets (`folder/filename`) when a short name could
  match more than one page.
- Add a link back from the target page when the relation is mutual.
- Remove links to pages that no longer exist.

## Related documentation

- [[docs/repo_docs/linking_completeness|Linking completeness]]
- [[docs/repo_docs/linking_patterns|Obsidian linking patterns]]
- [[docs/repo_docs/documentation_standards|Documentation standards]]
- [[docs/templates/template_guide|Template guide]]
